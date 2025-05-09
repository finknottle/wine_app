import json
import openai
import numpy as np
from pinecone import Pinecone
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
from dotenv import load_dotenv
import os
import streamlit as st

# --------------------------
# 🔑 Configuration & Initialization
# --------------------------
# Load environment variables for local testing if ENV=test
if os.getenv("ENV") == "test":
    load_dotenv() 
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
else: 
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME")
    EMBEDDING_MODEL = st.secrets.get("EMBED_MODEL", "text-embedding-3-small")

if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_INDEX_NAME:
    st.error("🚨 Missing API keys or Pinecone index name in configuration. Please check your .env file or Streamlit secrets.")
    st.stop() 

EMBEDDING_DIMENSION = 1536 
if EMBEDDING_MODEL == "text-embedding-3-large":
    EMBEDDING_DIMENSION = 3072

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    available_indexes = [idx_desc.name for idx_desc in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in available_indexes:
        st.error(f"🚨 Pinecone index '{PINECONE_INDEX_NAME}' not found. Available: {available_indexes}.")
        st.stop()
    index = pc.Index(PINECONE_INDEX_NAME)
    
    index_stats = index.describe_index_stats()
    if index_stats.dimension != EMBEDDING_DIMENSION:
        st.error(f"🚨 CRITICAL: Pinecone index dimension ({index_stats.dimension}) != script's EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION}). Fix config.")
        st.stop()

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    print("✅ Pinecone and OpenAI clients initialized successfully.")
except Exception as e:
    st.error(f"🚨 Failed to initialize API clients: {e}")
    print(f"❌ Error initializing API clients: {e}")
    st.stop()

DEFAULT_CONFIDENCE_SCORE = 85 

# -------------------------------------
# ✨ Helper Functions (Module Level)
# -------------------------------------
def get_field(metadata, field_key, default="N/A"):
    """Safely gets a field from metadata dictionary."""
    val = metadata.get(field_key, default)
    return val if val is not None else default

def get_review_snippets_for_prompt(reviews_json_str, num_snippets=2):
    """Extracts review text snippets for use in LLM prompts."""
    snippets = []
    try:
        reviews_list = json.loads(reviews_json_str)
        if isinstance(reviews_list, list):
            for rev in reviews_list[:num_snippets]:
                if isinstance(rev, dict) and rev.get("review"):
                    # Truncate for prompt to keep it manageable
                    snippets.append(f"\"{rev['review'][:200]}...\"") 
    except json.JSONDecodeError: 
        print(f"⚠️  Could not parse reviews_json_str in get_review_snippets_for_prompt: {reviews_json_str[:100]}")
        pass 
    return "; ".join(snippets) if snippets else "No detailed reviews available."

# -------------------------------------
# ✨ Create Comprehensive Text for Embedding
# -------------------------------------
def create_text_for_query_embedding(wine_metadata):
    if not wine_metadata or not isinstance(wine_metadata, dict): return "No wine data available."
    parts = []
    if wine_metadata.get("name"): parts.append(f"Wine: {wine_metadata['name']}.")
    if wine_metadata.get("brand"): parts.append(f"Producer: {wine_metadata['brand']}.")
    if wine_metadata.get("category"): parts.append(f"Type/Category: {wine_metadata['category']}.")
    if wine_metadata.get("country_of_origin"): parts.append(f"Origin: {wine_metadata['country_of_origin']}.")
    if wine_metadata.get("size"): parts.append(f"Bottle Size: {wine_metadata['size']}.")
    description_text = wine_metadata.get("description")
    if description_text: parts.append(f"\nProduct Description:\n{description_text}")
    keywords_list = wine_metadata.get("keywords_list") 
    if keywords_list and isinstance(keywords_list, list): parts.append(f"\nTags/Keywords: {', '.join(keywords_list)}.")
    
    reviews_section_for_embedding = [] 
    reviews_json_str = wine_metadata.get("reviews_json", "[]")
    try:
        reviews_data = json.loads(reviews_json_str)
        if isinstance(reviews_data, list):
            for review_item in reviews_data: 
                if isinstance(review_item, dict) and review_item.get("review"):
                    review_text = f"- Reviewer: {review_item.get('author', 'N/A')}"
                    if review_item.get('rating') is not None: review_text += f", Score: {review_item['rating']}"
                    review_text += f": \"{review_item['review']}\"" 
                    reviews_section_for_embedding.append(review_text)
    except json.JSONDecodeError: print(f"⚠️ Could not parse reviews_json for embedding: {reviews_json_str}")
    
    if reviews_section_for_embedding: parts.append("\nExpert Reviews and Tasting Notes:\n" + "\n".join(reviews_section_for_embedding))
    else: parts.append("\nGeneral characteristics based on type and origin.")
    
    agg_rating_json_str = wine_metadata.get("aggregateRating_json", "{}")
    if not reviews_section_for_embedding: 
        try:
            agg_rating_data = json.loads(agg_rating_json_str)
            if isinstance(agg_rating_data, dict) and agg_rating_data.get("ratingValue") is not None:
                rating_text = f"Overall Rating: {agg_rating_data['ratingValue']}"
                if agg_rating_data.get("bestRating"): rating_text += f" out of {agg_rating_data['bestRating']}"
                if agg_rating_data.get("reviewCount"): rating_text += f" (based on {agg_rating_data['reviewCount']} reviews)."
                parts.append(rating_text)
        except json.JSONDecodeError: print(f"⚠️ Could not parse aggregateRating_json for embedding: {agg_rating_json_str}")
    
    full_text = "\n".join(filter(None, parts)) 
    full_text = "\n".join([line.strip() for line in full_text.splitlines() if line.strip()])
    if not full_text.strip(): return f"Basic wine entry for {wine_metadata.get('name', 'Unknown Wine')}."
    return full_text

# --------------------------
# 2️⃣ Generate OpenAI Embeddings
# --------------------------
def generate_embedding(text_to_embed):
    if not text_to_embed or not isinstance(text_to_embed, str) or not text_to_embed.strip():
        print("⚠️ Input text for embedding is empty. Using default.")
        text_to_embed = "Generic item description."
    try:
        response = client.embeddings.create(input=[text_to_embed], model=EMBEDDING_MODEL)
        embedding = response.data[0].embedding
        if not embedding or len(embedding) != EMBEDDING_DIMENSION: raise ValueError("Invalid embedding.")
        return embedding
    except Exception as e: 
        print(f"❌ Error generating embedding: {e}.")
        return [0.0] * EMBEDDING_DIMENSION 

# --------------------------
# 🔍 1️⃣ Find Closest Wine Match
# --------------------------
def find_wine_by_name(user_wine_name_input, min_confidence_score=DEFAULT_CONFIDENCE_SCORE):
    print(f"\n🔎 Finding base wine for: '{user_wine_name_input}'")
    input_name_embedding = generate_embedding(user_wine_name_input)
    if np.all(np.array(input_name_embedding) == 0):
        st.warning("Could not pre-filter effectively due to name embedding issue.")
        return None
    try:
        candidate_matches = index.query(vector=input_name_embedding, top_k=20, include_metadata=True)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None
    if not candidate_matches or not candidate_matches.get("matches"): return None
    best_match_details = None; highest_fuzzy_score = 0
    choices = {m["id"]: m["metadata"]["name"] for m in candidate_matches["matches"] if m and m.get("metadata") and m["metadata"].get("name")}
    if not choices: return None
    extracted_result = process.extractOne(user_wine_name_input.lower(), {k: v.lower() for k, v in choices.items()}, scorer=fuzz.token_sort_ratio)
    if extracted_result:
        _, score, match_id_key = extracted_result
        if score >= min_confidence_score: 
            highest_fuzzy_score = score
            for m_obj in candidate_matches["matches"]:
                if m_obj["id"] == match_id_key: best_match_details = m_obj["metadata"]; break
    if best_match_details: print(f"✅ Fuzzy match: '{best_match_details['name']}' (Score: {highest_fuzzy_score})")
    else: print(f"❌ No strong fuzzy match (>{min_confidence_score}) found.")
    return best_match_details 

# --------------------------
# 3️⃣ Generate Enhanced RAG Explanation (Structured Comparison)
# --------------------------
def generate_enhanced_rag_explanation(base_wine_metadata, recommended_wine_metadata):
    """
    Generates a structured explanation including blurbs and detailed comparison.
    Returns a dictionary.
    """
    base_name = get_field(base_wine_metadata, "name", "The Selected Wine") 
    rec_name = get_field(recommended_wine_metadata, "name", "This Recommendation") 
    
    base_info_for_prompt = (
        f"Category: {get_field(base_wine_metadata, 'category')}, "
        f"Origin: {get_field(base_wine_metadata, 'country_of_origin')}, "
        f"Producer: {get_field(base_wine_metadata, 'brand')}, "
        f"Description: {get_field(base_wine_metadata, 'description', 'No description.')[:300]}..., "
        f"Reviews: {get_review_snippets_for_prompt(base_wine_metadata.get('reviews_json', '[]'))}"
    )
    rec_info_for_prompt = (
        f"Category: {get_field(recommended_wine_metadata, 'category')}, "
        f"Origin: {get_field(recommended_wine_metadata, 'country_of_origin')}, "
        f"Producer: {get_field(recommended_wine_metadata, 'brand')}, "
        f"Description: {get_field(recommended_wine_metadata, 'description', 'No description.')[:300]}..., "
        f"Reviews: {get_review_snippets_for_prompt(recommended_wine_metadata.get('reviews_json', '[]'))}"
    )

    prompt = f"""
You are an expert AI Sommelier. Your task is to provide insightful comparisons between two wines.

**Base Wine (User's Choice):** "{base_name}"
Details: {base_info_for_prompt}

**Recommended Wine:** "{rec_name}"
Details: {rec_info_for_prompt}

**Instructions:**

1.  **Base Wine Blurb:** In 1-2 sentences, summarize what the user likely enjoys about "{base_name}", based on its details (e.g., "You seem to enjoy wines that are [key characteristic 1], with notes of [key flavor/aroma], and a [key mouthfeel/body] character."). Focus on positive attributes inferable from the provided details.

2.  **Recommended Wine Blurb:** In 1-2 sentences, explain why "{rec_name}" is a good recommendation for someone who likes "{base_name}". Specifically highlight 1-2 key characteristics of "{rec_name}" that align with the likely preferences derived from "{base_name}" (e.g., "If you like {base_name}'s [specific quality like 'bold fruit' or 'crisp acidity'], you'll appreciate that {rec_name} also offers [similar/complementary specific quality like 'a similar rich dark fruit core' or 'a refreshing citrus zest'], making it an excellent alternative to explore."). Be specific about the shared or complementary appeal.

3.  **Detailed Comparison:** For this part, ONLY provide the Markdown formatted detailed comparison based on the categories below. Do NOT include any introductory text before the first H4 heading for this section.
    For each characteristic, describe the **Recommended Wine ("{rec_name}")** and how its attributes might appeal to someone who enjoys the **Base Wine ("{base_name}")**. If information for the recommended wine is not directly available, infer it where possible based on its category, origin, and reviews, or state 'Typically, wines like this offer...' or 'Not specified'. Use Markdown H4 headings (####) for each characteristic.

    #### 🍬 Sweetness
    (Describe "{rec_name}"'s perceived sweetness (e.g., Dry, Off-Dry, hints of sweetness). How does this compare or relate to the style of "{base_name}"?)

    #### 🍋 Acidity
    (Describe "{rec_name}"'s acidity (e.g., Bright and crisp, mellow, balanced). Would this provide a similar refreshment or structure to "{base_name}"?)

    #### 🧱 Tannins
    (For reds primarily: Describe "{rec_name}"'s tannins (e.g., Smooth and velvety, firm and structured, subtle). How might these tannins appeal to a fan of "{base_name}"?)

    #### 🍇 Fruity Flavor Profile
    (Describe the dominant fruit notes in "{rec_name}" (e.g., Ripe dark fruits, fresh red berries, zesty citrus). Are these flavors in a similar family or complementary to those likely found in "{base_name}"?)

    #### ⚖️ Body
    (Describe "{rec_name}"'s body (e.g., Light-bodied and elegant, medium-bodied and versatile, full-bodied and rich). How does this align with the expected body of "{base_name}"?)

    #### 👃 Nose/Aroma Profile
    (Describe "{rec_name}"'s key aromas beyond fruit (e.g., Floral notes, spicy undertones, earthy complexity, oak influences). Would these aromas be appreciated by someone who likes "{base_name}"?)

    #### 🎨 Color
    (Describe "{rec_name}"'s likely color based on its type/category (e.g., 'A vibrant ruby red typical of...', 'A pale lemon-green hue suggesting...'). If not inferable, state 'Color not specified, but typical for its style.')
    
    #### 🌍 Origin & Style Context
    (Comment on how "{rec_name}"'s origin or style (e.g., 'Classic Bordeaux style', 'New World fruit expression') relates to or complements the style of "{base_name}".)

**Output Format:**
You MUST respond with exactly three parts, separated by "||END_SECTION||":
BASE_WINE_BLURB: [Your 1-2 sentence blurb for the base wine here]||END_SECTION||
RECOMMENDED_WINE_BLURB: [Your 1-2 sentence blurb for the recommended wine here]||END_SECTION||
DETAILED_COMPARISON_MARKDOWN: [Your Markdown for the detailed comparison starting directly with '#### 🍬 Sweetness' and containing ONLY the detailed comparison content here]
"""
    explanation_dict = {
        "base_wine_blurb": f"Based on its profile, you likely enjoy {base_name} for its general characteristics.",
        "recommended_wine_blurb": f"{rec_name} is suggested as it shares some similar qualities with {base_name}.",
        "detailed_comparison_markdown": "A detailed comparison could not be generated at this time."
    }
    try:
        response = client.chat.completions.create( 
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, 
            max_tokens=950 # Slightly increased max_tokens for safety
        )
        full_response_text = response.choices[0].message.content.strip()
        
        parts = full_response_text.split("||END_SECTION||")
        if len(parts) == 3:
            base_blurb_text = parts[0].replace("BASE_WINE_BLURB:", "").strip()
            rec_blurb_text = parts[1].replace("RECOMMENDED_WINE_BLURB:", "").strip()
            # The third part is now expected to be purely the Markdown for detailed comparison
            detailed_markdown_text = parts[2].replace("DETAILED_COMPARISON_MARKDOWN:", "").strip()


            if base_blurb_text: explanation_dict["base_wine_blurb"] = base_blurb_text
            if rec_blurb_text: explanation_dict["recommended_wine_blurb"] = rec_blurb_text
            if detailed_markdown_text: explanation_dict["detailed_comparison_markdown"] = detailed_markdown_text
        else:
            print(f"⚠️ LLM response format unexpected. Parts found: {len(parts)}. Full response: {full_response_text[:500]}... Using fallback for RAG.")
            # If parsing fails, put the whole response into detailed comparison as a fallback,
            # hoping the user can still make sense of it or it contains some of the markdown.
            explanation_dict["detailed_comparison_markdown"] = "**AI Sommelier Notes (Unable to parse structured response):**\n\n" + full_response_text

    except Exception as e:
        print(f"⚠️ RAG Explanation generation failed: {e}")
    
    return explanation_dict


# --------------------------
# 4️⃣ Search for Similar Wines
# --------------------------
def search_similar_wines(base_wine_metadata, top_k=5, price_min=0.0, price_max=999999.0):
    print(f"🔎 Searching for wines similar to: '{base_wine_metadata.get('name', 'Unknown')}'")
    query_text = create_text_for_query_embedding(base_wine_metadata)
    query_vector = generate_embedding(query_text)
    if np.all(np.array(query_vector) == 0):
        st.error("Could not generate a search profile for your selected wine.")
        return []
    pinecone_filter = {}; price_filters = []
    try: p_min = float(price_min); p_max = float(price_max)
    except: p_min = 0.0; p_max = 999999.0 
    if p_min > 0: price_filters.append({"price": {"$gte": p_min}})
    if p_max < 999999.0: price_filters.append({"price": {"$lte": p_max}})
    if price_filters: pinecone_filter["$and"] = price_filters
    if pinecone_filter: print(f"Applying Pinecone filter: {pinecone_filter}")
    try:
        query_result = index.query(vector=query_vector, filter=pinecone_filter if pinecone_filter else None, top_k=top_k + 5, include_metadata=True)
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []
    if not query_result or "matches" not in query_result or not query_result["matches"]: return []
    ranked_results = []; base_wine_name_lower = base_wine_metadata.get("name", "").lower(); processed_ids = set() 
    for match in query_result["matches"]:
        if not match or not match.get("metadata"): continue 
        metadata = match["metadata"]; score = match.get("score", 0.0); match_id = match.get("id")
        if not match_id or match_id in processed_ids: continue 
        if base_wine_name_lower == metadata.get("name", "").lower(): continue
        
        explanation_data = generate_enhanced_rag_explanation(base_wine_metadata, metadata)
        metadata["rag_explanation_data"] = explanation_data 
        metadata["similarity_score"] = score 
        ranked_results.append(metadata) 
        processed_ids.add(match_id)
        if len(ranked_results) >= top_k: break 
    return ranked_results
    
# --------------------------
# 5️⃣ Main Recommendation Function
# --------------------------
def recommend_wines_for_streamlit(user_wine_name_input, top_k=5, price_min=0.0, price_max=999999.0, min_confidence_score=DEFAULT_CONFIDENCE_SCORE):
    print(f"\n🍇 Starting recommendation for: '{user_wine_name_input}'")
    base_wine_metadata = find_wine_by_name(user_wine_name_input, min_confidence_score)
    if not base_wine_metadata:
        st.warning(f"Sorry, couldn't find a close match for '{user_wine_name_input}'. Try a different name or check spelling.")
        return None, [], None 
    print(f"✅ Base wine identified: '{base_wine_metadata.get('name')}'")

    base_wine_blurb_prompt = f"""
Based on the following details for the wine "{get_field(base_wine_metadata, 'name', 'Unknown')}":
- Category: {get_field(base_wine_metadata, 'category')}
- Origin: {get_field(base_wine_metadata, 'country_of_origin')}
- Producer: {get_field(base_wine_metadata, 'brand')}
- Description: {get_field(base_wine_metadata, 'description', 'No description.')[:300]}...
- Reviews: {get_review_snippets_for_prompt(base_wine_metadata.get('reviews_json', '[]'))}

In 1-2 sentences, summarize what someone likely enjoys about this wine. Focus on positive attributes inferable from the provided details. Start with "You seem to enjoy wines that...".
"""
    base_wine_blurb = f"You likely enjoy {get_field(base_wine_metadata, 'name', 'this wine')} for its general characteristics and profile." 
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": base_wine_blurb_prompt}],
            temperature=0.5, max_tokens=100
        )
        blurb_text = response.choices[0].message.content.strip()
        if blurb_text: base_wine_blurb = blurb_text
    except Exception as e:
        print(f"⚠️ Failed to generate base wine blurb: {e}")

    recommendations = search_similar_wines(base_wine_metadata, top_k=top_k, price_min=price_min, price_max=price_max)
    return base_wine_metadata, recommendations, base_wine_blurb 

# --------------------------
# 🖥️ Streamlit UI (Example - for standalone testing)
# --------------------------
def _run_standalone_recommender_ui(): 
    st.set_page_config(page_title="🍷 Wine Recommender AI (Module Test)", layout="wide")
    st.title("🍇 AI Wine Recommender (Module Test)")
    with st.sidebar:
        st.header("Search Filters")
        user_input_wine = st.text_input("Enter a wine name:", placeholder="e.g., Opus One")
        num_recs = st.slider("Recommendations:", 1, 10, 3)
        price_range_val = st.slider("Price ($):", 0.0, 1000.0, (0.0, 500.0), 10.0)
        conf = st.slider("Name Match Confidence (%):", 50, 100, DEFAULT_CONFIDENCE_SCORE, 5)
    if user_input_wine and st.button("✨ Test Find Recommendations"):
        with st.spinner("Searching..."):
            base_meta, rec_list, base_blurb = recommend_wines_for_streamlit(user_input_wine, num_recs, price_range_val[0], price_range_val[1], conf)
        if base_meta:
            st.subheader(f"Base Wine: {base_meta.get('name', 'N/A')}")
            st.markdown(f"**What you might like:** {base_blurb}")
            st.json(base_meta) 
            if rec_list:
                st.subheader("Recommendations:")
                for rec in rec_list:
                    st.write(f"--- **{rec.get('name')}** (Score: {rec.get('similarity_score',0):.2f}) ---")
                    rag_data = rec.get("rag_explanation_data", {})
                    st.markdown(f"**Why try this?** {rag_data.get('recommended_wine_blurb', 'N/A')}")
                    st.markdown("**Detailed Comparison:**")
                    st.markdown(rag_data.get('detailed_comparison_markdown', "Not available."))
            else: st.info("No recommendations found.")
if __name__ == "__main__":
    print("Running wine_recommendation.py standalone for UI testing...")
    _run_standalone_recommender_ui()
