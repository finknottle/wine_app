import json
import openai
import numpy as np
from pinecone import Pinecone
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd 
import time 
import re 

# --------------------------
# 🔑 Configuration & Initialization
# --------------------------
IS_LOCAL_TEST = os.getenv("ENV") == "test"
if IS_LOCAL_TEST:
    print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} - RUNNING IN LOCAL TEST MODE ---")
    load_dotenv() 
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
else: 
    print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} - RUNNING IN STREAMLIT CLOUD MODE (or non-test local) ---")
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME")
    EMBEDDING_MODEL = st.secrets.get("EMBED_MODEL", "text-embedding-3-small")

print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} - CONFIGURATION ---")
print(f"PINECONE_API_KEY Loaded: {'Yes' if PINECONE_API_KEY else 'NO - MISSING!'}")
print(f"OPENAI_API_KEY Loaded: {'Yes' if OPENAI_API_KEY else 'NO - MISSING!'}")
print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"--- END CONFIGURATION ---")

if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_INDEX_NAME:
    error_message = "🚨 Missing API keys or Pinecone index name in configuration."
    print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}")
    if not IS_LOCAL_TEST: st.error(error_message + " Please check Streamlit secrets.")
    st.stop() 

EMBEDDING_DIMENSION = 1536 
if EMBEDDING_MODEL == "text-embedding-3-large": EMBEDDING_DIMENSION = 3072

try:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    available_indexes = [idx_desc.name for idx_desc in pc.list_indexes()]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Available Pinecone indexes: {available_indexes}")
    if PINECONE_INDEX_NAME not in available_indexes:
        error_message = f"🚨 Pinecone index '{PINECONE_INDEX_NAME}' not found. Available: {available_indexes}."
        print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}")
        if not IS_LOCAL_TEST: st.error(error_message); st.stop()
    index = pc.Index(PINECONE_INDEX_NAME) 
    index_stats = index.describe_index_stats()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone index stats: {index_stats}")
    if index_stats.dimension != EMBEDDING_DIMENSION:
        error_message = f"🚨 CRITICAL: Pinecone index dimension ({index_stats.dimension}) != script's EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION}). Fix config."
        print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}")
        if not IS_LOCAL_TEST: st.error(error_message); st.stop()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Initializing OpenAI client...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    print(f"✅ {time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone and OpenAI clients initialized successfully.")
except Exception as e:
    error_message = f"🚨 Failed to initialize API clients: {e}"
    print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}")
    if not IS_LOCAL_TEST: st.error(error_message); st.stop()

DEFAULT_CONFIDENCE_SCORE = 75 
MAX_SAME_PRODUCER_RECS = 1 
MAX_WINES_FROM_ANY_SINGLE_PRODUCER = 2 

@st.cache_data(ttl=3600) 
def get_all_wine_names():
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Attempting to fetch all wine names for autocomplete...")
    try:
        print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - `get_all_wine_names` is a placeholder. Implement efficient fetching.")
        if os.path.exists("wine_names.csv"): 
             print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Loading from wine_names.csv fallback...")
             df = pd.read_csv("wine_names.csv"); names = sorted(df["name"].dropna().unique().tolist())
             print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Loaded {len(names)} names from CSV.")
             return names
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - wine_names.csv not found. Autocomplete empty.")
        return [] 
    except Exception as e: print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - Error fetching wine names: {e}"); return [] 

def get_field(metadata, field_key, default="N/A"):
    val = metadata.get(field_key, default); return val if val is not None else default

def get_review_snippets_for_prompt(reviews_json_str, num_snippets=2):
    snippets = []
    try:
        reviews_list = json.loads(reviews_json_str)
        if isinstance(reviews_list, list):
            for rev in reviews_list[:num_snippets]:
                if isinstance(rev, dict) and rev.get("review"): snippets.append(f"\"{rev['review'][:200]}...\"") 
    except json.JSONDecodeError: print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - Could not parse reviews_json_str: {reviews_json_str[:100]}"); pass 
    return "; ".join(snippets) if snippets else "No detailed reviews available."

def extract_brand_from_name_heuristic(wine_name_str):
    if not wine_name_str or not isinstance(wine_name_str, str): return ""
    name_parts = wine_name_str.split(); non_brand_keywords = ["the", "le", "la", "les", "domaine", "chateau", "clos", "mas", "campo", "vina", "bodega", "cellers", "reserve", "cuvee", "old", "vine", "estate", "vineyard", "selection", "grand", "cru", "brut", "sec", "appellation", "contrôlée", "controlée", "cabernet", "sauvignon", "merlot", "chardonnay", "pinot", "noir", "gris", "grigio", "syrah", "shiraz", "zinfandel", "riesling", "malbec", "grenache", "blend", "rhône"]; year_prefix_indicators = ["de", "von", "van", "di"]; potential_brand = ""
    if not name_parts: return ""
    start_index = 0
    if re.match(r"^\d{4}$", name_parts[0]): start_index = 1
    brand_words_collected = []
    for i in range(start_index, len(name_parts)):
        part = name_parts[i]
        if any(part.startswith(prefix) for prefix in ["'", "\"", "“", "”"]): break
        if part.lower() in non_brand_keywords:
            if not brand_words_collected and part.lower() in ["domaine", "chateau", "clos", "mas", "vina", "bodega"]: brand_words_collected.append(part); continue
            elif brand_words_collected: break
            else: continue 
        if part[0].isupper() or part.lower().startswith("st.") or part.lower().startswith("st-"): brand_words_collected.append(part)
        elif brand_words_collected: break
        if len(brand_words_collected) >= 3: break
    potential_brand = " ".join(brand_words_collected)
    if potential_brand.endswith("'s"): potential_brand = potential_brand[:-2]
    elif potential_brand.endswith("'"): potential_brand = potential_brand[:-1]
    potential_brand = potential_brand.strip().rstrip(',')
    if len(potential_brand) < 2 and not potential_brand.isdigit(): return ""
    if potential_brand.lower() in non_brand_keywords and len(potential_brand.split()) == 1 : return ""
    if potential_brand.lower() in [term.lower() for term in ["rhone", "paso robles", "napa", "valley", "red", "white", "rose"]]: return ""
    return potential_brand.lower() 

def create_text_for_query_embedding(wine_metadata, *, include_identity: bool = True):
    """Build query text for embeddings.

    Key fix for KLWines-style catalogs:
    - If we include *too much identity* (producer/name/vintage), nearest neighbors
      become the same bottle across vintages.
    - For recommendation retrieval we want more *style-level* similarity.

    Set include_identity=False to down-weight exact producer/name matching.
    """

    if not wine_metadata or not isinstance(wine_metadata, dict):
        return "No wine data available."

    parts = []
    wine_name_str = wine_metadata.get("name")

    if include_identity and wine_name_str:
        parts.append(f"Wine: {wine_name_str}.")

    # Producer is helpful, but can dominate; only include when include_identity=True
    if include_identity:
        brand_for_embedding = get_field(wine_metadata, "brand")
        if brand_for_embedding == "N/A" or not brand_for_embedding:
            brand_for_embedding = extract_brand_from_name_heuristic(wine_name_str)
        if brand_for_embedding:
            parts.append(f"Producer: {brand_for_embedding}.")

    if wine_metadata.get("category"):
        parts.append(f"Type/Category: {wine_metadata['category']}.")
    if wine_metadata.get("country_of_origin"):
        parts.append(f"Origin: {wine_metadata['country_of_origin']}.")
    if wine_metadata.get("size"):
        parts.append(f"Bottle Size: {wine_metadata['size']}.")

    description_text = wine_metadata.get("description")
    if description_text:
        parts.append(f"\nTasting notes:\n{description_text}")

    keywords_list = wine_metadata.get("keywords_list")
    if keywords_list and isinstance(keywords_list, list):
        parts.append(f"\nTags/Keywords: {', '.join(keywords_list)}.")

    # Keep review text minimal (1–2 snippets) to reduce noise
    reviews_section_for_embedding = []
    reviews_json_str = wine_metadata.get("reviews_json", "[]")
    try:
        reviews_data = json.loads(reviews_json_str)
        if isinstance(reviews_data, list):
            for review_item in reviews_data[:2]:
                if isinstance(review_item, dict) and review_item.get("review"):
                    txt = str(review_item.get("review"))
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if len(txt) > 260:
                        txt = txt[:260].rstrip() + "…"
                    author = review_item.get("author", "Critic")
                    rating = review_item.get("rating")
                    if rating is not None:
                        reviews_section_for_embedding.append(f"Critic ({author}) {rating}: {txt}")
                    else:
                        reviews_section_for_embedding.append(f"Critic ({author}): {txt}")
    except Exception:
        pass

    if reviews_section_for_embedding:
        parts.append("\nCritic notes:\n" + "\n".join(reviews_section_for_embedding))

    full_text = "\n".join(filter(None, parts))
    full_text = "\n".join([line.strip() for line in full_text.splitlines() if line.strip()])
    if not full_text.strip():
        return f"Basic wine entry for {wine_metadata.get('name', 'Unknown Wine')}."
    return full_text

def generate_embedding(text_to_embed):
    if not text_to_embed or not isinstance(text_to_embed, str) or not text_to_embed.strip():
        print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - Input text for embedding is empty. Using default.")
        text_to_embed = "Generic item description."
    try:
        response = client.embeddings.create(input=[text_to_embed], model=EMBEDDING_MODEL)
        embedding = response.data[0].embedding
        if not embedding or len(embedding) != EMBEDDING_DIMENSION: raise ValueError("Invalid embedding.")
        return embedding
    except Exception as e: 
        print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - Error generating embedding: {e}.")
        return [0.0] * EMBEDDING_DIMENSION 

def find_wine_by_name(user_wine_name_input, min_confidence_score=DEFAULT_CONFIDENCE_SCORE): 
    print(f"\n🔎 {time.strftime('%Y-%m-%d %H:%M:%S')} - Finding base wine for: '{user_wine_name_input}'")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Step 1: Generating embedding for user input name...")
    input_name_embedding = generate_embedding(user_wine_name_input)
    if np.all(np.array(input_name_embedding) == 0):
        print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - Failed to generate embedding for input name.")
        st.warning("Could not process your wine name effectively.")
        return None
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Input name embedding generated.")
    print(f"🔍 {time.strftime('%Y-%m-%d %H:%M:%S')} - Step 2: Querying Pinecone for candidates...")
    try:
        candidate_matches_response = index.query(vector=input_name_embedding, top_k=20, include_metadata=True)
        candidate_matches_list = candidate_matches_response.get("matches", [])
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone query returned {len(candidate_matches_list)} candidates.")
    except Exception as e:
        print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone query for candidates failed: {e}")
        st.error(f"Search failed while fetching candidates: {e}")
        return None
    if not candidate_matches_list: print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - No candidates by Pinecone."); return None
    print(f"✨ {time.strftime('%Y-%m-%d %H:%M:%S')} - Step 3: Fuzzy matching on {len(candidate_matches_list)} candidates...")
    best_match_details = None; highest_fuzzy_score = 0
    choices = {m["id"]: m["metadata"]["name"] for m in candidate_matches_list if m and m.get("metadata") and m["metadata"].get("name")}
    if not choices: print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - No candidates with names for fuzzy matching."); return None
    extracted_result = process.extractOne(user_wine_name_input.lower(), {k: v.lower() for k, v in choices.items()}, scorer=fuzz.token_sort_ratio)
    if extracted_result:
        _, score, match_id_key_from_choices = extracted_result 
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FuzzyWuzzy best candidate: '{choices.get(match_id_key_from_choices)}' (Pinecone ID: {match_id_key_from_choices}) with score {score}")
        if score >= min_confidence_score: 
            highest_fuzzy_score = score
            for m_obj in candidate_matches_list: 
                if m_obj["id"] == match_id_key_from_choices: 
                    best_match_details = m_obj["metadata"]
                    if best_match_details is not None: best_match_details["pinecone_id"] = m_obj.get("id") 
                    break
    if best_match_details: print(f"✅ {time.strftime('%Y-%m-%d %H:%M:%S')} - Fuzzy match found: '{best_match_details.get('name')}' (Score: {highest_fuzzy_score}), Pinecone ID: {best_match_details.get('pinecone_id')}")
    else: print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - No strong fuzzy match (>{min_confidence_score}) found.")
    return best_match_details 

def generate_enhanced_rag_explanation(base_wine_metadata, recommended_wine_metadata):
    base_name = get_field(base_wine_metadata, "name", "The Selected Wine"); rec_name = get_field(recommended_wine_metadata, "name", "This Recommendation") 
    base_brand_for_prompt = get_field(base_wine_metadata, "brand"); rec_brand_for_prompt = get_field(recommended_wine_metadata, "brand")
    if base_brand_for_prompt == "N/A" or not base_brand_for_prompt : base_brand_for_prompt = extract_brand_from_name_heuristic(base_name); 
    if not base_brand_for_prompt: base_brand_for_prompt = "N/A"
    if rec_brand_for_prompt == "N/A" or not rec_brand_for_prompt: rec_brand_for_prompt = extract_brand_from_name_heuristic(rec_name); 
    if not rec_brand_for_prompt: rec_brand_for_prompt = "N/A"
    base_info_for_prompt = (f"Category: {get_field(base_wine_metadata, 'category')}, Origin: {get_field(base_wine_metadata, 'country_of_origin')}, Producer: {base_brand_for_prompt}, Description: {get_field(base_wine_metadata, 'description', 'No desc.')[:300]}..., Reviews: {get_review_snippets_for_prompt(base_wine_metadata.get('reviews_json', '[]'))}")
    rec_info_for_prompt = (f"Category: {get_field(recommended_wine_metadata, 'category')}, Origin: {get_field(recommended_wine_metadata, 'country_of_origin')}, Producer: {rec_brand_for_prompt}, Description: {get_field(recommended_wine_metadata, 'description', 'No desc.')[:300]}..., Reviews: {get_review_snippets_for_prompt(recommended_wine_metadata.get('reviews_json', '[]'))}")
    prompt = f"""
You are an expert AI Sommelier. Your task is to provide insightful comparisons between two wines.
**Base Wine (User's Choice):** "{base_name}" (Details: {base_info_for_prompt})
**Recommended Wine:** "{rec_name}" (Details: {rec_info_for_prompt})
**Instructions:**
1.  **Base Wine Blurb:** In 1-2 sentences, summarize what the user likely enjoys about "{base_name}", based on its details (e.g., "You seem to enjoy wines that are [key characteristic 1], with notes of [key flavor/aroma], and a [key mouthfeel/body] character."). Focus on positive attributes inferable from the provided details.
2.  **Recommended Wine Blurb:** In 1-2 sentences, explain why "{rec_name}" is a good recommendation for someone who likes "{base_name}". Specifically highlight 1-2 key characteristics of "{rec_name}" that align with the likely preferences derived from "{base_name}" (e.g., "If you like {base_name}'s [specific quality like 'bold fruit' or 'crisp acidity'], you'll appreciate that {rec_name} also offers [similar/complementary specific quality like 'a similar rich dark fruit core' or 'a refreshing citrus zest'], making it an excellent alternative to explore."). Be specific about the shared or complementary appeal.
3.  **Detailed Comparison:** For this part, ONLY provide the Markdown formatted detailed comparison based on the categories below. Do NOT include any introductory text before the first H4 heading for this section. For each characteristic, describe the **Recommended Wine ("{rec_name}")** and how its attributes might appeal to someone who enjoys the **Base Wine ("{base_name}")**. If information for the recommended wine is not directly available, infer it where possible based on its category, origin, and reviews, or state 'Typically, wines like this offer...' or 'Not specified'. Use Markdown H4 headings (####) for each characteristic.
    #### 🍬 Sweetness
    (Describe "{rec_name}"'s perceived sweetness. How does this relate to "{base_name}"?)
    #### 🍋 Acidity
    (Describe "{rec_name}"'s acidity. Would this appeal to a fan of "{base_name}"?)
    #### 🧱 Tannins
    (For reds: Describe "{rec_name}"'s tannins. How might these appeal to a fan of "{base_name}"?)
    #### 🍇 Fruity Flavor Profile
    (Describe dominant fruit notes in "{rec_name}". Are these similar or complementary to "{base_name}"?)
    #### ⚖️ Body
    (Describe "{rec_name}"'s body. How does this align with "{base_name}"?)
    #### 👃 Nose/Aroma Profile
    (Describe "{rec_name}"'s key aromas. Would these be appreciated by someone who likes "{base_name}"?)
    #### 🎨 Color
    (Describe "{rec_name}"'s likely color. If not inferable, state 'Color not specified, but typical for its style.')
    #### 🌍 Origin & Style Context
    (Comment on how "{rec_name}"'s origin or style relates to or complements "{base_name}".)
**Output Format:**
You MUST respond with exactly three parts, separated by "||END_SECTION||":
BASE_WINE_BLURB: [Your 1-2 sentence blurb for the base wine here]||END_SECTION||
RECOMMENDED_WINE_BLURB: [Your 1-2 sentence blurb for the recommended wine here]||END_SECTION||
DETAILED_COMPARISON_MARKDOWN: [Your Markdown for the detailed comparison starting directly with '#### 🍬 Sweetness' and containing ONLY the detailed comparison content here]
"""
    explanation_dict = {"base_wine_blurb": f"Enjoying {base_name}? Here's what makes it special based on its profile.", "recommended_wine_blurb": f"{rec_name} is a great pick if you like {base_name}, offering similar enjoyment.", "detailed_comparison_markdown": "Detailed comparison notes are being crafted..."}
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.4, max_tokens=950)
        full_response_text = response.choices[0].message.content.strip()
        parts = full_response_text.split("||END_SECTION||")
        if len(parts) == 3:
            base_blurb_text = parts[0].replace("BASE_WINE_BLURB:", "").strip(); rec_blurb_text = parts[1].replace("RECOMMENDED_WINE_BLURB:", "").strip(); detailed_markdown_text = parts[2].replace("DETAILED_COMPARISON_MARKDOWN:", "").strip()
            if base_blurb_text: explanation_dict["base_wine_blurb"] = base_blurb_text
            if rec_blurb_text: explanation_dict["recommended_wine_blurb"] = rec_blurb_text
            if detailed_markdown_text: explanation_dict["detailed_comparison_markdown"] = detailed_markdown_text
        else:
            print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - LLM format unexpected. Fallback. Parts: {len(parts)}. Resp: {full_response_text[:200]}...")
            explanation_dict["detailed_comparison_markdown"] = "**AI Notes:**\n" + full_response_text
    except Exception as e: print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - RAG Expl. failed: {e}")
    return explanation_dict

# Function to generate only the base wine blurb (to reduce initial load time)
def generate_base_wine_blurb(base_wine_metadata):
    if not base_wine_metadata: return "Could not determine details for your selected wine."
    
    base_name = get_field(base_wine_metadata, "name", "The Selected Wine")
    base_brand_for_prompt = get_field(base_wine_metadata, "brand")
    if base_brand_for_prompt == "N/A" or not base_brand_for_prompt : 
        base_brand_for_prompt = extract_brand_from_name_heuristic(base_wine_metadata.get("name", ""))
        if not base_brand_for_prompt: base_brand_for_prompt = "N/A" 

    base_info_for_prompt = (
        f"Category: {get_field(base_wine_metadata, 'category')}, "
        f"Origin: {get_field(base_wine_metadata, 'country_of_origin')}, "
        f"Producer: {base_brand_for_prompt}, "
        f"Description: {get_field(base_wine_metadata, 'description', 'No description.')[:200]}..., " 
        f"Reviews: {get_review_snippets_for_prompt(base_wine_metadata.get('reviews_json', '[]'), num_snippets=1)}" 
    )
    
    blurb_prompt = f"""
Based on the following details for the wine "{base_name}":
{base_info_for_prompt}

In 1-2 concise sentences, summarize what someone likely enjoys about this wine. Focus on its key positive attributes. Start with "You seem to enjoy wines that...".
Example: "You seem to enjoy wines that are full-bodied and fruit-forward, with hints of oak and a smooth finish."
"""
    default_blurb = f"You likely enjoy {base_name} for its general characteristics and profile." 
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": blurb_prompt}],
            temperature=0.5, max_tokens=100
        )
        blurb_text = response.choices[0].message.content.strip()
        return blurb_text if blurb_text else default_blurb
    except Exception as e:
        print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - Failed to generate base wine blurb: {e}")
        return default_blurb

def _normalize_cuvee_key(name: str) -> str:
    """Normalize a wine name to avoid recommending the same cuvée across vintages.

    Example: "2017 Booker 'Fracture' Paso Robles Syrah" and
             "2018 Booker 'Fracture' Paso Robles Syrah" -> same key.

    Heuristic only; ok if imperfect.
    """
    if not name:
        return ""
    s = name.lower()
    # drop vintage
    s = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", s)
    # normalize quotes
    s = s.replace("\"", " ").replace("'", " ").replace("’", " ")
    # drop punctuation-ish
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def search_similar_wines(base_wine_metadata, top_k=5, price_min=0.0, price_max=999999.0, 
                         max_same_producer_as_base=MAX_SAME_PRODUCER_RECS, 
                         max_per_any_producer=MAX_WINES_FROM_ANY_SINGLE_PRODUCER): 
    print(f"🔎 {time.strftime('%Y-%m-%d %H:%M:%S')} - Searching similar to: '{base_wine_metadata.get('name', 'Unknown')}'")
    # Style-first query to avoid returning the same producer/bottle across vintages.
    query_text = create_text_for_query_embedding(base_wine_metadata, include_identity=False)
    query_vector = generate_embedding(query_text)
    if np.all(np.array(query_vector) == 0): st.error("Could not generate search profile."); return []
    pinecone_filter = {}; price_filters = []
    try: p_min = float(price_min); p_max = float(price_max)
    except: p_min = 0.0; p_max = 999999.0 
    if p_min > 0: price_filters.append({"price": {"$gte": p_min}})
    if p_max < 999999.0: price_filters.append({"price": {"$lte": p_max}})
    if price_filters: pinecone_filter["$and"] = price_filters
    if pinecone_filter: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone filter: {pinecone_filter}")
    else: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - No price filter.")
    query_result = None
    # Fetch a lot more than top_k to avoid "same bottle across vintages" dominating the head.
    # KLWines-style catalogs often have many near-duplicates.
    num_candidates_to_fetch = max(top_k * 50, 250)
    try:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Querying Pinecone for {num_candidates_to_fetch} candidates...")
        query_result = index.query(vector=query_vector, filter=pinecone_filter if pinecone_filter else None, top_k=num_candidates_to_fetch, include_metadata=True)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone query found {len(query_result.get('matches', []))} raw matches.")
    except Exception as e: st.error(f"Search failed: {e}"); print(f"❌ {time.strftime('%Y-%m-%d %H:%M:%S')} - Pinecone query failed: {e}"); return []
    if not query_result or "matches" not in query_result or not query_result["matches"]: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - No matches from Pinecone."); return []
    
    final_recommendations = []
    base_wine_name_lower = base_wine_metadata.get("name", "").lower()
    base_cuvee_key = _normalize_cuvee_key(base_wine_metadata.get("name", ""))
    seen_cuvee_keys = set([base_cuvee_key]) if base_cuvee_key else set()
    base_wine_brand_from_meta = base_wine_metadata.get("brand")
    base_producer_for_diversity = extract_brand_from_name_heuristic(base_wine_metadata.get("name", "")) 
    if base_wine_brand_from_meta and isinstance(base_wine_brand_from_meta, str) and base_wine_brand_from_meta.strip() and base_wine_brand_from_meta != "N/A":
        base_producer_for_diversity = base_wine_brand_from_meta.lower()
    elif base_producer_for_diversity: base_producer_for_diversity = base_producer_for_diversity.lower()
    else: base_producer_for_diversity = "" 

    processed_ids = set(); same_producer_as_base_count = 0; producer_counts_in_recs = {}

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Processing {len(query_result['matches'])} candidates for diversity (top_k={top_k}, max_same_as_base={max_same_producer_as_base}, max_per_any={max_per_any_producer})...")
    print(f"--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Base Wine For Diversity: Name='{base_wine_metadata.get('name')}', Effective Producer='{base_producer_for_diversity}' ---")

    for match_idx, match in enumerate(query_result["matches"]):
        if len(final_recommendations) >= top_k: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Reached {top_k} recs."); break 
        if not match or not match.get("metadata"): print(f"⚠️ {time.strftime('%Y-%m-%d %H:%M:%S')} - Candidate {match_idx+1} malformed."); continue 
        
        metadata = match["metadata"]; score = match.get("score", 0.0); pinecone_vector_id = match.get("id")
        current_wine_name = metadata.get("name", ""); current_wine_name_lower = current_wine_name.lower()
        candidate_brand_from_meta = metadata.get("brand")
        candidate_producer_for_diversity = extract_brand_from_name_heuristic(current_wine_name) 
        if candidate_brand_from_meta and isinstance(candidate_brand_from_meta, str) and candidate_brand_from_meta.strip() and candidate_brand_from_meta != "N/A":
            candidate_producer_for_diversity = candidate_brand_from_meta.lower()
        elif candidate_producer_for_diversity: candidate_producer_for_diversity = candidate_producer_for_diversity.lower()
        else: candidate_producer_for_diversity = ""

        if not pinecone_vector_id or pinecone_vector_id in processed_ids:
            continue
        if base_wine_name_lower == current_wine_name_lower:
            continue

        # Prevent "same wine across vintages" spam.
        cuvee_key = _normalize_cuvee_key(current_wine_name)
        if cuvee_key and cuvee_key in seen_cuvee_keys:
            continue
        
        print(f"--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Eval Candidate {match_idx + 1}: '{current_wine_name}' (Score: {score:.4f}) ---")
        print(f"    Metadata Brand: '{candidate_brand_from_meta}', Heuristic Brand: '{extract_brand_from_name_heuristic(current_wine_name)}', Effective Candidate Producer: '{candidate_producer_for_diversity}'")
        
        is_same_as_base_producer = False
        if base_producer_for_diversity and candidate_producer_for_diversity and base_producer_for_diversity == candidate_producer_for_diversity:
            is_same_as_base_producer = True
            print(f"    Identified as SAME producer as base by direct brand/heuristic match: '{base_producer_for_diversity}'")
        # NOTE: We intentionally avoid substring "name contains producer" matching here.
    # It was too aggressive with the KLWines dataset and could collapse diversity,
    # yielding only 1-2 recommendations.
        
        if is_same_as_base_producer:
            if same_producer_as_base_count >= max_same_producer_as_base:
                print(f"    SKIPPED (Same Producer as Base Limit): '{current_wine_name}'. Limit {max_same_producer_as_base} reached for base's producer.")
                continue
            if candidate_producer_for_diversity and producer_counts_in_recs.get(candidate_producer_for_diversity, 0) >= max_per_any_producer:
                print(f"    SKIPPED (General Producer Limit for a same-as-base producer): '{current_wine_name}' from producer '{candidate_producer_for_diversity}'. General limit {max_per_any_producer} reached for this producer in recommendations.")
                continue
            same_producer_as_base_count += 1
            if candidate_producer_for_diversity: producer_counts_in_recs[candidate_producer_for_diversity] = producer_counts_in_recs.get(candidate_producer_for_diversity, 0) + 1
            print(f"    ACCEPTED (Same Producer as Base): '{current_wine_name}'. Base producer count: {same_producer_as_base_count}/{max_same_producer_as_base}. This producer in recs: {producer_counts_in_recs.get(candidate_producer_for_diversity, 0)}/{max_per_any_producer}")
        else: 
            if candidate_producer_for_diversity and producer_counts_in_recs.get(candidate_producer_for_diversity, 0) >= max_per_any_producer:
                print(f"    SKIPPED (General Producer Limit): '{current_wine_name}' from producer '{candidate_producer_for_diversity}'. General limit {max_per_any_producer} reached.")
                continue
            else:
                if candidate_producer_for_diversity: producer_counts_in_recs[candidate_producer_for_diversity] = producer_counts_in_recs.get(candidate_producer_for_diversity, 0) + 1
                print(f"    ACCEPTED (Different Producer): '{current_wine_name}'. This producer in recs: {producer_counts_in_recs.get(candidate_producer_for_diversity,0)}/{max_per_any_producer}")

        metadata["pinecone_id"] = pinecone_vector_id
        metadata["similarity_score"] = score
        final_recommendations.append(metadata)
        processed_ids.add(pinecone_vector_id)
        if cuvee_key:
            seen_cuvee_keys.add(cuvee_key)
            
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Finished. Returning {len(final_recommendations)} recommendations.")
    return final_recommendations
    
def recommend_wines_for_streamlit(user_wine_name_input, top_k=5, price_min=0.0, price_max=999999.0, min_confidence_score=DEFAULT_CONFIDENCE_SCORE):
    print(f"\n🍇 {time.strftime('%Y-%m-%d %H:%M:%S')} - Starting recommendation for: '{user_wine_name_input}'")
    base_wine_metadata = find_wine_by_name(user_wine_name_input, min_confidence_score)
    if not base_wine_metadata:
        # Message is handled by app.py
        return None, [], None 
    
    print(f"✅ {time.strftime('%Y-%m-%d %H:%M:%S')} - Base wine identified: '{base_wine_metadata.get('name')}' (Pinecone ID: {base_wine_metadata.get('pinecone_id')})")
    
    # Base wine blurb generation is now deferred to app.py
    recommendations = search_similar_wines(base_wine_metadata, top_k=top_k, price_min=price_min, price_max=price_max)
    return base_wine_metadata, recommendations, None # Return None for base_wine_blurb, app.py will fetch it


def recommend_wines_from_prompt(
    prompt_text: str,
    top_k: int = 8,
    price_min: float = 0.0,
    price_max: float = 999999.0,
    max_per_any_producer: int = MAX_WINES_FROM_ANY_SINGLE_PRODUCER,
):
    """Recommend wines directly from a free-text prompt ("vibe" mode).

    This skips base-wine identification and instead queries Pinecone using an
    embedding of the user's prompt.

    Returns: (pseudo_base, recommendations)
    """
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        return None, []

    print(f"\n🍇 {time.strftime('%Y-%m-%d %H:%M:%S')} - Starting vibe recommendation for prompt: '{prompt_text[:120]}'")
    query_vector = generate_embedding(prompt_text)
    if np.all(np.array(query_vector) == 0):
        st.warning("Could not process that prompt. Try describing the wine in a different way.")
        return None, []

    pinecone_filter = {}
    price_filters = []
    try:
        p_min = float(price_min)
        p_max = float(price_max)
    except Exception:
        p_min = 0.0
        p_max = 999999.0

    if p_min > 0:
        price_filters.append({"price": {"$gte": p_min}})
    if p_max < 999999.0:
        price_filters.append({"price": {"$lte": p_max}})
    if price_filters:
        pinecone_filter["$and"] = price_filters

    num_candidates_to_fetch = max(top_k * 12, 50)
    try:
        query_result = index.query(
            vector=query_vector,
            filter=pinecone_filter if pinecone_filter else None,
            top_k=num_candidates_to_fetch,
            include_metadata=True,
        )
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None, []

    matches = (query_result or {}).get("matches") or []
    if not matches:
        return None, []

    # Diversity: cap per producer
    producer_counts_in_recs = {}
    processed_ids = set()
    final_recommendations = []

    for match in matches:
        if len(final_recommendations) >= top_k:
            break
        if not match or not match.get("metadata"):
            continue

        metadata = match["metadata"]
        pinecone_vector_id = match.get("id")
        if not pinecone_vector_id or pinecone_vector_id in processed_ids:
            continue

        current_wine_name = metadata.get("name", "")
        candidate_brand_from_meta = metadata.get("brand")
        candidate_producer_for_diversity = extract_brand_from_name_heuristic(current_wine_name)
        if candidate_brand_from_meta and isinstance(candidate_brand_from_meta, str) and candidate_brand_from_meta.strip() and candidate_brand_from_meta != "N/A":
            candidate_producer_for_diversity = candidate_brand_from_meta.lower()
        elif candidate_producer_for_diversity:
            candidate_producer_for_diversity = candidate_producer_for_diversity.lower()
        else:
            candidate_producer_for_diversity = ""

        if candidate_producer_for_diversity and producer_counts_in_recs.get(candidate_producer_for_diversity, 0) >= max_per_any_producer:
            continue

        if candidate_producer_for_diversity:
            producer_counts_in_recs[candidate_producer_for_diversity] = producer_counts_in_recs.get(candidate_producer_for_diversity, 0) + 1

        metadata["pinecone_id"] = pinecone_vector_id
        metadata["similarity_score"] = match.get("score", 0.0)
        final_recommendations.append(metadata)
        processed_ids.add(pinecone_vector_id)

    pseudo_base = {
        "name": f"Your vibe: {prompt_text}",
        "description": "A sommelier-style prompt (not a specific bottle).",
        "brand": "AI Somm",
    }
    return pseudo_base, final_recommendations

def _run_standalone_recommender_ui(): 
    st.set_page_config(page_title="🍷 Wine Recommender AI (Module Test)", layout="wide")
    st.title("🍇 AI Wine Recommender (Module Test)")
    with st.sidebar: st.header("Search Filters"); user_input_wine = st.text_input("Enter a wine name:", placeholder="e.g., Opus One"); price_range_val = st.slider("Price ($):", 0.0, 1000.0, (0.0, 500.0), 10.0)
    if user_input_wine and st.button("✨ Test Find Recommendations"):
        with st.spinner("Searching..."): 
            base_meta, rec_list, _ = recommend_wines_for_streamlit(user_input_wine, top_k=5, price_min=price_range_val[0], price_max=price_range_val[1], min_confidence_score=75)
            test_base_blurb = "Test: Base blurb would be generated on demand in main app."
            if base_meta:
                test_base_blurb = generate_base_wine_blurb(base_meta) # Generate for test display
        if base_meta:
            st.subheader(f"Base Wine: {base_meta.get('name', 'N/A')}"); st.markdown(f"**What you might like:** {test_base_blurb}"); st.json(base_meta) 
            if rec_list:
                st.subheader("Recommendations:")
                for idx, rec in enumerate(rec_list):
                    st.write(f"--- **{rec.get('name')}** (Score: {rec.get('similarity_score',0):.2f}) ---"); rec_pinecone_id = rec.get("pinecone_id", f"test_rec_{idx}") 
                    if f"rag_data_test_{rec_pinecone_id}" not in st.session_state: st.session_state[f"rag_data_test_{rec_pinecone_id}"] = None 
                    if st.button(f"Show AI Notes for {rec.get('name', 'this wine')}", key=f"btn_test_rag_{rec_pinecone_id}"):
                        if not st.session_state[f"rag_data_test_{rec_pinecone_id}"]: 
                            with st.spinner("Generating AI Sommelier notes..."): st.session_state[f"rag_data_test_{rec_pinecone_id}"] = generate_enhanced_rag_explanation(base_meta, rec)
                    if st.session_state[f"rag_data_test_{rec_pinecone_id}"]:
                        rag_data = st.session_state[f"rag_data_test_{rec_pinecone_id}"]; st.markdown(f"**Why try this?** {rag_data.get('recommended_wine_blurb', 'N/A')}"); st.markdown("**Detailed Comparison:**"); st.markdown(rag_data.get('detailed_comparison_markdown', "Not available."))
            else: st.info("No recommendations found.")
if __name__ == "__main__": print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Running wine_recommendation.py standalone for UI testing..."); _run_standalone_recommender_ui()
