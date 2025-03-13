import json
import openai
import numpy as np
from pinecone import Pinecone
from fuzzywuzzy import fuzz  # For fuzzy string matching
from fuzzywuzzy import process
from dotenv import load_dotenv
import os
import streamlit as st

# --------------------------
# 🔑 Configuration
# --------------------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
EMBED_MODEL = st.secrets["EMBED_MODEL"] if "EMBED_MODEL" in st.secrets else "text-embedding-3-small"

DEFAULT_CONFIDENCE_SCORE = 0.9

# ✅ Initialize Pinecone & OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
openai.api_key = OPENAI_API_KEY

# --------------------------
# 🔍 1️⃣ Find Closest Wine Match
# --------------------------
def find_wine_by_name(wine_name, min_confidence_score=DEFAULT_CONFIDENCE_SCORE):
    """
    Find the closest wine match in Pinecone using vector search (semantic filter) + fuzzy matching on names.
    """
    print(f"\n🔎 Searching for wine name: {wine_name}")

    # Step 1: Removed Fetch by ID (not relevant for name-based fuzzy search in this data structure)

    # Step 2: Vector Search for semantic pre-filtering
    print(f"🔍 Using vector search for semantic filtering...")
    matches = index.query(
        vector=generate_embedding(wine_name),
        top_k=10, # Adjust top_k as needed
        include_metadata=True
    )

    if not matches["matches"]:
        print("❌ No semantically related wines found via embedding.")
        return None

    # Step 3: Apply Fuzzy Matching on Wine Names (Metadata)
    print(f"✨ Applying fuzzy matching on wine names...")
    best_match, best_score = None, 0
    for match in matches["matches"]:
        metadata_name = match["metadata"].get("name", "")
        score = fuzz.token_sort_ratio(wine_name.lower(), metadata_name.lower()) # Using token_sort_ratio

        print(f"🔍 Comparing '{wine_name}' ↔ '{metadata_name}' → Fuzzy Score (Token Sort Ratio): {score}")

        if score > best_score:
            best_score = score
            best_match = match

    if best_match and best_score >= min_confidence_score * 100:
        print(f"✅ Found match: {best_match['metadata']['name']} (Fuzzy Score: {best_score})")
        return best_match["metadata"]

    print("❌ No strong fuzzy match found after semantic filtering.")
    return None

# --------------------------
# 2️⃣ Generate OpenAI Embeddings
# --------------------------
def generate_embedding(text):
    """Generate an embedding using OpenAI. Returns a valid vector even on failure."""
    print(f"🔍 Generating embedding for: {text[:50]}...")  # Print first 50 chars

    # 🚨 Extra safety: Prevent empty or invalid inputs - Already handled in upsert, but good to keep here too.
    if not text or not isinstance(text, str) or text.strip() == "":
        text = "Generic wine description. No tasting notes available."

    try:
        response = openai.Embedding.create(input=[text.strip()[:3000]], model=EMBED_MODEL) # Truncate and strip input
        embedding = response["data"][0]["embedding"]
        if not embedding:
            raise ValueError("Empty embedding received!")
        return embedding
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return [0] * 1536  # Return a zero-vector if failed

# --------------------------
# 3️⃣ Generate RAG Explanation
# --------------------------
def generate_rag_explanation(base_wine, recommended_wine):
    """
    Use GPT-4 to explain why the recommended wine is similar.
    """
    print(f"📝 Generating RAG explanation for: {recommended_wine.get('name', 'Unknown')}")

    base_name = base_wine.get("name", "Unknown")
    rec_name = recommended_wine.get("name", "Unknown")
    base_reviews_json = base_wine.get("reviews", "[]") # Get JSON string, default to empty list string
    rec_reviews_json = recommended_wine.get("reviews", "[]") # Get JSON string, default to empty list string

    try:
        base_reviews = json.loads(base_reviews_json)
        rec_reviews = json.loads(rec_reviews_json)
    except json.JSONDecodeError:
        base_reviews = []
        rec_reviews = []

    # Format reviews for prompt -  take just the review text
    base_review_texts = [rev.get("review", "No review text") for rev in base_reviews]
    rec_review_texts = [rev.get("review", "No review text") for rev in rec_reviews]

    prompt = f"""
Imagine that you are a sommelier explaining wine recommendations to a customer.
The customer likes the wine: "{base_name}".
You are recommending: "{rec_name}" as a similar wine.

Here is some information to help you explain the similarity:

**"{base_name}" Details:**
- Category: {base_wine.get("category", "Unknown")}
- Country of Origin: {base_wine.get("country_of_origin", "Unknown")}
- Tasting Notes Snippets: {base_review_texts[:3]} # Showing up to 3 review snippets

**"{rec_name}" Details:**
- Category: {recommended_wine.get("category", "Unknown")}
- Country of Origin: {recommended_wine.get("country_of_origin", "Unknown")}
- Tasting Notes Snippets: {rec_review_texts[:3]} # Showing up to 3 review snippets

Explain in **2-3 sentences** why "{rec_name}" is a good recommendation for someone who likes "{base_name}". 
Focus on the *similarities* between the wines based on the provided details, especially tasting notes, category, country. 
Also include snippets of reviews as appropriate.
Be friendly and helpful, like a sommelier would be, but add details about the taste similarities.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # 🌟 Use gpt-3.5-turbo (cheaper) instead of gpt-4
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ RAG Explanation failed: {e}")
        return "This wine is recommended because it is similar in taste profile." # Generic fallback explanation

# --------------------------
# 4️⃣ Search for Similar Wines
# --------------------------
def search_similar_wines(base_wine, tasting_embedding, category_embedding=None, country_embedding=None, top_k=5, price_min=0.0, price_max=9999.0):
    """
    Search for similar wines using weighted embeddings and generate RAG explanations (AFTER price filter).
    """
    print(f"🔎 Searching for similar wines to: {base_wine['name']}")

    if not tasting_embedding:
        raise ValueError("❌ ERROR: tasting_embedding is missing! Cannot query Pinecone.")

    full_query_vector = np.array(tasting_embedding) * 0.75

    if category_embedding:
        full_query_vector += np.array(category_embedding) * 0.15
    if country_embedding:
        full_query_vector += np.array(country_embedding) * 0.1

    full_query_vector = full_query_vector.tolist()

    print(f"🔍 Querying Pinecone with vector size: {len(full_query_vector)}")

    query_result = index.query(
        vector=full_query_vector,
        top_k=top_k * 2, # Fetch more initially to filter later
        include_metadata=True
    )

    if "matches" not in query_result or not query_result["matches"]:
        print("❌ No matches found from Pinecone query.")
        return []

    ranked_results = []
    for match in query_result["matches"]:
        metadata, score = match["metadata"], match["score"]

        if base_wine["name"].lower() == metadata.get("name", "").lower():
            continue

        try:
            price = float(metadata.get("price", 0))
            if not (price_min <= price <= price_max):
                continue # ❌ Price Filter - Skip wines outside price range

            # ✅ Generate RAG Explanation AFTER Price Filter - Efficient!
            explanation = generate_rag_explanation(base_wine, metadata)
            metadata["rag_explanation"] = explanation
            metadata["score"] = score
            print(f"🔍 Found similar wine: {metadata.get('id', 'Unknown')} (Score: {score}) - RAG generated") # Indicate RAG generation

            ranked_results.append((score, metadata))

        except ValueError:
            continue

    ranked_results.sort(reverse=True, key=lambda x: x[0])
    return [r[1] for r in ranked_results[:top_k]]
    
# --------------------------
# 5️⃣ Main Recommendation Function
# --------------------------
def recommend_wines(wine_name, top_k=5, price_min=0.0, price_max=9999.0, min_confidence_score=DEFAULT_CONFIDENCE_SCORE, return_original_wine=False):
    """
    Find wines based on tasting profiles, generates RAG explanations, and optionally returns the original wine data.
    """
    print(f"\n🔎 Starting recommendation for: {wine_name}")

    base_wine = find_wine_by_name(wine_name, min_confidence_score)

    if not base_wine:
        print("❌ No base wine found.")
        return (None, None) if return_original_wine else None # Return tuple if requested

    tasting_embed = generate_embedding(base_wine.get("reviews", "No tasting notes available."))
    if not tasting_embed:
        return (base_wine, None) if return_original_wine else None # Return base wine and None recs

    category_embed = generate_embedding(base_wine.get("category", "")) if base_wine.get("category") else None
    country_embed = generate_embedding(base_wine.get("country_of_origin", "")) if base_wine.get("country_of_origin") else None

    recommendations = search_similar_wines(base_wine, tasting_embed, category_embed, country_embed, top_k, price_min, price_max)

    if return_original_wine:
        return base_wine, recommendations # Return both original wine and recommendations
    else:
        return recommendations # Return only recommendations
