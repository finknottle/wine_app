import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# --------------------------
# 🔑 Configuration (Load from .env)
# --------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HTML_FILES_DIR = "../klwines_pipeline/klwines_details_html" # 🌟 Path to your HTML files directory

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_wine_names_from_html_filenames(index, html_dir):
    """
    Fetches wine names from Pinecone using IDs extracted from HTML filenames.
    """
    print("Fetching wine names from Pinecone using HTML filenames...")
    wine_names = set()
    pinecone_ids_to_fetch = []

    # 1. Extract IDs from HTML filenames
    print(f"Scanning directory: {html_dir} to extract wine IDs...")
    if not os.path.isdir(html_dir):
        print(f"Error: Directory not found: {html_dir}")
        return []

    filenames = [f for f in os.listdir(html_dir) if f.endswith(".html")]
    extracted_ids = []
    for filename in filenames:
        base_name = filename[:-5] # Remove ".html" extension
        if base_name.isdigit(): # Check if it's a digit (to be safe)
            extracted_ids.append(base_name)
        else:
            print(f"Warning: Skipping non-numeric filename: {filename}")

    print(f"Extracted {len(extracted_ids)} wine IDs from HTML filenames.")

    # 2. Construct Pinecone vector IDs
    pinecone_ids_to_fetch = [f"wine_{id_str}" for id_str in extracted_ids]
    print(f"Constructed {len(pinecone_ids_to_fetch)} Pinecone vector IDs.")

    # 3. Fetch vectors from Pinecone using these IDs (in batches)
    fetched_count = 0
    batch_size = 100
    for i in range(0, len(pinecone_ids_to_fetch), batch_size):
        id_batch = pinecone_ids_to_fetch[i:i + batch_size]
        print(f"Fetching batch of Pinecone IDs: {id_batch[0]} to {id_batch[-1]}...")
        fetch_response = index.fetch(
            ids=id_batch,
            # include_metadata=True
        )

        if fetch_response.vectors:
            fetched_count += len(fetch_response.vectors)
            print(f"  Fetched {len(fetch_response.vectors)} vectors in this batch.")
            for vector in fetch_response.vectors.values():
                name = vector.metadata.get("name")
                if name:
                    wine_names.add(name)
        else:
            print("  No vectors fetched in this batch.")

    print(f"Total vectors fetched using HTML filenames: {fetched_count}")
    print(f"Fetched {len(wine_names)} unique wine names.")
    return sorted(list(wine_names))


def save_wine_names_to_csv(wine_names, csv_path="wine_names.csv"):
    """Saves wine names to CSV."""
    df = pd.DataFrame({"name": wine_names})
    df.to_csv(csv_path, index=False)
    print(f"Wine names saved to: {csv_path}")


if __name__ == "__main__":
    wine_names = get_wine_names_from_html_filenames(index, HTML_FILES_DIR)
    if wine_names:
        save_wine_names_to_csv(wine_names)
        print("✅ Successfully generated wine_names.csv")
    else:
        print("❌ Could not fetch wine names from Pinecone.")