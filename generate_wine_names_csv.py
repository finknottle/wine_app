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

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_all_wine_names_from_pinecone(index):
    """
    Fetches all wine names from Pinecone index metadata using index.query() with ROBUST pagination.
    """
    print("Fetching all wine names from Pinecone using index.query() with ROBUST pagination...")
    wine_names = set()
    all_matches = []

    top_k = 1000  # 🌟 Stay within top_k limit
    next_page_token = None

    page_number = 0 # Track page number for debugging
    while True:
        page_number += 1
        print(f"--- Page {page_number} ---")
        print(f"Fetching vectors (top_k={top_k}, pagination_token={next_page_token})...")

        query_response = index.query(
            vector=[0.0] * 1536,
            top_k=top_k,
            include_metadata=True,
            filter={},
            namespace=None,
            pagination_token=next_page_token
        )

        current_matches = query_response.matches or [] # Handle case where matches might be None/empty
        all_matches.extend(current_matches)
        print(f"  Received {len(current_matches)} matches in this batch.")


        if query_response.pagination: # 🌟 Robust check for pagination info
            next_page_token = query_response.pagination.next
            print(f"  Pagination info found. Next page token: {next_page_token}")
            if not next_page_token: # 🌟 Double check for None next_page_token (end of pagination)
                print("  next_page_token is None within pagination info. Ending pagination.")
                break
        else: # 🌟 No pagination info in response
            next_page_token = None
            print("  No pagination info in response. Ending pagination.")
            break # If no pagination info, assume no more pages

        if not next_page_token: # 🌟 Final check after potentially getting token from pagination info
            print("  next_page_token is None after processing pagination info. Ending pagination.")
            break


    print(f"Total matches received from Pinecone: {len(all_matches)}")

    if not all_matches:
        print("No matches (vectors) fetched from Pinecone query.")
        return []

    for match in all_matches:
        name = match.metadata.get("name")
        if name:
            wine_names.add(name)

    print(f"Fetched {len(wine_names)} unique wine names.")
    return sorted(list(wine_names))


def save_wine_names_to_csv(wine_names, csv_path="wine_names.csv"):
    """Saves a list of wine names to a CSV file."""
    df = pd.DataFrame({"name": wine_names})
    df.to_csv(csv_path, index=False)
    print(f"Wine names saved to: {csv_path}")


if __name__ == "__main__":
    wine_names = get_all_wine_names_from_pinecone(index)
    if wine_names:
        save_wine_names_to_csv(wine_names)
        print("✅ Successfully generated wine_names.csv")
    else:
        print("❌ Could not fetch wine names from Pinecone.")