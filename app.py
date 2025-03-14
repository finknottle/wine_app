import streamlit as st
import pandas as pd
import os
import json
from wine_recommendation import recommend_wines  # Assuming this is your backend module

#######################################
# 1️⃣ Load Wine Names for Autocomplete
#######################################
def load_wine_names(csv_path="wine_names.csv"):
    """Load unique wine names from CSV for autocomplete."""
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    return sorted(df["name"].dropna().unique().tolist())

#######################################
# 2️⃣ Helper function to display wine details in a card
#######################################
def display_wine_card(wine_data):
    """Displays wine details in a card format."""
    if not wine_data:
        st.error("Wine details not available.")
        return

    with st.container():
        # No images for now
        # col1, col2 = st.columns([0.3, 0.7])  # Adjust ratio for image and text
        # with col1:
        #     image_url = wine_data.get('image')
        #     if image_url:
        #         st.image(image_url, use_column_width=True)
        #     else:
        #         st.image("wine_placeholder.png", caption="No Image Available", use_column_width=True) # Replace with your placeholder

        # Modified for text-only card
        st.markdown(f"**🍷 {wine_data['name']}**", unsafe_allow_html=True)
        st.markdown(f"<small>{wine_data.get('country_of_origin', 'N/A')} - {wine_data.get('category', 'N/A')}</small>", unsafe_allow_html=True)
        st.markdown(f"**Price:** ${wine_data.get('price', 'N/A')}")

        rating_json_str = wine_data.get('aggregateRating')
        rating = None

        if rating_json_str:
            try:
                json_string_level_1_decoded = json.loads(rating_json_str)
                try:
                    rating = json.loads(json_string_level_1_decoded)
                except json.JSONDecodeError:
                    rating = None
            except json.JSONDecodeError:
                rating = None

        if rating and isinstance(rating, dict):
            rating_value = rating.get('ratingValue', 'N/A')
            rating_count = rating.get('ratingCount', 'N/A')
            st.markdown(f"**Rating:** {rating_value} ({rating_count} reviews)")
        else:
            st.markdown("**Rating:** N/A")

        description = wine_data.get('description', 'No description available.')
        if len(description) > 150:  # Show a snippet on the card
            st.markdown(f"<small>{description[:150]}...</small>", unsafe_allow_html=True) # Removed Read More link for now
        else:
            st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)

        rag_explanation = wine_data.get('rag_explanation')
        if rag_explanation:
            sentences = rag_explanation.split('. ')
            first_two_sentences = '. '.join(sentences[:2]) + '...' if len(sentences) > 2 else rag_explanation
            remaining_explanation = '. '.join(sentences[2:]) if len(sentences) > 2 else ""
            with st.expander("🤔 Why similar"):
                st.markdown(f"<small>{first_two_sentences} <br> {remaining_explanation}</small>", unsafe_allow_html=True)
        else:
            st.markdown("<small>No explanation available for why this wine is similar.</small>", unsafe_allow_html=True)

        reviews_json_str = wine_data.get('reviews')
        if reviews_json_str and reviews_json_str != "[]":
            try:
                reviews = json.loads(reviews_json_str)
                if reviews:
                    tasting_notes = "\n".join([review.get('review', '') for review in reviews])
                    if tasting_notes:
                        with st.expander("📝 Tasting Notes"):
                            st.markdown(f"<small>{tasting_notes}</small>", unsafe_allow_html=True)
            except json.JSONDecodeError:
                st.warning("Could not display tasting notes due to formatting issues.")

#######################################
# 3️⃣ Helper function to display full wine details with reviews
#######################################
def display_full_wine_details(wine_data, header_level=3):
    """Displays full wine details, including reviews."""
    if not wine_data:
        st.error("Wine details not available.")
        return

    st.markdown(f"<h{header_level} id='wine-details-{wine_data.get('name').replace(' ', '-')}' style='padding-top: 20px;'>🍷 {wine_data['name']}</h{header_level}>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Country:** {wine_data.get('country_of_origin', 'N/A')}")
        st.write(f"**Category:** {wine_data.get('category', 'N/A')}")
        st.write(f"**Price:** ${wine_data.get('price', 'N/A')}")

        rating_json_str = wine_data.get('aggregateRating')
        rating = None

        if rating_json_str:
            try:
                json_string_level_1_decoded = json.loads(rating_json_str)
                try:
                    rating = json.loads(json_string_level_1_decoded)
                except json.JSONDecodeError:
                    rating = None
            except json.JSONDecodeError:
                rating = None

        if rating and isinstance(rating, dict):
            rating_value = rating.get('ratingValue', 'N/A')
            rating_count = rating.get('ratingCount', 'N/A')
            st.write(f"**Rating:** {rating_value} ({rating_count} reviews)")
        else:
            st.write("**Rating:** N/A")

    with col2:
        description = wine_data.get('description', 'No description available.')
        st.write(f"**Description:** {description}")

    # Tasting notes are now handled in the card
    # reviews_json_str = wine_data.get('reviews')
    # if reviews_json_str and reviews_json_str != "[]":
    #     try:
    #         reviews = json.loads(reviews_json_str)
    #         if reviews:
    #             st.subheader("📝 Tasting Notes/Reviews")
    #             for review in reviews:
    #                 author = review.get('author', 'Unknown Author')
    #                 rating_value_review = review.get('rating', 'N/A')
    #                 review_text = review.get('review', 'No review text')

    #                 st.markdown(f"**{author} says:**")
    #                 st.write(f"Rating: {rating_value_review}")
    #                 st.write(f"> {review_text}")
    #                 st.write("---")
    #     except json.JSONDecodeError:
    #         st.warning("Could not display tasting notes due to formatting issues.")

#######################################
# 4️⃣ Streamlit App - Main function
#######################################
import json # Import JSON for handling reviews

def main():
    st.title("🍷 Wine Recommendation Engine")

    possible_wine_names = load_wine_names()

    st.subheader("🔍 Input Parameters")

    # 1️⃣ Wine Name Input
    wine_name = st.selectbox("Wine Name", options=possible_wine_names) if possible_wine_names else st.text_input("Wine Name")

    # 2️⃣ Price Range
    min_price, max_price = st.slider("Price Range", 0, 1000, (0, 1000))

    # 3️⃣ Number of Recommendations
    top_k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    # 4️⃣ Confidence Score
    confidence_score = st.slider("Confidence Score (Min)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

    st.write("---")

    #######################################
    # Get Recommendations and Display
    #######################################
    if st.button("🍷 Get Recommendations"):
        if not wine_name: # Simple validation
            st.error("Please enter or select a wine name.")
            return

        with st.spinner("🔄 Finding the best wines..."):
            original_wine, recommendations = recommend_wines( # Assuming recommend_wines now returns original wine & RAG explanation
                wine_name=wine_name,
                top_k=top_k,
                price_min=min_price,
                price_max=max_price,
                min_confidence_score=confidence_score,
                return_original_wine=True # Keep returning original wine
            )

        if not recommendations:
            st.error("❌ No recommendations found.")
        else:
            st.success(f"✅ Found {len(recommendations)} recommendations!")

            st.header("🔍 Original Wine")
            with st.container():
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    image_url = original_wine.get('image')
                    if image_url:
                        st.image(image_url, use_column_width=True)
                    else:
                        st.image("wine_placeholder.png", caption="No Image Available", use_column_width=True) # Replace with your placeholder
                with col2:
                    st.markdown(f"**🍷 {original_wine['name']}**", unsafe_allow_html=True)
                    st.markdown(f"<small>{original_wine.get('country_of_origin', 'N/A')} - {original_wine.get('category', 'N/A')}</small>", unsafe_allow_html=True)
                    st.markdown(f"**Price:** ${original_wine.get('price', 'N/A')}")
                    rating_json_str = original_wine.get('aggregateRating')
                    rating = None
                    if rating_json_str:
                        try:
                            json_string_level_1_decoded = json.loads(rating_json_str)
                            try:
                                rating = json.loads(json_string_level_1_decoded)
                            except json.JSONDecodeError:
                                rating = None
                        except json.JSONDecodeError:
                            rating = None
                    if rating and isinstance(rating, dict):
                        rating_value = rating.get('ratingValue', 'N/A')
                        rating_count = rating.get('ratingCount', 'N/A')
                        st.markdown(f"**Rating:** {rating_value} ({rating_count} reviews)")
                    else:
                        st.markdown("**Rating:** N/A")
                    st.markdown(f"<details><summary>📖 Description</summary><small>{original_wine.get('description', 'No description available.')}</small></details>", unsafe_allow_html=True)


            st.header("✨ Recommended Wines")
            for i, wine in enumerate(recommendations):
                with st.container():
                    display_wine_card(wine)
                    st.write("---") # Separator between recommendations

            # We are now displaying tasting notes in the card, so we can remove this section or keep it for full details
            # st.subheader("More Details:")
            # for wine in recommendations:
            #     display_full_wine_details(wine)


#######################################
# 5️⃣ Run Streamlit
#######################################
if __name__ == "__main__":
    st.set_page_config(page_title="Wine Recommender", layout="wide")
    main()


# import streamlit as st
# import pandas as pd
# import os
# import json
# from wine_recommendation import recommend_wines  # Assuming this is your backend module

# #######################################
# # 1️⃣ Load Wine Names for Autocomplete
# #######################################
# def load_wine_names(csv_path="wine_names.csv"):
#     """Load unique wine names from CSV for autocomplete."""
#     if not os.path.exists(csv_path):
#         return []
#     df = pd.read_csv(csv_path)
#     return sorted(df["name"].dropna().unique().tolist())

# #######################################
# # 2️⃣ Helper function to display wine details nicely
# #######################################
# def display_wine_details(wine_data, header_level=3, show_reviews=True):
#     """Displays wine details in a formatted way, including RAG explanation - Double JSON Decode."""
#     if not wine_data:
#         st.error("Wine details not available.")
#         return

#     st.markdown(f"<h{header_level}>🍷 {wine_data['name']}</h{header_level}>", unsafe_allow_html=True)

#     col1, col2 = st.columns(2)  # Create two columns layout

#     with col1:
#         st.write(f"**Country:** {wine_data.get('country_of_origin', 'N/A')}")
#         st.write(f"**Category:** {wine_data.get('category', 'N/A')}")
#         st.write(f"**Price:** ${wine_data.get('price', 'N/A')}")

#         rating_json_str = wine_data.get('aggregateRating')
#         rating = None

#         if rating_json_str:
#             try:
#                 # First JSON decode - to get from double-JSON-encoded string to single-JSON-encoded string
#                 json_string_level_1_decoded = json.loads(rating_json_str)

#                 try:
#                     # Second JSON decode - to get from single-JSON-encoded string to Python dictionary
#                     rating = json.loads(json_string_level_1_decoded)
#                 except json.JSONDecodeError as e2:
#                     st.warning(f"❌ Could not parse aggregate rating JSON data (Level 2): {e2}")
#                     st.warning("Failing JSON String (Level 2):", json_string_level_1_decoded)
#                     rating = None

#             except json.JSONDecodeError as e1:
#                 st.warning(f"❌ Could not parse aggregate rating JSON data (Level 1): {e1}")
#                 st.warning("Failing JSON String (Level 1):", rating_json_str)
#                 rating = None

#         if rating and isinstance(rating, dict): # Check if rating is now a dict after parsing
#             rating_value = rating.get('ratingValue', 'N/A')
#             rating_count = rating.get('ratingCount', 'N/A')
#             rating_str = f"**Rating:** {rating_value} ({rating_count} reviews)" # Removed "/5"
#             st.write(rating_str)
#         else:
#             st.write("**Rating:** N/A")

#     with col2:
#         description = wine_data.get('description', 'No description available.')
#         if len(description) > 300:
#             with st.expander("📖 Description"):
#                 st.write(description)
#         else:
#             st.write(f"**Description:** {description}")

#     rag_explanation = wine_data.get('rag_explanation')
#     if rag_explanation:
#         with st.expander("🤔 Why we think this wine is similar"):
#             st.write(rag_explanation)

#     if show_reviews:
#         reviews_json_str = wine_data.get('reviews') # Get reviews as JSON string
#         if reviews_json_str and reviews_json_str != "[]":
#             try:
#                 reviews = json.loads(reviews_json_str) # Parse reviews JSON string
#                 if reviews:
#                     with st.expander("📝 Tasting Notes/Reviews"):
#                         for review in reviews:
#                             author = review.get('author', 'Unknown Author')
#                             rating_value_review = review.get('rating', 'N/A') # Get review rating
#                             review_text = review.get('review', 'No review text')

#                             st.markdown(f"**{author} says:**")
#                             st.write(f"Rating: {rating_value_review}") # Removed "/5"
#                             st.write(f"> {review_text}")
#                             st.write("---")  # Separator for reviews
#             except json.JSONDecodeError:
#                 st.warning("Could not display tasting notes due to formatting issues.")

# #######################################
# # 3️⃣ Streamlit App - Main function
# #######################################
# import json # Import JSON for handling reviews

# def main():
#     st.title("🍷 Wine Recommendation Engine")

#     possible_wine_names = load_wine_names()

#     st.subheader("🔍 Input Parameters")

#     # 1️⃣ Wine Name Input
#     wine_name = st.selectbox("Wine Name", options=possible_wine_names) if possible_wine_names else st.text_input("Wine Name")

#     # 2️⃣ Price Range
#     min_price, max_price = st.slider("Price Range", 0, 1000, (0, 1000))

#     # 3️⃣ Number of Recommendations
#     top_k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

#     # 4️⃣ Confidence Score
#     confidence_score = st.slider("Confidence Score (Min)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

#     st.write("---")

#     #######################################
#     # Get Recommendations and Display
#     #######################################
#     if st.button("🍷 Get Recommendations"):
#         if not wine_name: # Simple validation
#             st.error("Please enter or select a wine name.")
#             return

#         with st.spinner("🔄 Finding the best wines..."):
#             original_wine, recommendations = recommend_wines( # Assuming recommend_wines now returns original wine & RAG explanation
#                 wine_name=wine_name,
#                 top_k=top_k,
#                 price_min=min_price,
#                 price_max=max_price,
#                 min_confidence_score=confidence_score,
#                 return_original_wine=True # Keep returning original wine
#             )

#         if not recommendations:
#             st.error("❌ No recommendations found.")
#         else:
#             st.success(f"✅ Found {len(recommendations)} recommendations!")

#             st.header("🔍 Original Wine") # Display Original Wine Section
#             display_wine_details(original_wine, header_level=2, show_reviews=True) # Show reviews for original wine

#             st.header("✨ Recommended Wines") # Display Recommendations Section
#             for i, wine in enumerate(recommendations):
#                 st.subheader(f"Recommendation #{i+1} (Score: {wine.get('score', 'N/A'):.2f})") # Recommendation Header with Score
#                 display_wine_details(wine, header_level=4, show_reviews=True) # Don't show reviews again for recommendations, can be toggled if needed
#                 st.write("---") # Separator between recommendations

# #######################################
# # 4️⃣ Run Streamlit
# #######################################
# if __name__ == "__main__":
#     st.set_page_config(page_title="Wine Recommender", layout="wide")
#     main()