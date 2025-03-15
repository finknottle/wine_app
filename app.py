import streamlit as st
import pandas as pd
import os
import json
import altair as alt
from wine_recommendation import recommend_wines  # Assuming this is your backend module

# Set page config at the very top
st.set_page_config(page_title="Wine Recommender", layout="wide")

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
        if len(description) > 150:
            st.markdown(f"<small>{description[:150]}...</small>", unsafe_allow_html=True)
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
                    with st.expander("📝 Tasting Notes"):
                        for review in reviews:
                            author = review.get('author', 'Anonymous')
                            rating = review.get('rating', 'N/A')
                            review_text = review.get('review', '')
                            st.markdown(f"**{author}** - Rating: {rating}")
                            st.markdown(f"<small>{review_text}</small>", unsafe_allow_html=True)
                            st.write("---")
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

    st.markdown(
        f"<h{header_level} id='wine-details-{wine_data.get('name').replace(' ', '-')}' style='padding-top: 20px;'>🍷 {wine_data['name']}</h{header_level}>",
        unsafe_allow_html=True
    )

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

#######################################
# 4️⃣ Streamlit App - Main function
#######################################
def main():
    st.title("🍷 AI Sommelier - Find similar wines")

    possible_wine_names = load_wine_names()
    # st.subheader("🔍 Input Parameters")

    # 1️⃣ Wine Name Input
    wine_name = st.selectbox("Wine Name", options=possible_wine_names) if possible_wine_names else st.text_input("Wine Name")

    # # 2️⃣ Price Range
    # min_price_slider, max_price_slider = st.slider("Price Range", 0, 1000, (0, 1000))

    # # 3️⃣ Number of Recommendations
    # top_k = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    # # 4️⃣ Confidence Score remains a float between 0.0 and 1.0
    # confidence_score = st.slider("Confidence Score (Min)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

    # st.write("---")

    #######################################
    # Get Recommendations and Display
    #######################################
    if st.button("🍷 Find Similar Wines"):
        if not wine_name:
            st.error("Please enter or select a wine name.")
            return

        with st.spinner("🔄 Finding the best wines..."):
            # Set default values for the inputs
            min_price_slider = min_price_slider if 'min_price_slider' in locals() else 0
            max_price_slider = max_price_slider if 'max_price_slider' in locals() else 200
            top_k = top_k if 'top_k' in locals() else 10
            confidence_score = confidence_score if 'confidence_score' in locals() else 0.9

            original_wine, recommendations = recommend_wines(
                wine_name=wine_name,
                top_k=top_k,
                price_min=min_price_slider,
                price_max=max_price_slider,
                min_confidence_score=confidence_score,
                return_original_wine=True
            )

        if not recommendations:
            st.error("❌ No recommendations found.")
        else:
            st.success(f"✅ Found {len(recommendations)} recommendations!")

            # Display original wine details
            st.header("🔍 Original Wine")
            with st.container():
                col1, col2 = st.columns([0.1, 0.7])
                with col1:
                    image_url = original_wine.get('image')
                    if image_url:
                        st.image(image_url, width=20)
                    else:
                        st.image("wine_placeholder.png", caption="No Image Available", width=20)
                with col2:
                    st.markdown(f"**🍷 {original_wine['name']}**", unsafe_allow_html=True)
                    st.markdown(
                        f"<small>{original_wine.get('country_of_origin', 'N/A')} - {original_wine.get('category', 'N/A')}</small>",
                        unsafe_allow_html=True
                    )
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
                    st.markdown(
                        f"<details><summary>📖 Description</summary><small>{original_wine.get('description', 'No description available.')}</small></details>",
                        unsafe_allow_html=True
                    )
                    # Display tasting notes for the original wine
                    reviews_json_str = original_wine.get('reviews')
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

            # Create a DataFrame from recommendations.
            df_recommendations = pd.DataFrame(recommendations)
            df_recommendations.columns = df_recommendations.columns.astype(str)
            
            # If recommendations don’t have 'similarity_score', set them to the slider value.
            if 'similarity_score' not in df_recommendations.columns:
                df_recommendations['similarity_score'] = confidence_score

            # Original wine's similarity score remains as 1.0 (100%)
            original_wine['similarity_score'] = 1.0
            
            # Concatenate original wine with recommendations.
            df_recommendations = pd.concat([df_recommendations, pd.DataFrame([original_wine])], ignore_index=True)
            df_recommendations['price'] = pd.to_numeric(df_recommendations['price'], errors='coerce')
            df_recommendations = df_recommendations.dropna(subset=['price'])
            
            # Convert similarity score to percentage for display (multiply by 100)
            df_recommendations['similarity_percentage'] = df_recommendations['similarity_score'] * 100

            # Set chart axis domains (adding a small margin)
            min_price_chart = df_recommendations['price'].min() - 10
            max_price_chart = df_recommendations['price'].max() + 10
            min_similarity_pct = df_recommendations['similarity_percentage'].min() - 5
            max_similarity_pct = df_recommendations['similarity_percentage'].max() + 5

            # Create the Altair scatter plot using the similarity_percentage field
            chart = alt.Chart(df_recommendations).mark_circle(size=100).encode(
                x=alt.X('price:Q', title='Price', scale=alt.Scale(domain=[min_price_chart, max_price_chart])),
                y=alt.Y('similarity_percentage:Q', 
                        title='Similarity Score (%)', 
                        scale=alt.Scale(domain=[min_similarity_pct, max_similarity_pct]),
                        axis=alt.Axis(format=".2f")),  # Shows two decimals
                color=alt.condition(
                    alt.datum.name == original_wine['name'],
                    alt.value('red'),  # Original wine in red
                    alt.value('blue')  # Recommended wines in blue
                ),
                tooltip=[
                    'name', 
                    'price', 
                    alt.Tooltip('similarity_percentage:Q', title='Similarity Score (%)', format=".2f")
                ]
            ).properties(
                width=600,
                height=400,
                title="Wine Price vs. Similarity Score (%)"
            )


            st.altair_chart(chart, use_container_width=True)

            st.header("✨ Recommended Wines")
            for i, wine in enumerate(recommendations):
                with st.container():
                    try:
                        wine_data = json.loads(wine) if isinstance(wine, str) else wine
                        display_wine_card(wine_data)
                    except json.JSONDecodeError:
                        st.error("Error parsing wine data.")
                    st.write("---")

if __name__ == "__main__":
    main()
