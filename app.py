import streamlit as st
# import pandas as pd # No longer needed if wine_names.csv is fully deprecated
import os
import json
# Ensure wine_recommendation.py is in the same directory or accessible in PYTHONPATH
import wine_recommendation # This will run the initialization code in wine_recommendation.py

# Set page config - This should be the first Streamlit command
st.set_page_config(page_title="🍷 AI Wine Recommender", layout="wide", initial_sidebar_state="expanded")

#######################################
# Card component (Adapted for new RAG explanation structure)
#######################################
def display_wine_card(wine_data, is_base_wine=False, base_wine_summary_blurb=None):
    """
    Displays a wine card with details.
    wine_data is expected to be a dictionary (metadata from Pinecone).
    base_wine_summary_blurb is for the initial blurb of the user's chosen wine.
    """
    if not wine_data or not isinstance(wine_data, dict):
        # This case should ideally be caught before calling this function
        st.warning("Wine details are missing or in an incorrect format.")
        return
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 4]) 

        with col1:
            image_url = wine_data.get('image_url') 
            if image_url:
                st.image(image_url, width=100, use_column_width='auto') 
            else:
                st.caption("No image")

        with col2:
            card_title = wine_data.get('name', 'Unknown Wine')
            if is_base_wine:
                st.subheader(f"⭐ Your Choice: {card_title}")
            else:
                st.subheader(card_title) 
            
            sub_details = []
            if wine_data.get('brand'): sub_details.append(f"**Producer:** {wine_data.get('brand')}")
            if wine_data.get('category'): sub_details.append(f"**Category:** {wine_data.get('category')}")
            if wine_data.get('country_of_origin'): sub_details.append(f"**Origin:** {wine_data.get('country_of_origin')}")
            if wine_data.get('size'): sub_details.append(f"**Size:** {wine_data.get('size')}")
            st.markdown(" | ".join(filter(None, sub_details)))

            meta_details = []
            if wine_data.get('price') is not None:
                meta_details.append(f"**Price:** ${wine_data.get('price'):.2f}")
            
            if wine_data.get('avg_rating') is not None:
                rating_text = f"{wine_data.get('avg_rating'):.1f}"
                if wine_data.get('best_rating_scale') is not None:
                    rating_text += f" / {wine_data.get('best_rating_scale'):.0f}"
                if wine_data.get('num_reviews') is not None:
                    rating_text += f" ({wine_data.get('num_reviews')} reviews)"
                meta_details.append(f"**Avg. Rating:** {rating_text}")

            if not is_base_wine and wine_data.get('similarity_score') is not None:
                 meta_details.append(f"**Similarity:** {wine_data.get('similarity_score'):.3f}")
            
            st.markdown(" | ".join(filter(None, meta_details)))

            # Display Blurbs
            if is_base_wine and base_wine_summary_blurb:
                st.markdown(f"**AI Sommelier on Your Pick:** _{base_wine_summary_blurb}_")
            
            rag_data = wine_data.get('rag_explanation_data', {}) # Get the dictionary
            if not is_base_wine:
                recommended_blurb = rag_data.get('recommended_wine_blurb')
                if recommended_blurb:
                    st.markdown(f"**Why You Might Like This:** _{recommended_blurb}_")

            # Detailed Comparison Expander (for recommended wines)
            if not is_base_wine:
                detailed_comparison_md = rag_data.get('detailed_comparison_markdown')
                if detailed_comparison_md:
                    with st.expander("💡 AI Sommelier's Detailed Comparison", expanded=False): # Start collapsed
                        st.markdown(detailed_comparison_md, unsafe_allow_html=True)
                else:
                    st.caption("No detailed comparison available.")
        
        # Common details expanders (Description, Reviews) for all cards
        description = wine_data.get('description', '') 
        if description: 
            with st.expander("📖 Product Description"):
                st.write(description)

        reviews_json_str = wine_data.get('reviews_json', '[]')
        if reviews_json_str and reviews_json_str != "[]":
            try:
                reviews = json.loads(reviews_json_str)
                if isinstance(reviews, list) and reviews: 
                    with st.expander("📝 Tasting Notes & Critic Reviews"):
                        for review in reviews[:3]: 
                            if isinstance(review, dict):
                                author = review.get('author', 'Anonymous')
                                rating = review.get('rating')
                                review_text = review.get('review', '')
                                review_md = f"**{author}**"
                                if rating is not None: review_md += f" (Rating: {rating})"
                                review_md += f": {review_text}"
                                st.markdown(review_md)
            except json.JSONDecodeError:
                st.warning("Could not parse tasting notes for display.")
        st.write("") # Adds a bit of vertical space after each card

#######################################
# Main Streamlit App Layout
#######################################
def main_app_layout():
    st.title("🍇 AI Wine Recommender")
    st.markdown("""
    Welcome! I'm your AI Sommelier. 
    Enter a wine you enjoy in the sidebar, adjust your preferences, and I'll find similar wines for you to explore.
    """)

    with st.sidebar:
        st.header("👨‍🍳 Your Preferences")
        user_input_wine_name = st.text_input("Enter a wine name you like:", placeholder="e.g., Caymus Cabernet Sauvignon")
        
        num_recommendations = st.slider("Number of recommendations:", 1, 10, 3, key="num_recs_slider")
        
        min_catalog_price = 0.0
        max_catalog_price = 2000.0 
        
        price_range = st.slider(
            "Price range ($):", 
            min_value=min_catalog_price, 
            max_value=max_catalog_price,  
            value=(min_catalog_price, 500.0), 
            step=10.0,
            format="$%.0f",
            key="price_slider"
        )
        price_min_filter, price_max_filter = price_range

        confidence_percent = st.slider(
            "Name Match Confidence (%):", 
            min_value=50, max_value=100, 
            value=wine_recommendation.DEFAULT_CONFIDENCE_SCORE, 
            step=5,
            help="How closely should the entered wine name match a wine in our catalog? Higher means stricter matching.",
            key="confidence_slider"
        )
        
        submit_button = st.button("✨ Find My Wine Matches!", type="primary", use_container_width=True)

    if submit_button and user_input_wine_name:
        with st.spinner(f"Consulting the cellar for '{user_input_wine_name}' and its companions... 🥂"):
            # Updated to receive three values
            base_wine_details, recommended_wines_list, base_wine_blurb = wine_recommendation.recommend_wines_for_streamlit(
                user_wine_name_input=user_input_wine_name,
                top_k=num_recommendations,
                price_min=price_min_filter,
                price_max=price_max_filter,
                min_confidence_score=confidence_percent
            )

        if base_wine_details:
            st.header(f"Based on Your Selection:")
            # Pass the base_wine_blurb to the card display
            display_wine_card(base_wine_details, is_base_wine=True, base_wine_summary_blurb=base_wine_blurb)
            st.markdown("---") 

            if recommended_wines_list:
                st.header(f"✨ Your AI Sommelier Recommends:")
                
                num_recs = len(recommended_wines_list)
                cols_to_display = min(num_recs, 3) 
                
                if cols_to_display > 0:
                    cols = st.columns(cols_to_display)
                    for i, rec_wine_meta in enumerate(recommended_wines_list):
                        with cols[i % cols_to_display]: 
                            display_wine_card(rec_wine_meta, is_base_wine=False)
                # else: # This case is unlikely if recommended_wines_list is not empty
                #      st.info("No recommendations to display in columns for the current results.")

            elif not recommended_wines_list: # Base wine found, but no recommendations
                st.info(f"I found '{base_wine_details.get('name', 'your chosen wine')}', but couldn't find other similar wines matching all your filter criteria. Perhaps try adjusting the price range or other filters?")
        
        # If base_wine_details is None, the message is handled by recommend_wines_for_streamlit via st.warning

    elif submit_button and not user_input_wine_name:
        st.warning("Please enter a wine name to get recommendations.")
    
if __name__ == "__main__":
    main_app_layout()
