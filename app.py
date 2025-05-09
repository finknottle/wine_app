import streamlit as st
# import pandas as pd # Only needed if get_all_wine_names falls back to CSV
import os # Keep os for potential CSV fallback in wine_recommendation
import json
# Ensure wine_recommendation.py is in the same directory or accessible in PYTHONPATH
import wine_recommendation # This will run the initialization code in wine_recommendation.py

# Set page config - This should be the first Streamlit command
st.set_page_config(page_title="AI Somm 🍇 Wine Recommender", layout="wide", initial_sidebar_state="expanded") # Updated title

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
        st.warning("Wine details are missing or in an incorrect format.")
        return
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 4]) 

        with col1:
            image_url = wine_data.get('image_url') 
            if image_url:
                st.image(image_url, width=80) 
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
            if sub_details: 
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

            # Removed Similarity Score display from here
            # if not is_base_wine and wine_data.get('similarity_score') is not None:
            #      meta_details.append(f"**Similarity:** {wine_data.get('similarity_score'):.3f}")
            
            if meta_details: 
                st.markdown(" | ".join(filter(None, meta_details)))

            if is_base_wine and base_wine_summary_blurb:
                st.markdown(f"**AI Somm on Your Pick:** _{base_wine_summary_blurb}_")
            
            rag_data = wine_data.get('rag_explanation_data', {}) 
            if not is_base_wine:
                recommended_blurb = rag_data.get('recommended_wine_blurb')
                if recommended_blurb:
                    st.markdown(f"**Why You Might Like This:** _{recommended_blurb}_")

            if not is_base_wine:
                detailed_comparison_md = rag_data.get('detailed_comparison_markdown')
                if detailed_comparison_md:
                    with st.expander("💡 AI Somm's Detailed Comparison", expanded=False): 
                        st.markdown(detailed_comparison_md, unsafe_allow_html=True)
        
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
        st.write("") 

#######################################
# Main Streamlit App Layout
#######################################
def main_app_layout():
    # --- Updated App Name and Introductory Text ---
    st.title("🍇 AI Somm: Your Personal Wine Guide") 
    st.markdown("""
    Ready to uncork your next favorite wine? 🍾
    
    Tell me a wine you already love, and I'll use a dash of AI magic to suggest others that might tantalize your tastebuds. 
    Adjust the price filter in the sidebar to match your budget. Let the discovery begin!
    """)
    st.markdown("---")

    wine_names_list = wine_recommendation.get_all_wine_names() 
    
    col_select, col_button_space = st.columns([3,1]) 

    with col_select:
        if not wine_names_list:
            st.info("Loading wine names for selection... If this is the first run or after a cache refresh, it might take a moment. If no names appear, please check the console for data loading messages from `wine_recommendation.py` and ensure the `get_all_wine_names` function is correctly implemented.")
            user_selected_wine_name = st.text_input(
                "Enter a wine name you like (autocomplete list unavailable):", 
                placeholder="e.g., Caymus Cabernet Sauvignon",
                key="wine_text_input_fallback"
            )
        else:
            user_selected_wine_name = st.selectbox(
                "Select or type a wine you enjoy:",
                options=[""] + wine_names_list,  
                index=0, 
                help="Start typing to search, or select from the list.",
                key="wine_selectbox"
            )

    with st.sidebar:
        st.header("💰 Your Price Filter")
        min_catalog_price = 0.0
        max_slider_price = 200.0 
        
        price_range = st.slider(
            "Preferred price range ($):", 
            min_value=min_catalog_price, 
            max_value=max_slider_price,  
            value=(min_catalog_price, 100.0), 
            step=5.0, 
            format="$%.0f",
            key="price_slider"
        )
        price_min_filter, price_max_filter = price_range

    submit_button = st.button("✨ Find My Wine Matches!", type="primary", use_container_width=True)
    st.markdown("---")


    if submit_button and user_selected_wine_name:
        with st.spinner(f"Consulting the cellar for '{user_selected_wine_name}' and its companions... 🥂"):
            base_wine_details, recommended_wines_list, base_wine_blurb = wine_recommendation.recommend_wines_for_streamlit(
                user_wine_name_input=user_selected_wine_name,
                top_k=5, 
                price_min=price_min_filter,
                price_max=price_max_filter,
                min_confidence_score=wine_recommendation.DEFAULT_CONFIDENCE_SCORE 
            )

        if base_wine_details:
            st.header(f"Based on Your Selection:")
            display_wine_card(base_wine_details, is_base_wine=True, base_wine_summary_blurb=base_wine_blurb)
            st.markdown("---") 

            if recommended_wines_list:
                st.header(f"✨ Your AI Somm Recommends:")
                
                num_recs = len(recommended_wines_list)
                cols_to_display = min(num_recs, 3) if num_recs > 0 else 1 
                
                if num_recs > 0 :
                    cols = st.columns(cols_to_display)
                    for i, rec_wine_meta in enumerate(recommended_wines_list):
                        with cols[i % cols_to_display]: 
                            display_wine_card(rec_wine_meta, is_base_wine=False)

            elif not recommended_wines_list: 
                st.info(f"I found '{base_wine_details.get('name', 'your chosen wine')}', but couldn't find other similar wines matching all your filter criteria. Perhaps try adjusting the price range?")
        
    elif submit_button and not user_selected_wine_name:
        st.warning("Please select or enter a wine name to get recommendations.")
    
if __name__ == "__main__":
    main_app_layout()
