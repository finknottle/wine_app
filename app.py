import streamlit as st
import os 
import json
import wine_recommendation # This will run the initialization code in wine_recommendation.py
import time 

st.set_page_config(page_title="AI Somm 🍇 Wine Recommender", layout="wide", initial_sidebar_state="expanded")

#######################################
# Card component 
#######################################
def display_wine_card(wine_data, card_key_prefix, is_base_wine=False, base_wine_summary_blurb=None, rag_explanation_content=None):
    """
    Displays a wine card with details.
    rag_explanation_content is the pre-fetched RAG data for this card.
    """
    if not wine_data or not isinstance(wine_data, dict):
        st.warning("Wine details are missing or in an incorrect format.")
        return
    
    wine_id_for_key = wine_data.get('pinecone_id', wine_data.get('name', str(time.time()))) 
    
    # All cards use a bordered container for a consistent look
    # The content of the card will be placed here.
    # For horizontal scrolling, the card itself should not force full width.
    
    with st.container(border=True): 
        col1, col2 = st.columns([1, 3]) # Image column, Text column

        with col1:
            image_url = wine_data.get('image_url') 
            if image_url:
                st.image(image_url, width=100) # Slightly smaller image
            else:
                st.caption("No image")

        with col2:
            card_title = wine_data.get('name', 'Unknown Wine')
            if is_base_wine:
                st.subheader(f"⭐ Your Starting Point: {card_title}")
            else:
                st.subheader(card_title) 
            
            details_to_display = []
            if wine_data.get('brand'): details_to_display.append(f"**🍷 Producer:** {wine_data.get('brand')}")
            if wine_data.get('price') is not None: # Price is important
                details_to_display.append(f"**💲 Price:** ${wine_data.get('price'):.2f}")
            if wine_data.get('avg_rating') is not None: # Rating is important
                rating_text = f"{wine_data.get('avg_rating'):.1f}"
                if wine_data.get('best_rating_scale') is not None: rating_text += f" / {wine_data.get('best_rating_scale'):.0f}"
                if wine_data.get('num_reviews') is not None: rating_text += f" ({wine_data.get('num_reviews')} reviews)"
                details_to_display.append(f"**🌟 Avg. Rating:** {rating_text}")
            
            # Less prominent details, can be shorter or in expander if needed
            # For now, keep them, but be mindful of card height
            if wine_data.get('category'): details_to_display.append(f"**🍇 Category:** {wine_data.get('category')}")
            if wine_data.get('country_of_origin'): details_to_display.append(f"**🌍 Origin:** {wine_data.get('country_of_origin')}")
            if wine_data.get('size'): details_to_display.append(f"**🍾 Size:** {wine_data.get('size')}") 
            
            for detail in details_to_display:
                st.markdown(f"<div style='font-size: 0.9rem; margin-bottom: 0.1rem;'>{detail}</div>", unsafe_allow_html=True)
            # st.write("") # Removed to reduce vertical space

            if is_base_wine and base_wine_summary_blurb:
                st.markdown(f"<div style='font-size: 0.95rem; margin-top: 0.5rem;'>**AI Somm on Your Pick:** _{base_wine_summary_blurb}_</div>", unsafe_allow_html=True)
            
            if not is_base_wine: 
                if rag_explanation_content and rag_explanation_content.get('recommended_wine_blurb'):
                    st.markdown(f"<div style='font-size: 0.95rem; margin-top: 0.5rem;'>**Why You'll Love This Wine:** _{rag_explanation_content.get('recommended_wine_blurb')}_</div>", unsafe_allow_html=True)
                elif not rag_explanation_content: 
                    st.caption("⏳ AI Somm is tasting this wine... notes coming in a few seconds, stay tuned!")


        # Detailed Comparison Expander (for recommended wines)
        if not is_base_wine and rag_explanation_content: 
            detailed_comparison_md = rag_explanation_content.get('detailed_comparison_markdown')
            if detailed_comparison_md:
                expander_key = f"expander_details_{card_key_prefix}_{wine_id_for_key}"
                with st.expander("💡 AI Somm's Detailed Comparison", expanded=False): 
                    st.markdown(detailed_comparison_md, unsafe_allow_html=True)
        
        # Combined Description and Reviews, always in an expander, collapsed by default for ALL cards
        description = wine_data.get('description', '') 
        reviews_json_str = wine_data.get('reviews_json', '[]')
        reviews = []
        if reviews_json_str and reviews_json_str != "[]":
            try:
                loaded_reviews = json.loads(reviews_json_str)
                if isinstance(loaded_reviews, list):
                    reviews = loaded_reviews
            except json.JSONDecodeError:
                st.warning("Could not parse tasting notes for display.")

        if description or reviews:
            expander_title = "📝 Description, Tasting Notes & Critic Reviews"
            with st.expander(expander_title, expanded=False): 
                if description:
                    st.markdown(f"**Product Description:**\n{description}")
                    if reviews: 
                        st.markdown("---") 
                
                if reviews:
                    if description : st.markdown(f"**Tasting Notes & Critic Reviews:**") 
                    elif not description: st.markdown(f"**Tasting Notes & Critic Reviews:**")

                    for review in reviews[:3]: 
                        if isinstance(review, dict):
                            author = review.get('author', 'Anonymous'); rating = review.get('rating'); review_text = review.get('review', '')
                            review_md = f"**{author}**"; 
                            if rating is not None: review_md += f" (Rating: {rating})"
                            review_md += f": {review_text}"; st.markdown(review_md)
        # st.write("") # Removed to reduce vertical space

#######################################
# Main Streamlit App Layout
#######################################
def main_app_layout():
    st.title("🍇 AI Somm") 
    st.markdown("### Let us help you find your next favorite wine!") 
    st.markdown("""
    Tell AI Somm a wine you already love, and our AI will suggest others that might tantalize your tastebuds. 
    Adjust the price filter in the sidebar to match your budget. Let the discovery begin! 🍾
    """)
    st.markdown("---")

    # Session state initialization
    if 'base_wine_for_display' not in st.session_state: st.session_state.base_wine_for_display = None
    if 'base_blurb_for_display' not in st.session_state: st.session_state.base_blurb_for_display = None
    if 'recommendations_list' not in st.session_state: st.session_state.recommendations_list = [] 
    if 'rag_explanations' not in st.session_state: st.session_state.rag_explanations = {} 
    if 'fetch_rag_next_idx' not in st.session_state: st.session_state.fetch_rag_next_idx = 0
    if 'new_search_triggered' not in st.session_state: st.session_state.new_search_triggered = False
    if 'initial_load_complete' not in st.session_state: st.session_state.initial_load_complete = False
    if 'base_blurb_fetched' not in st.session_state: st.session_state.base_blurb_fetched = False

    wine_names_list = wine_recommendation.get_all_wine_names() 
    
    user_selected_wine_name = st.selectbox(
        "Tell us a wine you like...", 
        options=[""] + wine_names_list,  
        index=0, 
        help="Start typing to search, or select from the list.",
        key="wine_selectbox_main"
    )

    with st.sidebar:
        st.header("💰 Price Filter")
        min_catalog_price = 0.0; max_slider_price = 200.0 
        price_range = st.slider("Preferred price range ($):", 
                                min_value=min_catalog_price, max_value=max_slider_price, 
                                value=(min_catalog_price, 100.0), step=5.0, format="$%.0f", 
                                key="price_slider")
        price_min_filter, price_max_filter = price_range

    submit_button = st.button("✨ Find My Wine Matches!", type="primary", use_container_width=True)
    st.markdown("---")

    if submit_button and user_selected_wine_name:
        print(f"DEBUG: Submit button clicked for wine: {user_selected_wine_name}")
        st.session_state.new_search_triggered = True 
        st.session_state.base_wine_for_display = None
        st.session_state.recommendations_list = []
        st.session_state.base_blurb_for_display = None 
        st.session_state.rag_explanations = {} 
        st.session_state.fetch_rag_next_idx = 0
        st.session_state.initial_load_complete = False 
        st.session_state.base_blurb_fetched = False 

        with st.spinner(f"Finding wines similar to '{user_selected_wine_name}'... 🥂"):
            base_details, rec_list_meta_only, _ = wine_recommendation.recommend_wines_for_streamlit(
                user_wine_name_input=user_selected_wine_name, top_k=5, 
                price_min=price_min_filter, price_max=price_max_filter,
                min_confidence_score=wine_recommendation.DEFAULT_CONFIDENCE_SCORE 
            )
        st.session_state.base_wine_for_display = base_details
        st.session_state.recommendations_list = rec_list_meta_only 
        st.session_state.initial_load_complete = True 
        print(f"DEBUG: Initial fetch complete. Base wine: {base_details.get('name') if base_details else 'None'}. Recs count: {len(rec_list_meta_only)}")
        st.rerun() 

    if st.session_state.base_wine_for_display:
        if not st.session_state.base_blurb_fetched and st.session_state.initial_load_complete:
            print(f"DEBUG: Fetching blurb for base wine: {st.session_state.base_wine_for_display.get('name')}")
            st.session_state.base_blurb_for_display = wine_recommendation.generate_base_wine_blurb(st.session_state.base_wine_for_display)
            st.session_state.base_blurb_fetched = True
            print(f"DEBUG: Base blurb fetched. Rerunning to display.")
            st.rerun() 

        st.markdown("## 🍷 Your Wine Selection") 
        display_wine_card(
            st.session_state.base_wine_for_display, 
            card_key_prefix="base", 
            is_base_wine=True, 
            base_wine_summary_blurb=st.session_state.base_blurb_for_display
        )
        st.markdown("---") 

        if st.session_state.recommendations_list:
            st.markdown("## ✨ AI Somm's Picks") 
            
            # --- Horizontal Scroll for Recommendations ---
            # Inject CSS for horizontal scrolling flex container
            # The key is that the items inside this container should not force full width.
            # display_wine_card now uses st.container(border=True) which should behave well.
            st.markdown("""
                <style>
                div.stButton > button {
                    width: 100%;
                }
                .horizontal-scroll-container {
                    display: flex;
                    overflow-x: auto; /* Enables horizontal scrolling */
                    overflow-y: hidden; /* Hides vertical scrollbar on the container itself */
                    padding: 10px 0px 10px 0px; /* Add some padding */
                    gap: 1rem; /* Space between cards */
                    width: 100%; /* Ensure the container takes full width */
                }
                .scrollable-card {
                    min-width: 320px; /* Minimum width of each card */
                    max-width: 360px; /* Maximum width of each card */
                    flex: 0 0 auto;   /* Prevents cards from shrinking/growing, maintains their width */
                    /* border: 1px solid #e0e0e0; /* Moved border to st.container in display_wine_card */
                    /* border-radius: 0.5rem; */
                    /* padding: 1rem; */ /* Padding is handled by st.container now */
                    /* background-color: #ffffff; */ /* Background is handled by st.container */
                    /* margin-bottom: 10px; */ /* Redundant due to gap */
                }
                /* Hide scrollbar for Chrome, Safari and Opera */
                .horizontal-scroll-container::-webkit-scrollbar {
                    display: none;
                }
                /* Hide scrollbar for IE, Edge and Firefox */
                .horizontal-scroll-container {
                    -ms-overflow-style: none;  /* IE and Edge */
                    scrollbar-width: none;  /* Firefox */
                }
                </style>
            """, unsafe_allow_html=True)

            # Create the flex container
            st.markdown('<div class="horizontal-scroll-container">', unsafe_allow_html=True)
            for i, rec_wine_meta in enumerate(st.session_state.recommendations_list):
                wine_pinecone_id = rec_wine_meta.get('pinecone_id') 
                rag_content_for_card = st.session_state.rag_explanations.get(wine_pinecone_id)
                
                # Each card is an item in the flex container
                # We use st.markdown to inject the div for each card, then call display_wine_card
                # The display_wine_card will create its own bordered st.container inside this div
                st.markdown('<div class="scrollable-card">', unsafe_allow_html=True)
                display_wine_card(
                    rec_wine_meta, 
                    card_key_prefix=f"rec_{i}", 
                    is_base_wine=False,
                    rag_explanation_content=rag_content_for_card 
                )
                st.markdown('</div>', unsafe_allow_html=True) # Close scrollable-card div
            st.markdown('</div>', unsafe_allow_html=True) # Close horizontal-scroll-container

        
        elif st.session_state.initial_load_complete and not st.session_state.recommendations_list: 
             if st.session_state.base_wine_for_display: 
                st.info(f"I found '{st.session_state.base_wine_for_display.get('name', 'your chosen wine')}', but couldn't find other similar wines matching all your filter criteria. Perhaps try adjusting the price range?")

    # --- Progressive RAG data fetching LOGIC ---
    if st.session_state.initial_load_complete and st.session_state.base_blurb_fetched and \
       st.session_state.recommendations_list and st.session_state.base_wine_for_display:
        
        idx_to_fetch = st.session_state.get('fetch_rag_next_idx', 0) 
        if idx_to_fetch < len(st.session_state.recommendations_list):
            wine_to_explain = st.session_state.recommendations_list[idx_to_fetch]
            wine_pinecone_id_to_fetch = wine_to_explain.get('pinecone_id') 
            if wine_pinecone_id_to_fetch and wine_pinecone_id_to_fetch not in st.session_state.rag_explanations:
                print(f"Progressive RAG (app.py): >>> ATTEMPTING TO FETCH RAG for Pinecone ID: {wine_pinecone_id_to_fetch}, Name: {wine_to_explain.get('name')}, Index: {idx_to_fetch} <<<")
                rag_data = wine_recommendation.generate_enhanced_rag_explanation(st.session_state.base_wine_for_display, wine_to_explain)
                st.session_state.rag_explanations[wine_pinecone_id_to_fetch] = rag_data
                st.session_state.fetch_rag_next_idx = idx_to_fetch + 1 
                print(f"Progressive RAG (app.py): Fetched for {wine_to_explain.get('name')}. Next index: {st.session_state.fetch_rag_next_idx}. Rerunning.")
                st.rerun() 
        elif idx_to_fetch >= len(st.session_state.recommendations_list) and st.session_state.new_search_triggered: 
            print("DEBUG (app.py): All RAGs fetched for new search.")
            st.session_state.new_search_triggered = False 
    
    elif submit_button and not user_selected_wine_name: 
        st.warning("Please select or enter a wine name to get recommendations.")
    
if __name__ == "__main__":
    main_app_layout()
