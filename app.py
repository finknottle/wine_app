import streamlit as st
import json
import time

import wine_recommendation  # This runs initialization code in wine_recommendation.py

st.set_page_config(
    page_title="AI Somm 🍇 Wine Recommender",
    layout="wide",
    initial_sidebar_state="auto",
)


SESSION_DEFAULTS = {
    "base_wine_for_display": None,
    "base_blurb_for_display": None,
    "recommendations_list": [],
    "rag_explanations": {},
    "fetch_rag_next_idx": 0,
    "new_search_triggered": False,
    "initial_load_complete": False,
    "base_blurb_fetched": False,

    # Sommelier UX: two entry modes
    "entry_mode": "Wine I love",
    "vibe_text": "",

    # Taste profile sliders (session only)
    "taste_body": 50,          # 0 light -> 100 full
    "taste_acidity": 50,       # 0 low  -> 100 high
    "taste_tannin": 50,        # 0 soft -> 100 firm
    "taste_fruit_earth": 50,   # 0 fruit -> 100 earth
    "taste_oak": 30,           # 0 none -> 100 oaky

    # Session-only preference signals
    "liked_wines": [],  # list[str] of wine names the user liked
    "disliked_ids": set(),  # set[str] of pinecone_ids to hide
    "blocked_producers": set(),  # set[str] of producers to hide (session only)
}


def init_session_state():
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            # Avoid sharing mutable defaults like set()/list() across sessions
            if isinstance(v, (list, dict, set)):
                st.session_state[k] = v.__class__()
            else:
                st.session_state[k] = v


def reset_session_state():
    # Reset our own keys
    for k, v in SESSION_DEFAULTS.items():
        # Avoid sharing mutable defaults like set()/list() across sessions
        if isinstance(v, (list, dict, set)):
            st.session_state[k] = v.__class__()
        else:
            st.session_state[k] = v

    # Reset input widgets
    st.session_state["wine_selectbox_main"] = ""
    st.session_state["wine_query"] = ""
    st.session_state["vibe_text"] = ""


#######################################
# Card component
#######################################

def display_wine_card(
    wine_data,
    card_key_prefix,
    is_base_wine=False,
    base_wine_summary_blurb=None,
    rag_explanation_content=None,
):
    """Displays a wine card with details."""

    if not wine_data or not isinstance(wine_data, dict):
        st.warning("Wine details are missing or in an incorrect format.")
        return

    wine_id_for_key = wine_data.get("pinecone_id", wine_data.get("name", str(time.time())))

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            image_url = wine_data.get("image_url")
            if image_url:
                st.image(image_url, width=120)
            else:
                st.caption("No image")

        with col2:
            card_title = wine_data.get("name", "Unknown Wine")
            if is_base_wine:
                st.subheader(f"⭐ Your Starting Point: {card_title}")
            else:
                st.subheader(card_title)

            details_to_display = []
            if wine_data.get("brand"):
                details_to_display.append(f"**🍷 Producer:** {wine_data.get('brand')}")
            if wine_data.get("price") is not None:
                details_to_display.append(f"**💲 Price:** ${wine_data.get('price'):.2f}")
            if wine_data.get("size"):
                details_to_display.append(f"**🍾 Size:** {wine_data.get('size')}")
            if wine_data.get("avg_rating") is not None:
                rating_text = f"{wine_data.get('avg_rating'):.1f}"
                if wine_data.get("best_rating_scale") is not None:
                    rating_text += f" / {wine_data.get('best_rating_scale'):.0f}"
                if wine_data.get("num_reviews") is not None:
                    rating_text += f" ({wine_data.get('num_reviews')} reviews)"
                details_to_display.append(f"**🌟 Avg. Rating:** {rating_text}")

            for detail_text in details_to_display:
                st.markdown(detail_text)

            if is_base_wine:
                if base_wine_summary_blurb:
                    st.markdown(f"**AI Somm on Your Pick:** _{base_wine_summary_blurb}_")
                else:
                    st.caption("⏳ AI Somm is tasting your pick…")
            else:
                # Keep this area stable so cards don't jump around.
                if rag_explanation_content and rag_explanation_content.get("recommended_wine_blurb"):
                    st.markdown(
                        f"**Why You Might Like This:** _{rag_explanation_content.get('recommended_wine_blurb')}_"
                    )
                else:
                    st.markdown("**Why You Might Like This:** _AI Somm is tasting this wine…_")

        # Detailed Comparison Expander (recommended wines)
        if not is_base_wine and rag_explanation_content:
            detailed_comparison_md = rag_explanation_content.get("detailed_comparison_markdown")
            if detailed_comparison_md:
                with st.expander("💡 AI Somm's Detailed Comparison", expanded=False):
                    st.markdown(detailed_comparison_md, unsafe_allow_html=True)

        # Description + Reviews expander (all wines)
        description = wine_data.get("description", "")
        reviews_json_str = wine_data.get("reviews_json", "[]")
        reviews = []
        if reviews_json_str and reviews_json_str != "[]":
            try:
                loaded_reviews = json.loads(reviews_json_str)
                if isinstance(loaded_reviews, list):
                    reviews = loaded_reviews
            except json.JSONDecodeError:
                st.warning("Could not parse tasting notes for display.")

        if description or reviews:
            with st.expander("📝 Description, Tasting Notes & Critic Reviews", expanded=False):
                if description:
                    st.markdown(f"**Product Description:**\n{description}")
                    if reviews:
                        st.markdown("---")

                if reviews:
                    st.markdown("**Tasting Notes & Critic Reviews:**")
                    for review in reviews[:3]:
                        if isinstance(review, dict):
                            author = review.get("author", "Anonymous")
                            rating = review.get("rating")
                            review_text = review.get("review", "")
                            review_md = f"**{author}**"
                            if rating is not None:
                                review_md += f" (Rating: {rating})"
                            review_md += f": {review_text}"
                            st.markdown(review_md)

        st.write("")


#######################################
# Main Streamlit App Layout
#######################################

def _assign_flight_tiers(recs: list[dict]) -> list[dict]:
    """Tag recs as a 'flight': core / explore / wildcard."""
    out = []
    for i, r in enumerate(recs or []):
        r2 = dict(r)
        if i < 3:
            r2["flight_tier"] = "core"
        elif i < 5:
            r2["flight_tier"] = "explore"
        else:
            r2["flight_tier"] = "wildcard"
        out.append(r2)
    return out


def _apply_session_filters(recs: list[dict]) -> list[dict]:
    filtered = []
    for w in recs or []:
        pid = w.get("pinecone_id")
        producer = (w.get("brand") or "").strip().lower()
        if pid and pid in st.session_state.disliked_ids:
            continue
        if producer and producer in st.session_state.blocked_producers:
            continue
        filtered.append(w)
    return filtered


def run_new_search(wine_name: str, price_min: float, price_max: float):
    """Start a brand-new search and reset result-related state."""
    st.session_state.new_search_triggered = True
    st.session_state.base_wine_for_display = None
    st.session_state.recommendations_list = []
    st.session_state.base_blurb_for_display = None
    st.session_state.rag_explanations = {}
    st.session_state.fetch_rag_next_idx = 0
    st.session_state.initial_load_complete = False
    st.session_state.base_blurb_fetched = False

    with st.spinner(f"Finding wines similar to '{wine_name}'… 🥂"):
        base_details, rec_list_meta_only, _ = wine_recommendation.recommend_wines_for_streamlit(
            user_wine_name_input=wine_name,
            top_k=6,
            price_min=price_min,
            price_max=price_max,
            min_confidence_score=wine_recommendation.DEFAULT_CONFIDENCE_SCORE,
        )

    st.session_state.base_wine_for_display = base_details
    filtered = _apply_session_filters(rec_list_meta_only or [])
    st.session_state.recommendations_list = _assign_flight_tiers(filtered)
    st.session_state.initial_load_complete = True


def _taste_words():
    def bucket(val, low, high, low_word, mid_word, high_word):
        if val <= low:
            return low_word
        if val >= high:
            return high_word
        return mid_word

    body = bucket(st.session_state.taste_body, 30, 70, "light-bodied", "medium-bodied", "full-bodied")
    acid = bucket(st.session_state.taste_acidity, 30, 70, "low-acid", "balanced acidity", "high-acid")
    tann = bucket(st.session_state.taste_tannin, 30, 70, "soft tannins", "moderate tannins", "firm tannins")
    fe = bucket(st.session_state.taste_fruit_earth, 30, 70, "fruit-forward", "balanced fruit/earth", "earthy/savory")
    oak = bucket(st.session_state.taste_oak, 25, 65, "little/no oak", "some oak", "noticeable oak")
    return body, acid, tann, fe, oak


def build_vibe_prompt(vibe_text: str) -> str:
    body, acid, tann, fe, oak = _taste_words()
    vibe_text = (vibe_text or "").strip()
    return (
        f"{vibe_text}\n\n"
        f"Preferences: {body}; {acid}; {tann}; {fe}; {oak}. "
        f"Recommend a small curated flight with a few close matches and a couple of adjacent explorations."
    ).strip()


def run_new_vibe_search(vibe_text: str, price_min: float, price_max: float):
    st.session_state.new_search_triggered = True
    st.session_state.base_wine_for_display = None
    st.session_state.recommendations_list = []
    st.session_state.base_blurb_for_display = None
    st.session_state.rag_explanations = {}
    st.session_state.fetch_rag_next_idx = 0
    st.session_state.initial_load_complete = False
    st.session_state.base_blurb_fetched = True  # no base blurb in vibe mode

    prompt = build_vibe_prompt(vibe_text)

    with st.spinner("AI Somm is thinking about your vibe…"):
        pseudo_base, recs = wine_recommendation.recommend_wines_from_prompt(
            prompt_text=prompt,
            top_k=6,
            price_min=price_min,
            price_max=price_max,
        )

    st.session_state.base_wine_for_display = pseudo_base
    filtered = _apply_session_filters(recs or [])
    st.session_state.recommendations_list = _assign_flight_tiers(filtered)
    st.session_state.initial_load_complete = True


APP_VERSION = "d24e2b5"


def main_app_layout():
    init_session_state()

    st.title("🍇 AI Somm")
    st.caption(f"Version: {APP_VERSION}")
    st.markdown("### Let us help you find your next favorite wine!")
    st.markdown(
        """
        Tell AI Somm a wine you already love, and our AI will suggest others that might tantalize your tastebuds.
        Adjust the price filter in the sidebar to match your budget. Let the discovery begin! 🍾
        """
    )
    st.markdown("---")

    wine_names_list = wine_recommendation.get_all_wine_names()

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Sommelier Controls")

        st.session_state.entry_mode = st.radio(
            "How do you want to start?",
            options=["Wine I love", "Describe a vibe"],
            horizontal=False,
            key="entry_mode",
        )

        if st.button("🔄 Reset", use_container_width=True):
            reset_session_state()
            st.rerun()

        if st.session_state.liked_wines:
            st.caption("Recent likes")
            for w in st.session_state.liked_wines[-5:][::-1]:
                st.write(f"• {w}")

        if st.session_state.entry_mode == "Describe a vibe":
            st.header("Your Taste Profile")
            st.caption("Optional: tweak these if you know your preferences.")

            st.slider("Body", 0, 100, st.session_state.taste_body, key="taste_body")
            st.slider("Acidity", 0, 100, st.session_state.taste_acidity, key="taste_acidity")
            st.slider("Tannin (reds)", 0, 100, st.session_state.taste_tannin, key="taste_tannin")
            st.slider("Fruit ↔ Earth", 0, 100, st.session_state.taste_fruit_earth, key="taste_fruit_earth")
            st.slider("Oak", 0, 100, st.session_state.taste_oak, key="taste_oak")

        st.header("💰 Price")
        min_catalog_price = 0.0
        max_slider_price = 200.0
        price_range = st.slider(
            "Preferred price range ($):",
            min_value=min_catalog_price,
            max_value=max_slider_price,
            value=(min_catalog_price, 100.0),
            step=5.0,
            format="$%.0f",
            key="price_slider",
        )
        price_min_filter, price_max_filter = price_range

        # Progress indicator for the "AI tasting" pass
        if st.session_state.initial_load_complete and st.session_state.recommendations_list:
            total = len(st.session_state.recommendations_list)
            done = len(st.session_state.rag_explanations)
            st.caption(f"AI Somm tasting notes: {done}/{total}")
            st.progress(done / max(total, 1))

    # --- Entry UI ---
    user_selected_wine_name = ""

    if st.session_state.entry_mode == "Wine I love":
        st.text_input(
            "Search (optional)",
            value="",
            placeholder="Try: Pinot Noir, Ridge, Chablis…",
            key="wine_query",
            help="Type a few characters to narrow the wine list.",
        )
        query = (st.session_state.get("wine_query") or "").strip().lower()
        if query:
            filtered_wines = [w for w in wine_names_list if query in w.lower()]
            filtered_wines = filtered_wines[:250]
        else:
            filtered_wines = wine_names_list

        user_selected_wine_name = st.selectbox(
            "Tell me a wine you love…",
            options=[""] + filtered_wines,
            index=0,
            help="Type to search, then pick a wine.",
            key="wine_selectbox_main",
        )

        if not wine_names_list:
            st.caption("Wine list for autocomplete is loading or unavailable.")

        submit_button = st.button(
            "✨ Pour me a flight",
            type="primary",
            use_container_width=True,
        )

        if submit_button and user_selected_wine_name:
            run_new_search(user_selected_wine_name, price_min_filter, price_max_filter)
            st.rerun()

    else:
        st.text_area(
            "Describe what you're in the mood for",
            placeholder="Examples: 'crisp white with high acidity for oysters' / 'juicy red, not too oaky, pizza night' / 'sparkling, celebratory, not too sweet'",
            key="vibe_text",
            height=100,
        )

        body, acid, tann, fe, oak = _taste_words()
        st.caption(f"Taste profile: {body} • {acid} • {tann} • {fe} • {oak}")

        submit_button = st.button(
            "✨ Pour me a flight",
            type="primary",
            use_container_width=True,
        )

        if submit_button and (st.session_state.get("vibe_text") or "").strip():
            run_new_vibe_search(st.session_state.vibe_text, price_min_filter, price_max_filter)
            st.rerun()

    st.markdown("---")

    # --- Display results ---
    if st.session_state.base_wine_for_display:
        if not st.session_state.base_blurb_fetched and st.session_state.initial_load_complete:
            st.session_state.base_blurb_for_display = wine_recommendation.generate_base_wine_blurb(
                st.session_state.base_wine_for_display
            )
            st.session_state.base_blurb_fetched = True
            st.rerun()

        st.markdown("## 🍷 Your Wine Selection")
        display_wine_card(
            st.session_state.base_wine_for_display,
            card_key_prefix="base",
            is_base_wine=True,
            base_wine_summary_blurb=st.session_state.base_blurb_for_display,
        )
        st.markdown("---")

        if st.session_state.recommendations_list:
            st.markdown("## ✨ AI Somm's Flight")
            st.caption("A curated set: close matches first, then a couple of explorations.")

            tier_labels = {
                "core": "Core matches",
                "explore": "Adjacent explorations",
                "wildcard": "Wildcard",
            }

            # Group by tier, preserve order
            tiers = ["core", "explore", "wildcard"]
            for tier in tiers:
                tier_recs = [r for r in st.session_state.recommendations_list if r.get("flight_tier") == tier]
                if not tier_recs:
                    continue

                st.markdown(f"### {tier_labels.get(tier, tier.title())}")

                num_recs_to_show = len(tier_recs)
                cols_to_display = min(num_recs_to_show, 3) if num_recs_to_show > 0 else 1
                cols = st.columns(cols_to_display)

                for i, rec_wine_meta in enumerate(tier_recs):
                    wine_pinecone_id = rec_wine_meta.get("pinecone_id")
                    rag_content_for_card = (
                        st.session_state.rag_explanations.get(wine_pinecone_id) if wine_pinecone_id else None
                    )

                    with cols[i % cols_to_display]:
                        display_wine_card(
                            rec_wine_meta,
                            card_key_prefix=f"rec_{tier}_{i}",
                            is_base_wine=False,
                            rag_explanation_content=rag_content_for_card,
                        )

                        if wine_pinecone_id:
                            fcol1, fcol2 = st.columns(2)

                            with fcol1:
                                if st.button("👍 More like this", key=f"fb_up_{wine_pinecone_id}"):
                                    rec_name = (rec_wine_meta.get("name") or "").strip()
                                    if rec_name:
                                        st.session_state.liked_wines.append(rec_name)
                                        run_new_search(rec_name, price_min_filter, price_max_filter)
                                        st.rerun()

                            with fcol2:
                                if st.button("👎 Not my style", key=f"fb_down_{wine_pinecone_id}"):
                                    st.session_state.disliked_ids.add(wine_pinecone_id)
                                    producer = (rec_wine_meta.get("brand") or "").strip().lower()
                                    if producer:
                                        st.session_state.blocked_producers.add(producer)

                                    st.session_state.recommendations_list = [
                                        w for w in st.session_state.recommendations_list
                                        if w.get("pinecone_id") != wine_pinecone_id
                                    ]
                                    st.rerun()

        elif st.session_state.initial_load_complete and not st.session_state.recommendations_list:
            st.info(
                f"I found '{st.session_state.base_wine_for_display.get('name', 'your chosen wine')}', but couldn't find other similar wines matching your filters. Try widening the price range."
            )

    # --- Progressive RAG fetching ---
    if (
        st.session_state.initial_load_complete
        and st.session_state.base_blurb_fetched
        and st.session_state.recommendations_list
        and st.session_state.base_wine_for_display
    ):
        idx_to_fetch = st.session_state.get("fetch_rag_next_idx", 0)
        if idx_to_fetch < len(st.session_state.recommendations_list):
            wine_to_explain = st.session_state.recommendations_list[idx_to_fetch]
            wine_pinecone_id_to_fetch = wine_to_explain.get("pinecone_id")

            if wine_pinecone_id_to_fetch and wine_pinecone_id_to_fetch not in st.session_state.rag_explanations:
                rag_data = wine_recommendation.generate_enhanced_rag_explanation(
                    st.session_state.base_wine_for_display,
                    wine_to_explain,
                )
                st.session_state.rag_explanations[wine_pinecone_id_to_fetch] = rag_data
                st.session_state.fetch_rag_next_idx = idx_to_fetch + 1
                st.rerun()

        elif idx_to_fetch >= len(st.session_state.recommendations_list) and st.session_state.new_search_triggered:
            st.session_state.new_search_triggered = False

    elif submit_button and not user_selected_wine_name:
        st.warning("Please select a wine name to get recommendations.")


if __name__ == "__main__":
    main_app_layout()
