from pathlib import Path

import streamlit as st

from food_recommendation_functions import (
    create_similarity_search_collection,
    load_food_data,
    perform_filtered_similarity_search,
    perform_similarity_search,
    populate_similarity_collection,
)

DATA_PATH = Path(__file__).parent / "FoodDataSet.json"
EXAMPLE_SEARCHES = [
    "chocolate dessert",
    "Italian food",
    "low calorie",
    "sweet treats",
    "baked goods",
    "hearty meal",
]


@st.cache_resource(show_spinner="Loading food database and building search index...")
def init_food_search():
    food_items = load_food_data(str(DATA_PATH))
    collection = create_similarity_search_collection(
        "interactive_food_search",
        {"description": "A collection for interactive food search"},
    )
    populate_similarity_collection(collection, food_items)
    cuisines = sorted({item.get("cuisine_type", "Unknown") for item in food_items})
    return collection, food_items, cuisines


def format_results_markdown(query: str, results: list[dict]) -> str:
    if not results:
        return (
            f"No matching foods found for **{query}**.\n\n"
            "Try different keywords such as cuisine types (`Italian`, `American`), "
            "ingredients (`chocolate`, `cheese`), or descriptors (`sweet`, `baked`, `dessert`)."
        )

    lines = [f"Found **{len(results)}** recommendations for **{query}**:\n"]
    for index, result in enumerate(results, start=1):
        score = result["similarity_score"] * 100
        lines.extend(
            [
                f"### {index}. {result['food_name']}",
                f"**Match:** {score:.1f}% · **Cuisine:** {result['cuisine_type']} · "
                f"**Calories:** {result['food_calories_per_serving']} per serving",
                f"{result['food_description']}",
                "",
            ]
        )

    cuisines = sorted({result["cuisine_type"] for result in results})
    suggestions = [f"- Try `{cuisine} dishes` for more {cuisine} options" for cuisine in cuisines[:3]]
    avg_calories = sum(result["food_calories_per_serving"] for result in results) / len(results)
    if avg_calories > 350:
        suggestions.append("- Try `low calorie` for lighter options")
    else:
        suggestions.append("- Try `hearty meal` for more substantial dishes")

    lines.append("**Related searches:**")
    lines.extend(suggestions)
    return "\n".join(lines)


def run_search(collection, query: str, cuisine_filter: str | None, max_calories: int | None, n_results: int):
    if cuisine_filter or max_calories:
        return perform_filtered_similarity_search(
            collection,
            query,
            cuisine_filter=cuisine_filter,
            max_calories=max_calories,
            n_results=n_results,
        )
    return perform_similarity_search(collection, query, n_results)


def handle_user_query(collection, query: str, cuisine_filter: str | None, max_calories: int | None, n_results: int):
    normalized = query.strip().lower()
    if normalized in {"help", "h"}:
        return (
            "**How to use this chatbot**\n\n"
            "- Type any food name, ingredient, or description to search.\n"
            "- Use the sidebar to filter by cuisine or maximum calories.\n"
            "- Example searches: `chocolate dessert`, `Italian food`, `low calorie`.\n"
            "- Use **Clear chat** in the sidebar to reset the conversation."
        )

    results = run_search(collection, query, cuisine_filter, max_calories, n_results)
    return format_results_markdown(query, results)


st.set_page_config(
    page_title="Food Recommendation Chatbot",
    page_icon="🍽️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .food-header {
        margin-bottom: 0.25rem;
    }
    .stChatMessage {
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍽️ Food Recommendation Chatbot")
st.caption("Semantic search over your food database powered by ChromaDB")

try:
    collection, food_items, cuisines = init_food_search()
except Exception as error:
    st.error(f"Failed to initialize the food recommendation system: {error}")
    st.stop()

with st.sidebar:
    st.header("Search options")
    st.metric("Foods indexed", len(food_items))

    cuisine_filter = st.selectbox(
        "Cuisine filter",
        options=["Any"] + cuisines,
        index=0,
    )
    max_calories = st.slider(
        "Maximum calories per serving",
        min_value=0,
        max_value=800,
        value=0,
        step=25,
        help="Set to 0 to disable the calorie filter.",
    )
    n_results = st.slider("Results per search", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Try an example")
    for example in EXAMPLE_SEARCHES:
        if st.button(example, use_container_width=True, key=f"example_{example}"):
            st.session_state.pending_query = example

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! Ask me for food recommendations by typing a dish, ingredient, "
                "cuisine, or description. For example: `spicy Asian noodles` or `low calorie dessert`."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

selected_cuisine = None if cuisine_filter == "Any" else cuisine_filter
selected_max_calories = None if max_calories == 0 else max_calories

prompt = st.chat_input("Search for food...")
if "pending_query" in st.session_state and st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for recommendations..."):
            reply = handle_user_query(
                collection,
                prompt,
                selected_cuisine,
                selected_max_calories,
                n_results,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
