import streamlit as st

from food_advanced_search import (
    DEMONSTRATIONS,
    format_results_markdown,
    get_advanced_help_markdown,
    init_advanced_food_search,
    run_advanced_search,
)


@st.cache_resource(show_spinner="Loading food database and building advanced search index...")
def init_food_search():
    return init_advanced_food_search()


def handle_user_query(
    collection,
    query: str,
    cuisine_filter: str | None,
    max_calories: int | None,
    n_results: int,
):
    normalized = query.strip().lower()
    if normalized in {"help", "h"}:
        return get_advanced_help_markdown()

    results = run_advanced_search(
        collection,
        query,
        cuisine_filter=cuisine_filter,
        max_calories=max_calories,
        n_results=n_results,
    )
    return format_results_markdown(query, results)


def run_demonstration(collection, demo: dict, n_results: int):
    filters = []
    if demo["cuisine_filter"]:
        filters.append(f"cuisine: {demo['cuisine_filter']}")
    if demo["max_calories"]:
        filters.append(f"max calories: {demo['max_calories']}")
    filter_text = ", ".join(filters) if filters else "no filters"
    title = f"**{demo['title']}** (`{demo['query']}`, {filter_text})"

    results = run_advanced_search(
        collection,
        demo["query"],
        cuisine_filter=demo["cuisine_filter"],
        max_calories=demo["max_calories"],
        n_results=n_results,
    )
    return format_results_markdown(demo["query"], results, title=title)


st.set_page_config(
    page_title="Advanced Food Search",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stChatMessage {
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Advanced Food Search")
st.caption("Filtered semantic search powered by ChromaDB")

try:
    collection, food_items, cuisines = init_food_search()
except Exception as error:
    st.error(f"Failed to initialize the advanced food search system: {error}")
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
    st.subheader("Demonstrations")
    for index, demo in enumerate(DEMONSTRATIONS):
        if st.button(demo["title"], use_container_width=True, key=f"demo_{index}"):
            st.session_state.pending_demo = demo

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_demo = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to the advanced food search chatbot. "
                "Type a dish, ingredient, or description to search, "
                "or use the sidebar to apply cuisine and calorie filters. "
                "Try the demonstration examples to see combined filters in action."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

selected_cuisine = None if cuisine_filter == "Any" else cuisine_filter
selected_max_calories = None if max_calories == 0 else max_calories

if "pending_demo" in st.session_state and st.session_state.pending_demo:
    demo = st.session_state.pending_demo
    st.session_state.pending_demo = None

    user_message = f"Demo: {demo['title']} — {demo['query']}"
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Running demonstration..."):
            reply = run_demonstration(collection, demo, n_results)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

prompt = st.chat_input("Search for food...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            reply = handle_user_query(
                collection,
                prompt,
                selected_cuisine,
                selected_max_calories,
                n_results,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
