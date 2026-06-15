import streamlit as st

from chatbot import clean_response, create_conversation


st.set_page_config(page_title="LangChain Chatbot", page_icon="💬")
st.title("LangChain Chatbot")
st.caption("Free local model with conversation memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    with st.spinner("Loading model (first run may take a minute)..."):
        st.session_state.conversation = create_conversation(verbose=False)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.invoke(input=prompt)
            reply = clean_response(response["response"])
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    with st.spinner("Resetting conversation..."):
        st.session_state.conversation = create_conversation(verbose=False)
    st.rerun()
