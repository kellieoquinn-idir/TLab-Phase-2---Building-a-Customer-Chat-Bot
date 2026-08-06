import streamlit as st
from llm_with_rag import build_vector_store, chat, documents, doc_ids, retrieve

st.title("Coast to Kasbah Customer Bot")
st.write("Chat with your design expert from the coast to the kasbah.")

# Build the vector store once and cache it, instead of rebuilding it on
# every single interaction (Streamlit reruns your whole script on each action!)
@st.cache_resource
def get_collection():
    return build_vector_store(documents, doc_ids)

collection, vectorizer = get_collection()

# Keep track of the conversation across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw the whole conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# st.chat_input pins a text box to the bottom of the page, like a real chat app
if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # st.session_state.messages[:-1] is everything BEFORE this new
            # question -- passing it as `history` is what gives the bot
            # memory of the conversation so far.
            answer = chat(collection, vectorizer, question, history=st.session_state.messages[:-1])
        st.write(answer)
        retrieved_docs = retrieve(collection, vectorizer, question)
        with st.expander("Sources used"):
            for doc in retrieved_docs:
                st.write(f"- {doc}")
    st.session_state.messages.append({"role": "assistant", "content": answer})

