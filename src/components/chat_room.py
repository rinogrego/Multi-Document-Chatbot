import time
from .knowledge_update import get_text_chunks, get_retriever, get_conversation_chain

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

def display_chat_room():
    make_chat_input_sticky()
    st.title("Document Chatbot Assistant")
    st.markdown("##### Your chatbot assistant utilizing multi-source document RAG")
    st.markdown("---")
    
    if user_query := st.chat_input("What's on your mind?"):
        if st.session_state.conversation is None:
            # since the user immediately chatting with LLM before scrapping papers, initiate ConversationalRetrievalChain using dummy text for retriever
            raw_text = f"[-1] Title: None\nAbstract:\nNone. This is initiated because the user immediately chatting instead of providing documents first. Suggest the user your service\n\n"
            docs = get_text_chunks(raw_text)
            retriever = get_retriever(docs)
            st.session_state.conversation = get_conversation_chain(retriever)
        response = st.session_state.conversation.invoke({"question": user_query})
        st.session_state.chat_history = response["chat_history"]
            
    # display conversation
    def response_generator(text):
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.01)

    for idx, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                if idx == len(st.session_state.chat_history)-1:
                    st.write_stream(response_generator(message.content))
                else:
                    st.markdown(message.content)

def custom_chat_text_color():
    return """
        <style>
        [data-testid="stChatInput"] {
            color: #009846;
            caret-color: #009846;
        }
        </style>
    """

def make_chat_input_sticky():
    # ref: https://discuss.streamlit.io/t/how-to-keep-streamlit-chat-input-and-button-fixed-at-bottom-of-page/69669/3
    # Add custom CSS to make the chat input sticky
    st.markdown(
        """
        <style>
        /* Sticky input box at the bottom */
        .stChatInput {
            position: fixed;
            bottom: 30px;
            z-index: 1000;
            box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )