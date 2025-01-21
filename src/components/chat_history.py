import streamlit as st

from src.utils import save_chat_history, load_chat_history

def display_chat_history():
    st.markdown("### Manage Chat History")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Save Chat History")
        filename = st.text_input(label="JSON filename", value="chat_history")
        filename += ".json"
        if st.button("Save Chat History"):
            save_chat_history(filename)
    with col2:
        st.markdown("##### Load Chat History")
        json_filename = st.text_input("json file path containing chat history")
        json_filename += ".json"
        if st.button("Load Chat History"):
            load_chat_history(json_filename)
    
    st.markdown("### Current Chat History")
    st.write(st.session_state.chat_history)