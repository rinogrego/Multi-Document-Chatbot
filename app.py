import os

from src.components.knowledge_base import display_knowledge_base
from src.components.knowledge_update import display_knowledge_update
from src.components.chat_room import display_chat_room
from src.components.chat_history import display_chat_history

from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Document Chatbot Assistant",
    layout="centered",
)

def init():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pubmed_papers_keywords" not in st.session_state:
        st.session_state.pubmed_papers_keywords = []
        st.session_state.pubmed_papers_scrap_results = {}
        st.session_state.pubmed_papers_scrap_results_df_total = pd.DataFrame()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_titles" not in st.session_state:
        st.session_state.pdf_titles = []
        st.session_state.pdf_titles_metadata = {}
        st.session_state.pdf_contents = {}
    if "num_rag_contexts" not in st.session_state:
        st.session_state.num_rag_contexts = 10
    if "llm_model" not in st.session_state:
        st.session_state.llm_model_name = "gpt-4o-mini"
        st.session_state.llm_model = ChatOpenAI(
            model = st.session_state.llm_model_name, 
            temperature = 0.7,
            max_tokens = 2048,
            timeout = None,
            max_retries = 2,
            top_p = 0.95,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    if "pmcids" not in st.session_state:
        st.session_state.pmcids = []
        st.session_state.pmcids_metadata = {}
        st.session_state.pmcid_articles = {}
    if "wikipedia_searches" not in st.session_state:
        st.session_state.wikipedia_searches = {}

home_tab, tab1, tab2, tab3, tab4 = st.tabs(["Welcome", "Knowledge Base", "Add Knowledge", "Chat Room", "Chat History"])
with home_tab:
    st.header("Hello there!")
    st.markdown("""
        <p>Welcome! We're excited to have you here exploring our LLM app.</p>
        <p>Think of this space as your go-to companion for finding the information you need, pulling together knowledge from different corners to give you clear, meaningful answers. </p>
        <p>Whether you’re here to dig deep into a topic, automate some of your workflows, or simply see what our AI can uncover for you, we hope you find the experience both intuitive and enlightening.</p>
        <p>If there's anything you'd like to know or explore, don't hesitate to dive in—this space is built for curiosity and discovery, and we’re here to make that journey as seamless and insightful as possible for you!</p>
    """, unsafe_allow_html=True)

def main():
        
    with tab3:
        display_chat_room()

    with tab2:
        display_knowledge_update()
        
    with tab1:
        display_knowledge_base()
        
    with tab4:
        display_chat_history()

if __name__ == "__main__":
    init()
    main()