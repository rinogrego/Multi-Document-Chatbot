import os
from src.scrapper import scrap_pubmed_abstract, scrap_pubmed_article
from src.utils import get_pdf_text, get_text_chunks, get_retriever, get_conversation_chain

from langchain_community.retrievers import WikipediaRetriever
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def load_abstracts_from_pubmed(
    keywords: str, 
    retmax: int = 10,
    attempt: int = 1,
) -> pd.DataFrame:
    try:
        df = scrap_pubmed_abstract(query=keywords, retmax=retmax)
        return df[['Title', 'Abstract', 'Journal', 'Year', 'Month', 'PMID', 'DOI']], keywords
    except RuntimeError as e: # Supplied id parameter is empty.
        # if previous scrapping attempt is not successful, reduce keywords to increase the chance of successful scrap
        st.write("Encountered Error")
        st.write(e)
        llm = st.session_state.llm_model
        messages = [
            ("system", f"You are a helpful assistant that paraphrase or reduce the pubmed search keywords without losing most information. Paraphrase or reduce the keywords of this failed keywords to extract pubmed abstracts"),
            ("human", keywords),
        ]
        ai_msg = llm.invoke(messages)
        new_keywords = ai_msg.content
        st.write(f"Attempt      : {attempt+1}")
        st.write(f"New Keywords : {new_keywords}")
        return load_abstracts_from_pubmed(keywords=new_keywords, retmax=retmax, attempt=attempt+1)

def display_knowledge_update():
    st.subheader("Global State", divider = True)
    col1, col2 = st.columns(2)
    with col1:
        table_display = f"""
        | Criteria | Value |
        | --- | --- |
        | OpenAI LLM Used | {st.session_state.llm_model_name} |
        | Number of RAG Contexts | {st.session_state.num_rag_contexts} |
        """
        st.write(table_display)
    with col2:
        form_global_state  = st.form(key="form_global_state")
        llm_model = form_global_state.selectbox(
            "Choose your model", 
            options=["gpt-4o-mini", "gpt-4o"],
        )
        k_docs_for_rag = form_global_state.number_input(
            "Number of documents for RAG contexts", 
            min_value = 1,
            max_value = 50,
            value = 10,
            step = 1,
            key = "k_docs_for_rag"
    )
        if form_global_state.form_submit_button("Update Global State"):
            st.session_state.num_rag_contexts = k_docs_for_rag
            from src.utils import change_model
            change_model(llm_model)
            st.rerun()
    
    # Custom PDF files
    st.markdown("---")
    st.subheader("Upload PDF Files", divider = True)
    form_pdf_uploads = st.form(key="form_pdf_uploads")
    pdf_docs = form_pdf_uploads.file_uploader("**Upload your PDFs here and click on Process**", accept_multiple_files=True)
    st.write("**WARNING**: The citation prompt is still bad")
    if form_pdf_uploads.form_submit_button("Process"):
        raw_text, docs = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        retriever = get_retriever(text_chunks)
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(retriever)
        else:
            st.session_state.conversation.retriever = retriever
    
    # PubMed Section
    st.markdown("---")
    st.subheader("Scrap PubMed Abstracts", divider=True)
    form = st.form(key="form_pubmed_search")
    search_query = form.text_input("**Enter your keywords:**", placeholder="e.g.: diabetes machine learning")
    pubmed_retmax = form.number_input(
        "Number of pubmed abstracts to retrieve", 
        min_value = 5,
        max_value = 100,
        value = 10,
        step = 1,
        key = "pubmed_retmax"
    )
    if form.form_submit_button("Search"):
        st.subheader("Search query entered")
        st.write(
            "Parameter config:<br>Entrez's retmax: {}".format(pubmed_retmax),
            unsafe_allow_html=True
        )
        with st.spinner("Searching Relevant Papers"):
            if search_query not in st.session_state.pubmed_papers_keywords:
                df_title_abstracts, keywords = load_abstracts_from_pubmed(
                    search_query, 
                    retmax = pubmed_retmax,
                )
                st.session_state.pubmed_papers_keywords.append(keywords)
                st.session_state.pubmed_papers_scrap_results[keywords] = {
                    "df_title_abstracts": df_title_abstracts
                }
                st.session_state.pubmed_papers_scrap_results_df_total = pd.concat([st.session_state.pubmed_papers_scrap_results_df_total, df_title_abstracts], axis=0)
            else:
                df_title_abstracts = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]
            
            # constructing raw text for contexts to create vectorstore
            # NOTE: may change this method into create a Document object for each paper and its metadata
            raw_text = ""
            idx = 1
            for search_query in st.session_state.pubmed_papers_keywords: 
                paper_titles = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]["Title"]
                abstracts = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]["Abstract"]
                journals = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['Journal']
                years = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['Year']
                months = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['Month']
                pmids = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['PMID']
                dois = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['DOI']
                for title, abs, journal, year, month, pmid, doi in zip(paper_titles, abstracts, journals, years, months, pmids, dois):
                    raw_text += f"[{idx}] Title: {title}\nAbstract:\n{abs}\n\n"
                    idx += 1
                                
            docs = get_text_chunks(raw_text)
            retriever = get_retriever(docs)
            # somehow this part is reloaded whenever doing chat so implement checking to prevent resetting session_state
            if st.session_state.conversation is None:
                st.session_state.conversation = get_conversation_chain(retriever)
            else:
                # update retriever to accomodate new documents
                st.session_state.conversation.retriever = retriever

    # PubMed Full Articles
    st.markdown("---")
    st.subheader("Crawl PubMed Article", divider=True)
    form_pubmed_article = st.form(key="form_pubmed_full_papers_search")
    pmcid = form_pubmed_article.text_input("**Enter PMCID:**", placeholder="e.g.: PMC8822225")
    if form_pubmed_article.form_submit_button("Get Article"): 
        if pmcid not in st.session_state.pmcids:
            raw_text, metadata = scrap_pubmed_article(pmcid=pmcid)
            st.session_state.pmcids.append(pmcid)
            st.session_state.pmcids_metadata[pmcid] = metadata
            st.session_state.pmcid_articles[pmcid] = {
                "raw_text": raw_text
            }
        else:
            raw_text = st.session_state.pmcid_articles[pmcid]["raw_text"]
        # raw_text, docs = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        retriever = get_retriever(text_chunks)
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(retriever)
        else:
            st.session_state.conversation.retriever = retriever
    
    # Wikipedia Article
    st.markdown("---")
    st.subheader("Scrap Wikipedia Articles", divider=True)
    wiki_article = st.form(key="wikipedia_articles_search")
    wiki_keywords = wiki_article.text_input("**Enter Wikipedia Keywords:**", placeholder="e.g.: Bioinformatics")
    if wiki_article.form_submit_button("Get Wikipedia Article"): 
        if wiki_keywords.lower() not in st.session_state.wikipedia_searches:
            wiki_retriever = WikipediaRetriever()
            docs = wiki_retriever.invoke(wiki_keywords)
            st.session_state.wikipedia_searches[wiki_keywords.lower()] = docs
        else:
            docs = st.session_state.wikipedia_searches[wiki_keywords.lower()]
        retriever = get_retriever(docs)
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(retriever)
        else:
            st.session_state.conversation.retriever = retriever
    
    # # arXiv paper
    # st.markdown("---")        
    # st.subheader("Scrap arXiv Articles", divider=True)
    
    # # arXiv Abstracts
    # st.markdown("---")
    # st.subheader("Scrap arXiv Abstracts", divider=True)
    
    # # Duckduckgo search
    # st.markdown("---")
    # st.subheader("Duck Duck Go Search", divider=True)
