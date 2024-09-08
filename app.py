import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever

from dotenv import load_dotenv
from utils.scrap_pubmed_abstracts import scrap_pubmed
import pandas as pd

import os
import base64
from typing import Tuple, List

load_dotenv()

st.set_page_config(
    page_title="Document Chatbot Assistant",
    layout="centered"
)

tab1, tab2, tab3 = st.tabs(["Knowledge Base", "Add Knowledge", "Chat Room"])


def get_pdf_text(pdf_docs) -> Tuple[str, Document]:
    # from: https://stackoverflow.com/a/76816979/13283654
    # process pdf_docs to langchain's Document object
    docs = []
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        title = reader.metadata.title if reader.metadata.title else "Untitled"
        # If metadata is not available, try to infer the title from the first page
        if title == "Untitled" and reader.pages:
            first_page_text = reader.pages[0].extract_text()
            # Consider the first line or the first few words as the title
            title = first_page_text.split('\n')[0].strip() if first_page_text else "Untitled"
        
        if title not in st.session_state.pdf_titles:
            st.session_state.pdf_titles.append(title)
            st.session_state.pdf_titles_metadata[title] = {"num_pages": len(reader.pages)}
            st.session_state.pdf_contents[title] = pdf
        
        text += f"Title: {title}"
        
        for page in reader.pages:
            page_text = page.extract_text()
            text += f"Page: {page.page_number}"
            text += f"Text: {page_text}"
            docs.append(
                Document(
                    page_content = page_text, 
                    metadata = {
                        'page': page.page_number, 
                        'title': title
                    }
                )
            )
    return text, docs

def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=4000,
        chunk_overlap=400,
        length_function=len
    )
    docs = text_splitter.create_documents([text])
    docs = text_splitter.split_documents(documents=docs)
    return docs

def get_retriever(docs, update_docs=True):
    if update_docs:
        # update the vectorstore
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        else:
            st.session_state.vectorstore.add_documents(documents=docs)
            
        faiss_index = st.session_state.vectorstore
        # Retrieve all data from the FAISS index
        all_data = faiss_index.docstore._dict
        # Print all data
        docs = []
        for doc_id, doc in all_data.items():
            print(f"Document ID : {doc_id}")
            print(f"Doc         : {doc}")
            # print(f"Content: {doc['page_content']}")
            # print(f"Metadata: {doc['metadata']}")
            print(type(doc))
            print("\n" + "-"*50 + "\n")
            docs.append(doc)
    
    # from: https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22
    retriever_vectordb = st.session_state.vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": st.session_state.num_rag_contexts}
    )
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k =  st.session_state.num_rag_contexts
    retriever = EnsembleRetriever(
        retrievers = [retriever_vectordb, keyword_retriever],
        weights = [0.5, 0.5]
    )
    # maybe implement custom invoke for EnsembleRetriever
    # source: https://api.python.langchain.com/en/latest/core/retrievers/langchain_core.retrievers.BaseRetriever.html#langchain_core.retrievers.BaseRetriever
    return retriever

def get_conversation_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question",
        output_key="answer", # because issue: https://github.com/langchain-ai/langchain/issues/2303#issuecomment-1508973646
        return_messages=True
    )
    
    ### SYSTEM PROMPT
    # from: https://github.com/langchain-ai/langchain/issues/5462#issuecomment-1569923207
    system_template = """You are a helpful chatbot assistant. 
    You are equipped with various knowledges from uploaded pdfs to pubmed abstracts.
    Use the following pieces of context to answer the users question.
    If you cannot find the answer from the pieces of context, just say that you don't know and state the reason why you don't know. Don't try to make up an answer.
    The context given came from various sources. 
    Always cite the source of your answer by using [title_of_document].
    If there is a number page, then cite using (title_of_document, page page_number).
    If there are multiple statements from different pages, cite properly for each statement.
    If the user is not asking something regarding pubmed literatures, engage in normal conversation, but suggest the user to leverage your service.
    If the user ask how to use your service, guide the user to follow the instructions. Write each step by new lines. Also paraphrase to make it more friendly.
    <ol>
        <li>Go to "Add Knowledge" tab to add information</li>
        <li>You can either add my knowledge by providing me with your uploaded PDFs, or by give me keywords so that I can try scrap paper abstracts from PubMed</li>
        <li>Wait until the loading or scrapping process is completed</li>
        <li>You can look at the current state of information that I can consider use to help you in "Knowledge Base" tab</li>
        <li>If you want to add more information, just repeat the same process of adding knowledge</li>
    </ol>
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    ### CONDENSE QUESTION PROMPT
    # from: https://github.com/langchain-ai/langchain/issues/4076#issuecomment-1563138403
    condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Preserve the original question in the answer sentiment during rephrasing.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    condense_question_prompt = PromptTemplate.from_template(condense_question_template)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
        combine_docs_chain_kwargs = {"prompt": qa_prompt},
        condense_question_prompt = condense_question_prompt,
        
    )
    return conversation_chain

def load_abstracts_from_pubmed(
    keywords: str, 
    retmax: int = 10,
    attempt: int = 1,
) -> pd.DataFrame:
    try:
        df = scrap_pubmed(query=keywords, retmax=retmax)
        return df[['Title', 'Abstract', 'Journal', 'Year', 'Month', 'PMID', 'DOI']], keywords
    except RuntimeError as e: # Supplied id parameter is empty.
        # if previous scrapping attempt is not successful, reduce keywords to increase the chance of successful scrap
        st.write("Encountered Error")
        st.write(e)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        messages = [
            ("system", f"You are a helpful assistant that paraphrase or reduce the pubmed search keywords without losing most information. Paraphrase or reduce the keywords of this failed keywords to extract pubmed abstracts"),
            ("human", keywords),
        ]
        ai_msg = llm.invoke(messages)
        new_keywords = ai_msg.content
        st.write(f"Attempt      : {attempt+1}")
        st.write(f"New Keywords : {new_keywords}")
        return load_abstracts_from_pubmed(keywords=new_keywords, retmax=retmax, attempt=attempt+1)

def main():
    import time
    
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
        
    with tab3:
        st.title("Document Chatbot Assistant")
        st.markdown("##### Your chatbot assistant utilizing multi-source document RAG")
        st.markdown("---")
        
        # user input
        user_query = st.chat_input("What's on your mind?")
        if user_query is not None and user_query != "":
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
                time.sleep(0.05)
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
                    
    with tab2:
        st.subheader("RAG Contexts", divider = True)
        form_rag_contexts  = st.form(key="form_rag_contexts")
        k_docs_for_rag = form_rag_contexts.number_input(
            "Number of documents for RAG contexts", 
            min_value = 1,
            max_value = 50,
            value = 5,
            step = 1,
            key = "k_docs_for_rag"
        )
        if form_rag_contexts.form_submit_button("Update Context Number"):
            st.session_state.num_rag_contexts = k_docs_for_rag
        
        st.subheader("Upload PDF Files", divider = True)
        pdf_docs = st.file_uploader("**Upload your PDFs here and click on Process**", accept_multiple_files=True)
        st.write("**WARNING**: The citation prompt is still bad")
        if pdf_docs:
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text, docs = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    retriever = get_retriever(text_chunks)
                    if st.session_state.conversation is None:
                        st.session_state.conversation = get_conversation_chain(retriever)
                    else:
                        st.session_state.conversation.retriever = retriever
        st.markdown("---")
        
        # PubMed Section
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

    with tab1:
        st.subheader("PDFs Uploaded", divider = True)
        # display_pdfs_uploaded
        # # print relevant page
        # page_num = 5
        # for doc in documents:
        #     if doc.metadata["page"] == page_num:
        #         page_doc = f"<h4>Page {page_num}</h4>{doc.page_content}"
        #         break
        #     else:
        #         page_doc = "This document don't have that page number"
        # st.write(page_doc, unsafe_allow_html=True)
        # st.markdown("---")
        if len(st.session_state.pdf_titles) == 0:
            no_pdf_html = """
            <div style="display: flex; justify-content: center; align-items: center; color: #ff4b4b; margin-top: 5px;">
                <div style="padding: 5px">
                    <p style="font-size: 20px; text-align: center;">No PDF uploaded yet.</p>
                    <p style="font-size: 16px; text-align: center;">You can upload your PDF in Add Knowledge tab</p>
                </div>
            </div>
            """
            st.write(no_pdf_html, unsafe_allow_html=True)
        else:
            
            for title in st.session_state.pdf_titles:
                pdf_html = f"""
                <h5>{title}</h5>
                <p>Pages: {st.session_state.pdf_titles_metadata[title]["num_pages"]}</p>
                <hr>
                """
                st.write(pdf_html, unsafe_allow_html=True)
            
            # # from: https://discuss.streamlit.io/t/rendering-pdf-on-ui/13505
            # pdf_title = st.session_state.pdf_titles[0]
            # st.write(pdf_title)
            # pdf = st.session_state.pdf_contents[pdf_title]
            # # Read the PDF file as binary and encode it in base64
            # # base64_pdf = base64.b64encode(pdf.read()).decode('utf-8')
            # pdf_viewer(pdf.getvalue())
            # # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="500" type="application/pdf"></iframe>'
            # # st.markdown(pdf_display, unsafe_allow_html=True)
            
        st.markdown("---")
        
        st.subheader("PubMed Abstracts Scrapped", divider = True)
        if len(st.session_state.pubmed_papers_keywords) == 0:
            no_pubmed_html = """
            <div style="display: flex; justify-content: center; align-items: center; color: #ff4b4b; margin-top: 5px;">
                <div style="padding: 5px">
                    <p style="font-size: 20px; text-align: center;">No PubMed abstracts scrapped yet.</p>
                    <p style="font-size: 16px; text-align: center;">You can scrap PubMed abstracts in Add Knowledge tab</p>
                </div>
            </div>
            """
            st.write(no_pubmed_html, unsafe_allow_html=True)
        else:
            # bonus with author info please
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
                expander = st.expander(f"**Keywords: {search_query} ({idx}-{idx-1+len(abstracts)})**")
                for title, abs, journal, year, month, pmid, doi in zip(paper_titles, abstracts, journals, years, months, pmids, dois):
                    expander.write(
                        f"""<h5>[{idx}] {title}</h5>
                        <p>{abs}</p>
                        <p>
                            Journal : {journal} <br>
                            Date &emsp;&nbsp;: {month}, {year} <br>
                            PMID &emsp;: <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">{pmid}</a><br>
                            doi&emsp;&emsp;: <a href="https://www.doi.org/{doi}">{doi}</a>
                        </p>
                        <hr>""", 
                        unsafe_allow_html=True
                    )
                    idx += 1
        
        st.markdown("---")
if __name__ == "__main__":
    main()