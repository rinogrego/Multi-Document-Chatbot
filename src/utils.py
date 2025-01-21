import os
from typing import Tuple
import json

from src.config import EMBEDDING_MODEL

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs) -> Tuple[str, Document]:
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
        embeddings = EMBEDDING_MODEL
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
            # print(f"Document ID : {doc_id}")
            # print(f"Doc         : {doc}")
            # print(f"Content: {doc['page_content']}")
            # print(f"Metadata: {doc['metadata']}")
            # print("\n" + "-"*50 + "\n")
            docs.append(doc)
    
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
    return retriever

def get_conversation_chain(retriever):
    llm = st.session_state.llm_model
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question",
        output_key="answer", # because issue: https://github.com/langchain-ai/langchain/issues/2303#issuecomment-1508973646
        return_messages=True
    )
    
    ### SYSTEM PROMPT
    SYSTEM_TEMPLATE = """You are a knowledgeable and helpful chatbot assistant. 
    You have access to a variety of information, including uploaded PDFs and PubMed abstracts. 
    Use the provided context to answer the user's question as accurately as possible.

    If the answer cannot be found within the given context, be honest and state that you don't know, along with the reason. Do not attempt to generate or fabricate information.

    The context comes from multiple sources. Always cite the source of your answer using the format:
    - [Title_of_Document]
    - If a page number is available, cite it as (Title_of_Document, page Page_Number).
    - When referencing multiple statements from different pages, cite each one appropriately.

    If the user asks about topics unrelated to PubMed literature, engage in a normal conversation while subtly suggesting that they take advantage of your research capabilities.

    If the user inquires about how to use your service, provide clear, step-by-step guidance in a friendly and paraphrased manner:

    1. Navigate to the "Add Knowledge" tab to add new information.
    2. You can either upload PDFs for me to learn from or provide keywords so I can retrieve paper abstracts from PubMed.
    3. Wait for the processing or scraping to complete.
    4. Check the "Knowledge Base" tab to see the current collection of information I can use to assist you.
    5. If you need to add more knowledge, simply repeat the process.

    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
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

def change_model(model_name):
    st.session_state.llm_model_name = model_name
    st.session_state.llm_model = ChatOpenAI(
        model = model_name, 
        temperature = 0.7,
        max_tokens = 2048,
        timeout = None,
        max_retries = 2,
        top_p = 0.95,
        api_key=os.getenv("OPENAI_API_KEY")
    )

def serialize_chat_history(chat_history):
    """
    Convert chat history objects into a JSON serializable format.
    """
    serialized = []
    for message in chat_history:
        if isinstance(message, AIMessage):
            serialized.append({"AI": message.content})
        elif isinstance(message, HumanMessage):
            serialized.append({"Human": message.content})
        else:
            serialized.append({"Unknown": str(message)})
    return serialized

def deserialize_chat_history(chat_history):
    """
    Convert JSON data back into chat history objects.
    """
    deserialized = []
    for entry in chat_history:
        if "AI" in entry:
            deserialized.append(AIMessage(content=entry["AI"]))
        elif "Human" in entry:
            deserialized.append(HumanMessage(content=entry["Human"]))
        else:
            deserialized.append(entry)
    return deserialized

def save_chat_history(filename):
    """
    Save the chat history stored in st.session_state.chat_history to a JSON file.
    """
    if "chat_history" in st.session_state:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serialize_chat_history(st.session_state.chat_history), f, indent=4, ensure_ascii=False)
        st.success("Chat history saved successfully!")
    else:
        st.warning("No chat history found to save.")

def load_chat_history(filename):
    """
    Load the chat history from a JSON file into st.session_state.chat_history.
    """
    filename = os.path.join(os.getcwd(), filename)
    try:
        st.write(filename)
        with open(filename, "r", encoding="utf-8") as f:
            st.session_state.chat_history = deserialize_chat_history(json.load(f))
        st.success("Chat history loaded successfully!")
    except FileNotFoundError:
        st.warning("No saved chat history found.")
    except json.JSONDecodeError:
        st.error("Error loading chat history. The file may be corrupted.")

