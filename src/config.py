import os

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=os.getenv("OPENAI_API_KEY")
)

PROJECT_ID = "project_init"
VECTOR_DB_NAME = "faiss_index"
PROJECT_DIR_PATH = f"storage/{PROJECT_ID}/"                             # folter to store all necessities of a Project
VECTORSTORE_PATH = f"storage/{PROJECT_ID}/vectorstore/{VECTOR_DB_NAME}" # folder to store the vectordb of a Project
FILES_STORAGE_PATH = f"storage/{PROJECT_ID}/files/{VECTOR_DB_NAME}"     # folder to store all files uploaded on a Project
CONVERATION_DIR_PATH = f"storage/{PROJECT_ID}/conversations"            # folder to store chat_history for each converstation on a Project
DATABASE_DIR_PATH = f"storage/tables"                                   # folder to store relational databases for the App: Project_Table, Conversation_Table