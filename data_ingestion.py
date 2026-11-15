import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle

load_dotenv()

DATA_DIR = "./resources/organisation_data"

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = 'org_data'
EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BM25_DIR = "./bm25_index"

CHUNK_SIZE = 1000           # chunk size (characters)
CHUNK_OVERLAP = 100         # chunk overlap (characters)


def load_files(dir_name):
    all_docs = []

    for department in os.listdir(dir_name):
        dept_path = os.path.join(dir_name, department)

        if os.path.isdir(dept_path):
            for file in os.listdir(dept_path):
                file_path = os.path.join(dept_path, file)
                docs = []
                
                if file.endswith(".md"):
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()   
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                for doc in docs:
                    if 'page' in doc.metadata:
                        page_number = doc.metadata['page'] + 1
                    else:
                        page_number = None     # Markdown has no concept of 'page'
                    doc.metadata = {
                        'role': department.lower(),
                        'source': file,
                        'page': page_number
                    }

                all_docs.extend(docs)

    return all_docs

def chunk_docs(docs, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,        # chunk size (characters)
        chunk_overlap = chunk_overlap   # chunk overlap (characters)
    )
    
    split_docs = text_splitter.split_documents(docs)

    return split_docs

def ingest_in_vector_db(docs):
    if not docs:
        print("No documents to ingest into ChromaDB.")
        return

    vector_store = Chroma( 
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=EMBEDDING_FUNCTION
    )
    try:
        vector_store.delete_collection()
        print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
    except Exception as e:
        print(f"Problem occured while deleting : {e}")
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=EMBEDDING_FUNCTION
    )
    vector_store.add_documents(documents=docs)

    print('='*100)
    print(f'\nSUCCESS : {len(docs)} Embeddings from provided documents have been stored in Vector DB successfully.\n')
    print('='*100)

def create_bm25_index(docs):
    if not docs:
        print("No documents to build BM25 index.")
        return

    os.makedirs(BM25_DIR, exist_ok=True)

    # Building separate BM25 indexes per department because LangChain's BM25Retriever does not support metadata filtering (like vector db)
    
    # Complete Index (For C-Level)
    with open(os.path.join(BM25_DIR, "bm25_index_complete.pkl"), "wb") as f:
        pickle.dump(BM25Retriever.from_documents(docs), f)

    # Role-specific Indexes
    roles = {d.metadata.get("role") for d in docs if d.metadata.get("role")}
    for role in roles:
        role_docs = [d for d in docs if d.metadata.get("role") == role]
        if role_docs:
            file_path = os.path.join(BM25_DIR, f"bm25_index_{role}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(BM25Retriever.from_documents(role_docs), f)    

    print('='*100)
    print(f'\nSUCCESS : BM25 indexing complete.\n')
    print('='*100)

def main():
    loaded_docs = load_files(DATA_DIR)
    all_split_docs = chunk_docs(loaded_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    ingest_in_vector_db(all_split_docs)
    create_bm25_index(all_split_docs)

if __name__=="__main__":
    main()