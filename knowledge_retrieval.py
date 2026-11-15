from dotenv import load_dotenv
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableParallel
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.tools import tool

load_dotenv()

#RAG_MODEL = "qwen/qwen3.8-27b"
RAG_MODEL = "openai/gpt-oss-120b"

RAG_LLM = ChatGroq(model=RAG_MODEL, temperature=0)

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = 'org_data'
EMBEDDING_FUNCTION = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

COMPRESSOR = CohereRerank(model="rerank-english-v3.0", top_n=5)

ROLE_CONFIGS = {
    'c-level': {
        'allowed_roles' : ['general', 'engineering', 'marketing', 'finance'],
        'index_path'    : "./bm25_index/bm25_index_complete.pkl"
    },
    'engineering': {
        'allowed_roles' : ['general', 'engineering'],
        'index_path'    : "./bm25_index/bm25_index_engineering.pkl"
    },
    'marketing': {
        'allowed_roles' : ['general', 'marketing'],
        'index_path'    : "./bm25_index/bm25_index_marketing.pkl"
    },
    'finance': {
        'allowed_roles' : ['general', 'finance'],
        'index_path'    : "./bm25_index/bm25_index_finance.pkl"
    },
    'general': {
        'allowed_roles' : ['general'],
        'index_path'    : "./bm25_index/bm25_index_general.pkl"
    }
}

VECTOR_STORE = Chroma( 
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=EMBEDDING_FUNCTION
)

PROMPT_FOR_RAG_LLM = ChatPromptTemplate.from_template(
"""
You are an information retrieval assistant for a FinTech company.
If the question is out of scope, say: "I can't help you with that. Please ask questions relevant to your role."
Answer the question using ONLY the retrieved context.
If the answer is not in context, say: "This information is not available in the documents accessible to your role."
Don't hallucinate or make things up.
Always give source of information using the metadata like document name, page number.
Use a friendly and clear conversational style, like you're explaining it to a colleague.
Avoid giving unnecessarily long answers.

Context:
{context}

Question:
{input}
""")

def load_bm25_retriever(file_path: str):
    with open(file_path, "rb") as f:
        retriever = pickle.load(f)
    return retriever  

def format_docs(docs):
    """
    Formats document content and injects metadata cleanly for the LLM prompt.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Document')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"Source: {source} (Page {page}) Content: {doc.page_content}")
    return "\n\n".join(formatted)
    # compression_retriever passes a list of raw Document objects straight to prompt template string.
    # LangChain prompt template expects a string for variables, which means documents will implicitly cast to strings,
    # ....  outputting ugly code-like representations (e.g., [Document(page_content='...', metadata=...)]) into the LLM context window.

def build_rag_chain(ensemble_retriever):
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=COMPRESSOR, 
        base_retriever=ensemble_retriever
    )

    retrieval_step = RunnableParallel(
        docs=compression_retriever,
        input=RunnablePassthrough()
    )

    rag_chain = retrieval_step | RunnableParallel(
        answer=(
            {
                "context": lambda x: format_docs(x["docs"]),
                "input": lambda x: x["input"]
            }
            | PROMPT_FOR_RAG_LLM
            | RAG_LLM
            | (lambda msg: msg.content)
        ),
        docs=lambda x: x["docs"]
    )

    return rag_chain

# Pre-building chains at application startup to save runtime execution latency
CHAIN_REGISTRY = {}
def initialize_chains():
    for user_role in ROLE_CONFIGS.keys():

        metadata_filter_config = ROLE_CONFIGS.get(user_role)
        
        allowed_roles = metadata_filter_config['allowed_roles']
        bm25_index_path = metadata_filter_config['index_path']
    
        vector_retriever = VECTOR_STORE.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 10,
                'filter': {'role': {"$in": allowed_roles}}
            }
        )
    
        bm25_retriever = load_bm25_retriever(bm25_index_path)
        bm25_retriever.k = 10
    
        ensemble_retriever = EnsembleRetriever(
            retrievers = [bm25_retriever, vector_retriever],
            weights = [0.5, 0.5]
        )
        
        CHAIN_REGISTRY[user_role] = build_rag_chain(ensemble_retriever)
# Run initialization once at module load
initialize_chains()


@tool(response_format="content_and_artifact")
def search_company_docs(query: str, config: RunnableConfig) -> tuple[str, list]:
    """
    Queries the internal organization documents to answer questions about 
    financial reports, engineering documentation, marketing plans, policies, corporate data etc.

    Args:
        query (str): The question to ask the knowledge base.
    """
    # Extract user_role safely (without llm seeing it) from the runtime configuration. If not provided, fallback to default
    user_role = config.get("configurable", {}).get("user_role", "general")

    # Fetch the pre-built chain from registry. Fallback dynamically if missing.
    rag_chain = CHAIN_REGISTRY.get(user_role, CHAIN_REGISTRY['general'])
    
    result = rag_chain.invoke(query)

    return result["answer"], result["docs"]

'''
Token usage has NOT increased by a single token.
Here is why:
1. How response_format="content_and_artifact" works in LangChain:
When a @tool has response_format="content_and_artifact" and returns (result["answer"], result["docs"]), LangChain splits that tuple inside ToolMessage:
	• ToolMessage.content = result["answer"] (the short string)
	• ToolMessage.artifact = result["docs"] (stored locally in Python memory only)
2. What is actually sent to Groq (AGENT_LLM):
When LangChain formats ToolMessage to send back to the Groq API (AGENT_LLM), it completely ignores .artifact and ONLY sends .content. The retrieved docs are never serialized into the LLM prompt.
'''