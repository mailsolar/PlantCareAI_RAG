import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from fastapi import HTTPException 

# --- Configuration ---
VECTOR_DB_DIR = "vector_db"
POLICY_DIRECTORY_PATH = "policy/" 

# --- Global Components (Initialized to None) ---
# These must be None initially so the server can start instantly.
vectorstore = None
rag_chain = None

# --- Initialization Helpers ---

def get_gemini_api_key():
    """Gets the API key from environment variables, checking both common names."""
    # Check for GEMINI_API_KEY first, then GOOGLE_API_KEY
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("API Key (GEMINI_API_KEY or GOOGLE_API_KEY) not found. Check your environment.")
    return gemini_api_key

def create_rag_clients():
    """Initializes the LLM and Embedding clients."""
    gemini_api_key = get_gemini_api_key()
    
    # Force the key into the environment variable that LangChain is often hard-coded to check.
    os.environ["GOOGLE_API_KEY"] = gemini_api_key 

    embeddings_client = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    ) 

    llm_client = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1
    )
    return embeddings_client, llm_client

def create_vector_store(embeddings_client):
    """Loads all documents, chunks them, and creates/persists the Chroma vector store."""
    all_documents = []
    
    print(f"Loading documents from directory: {POLICY_DIRECTORY_PATH}")
    for filename in os.listdir(POLICY_DIRECTORY_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(POLICY_DIRECTORY_PATH, filename)
            print(f"  - Loading {filename}...")
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                # Add source metadata for citation
                doc.metadata['source'] = filename
            
            all_documents.extend(documents)

    print(f"Successfully loaded a total of {len(all_documents)} pages from all PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"Split document corpus into {len(texts)} chunks for indexing.")

    print(f"Creating and persisting vector store to {VECTOR_DB_DIR}...")
    vectorstore_instance = Chroma.from_documents(
        documents=texts,
        embedding=embeddings_client,
        persist_directory=VECTOR_DB_DIR
    )
    vectorstore_instance.persist()
    print("Vector store creation complete.")
    return vectorstore_instance

def get_or_create_vector_store(embeddings_client):
    """Checks if the vector store exists; if not, creates it."""
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Loading existing vector store from {VECTOR_DB_DIR}...")
        vectorstore_instance = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings_client
        )
        print("Vector store loaded successfully.")
    else:
        print("Vector store not found. Creating a new one...")
        vectorstore_instance = create_vector_store(embeddings_client)

    return vectorstore_instance

def create_rag_chain(vector_store_instance, llm_client):
    """Creates the Conversational Retrieval Chain."""
    retriever = vector_store_instance.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_client,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- CRITICAL: Function to be run in the background thread ---
def initialize_rag_components():
    """Initializes the heavy RAG components asynchronously."""
    global vectorstore
    global rag_chain
    
    try:
        embeddings_client, llm_client = create_rag_clients()
        
        # 1. Initialize Vector Store (the heavy part)
        vectorstore = get_or_create_vector_store(embeddings_client)
        
        # 2. Create the RAG chain
        rag_chain = create_rag_chain(vectorstore, llm_client)
        
        print("RAG components fully initialized and ready for queries.")
    except Exception as e:
        print(f"CRITICAL RAG INITIALIZATION FAILURE: {e}")
        # Set to False to signal permanent failure
        rag_chain = False 


# Initialize chat history for the RAG chain
chat_history = []

def query_rag_system(question: str):
    """
    Executes a query against the RAG system and updates the chat history.
    """
    global chat_history
    global rag_chain
    
    # CRITICAL: If initialization is not complete, throw an HTTP error.
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Service is starting up. RAG components are still initializing (503)."
        )
    if rag_chain is False:
        raise HTTPException(
            status_code=500,
            detail="RAG initialization failed. Check server logs for critical errors."
        )

    result = rag_chain.invoke(
        {"question": question, "chat_history": chat_history}
    )
    
    chat_history.append((question, result["answer"]))

    return {
        "answer": result["answer"],
        "source_documents": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
    }

def clear_chat_history():
    """Clears the session's chat history."""
    global chat_history
    global rag_chain

    if rag_chain is None or rag_chain is False:
        return {"status": "RAG not initialized, but history cleared."}
        
    chat_history = []
    return {"status": "Chat history cleared."}
