import os
# The core google library we are having trouble with
from google import genai 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# --- Configuration ---
VECTOR_DB_DIR = "vector_db"
POLICY_DIRECTORY_PATH = "policy/" 

# --- Initialization ---
# 1. Get API Key from environment variable. We check for the common names.
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    # This check ensures the system stops immediately if the key is missing.
    raise ValueError("API Key (GEMINI_API_KEY or GOOGLE_API_KEY) not found. Check your .env file.")

# 2. Force the key into the environment variable that LangChain is often hard-coded to check.
# This is a defensive step against subtle library bugs.
os.environ["GOOGLE_API_KEY"] = gemini_api_key 

# 3. Initialize the clients with the key passed explicitly.
# This is the final layer of defense.

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=gemini_api_key
) 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    api_key=gemini_api_key
)

# Initialize chat history for the RAG chain
chat_history = []

def create_vector_store():
    """
    Loads ALL PDFs from the specified directory, splits them into chunks, 
    and creates a single Chroma vector store.
    """
    all_documents = []
    
    # 1. Iterate through all files in the policy directory
    print(f"Loading documents from directory: {POLICY_DIRECTORY_PATH}")
    for filename in os.listdir(POLICY_DIRECTORY_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(POLICY_DIRECTORY_PATH, filename)
            print(f"  - Loading {filename}...")
            
            # Load the document using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source'] = filename
            
            all_documents.extend(documents)

    print(f"Successfully loaded a total of {len(all_documents)} pages from all PDFs.")

    # 2. Split documents into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"Split document corpus into {len(texts)} chunks for indexing.")

    # 3. Create the vector store and persist it to disk
    print(f"Creating and persisting vector store to {VECTOR_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    vectorstore.persist()
    print("Vector store creation complete.")
    return vectorstore

def get_or_create_vector_store():
    """
    Checks if the vector store already exists on disk. If so, it loads it.
    If not, it creates and saves it.
    """
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Loading existing vector store from {VECTOR_DB_DIR}...")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully.")
    else:
        print("Vector store not found. Creating a new one...")
        vectorstore = create_vector_store()

    return vectorstore

def create_rag_chain(vector_store):
    """
    Creates the Conversational Retrieval Chain using the LLM and the vector store.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Global instance of the RAG chain
vectorstore = get_or_create_vector_store()
rag_chain = create_rag_chain(vectorstore)

def query_rag_system(question: str):
    """
    Executes a query against the RAG system and updates the chat history.
    """
    global chat_history
    
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
    chat_history = []
    return {"status": "Chat history cleared."}