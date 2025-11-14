from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the new initialization function
from rag_model import query_rag_system, clear_chat_history, initialize_rag_components
import threading 
import os

app = FastAPI(
    title="PMKSY Policy RAG System",
    description="Backend for querying the Pradhan Mantri Krishi Sinchayee Yojana policy document.",
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: list

class StatusResponse(BaseModel):
    status: str
    
# --- Startup Logic (FIXED) ---
@app.on_event("startup")
async def startup_event():
    print("FastAPI application starting up. Initializing RAG components in background thread...")
    
    # Start the potentially long-running RAG initialization in a background thread.
    # This ensures the main server thread can start immediately and avoid the 502 timeout.
    threading.Thread(target=initialize_rag_components).start() 
    
    print("Server started successfully, RAG components initializing...")


# --- API Endpoints ---

@app.get("/", response_model=StatusResponse)
async def health_check():
    """A simple health check endpoint."""
    return {"status": "RAG system is online and operational. RAG components may still be initializing."}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Submits a question to the RAG system and returns an answer 
    with relevant source chunks.
    """
    try:
        # query_rag_system now handles the 503/500 check internally
        response = query_rag_system(request.question)
        return response
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing the query. Check backend logs."
        )

@app.post("/clear_history", response_model=StatusResponse)
async def reset_chat_history():
    """
    Clears the conversational chat history for the current session. 
    """
    return clear_chat_history()
