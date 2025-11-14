from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_model import query_rag_system, clear_chat_history
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
    
# --- Startup Logic ---
@app.on_event("startup")
async def startup_event():
    print("FastAPI application started. RAG model initialized.")

# --- API Endpoints ---

@app.get("/", response_model=StatusResponse)
async def health_check():
    """A simple health check endpoint."""
    return {"status": "RAG system is online and operational."}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Submits a question to the RAG system and returns an answer 
    with relevant source chunks.
    """
    try:
        response = query_rag_system(request.question)
        return response
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing the query. Check backend logs."
        )

@app.post("/clear_history", response_model=StatusResponse)
async def reset_chat_history():
    """
    Clears the conversational chat history for the current session. 
    Call this when a new user starts or after a period of inactivity.
    """
    return clear_chat_history()