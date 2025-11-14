# File: start.py

import uvicorn
from dotenv import load_dotenv
import os

# 1. GUARANTEE: Load environment variables before Uvicorn starts.
load_dotenv()

# --- DIAGNOSTIC CHECK (KEEP FOR LOCAL RUNS) ---
key = os.getenv("GEMINI_API_KEY")
print(f"DIAGNOSTIC: GEMINI_API_KEY value is: {key[:5]}... (length: {len(key) if key else 0})")
# --- END DIAGNOSTIC CHECK ---

# 2. Launch Uvicorn
if __name__ == "__main__":
    print("Starting Uvicorn server with environment loaded...")
    # This host setting is CRITICAL for Docker/Railway
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
