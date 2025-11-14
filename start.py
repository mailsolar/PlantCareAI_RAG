# File: start.py

import uvicorn
from dotenv import load_dotenv
import os # Import os here

# 1. GUARANTEE: Load environment variables before Uvicorn starts.
load_dotenv()

# --- TEMPORARY DIAGNOSTIC CHECK ---
# Print the key value to confirm it loaded correctly
key = os.getenv("GEMINI_API_KEY")
print(f"DIAGNOSTIC: GEMINI_API_KEY value is: {key[:5]}... (length: {len(key) if key else 0})")
if not key or len(key) < 30:
    print("CRITICAL ERROR: Key is missing or too short. Check your .env file.")
# --- END DIAGNOSTIC CHECK ---

# 2. Launch Uvicorn
if __name__ == "__main__":
    print("Starting Uvicorn server with environment loaded...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")