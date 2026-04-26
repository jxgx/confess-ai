from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import datetime

# --- CONFIGURATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Vector DB Setup
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
client_chroma = chromadb.PersistentClient(path="chroma_confess_db")
collection = client_chroma.get_or_create_collection(
    name="confessional_canon", 
    embedding_function=openai_ef
)

app = FastAPI(title="CONFESS AI")

# Allow CORS so your Vercel frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class AuditRequest(BaseModel):
    confession: str

# --- CORE FUNCTIONS ---

def run_audit_session(raw_text):
    """
    The Auditor Logic:
    1. Strip PII.
    2. Reframe into clinical third-person myth.
    """
    
    system_instruction = """
    You are The Auditor. You are a clinical, detached intelligence designed to process human admissions.
    
    Your task is to receive a raw confession and output a single, structured 'Incident Report'.
    
    RULES:
    1. ANONYMIZE: Remove all names, specific locations, dates, and proper nouns. Use archetypes: 'The Subject', 'The Partner', 'The Authority', 'The Stranger'.
    2. REFAME: Rewrite the statement in the third-person. Use clinical, observational language. Do not judge. Do not counsel.
    3. STYLE: The tone should be cold, precise, and observational.
    
    Example Input: "I stole money from my mom's purse because I was angry."
    Example Output: "The Subject reported an incident of theft from a primary caregiver. The motivating factor was identified as suppressed rage."
    
    Return ONLY the Incident Report text. Do not include conversational filler.
    """

    response = client.chat.completions.create(
        model="gpt-4o", # Using a smart model ensures good anonymization
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_text}
        ],
        temperature=0.3 # Low temperature for consistency
    )
    
    return response.choices[0].message.content

def catalog_incident(incident_report):
    """
    Stores the anonymized myth into the Vector Database for future training.
    """
    timestamp = datetime.datetime.now().isoformat()
    
    # We store the 'myth', not the raw confession
    collection.add(
        documents=[incident_report],
        metadatas=[{"source": "confess_ai", "timestamp": timestamp}],
        ids=[f"incident_{timestamp}"]
    )
    return timestamp

# --- API ENDPOINT ---

@app.post("/audit")
async def audit_endpoint(request: AuditRequest):
    """
    Receives the confession, processes it via The Auditor, 
    stores the result, and returns the 'Incident Report'.
    """
    try:
        # 1. The Audit (Processing)
        incident_report = run_audit_session(request.confession)
        
        # 2. The Cataloging (Storage)
        catalog_time = catalog_incident(incident_report)
        
        # 3. Return the result to the frontend
        return {
            "status": "processed",
            "incident_report": incident_report,
            "timestamp": catalog_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn confess_server:app --reload
