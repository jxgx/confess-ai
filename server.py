from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import datetime

# 1. Load Config
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Setup ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
client_chroma = chromadb.PersistentClient(path="chroma_db")
collection = client_chroma.get_or_create_collection(
    name="confessional_canon", 
    embedding_function=openai_ef
)

# 3. Setup FastAPI
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Data Structure
class AuditRequest(BaseModel):
    confession: str

# 5. Logic Functions
def run_audit_session(raw_text):
    system_instruction = """
    You are The Auditor. You are a clinical, detached intelligence designed to process human admissions.
    
    Your task is to receive a raw confession and output a single, structured 'Incident Report'.
    
    RULES:
    1. ANONYMIZE: Remove all names, specific locations, dates, and proper nouns. Use archetypes: 'The Subject', 'The Partner', 'The Authority'.
    2. REFAME: Rewrite the statement in the third-person. Use clinical, observational language.
    
    Example Input: "I stole money from my mom."
    Example Output: "The Subject reported an incident of theft from a primary caregiver."
    
    Return ONLY the Incident Report text.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": raw_text}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def catalog_incident(incident_report):
    timestamp = datetime.datetime.now().isoformat()
    collection.add(
        documents=[incident_report],
        metadatas=[{"source": "confess_ai", "timestamp": timestamp}],
        ids=[f"incident_{timestamp}"]
    )

# 6. The Endpoint
@app.post("/audit")
async def audit_endpoint(request: AuditRequest):
    try:
        incident_report = run_audit_session(request.confession)
        catalog_incident(incident_report)
        return {
            "status": "processed",
            "incident_report": incident_report
        }
    except Exception as e:
        return {"error": str(e)}
