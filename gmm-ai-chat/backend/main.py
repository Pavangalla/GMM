import os
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .models import ChatRequest, ChatResponse
from .chat import chat
from .data_loader import build_database, build_embeddings, DB_PATH, EMBEDDINGS_PATH

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

import requests

DRIVE_DB_URL = "https://drive.google.com/uc?export=download&id=1d58EOGubF0dyTL8B0YMJQDM_9XjxbGgo"
DRIVE_NPZ_URL = "https://drive.google.com/uc?export=download&id=1WjwPLXPjeDPasUYx7-G4UQDvMdXEf7c8"

import shutil

@app.post("/clear-disk")
def clear_disk():
    try:
        shutil.rmtree("/data")
        os.makedirs("/data", exist_ok=True)
        return {"status": "disk cleared"}
    except Exception as e:
        return {"error": str(e)}


def ensure_embeddings():
    if not os.path.exists(DB_PATH):
        print("Downloading DB from Google Drive...")
        r = requests.get(DRIVE_DB_URL)
        with open(DB_PATH, "wb") as f:
            f.write(r.content)

    if not os.path.exists(EMBEDDINGS_PATH):
        print("Downloading NPZ from Google Drive...")
        r = requests.get(DRIVE_NPZ_URL)
        with open(EMBEDDINGS_PATH, "wb") as f:
            f.write(r.content)

    print("Embeddings ready.")

async def lifespan(app):
    ensure_embeddings()
    yield


app = FastAPI(title="GMM AI Chat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "db_exists": os.path.exists(DB_PATH)}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-db")
def rebuild():
    """Admin endpoint — call after GMM data updates."""
    build_database()
    return {"status": "rebuilt"}
