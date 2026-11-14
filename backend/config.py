import os
from dotenv import load_dotenv

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/autodev_db")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace_output")
    MAX_RETRY_LIMIT = int(os.getenv("MAX_RETRY_LIMIT", 3))
    DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "python:3.11-slim")

config = Config()

config = Config()