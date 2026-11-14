import os
from dotenv import load_dotenv

# Load environment variables from the project's .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")

    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")

    DATABASE_URL = os.getenv("DATABASE_URL")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace_output")


config = Config()