from pathlib import Path
import os

from dotenv import load_dotenv


# Project root: cine-reason-assistant/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        f"GROQ_API_KEY not found in {ENV_FILE}"
    )
