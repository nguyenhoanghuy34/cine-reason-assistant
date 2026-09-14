from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_API_KEY


def create_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )