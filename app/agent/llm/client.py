from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY


def create_llm() -> ChatGroq:
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=512,
    )
