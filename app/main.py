import warnings

# Hide requests dependency warning
warnings.filterwarnings(
    "ignore",
    message="urllib3.*doesn't match a supported version!",
)

# Hide Gemini fixed sampling warning
warnings.filterwarnings(
    "ignore",
    message="Model 'gemini-3.6-flash' uses fixed sampling defaults.*",
)

# Hide Gemini AFC warning
warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling.*",
)


from app.agent.graph import agent


def run_query(query: str, user_id: int | None = None):
    result = agent.invoke({
        "user_id": user_id,
        "query": query,
    })

    print("=" * 70)
    print(f"USER ID : {user_id}")
    print(f"QUERY   : {query}")
    print(f"INTENT  : {result.get('intent')}")
    print(f"REASON  : {result.get('intent_reason')}")
    print("-" * 70)
    print(result.get("response"))
    print("=" * 70)


if __name__ == "__main__":
    run_query(
        "What should I watch tonight?",
        user_id=1,
    )

    run_query(
        "What do people with similar taste to me think about Inception?",
        user_id=1,
    )

    run_query(
        "What is Inception about?",
        user_id=1,
    )

    run_query(
        "What genres am I missing?",
        user_id=1,
    )