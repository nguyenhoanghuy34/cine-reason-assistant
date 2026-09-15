import warnings
import logging
from textwrap import fill


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", message=".*Automatic Function Calling.*")
warnings.filterwarnings("ignore", category=Warning, module="requests")
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

from app.agent.graph import invoke_with_memory
from app.agent.tools.evidence_guard_tools import check_recommendation_grounding


def _compact_text(text: str, width: int = 88) -> str:
    paragraphs = [part.strip() for part in str(text).splitlines() if part.strip()]
    return "\n".join(fill(paragraph, width=width) for paragraph in paragraphs)


def _first_items(items, limit: int = 3):
    return list(items or [])[:limit]


def _format_evidence(result: dict) -> str:
    evidence = result.get("evidence") or {}
    lines = [
        f"intent={result.get('intent', 'UNKNOWN')}",
        f"user_id={result.get('user_id', evidence.get('user_id', evidence.get('current_user_id', 'unknown')))}",
    ]

    recommendations = evidence.get("recommendations")
    if isinstance(recommendations, dict):
        candidates = recommendations.get("candidate_movies", [])
        if candidates:
            titles = [
                f"{movie.get('title')} ({movie.get('average_rating', 'n/a')})"
                for movie in _first_items(candidates)
            ]
            lines.append("candidates=" + "; ".join(titles))
        constraints = recommendations.get("constraints") or {}
        active_constraints = {key: value for key, value in constraints.items() if value}
        if active_constraints:
            lines.append(f"constraints={active_constraints}")

    movie_info = evidence.get("movie_info")
    if isinstance(movie_info, dict) and movie_info.get("movies"):
        titles = [
            f"{movie.get('title')} ({movie.get('average_rating', 'n/a')})"
            for movie in _first_items(movie_info.get("movies"))
        ]
        lines.append("movie_info=" + "; ".join(titles))

    movie_summary = evidence.get("movie_summary")
    if isinstance(movie_summary, dict) and movie_summary.get("summaries"):
        titles = [str(movie.get("title")) for movie in _first_items(movie_summary.get("summaries"))]
        lines.append("movie_summary=" + "; ".join(titles))

    user_history = evidence.get("user_rating_history")
    if isinstance(user_history, dict) and user_history.get("top_rated_movies"):
        titles = [
            f"{movie.get('title')} ({movie.get('rating')})"
            for movie in _first_items(user_history.get("top_rated_movies"))
        ]
        lines.append("user_top_ratings=" + "; ".join(titles))

    movie_opinions = evidence.get("movie_opinions")
    if isinstance(movie_opinions, dict):
        for item in _first_items(movie_opinions.get("results", []), limit=2):
            ratings = item.get("ratings", [])
            if ratings:
                shown = [
                    f"user {row.get('userId')} rated {row.get('rating')}"
                    for row in _first_items(ratings)
                ]
                lines.append(f"{item.get('requested_title')}: " + "; ".join(shown))

    similar = evidence.get("similar_users")
    if isinstance(similar, dict):
        related_ids = similar.get("related_user_ids", [])
        if related_ids:
            lines.append("similar_users=" + ", ".join(map(str, _first_items(related_ids, 5))))
        movies = similar.get("movies", [])
        if movies:
            titles = [
                f"{movie.get('title')} ({movie.get('average_rating')})"
                for movie in _first_items(movies)
            ]
            lines.append("similar_users_top_movies=" + "; ".join(titles))

    genre_analysis = evidence.get("genre_analysis")
    if isinstance(genre_analysis, dict):
        blind_spots = (
            genre_analysis.get("unwatched_genres")
            or genre_analysis.get("underexposed_genres")
            or []
        )
        if blind_spots:
            lines.append("blind_spots=" + ", ".join(map(str, _first_items(blind_spots, 5))))

    grounding = check_recommendation_grounding(
        result.get("response", ""),
        evidence,
    )
    if grounding["status"] != "not_applicable":
        lines.append(f"grounding={grounding['status']}")

    return "\n".join(f"- {line}" for line in lines)


def print_result(result: dict) -> None:
    print()
    print("=" * 88)
    print("ASSISTANT")
    print("-" * 88)
    print(_compact_text(result.get("response", "")) or "(no answer)")
    print()
    print("EVIDENCE")
    print("-" * 88)
    print(_format_evidence(result))
    print("=" * 88)
    print()


def run_chat(user_id: int, thread_id: str = "default") -> None:
    print("Nhap cau hoi ve phim; nhap /exit de ket thuc.")
    print(f"User {user_id} | Session: {thread_id}")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"/exit", "/quit"}:
            return
        if not query:
            continue
        try:
            result = invoke_with_memory(user_id, query, thread_id)
        except Exception as exc:
            print(f"Khong hoan tat luot hoi: {exc}")
            continue
        print_result(result)


if __name__ == "__main__":
    try:
        while True:
            try:
                user_id = int(input("User ID: ").strip())
                break
            except ValueError:
                print("User ID phai la so nguyen.")
        thread_id = input("Session name (Enter = default, reuse name to continue): ").strip() or "default"
        run_chat(user_id, thread_id)
    except (KeyboardInterrupt, EOFError):
        print("\nDa ket thuc hoi thoai.")
