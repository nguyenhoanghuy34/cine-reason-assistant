from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
)

CLEAN_DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
)

SIMILARITY_FILE = CLEAN_DATA_DIR / "user_similarity.parquet"
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_FILE = DATA_DIR / "movies.csv"
TAGS_FILE = DATA_DIR / "tags.csv"
LINKS_FILE = DATA_DIR / "links.csv"
MOVIES_PLOTS_FILE = DATA_DIR / "movies_with_plots.csv"


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def source(path: Path) -> None:
    print(f"Source file : {path}")
    print(f"Exists      : {path.exists()}")

    if path.exists():
        print(f"File size   : {path.stat().st_size:,} bytes")


# ============================================================
# 1. USER SIMILARITY
# ============================================================

def inspect_similarity(user_id: int) -> list[int]:

    section("1. USER SIMILARITY")

    source(SIMILARITY_FILE)

    df = pd.read_parquet(SIMILARITY_FILE)

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("DataFrame shape:")
    print(df.shape)

    print()
    print("First 20 rows:")
    print(df.head(20).to_string(index=False))

    print()
    print(f"Searching user_id = {user_id}")

    row = df[df["user_id"] == user_id]

    if row.empty:
        raise ValueError(
            f"user_id {user_id} not found in {SIMILARITY_FILE}"
        )

    related_ids = row.iloc[0]["related_user_ids"]

    print()
    print("Matched row:")
    print(row.to_string(index=False))

    print()
    print("Related user IDs:")
    print(related_ids)

    print()
    print(f"Number of related users: {len(related_ids)}")

    return [
        int(uid)
        for uid in related_ids
        if int(uid) != user_id
    ]


# ============================================================
# 2. RATINGS
# ============================================================

def inspect_ratings(
    user_id: int,
    related_user_ids: list[int],
    movie_id: int,
) -> pd.DataFrame:

    section("2. RATINGS")

    source(RATINGS_FILE)

    ratings = pd.read_csv(RATINGS_FILE)

    print()
    print("Columns:")
    print(ratings.columns.tolist())

    print()
    print("DataFrame shape:")
    print(ratings.shape)

    print()
    print("First 20 rows:")
    print(ratings.head(20).to_string(index=False))

    # --------------------------------------------------------
    # Current user
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print(f"ALL RATINGS FROM CURRENT USER {user_id}")
    print("-" * 100)

    current_user_ratings = ratings[
        ratings["userId"] == user_id
    ].copy()

    print(
        current_user_ratings.to_string(index=False)
    )

    print()
    print(
        f"Current user total ratings: "
        f"{len(current_user_ratings)}"
    )

    # --------------------------------------------------------
    # Related users
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("RELATED USER IDs")
    print("-" * 100)

    print(related_user_ids)

    related_ratings = ratings[
        ratings["userId"].isin(related_user_ids)
    ].copy()

    print()
    print(
        f"Total ratings from related users: "
        f"{len(related_ratings)}"
    )

    print()
    print("ALL RATINGS FROM RELATED USERS:")
    print(
        related_ratings.to_string(index=False)
    )

    # --------------------------------------------------------
    # Target movie
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print(f"RATINGS OF MOVIE ID {movie_id} FROM RELATED USERS")
    print("-" * 100)

    movie_ratings = related_ratings[
        related_ratings["movieId"] == movie_id
    ].copy()

    if movie_ratings.empty:
        print("NO RELATED USER RATED THIS MOVIE.")
    else:
        print(
            movie_ratings.to_string(index=False)
        )

    return movie_ratings


# ============================================================
# 3. MOVIES
# ============================================================

def inspect_movies(movie_name: str) -> tuple[int, pd.Series]:

    section("3. MOVIES")

    source(MOVIES_FILE)

    movies = pd.read_csv(MOVIES_FILE)

    print()
    print("Columns:")
    print(movies.columns.tolist())

    print()
    print("DataFrame shape:")
    print(movies.shape)

    print()
    print("First 20 rows:")
    print(
        movies.head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # Search movie
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print(f"SEARCH MOVIE: {movie_name}")
    print("-" * 100)

    matched = movies[
        movies["title"].str.contains(
            movie_name,
            case=False,
            na=False,
        )
    ].copy()

    if matched.empty:
        raise ValueError(
            f"Movie '{movie_name}' not found."
        )

    print(
        matched.to_string(index=False)
    )

    if len(matched) > 1:
        print()
        print(
            "WARNING: Multiple movies matched."
        )

    movie = matched.iloc[0]

    movie_id = int(movie["movieId"])

    print()
    print("SELECTED MOVIE:")
    print(f"movieId : {movie_id}")
    print(f"title   : {movie['title']}")
    print(f"genres  : {movie['genres']}")

    return movie_id, movie


# ============================================================
# 4. MOVIES WITH PLOTS
# ============================================================

def inspect_movie_plot(movie_id: int) -> None:

    section("4. MOVIES WITH PLOTS")

    source(MOVIES_PLOTS_FILE)

    movies = pd.read_csv(MOVIES_PLOTS_FILE)

    print()
    print("Columns:")
    print(movies.columns.tolist())

    print()
    print("DataFrame shape:")
    print(movies.shape)

    movie = movies[
        movies["movieId"] == movie_id
    ]

    if movie.empty:
        print(
            f"No plot found for movieId={movie_id}"
        )
        return

    print()
    print("TARGET MOVIE DATA:")
    print(
        movie.to_string(index=False)
    )


# ============================================================
# 5. TAGS
# ============================================================

def inspect_tags(
    movie_id: int,
    related_user_ids: list[int],
) -> None:

    section("5. TAGS")

    source(TAGS_FILE)

    tags = pd.read_csv(TAGS_FILE)

    print()
    print("Columns:")
    print(tags.columns.tolist())

    print()
    print("DataFrame shape:")
    print(tags.shape)

    print()
    print("First 20 rows:")
    print(
        tags.head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # Target movie tags
    # --------------------------------------------------------

    movie_tags = tags[
        tags["movieId"] == movie_id
    ]

    print()
    print("-" * 100)
    print(f"TAGS FOR MOVIE ID {movie_id}")
    print("-" * 100)

    if movie_tags.empty:
        print("No tags for this movie.")
    else:
        print(
            movie_tags.to_string(index=False)
        )

    # --------------------------------------------------------
    # Related users' tags
    # --------------------------------------------------------

    related_tags = tags[
        tags["userId"].isin(related_user_ids)
    ]

    print()
    print("-" * 100)
    print("TAGS CREATED BY RELATED USERS")
    print("-" * 100)

    if related_tags.empty:
        print("No tags from related users.")
    else:
        print(
            related_tags.to_string(index=False)
        )


# ============================================================
# 6. LINKS
# ============================================================

def inspect_links(movie_id: int) -> None:

    section("6. LINKS")

    source(LINKS_FILE)

    links = pd.read_csv(LINKS_FILE)

    print()
    print("Columns:")
    print(links.columns.tolist())

    print()
    print("DataFrame shape:")
    print(links.shape)

    print()
    print("First 20 rows:")
    print(
        links.head(20).to_string(index=False)
    )

    movie_link = links[
        links["movieId"] == movie_id
    ]

    print()
    print("-" * 100)
    print(f"LINK INFORMATION FOR MOVIE ID {movie_id}")
    print("-" * 100)

    if movie_link.empty:
        print("No link information.")
    else:
        print(
            movie_link.to_string(index=False)
        )


# ============================================================
# 7. FINAL DATA FLOW
# ============================================================

def print_final_summary(
    user_id: int,
    related_user_ids: list[int],
    movie_id: int,
    movie: pd.Series,
    movie_ratings: pd.DataFrame,
) -> None:

    section("7. FINAL DATA FLOW")

    print("USER")
    print(f"  user_id = {user_id}")

    print()
    print("        ↓")

    print("SIMILAR USERS")
    print(f"  Source: {SIMILARITY_FILE}")
    print(f"  related_user_ids = {related_user_ids}")

    print()
    print("        ↓")

    print("TARGET MOVIE")
    print(f"  Source: {MOVIES_FILE}")
    print(f"  movieId = {movie_id}")
    print(f"  title   = {movie['title']}")
    print(f"  genres  = {movie['genres']}")

    print()
    print("        ↓")

    print("RELATED USERS' RATINGS")
    print(f"  Source: {RATINGS_FILE}")
    print(
        f"  Number of ratings for this movie: "
        f"{len(movie_ratings)}"
    )

    if not movie_ratings.empty:

        print(
            f"  Average rating: "
            f"{movie_ratings['rating'].mean():.2f}"
        )

        print()
        print("  Individual ratings:")

        for _, row in movie_ratings.iterrows():
            print(
                f"    user {int(row['userId'])}"
                f" -> {float(row['rating'])}/5"
            )

    print()
    print("        ↓")

    print("EVIDENCE AVAILABLE FOR LLM")

    print()
    print("The LLM should reason from the above evidence.")
    print("The LLM should NOT invent ratings or user opinions.")


# ============================================================
# MAIN
# ============================================================

def run_test(
    user_id: int,
    movie_name: str,
) -> None:

    print("=" * 100)
    print("CINE REASON ASSISTANT")
    print("RELATED USERS - FULL DATA TRACE TEST")
    print("=" * 100)

    print()
    print(f"User ID    : {user_id}")
    print(f"Movie query: {movie_name}")

    # 1. Similarity
    related_user_ids = inspect_similarity(user_id)

    # 2. Movie
    movie_id, movie = inspect_movies(movie_name)

    # 3. Ratings
    movie_ratings = inspect_ratings(
        user_id=user_id,
        related_user_ids=related_user_ids,
        movie_id=movie_id,
    )

    # 4. Plot
    inspect_movie_plot(movie_id)

    # 5. Tags
    inspect_tags(
        movie_id=movie_id,
        related_user_ids=related_user_ids,
    )

    # 6. Links
    inspect_links(movie_id)

    # 7. Summary
    print_final_summary(
        user_id=user_id,
        related_user_ids=related_user_ids,
        movie_id=movie_id,
        movie=movie,
        movie_ratings=movie_ratings,
    )

    print()
    print("=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("CINE REASON ASSISTANT - FULL DATA TRACE")
    print("=" * 100)

    while True:
        try:
            user_id = int(
                input("Enter User ID: ").strip()
            )
            break
        except ValueError:
            print("User ID must be an integer.")

    movie_name = input(
        "Enter movie name: "
    ).strip()

    if not movie_name:
        movie_name = "Pulp Fiction"

    try:
        run_test(
            user_id=user_id,
            movie_name=movie_name,
        )

    except Exception as exc:
        print()
        print("=" * 100)
        print("ERROR")
        print("=" * 100)
        print(exc)
        print("=" * 100)