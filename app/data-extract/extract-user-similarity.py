"""
Extract user similarity relationships from MovieLens data.

Definition:
    For each source user, find other users who share at least one genre
    where BOTH users have rated movies in that genre >= 4 stars.

Output:
    app/data/clean-data/user_similarity.parquet

Schema:
    - user_id
    - related_user_ids

Example:
    user_id = 1
    related_user_ids = [5, 18, 42, ...]

The output is intended to identify users with similar movie preferences.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

SOURCE_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"

OUTPUT_DIR = PROJECT_ROOT / "app" / "data" / "clean-data"

RATINGS_PATH = SOURCE_DIR / "ratings.csv"
MOVIES_PATH = SOURCE_DIR / "movies_with_plots.csv"

OUTPUT_PATH = OUTPUT_DIR / "user_similarity.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_RATING_THRESHOLD = 4.0

EXPECTED_COLUMNS = [
    "user_id",
    "related_user_ids",
]


# ============================================================
# HELPERS
# ============================================================

def split_genres(genres: str) -> list[str]:
    """
    Convert MovieLens genre string into a clean list.

    Example:
        "Action|Thriller|Crime"
        ->
        ["Action", "Thriller", "Crime"]
    """

    if pd.isna(genres):
        return []

    genres = str(genres).strip()

    if not genres:
        return []

    return [
        genre.strip()
        for genre in genres.split("|")
        if genre.strip()
    ]


def print_progress(
    current: int,
    total: int,
    user_id: int,
) -> None:
    """
    Print extraction progress.
    """

    percentage = current / total * 100

    print(
        f"\r    Processing user "
        f"{current:,}/{total:,} "
        f"({percentage:6.2f}%) | user_id={user_id}",
        end="",
        flush=True,
    )

    if current == total:
        print()


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("=" * 70)
    print("USER SIMILARITY EXTRACTION")
    print("=" * 70)

    print("\n[1/7] Loading source files...")

    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing source file:\n{RATINGS_PATH}"
        )

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Missing source file:\n{MOVIES_PATH}"
        )

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    print(f"  ratings.csv           : {len(ratings):,} rows")
    print(f"  movies_with_plots.csv : {len(movies):,} rows")

    return ratings, movies


# ============================================================
# VALIDATE SOURCE DATA
# ============================================================

def validate_source_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:

    print("\n[2/7] Validating source data...")

    required_rating_columns = {
        "userId",
        "movieId",
        "rating",
    }

    required_movie_columns = {
        "movieId",
        "genres",
    }

    missing_rating_columns = (
        required_rating_columns - set(ratings.columns)
    )

    missing_movie_columns = (
        required_movie_columns - set(movies.columns)
    )

    if missing_rating_columns:
        raise ValueError(
            "ratings.csv is missing columns: "
            f"{sorted(missing_rating_columns)}"
        )

    if missing_movie_columns:
        raise ValueError(
            "movies_with_plots.csv is missing columns: "
            f"{sorted(missing_movie_columns)}"
        )

    if ratings["userId"].isna().any():
        raise ValueError("ratings.csv contains NULL userId.")

    if ratings["movieId"].isna().any():
        raise ValueError("ratings.csv contains NULL movieId.")

    if ratings["rating"].isna().any():
        raise ValueError("ratings.csv contains NULL rating.")

    if movies["movieId"].duplicated().any():
        duplicate_count = movies["movieId"].duplicated().sum()

        raise ValueError(
            "movies_with_plots.csv contains duplicate movieId values: "
            f"{duplicate_count}"
        )

    print("  Required columns       : OK")
    print("  NULL checks            : OK")
    print("  movieId uniqueness     : OK")

    print(
        f"  Unique users           : "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"  Unique movies          : "
        f"{ratings['movieId'].nunique():,}"
    )


# ============================================================
# PREPARE HIGH-RATED DATA
# ============================================================

def prepare_high_rated_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:

    print("\n[3/7] Preparing high-rated movie data...")

    ratings = ratings.copy()
    movies = movies.copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    movies["movieId"] = movies["movieId"].astype(int)

    movies["genres_list"] = movies["genres"].apply(split_genres)

    # --------------------------------------------------------
    # Keep only ratings >= 4
    # --------------------------------------------------------

    high_rated = ratings[
        ratings["rating"] >= HIGH_RATING_THRESHOLD
    ].copy()

    print(
        f"  Ratings >= {HIGH_RATING_THRESHOLD:.0f} "
        f"stars                : {len(high_rated):,}"
    )

    # --------------------------------------------------------
    # Attach movie genres
    # --------------------------------------------------------

    high_rated = high_rated.merge(
        movies[
            [
                "movieId",
                "genres_list",
            ]
        ],
        on="movieId",
        how="inner",
        validate="many_to_one",
    )

    print(
        f"  Matched high-rated rows : "
        f"{len(high_rated):,}"
    )

    # --------------------------------------------------------
    # Remove movies without usable genres
    # --------------------------------------------------------

    high_rated = high_rated[
        high_rated["genres_list"].map(len) > 0
    ].copy()

    print(
        f"  Rows with valid genres   : "
        f"{len(high_rated):,}"
    )

    return high_rated


# ============================================================
# BUILD USER-GENRE INDEX
# ============================================================

def build_user_genre_index(
    high_rated: pd.DataFrame,
) -> dict[int, set[str]]:

    print("\n[4/7] Building user -> high-rated genres index...")

    user_genres: dict[int, set[str]] = defaultdict(set)

    for row in high_rated.itertuples(index=False):

        user_id = int(row.userId)

        for genre in row.genres_list:
            user_genres[user_id].add(genre)

    print(
        f"  Users with >= 1 high-rated genre : "
        f"{len(user_genres):,}"
    )

    genre_count = len(
        {
            genre
            for genres in user_genres.values()
            for genre in genres
        }
    )

    print(
        f"  Unique genres                      : "
        f"{genre_count:,}"
    )

    return dict(user_genres)


# ============================================================
# BUILD GENRE -> USERS INDEX
# ============================================================

def build_genre_user_index(
    user_genres: dict[int, set[str]],
) -> dict[str, set[int]]:

    print("\n[5/7] Building genre -> users index...")

    genre_users: dict[str, set[int]] = defaultdict(set)

    for user_id, genres in user_genres.items():

        for genre in genres:
            genre_users[genre].add(user_id)

    for genre in sorted(genre_users):
        print(
            f"  {genre:<20} "
            f"{len(genre_users[genre]):,} users"
        )

    return dict(genre_users)


# ============================================================
# EXTRACT USER SIMILARITY
# ============================================================

def extract_similarity(
    user_genres: dict[int, set[str]],
    genre_users: dict[str, set[int]],
    all_user_ids: list[int],
) -> pd.DataFrame:

    print("\n[6/7] Extracting user similarity...")
    print(
        "  Rule: share >= 1 genre where both users "
        "rated >= 4 stars."
    )
    print()

    records: list[dict] = []

    total_users = len(all_user_ids)

    for current_index, user_id in enumerate(
        all_user_ids,
        start=1,
    ):

        print_progress(
            current_index,
            total_users,
            user_id,
        )

        current_genres = user_genres.get(
            user_id,
            set(),
        )

        related_users: set[int] = set()

        # ----------------------------------------------------
        # Find all users sharing at least one high-rated genre
        # ----------------------------------------------------

        for genre in current_genres:

            users_in_same_genre = genre_users.get(
                genre,
                set(),
            )

            related_users.update(
                users_in_same_genre
            )

        # ----------------------------------------------------
        # A user cannot be related to themselves
        # ----------------------------------------------------

        related_users.discard(user_id)

        # ----------------------------------------------------
        # Deterministic ordering
        # ----------------------------------------------------

        related_user_ids = sorted(
            related_users
        )

        records.append(
            {
                "user_id": user_id,
                "related_user_ids": related_user_ids,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_output(
    similarity: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_genres: dict[int, set[str]],
) -> None:

    print("\n\n[7/7] Validating generated user similarity...")

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    actual_columns = similarity.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        raise AssertionError(
            "Invalid schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    print("  Schema                         : OK")

    # --------------------------------------------------------
    # One row per user
    # --------------------------------------------------------

    if similarity["user_id"].duplicated().any():
        raise AssertionError(
            "Duplicate user_id found in output."
        )

    source_users = set(
        ratings["userId"].astype(int).unique()
    )

    output_users = set(
        similarity["user_id"].astype(int)
    )

    if source_users != output_users:
        missing = source_users - output_users
        extra = output_users - source_users

        raise AssertionError(
            "User transfer mismatch.\n"
            f"Missing users: {sorted(missing)}\n"
            f"Extra users:   {sorted(extra)}"
        )

    print(
        f"  All source users transferred     : "
        f"{len(output_users):,} / {len(source_users):,}"
    )

    # --------------------------------------------------------
    # Validate every relationship
    # --------------------------------------------------------

    print(
        "\n  Checking relationship correctness..."
    )

    total_users = len(similarity)

    for current_index, row in enumerate(
        similarity.itertuples(index=False),
        start=1,
    ):

        user_id = int(row.user_id)
        related_users = list(row.related_user_ids)

        # Progress every user
        print_progress(
            current_index,
            total_users,
            user_id,
        )

        # ----------------------------------------------------
        # No self relationship
        # ----------------------------------------------------

        if user_id in related_users:
            raise AssertionError(
                f"Self relationship found for user "
                f"{user_id}."
            )

        # ----------------------------------------------------
        # No duplicate related users
        # ----------------------------------------------------

        if len(related_users) != len(set(related_users)):
            raise AssertionError(
                f"Duplicate related user found for "
                f"user {user_id}."
            )

        # ----------------------------------------------------
        # Calculate expected relationships independently
        # ----------------------------------------------------

        current_genres = user_genres.get(
            user_id,
            set(),
        )

        expected_related: set[int] = set()

        for other_user_id, other_genres in user_genres.items():

            if other_user_id == user_id:
                continue

            if current_genres.intersection(
                other_genres
            ):
                expected_related.add(
                    other_user_id
                )

        expected_related_list = sorted(
            expected_related
        )

        if related_users != expected_related_list:
            raise AssertionError(
                f"Relationship mismatch for user "
                f"{user_id}.\n"
                f"Expected: {expected_related_list}\n"
                f"Actual:   {related_users}"
            )

    # --------------------------------------------------------
    # Validate every related user exists in source
    # --------------------------------------------------------

    source_user_ids = set(
        ratings["userId"].astype(int).unique()
    )

    for row in similarity.itertuples(index=False):

        invalid_users = (
            set(row.related_user_ids)
            - source_user_ids
        )

        if invalid_users:
            raise AssertionError(
                f"User {row.user_id} contains invalid "
                f"related user IDs: "
                f"{sorted(invalid_users)}"
            )

    print(
        "\n  Relationship correctness         : OK"
    )

    # --------------------------------------------------------
    # Null checks
    # --------------------------------------------------------

    if similarity["user_id"].isna().any():
        raise AssertionError(
            "user_id contains NULL values."
        )

    if similarity["related_user_ids"].isna().any():
        raise AssertionError(
            "related_user_ids contains NULL values."
        )

    print("  NULL checks                      : OK")

    # --------------------------------------------------------
    # Every related user must have at least one shared
    # high-rated genre
    # --------------------------------------------------------

    print(
        "  Shared high-rated genre rule     : OK"
    )


# ============================================================
# SAVE
# ============================================================

def save_output(
    similarity: pd.DataFrame,
) -> None:

    print("\nSaving output...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    similarity.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    print(
        f"  Saved to:\n"
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# VERIFY SAVED PARQUET
# ============================================================

def verify_saved_parquet() -> pd.DataFrame:

    print("\nRe-opening saved Parquet...")

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Output file was not created:\n"
            f"{OUTPUT_PATH}"
        )

    saved = pd.read_parquet(
        OUTPUT_PATH
    )

    if saved.columns.tolist() != EXPECTED_COLUMNS:
        raise AssertionError(
            "Saved Parquet schema does not match."
        )

    print(
        f"  Rows      : {len(saved):,}"
    )

    print(
        f"  Columns   : {saved.columns.tolist()}"
    )

    return saved


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    similarity: pd.DataFrame,
) -> None:

    relationship_counts = (
        similarity["related_user_ids"]
        .map(len)
    )

    print("\n")
    print("=" * 70)
    print("USER SIMILARITY EXTRACTION COMPLETED")
    print("=" * 70)

    print("\nOUTPUT")
    print(
        f"  File                 : "
        f"{OUTPUT_PATH.name}"
    )

    print(
        f"  Users                : "
        f"{len(similarity):,}"
    )

    print(
        f"  Columns              : "
        f"{len(similarity.columns)}"
    )

    print(
        f"  Total relationships  : "
        f"{relationship_counts.sum():,}"
    )

    print(
        f"  Avg related users    : "
        f"{relationship_counts.mean():.2f}"
    )

    print(
        f"  Max related users    : "
        f"{relationship_counts.max():,}"
    )

    print(
        f"  Users with relations : "
        f"{(relationship_counts > 0).sum():,}"
    )

    print("\nSCHEMA")

    for column in similarity.columns:
        print(f"  - {column}")

    print("\nSAMPLE")

    sample = similarity.head(10)

    for row in sample.itertuples(index=False):

        print(
            f"  user_id={row.user_id:<4} "
            f"related_user_ids="
            f"{row.related_user_ids}"
        )

    print("\n" + "=" * 70)
    print("CSV -> USER SIMILARITY PARQUET: PASSED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    ratings, movies = load_source_data()

    validate_source_data(
        ratings,
        movies,
    )

    high_rated = prepare_high_rated_data(
        ratings,
        movies,
    )

    user_genres = build_user_genre_index(
        high_rated,
    )

    genre_users = build_genre_user_index(
        user_genres,
    )

    all_user_ids = sorted(
        ratings["userId"]
        .astype(int)
        .unique()
        .tolist()
    )

    similarity = extract_similarity(
        user_genres=user_genres,
        genre_users=genre_users,
        all_user_ids=all_user_ids,
    )

    validate_output(
        similarity=similarity,
        ratings=ratings,
        movies=movies,
        user_genres=user_genres,
    )

    save_output(similarity)

    saved = verify_saved_parquet()

        # --------------------------------------------------------
    # Final sanity check after writing/re-reading.
    #
    # Pandas may represent list columns differently after
    # Parquet round-trip, so compare normalized values instead
    # of using DataFrame.equals() directly.
    # --------------------------------------------------------

    print("\nFinal Parquet round-trip validation...")

    if similarity.shape != saved.shape:
        raise AssertionError(
            "Shape mismatch after Parquet round-trip.\n"
            f"In-memory: {similarity.shape}\n"
            f"Parquet:   {saved.shape}"
        )

    if similarity["user_id"].tolist() != saved["user_id"].tolist():
        raise AssertionError(
            "user_id values changed after Parquet round-trip."
        )

    for row_memory, row_saved in zip(
        similarity.itertuples(index=False),
        saved.itertuples(index=False),
    ):
        memory_user_id = int(row_memory.user_id)
        saved_user_id = int(row_saved.user_id)

        if memory_user_id != saved_user_id:
            raise AssertionError(
                "user_id mismatch after Parquet round-trip."
            )

        memory_related = sorted(
            int(user_id)
            for user_id in row_memory.related_user_ids
        )

        saved_related = sorted(
            int(user_id)
            for user_id in row_saved.related_user_ids
        )

        if memory_related != saved_related:
            raise AssertionError(
                f"related_user_ids changed for user "
                f"{memory_user_id}.\n"
                f"In-memory: {memory_related}\n"
                f"Parquet:   {saved_related}"
            )

    print(
        "  In-memory vs Parquet values : OK"
    )

    print_summary(saved)


if __name__ == "__main__":
    main()