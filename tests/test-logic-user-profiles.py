from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Subject\HOME_TEST\cine-reason-assistant")

RAW_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
PARQUET_FILE = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
    / "user_profiles.parquet"
)

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies.csv"
TAGS_FILE = RAW_DIR / "tags.csv"


# ============================================================
# CONFIG
# ============================================================

HIGH_RATING_THRESHOLD = 4.0
LOW_RATING_THRESHOLD = 3.0
TOP_N_GENRES = 2


# ============================================================
# HELPERS
# ============================================================

def split_genres(genres):
    """Convert 'Action|Comedy' -> ['Action', 'Comedy']."""
    if pd.isna(genres):
        return []

    return [
        genre.strip()
        for genre in str(genres).split("|")
        if genre.strip()
    ]


def unique_preserve_order(values):
    """Remove duplicates while preserving original order."""
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def as_list(value):
    """
    Convert Parquet list/ndarray representation to Python list
    ONLY for comparison.

    This does NOT modify the loaded Parquet DataFrame.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def source_data():
    """Load original CSV files."""
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    tags = pd.read_csv(TAGS_FILE)

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    movies["movieId"] = movies["movieId"].astype(int)

    tags["userId"] = tags["userId"].astype(int)
    tags["movieId"] = tags["movieId"].astype(int)

    return {
        "ratings": ratings,
        "movies": movies,
        "tags": tags,
    }


@pytest.fixture(scope="module")
def profiles():
    """Load the generated Parquet file."""
    return pd.read_parquet(PARQUET_FILE)


# ============================================================
# FILE EXISTENCE
# ============================================================

def test_source_files_exist():
    """Verify all source CSV files exist."""
    assert RATINGS_FILE.exists(), f"Missing: {RATINGS_FILE}"
    assert MOVIES_FILE.exists(), f"Missing: {MOVIES_FILE}"
    assert TAGS_FILE.exists(), f"Missing: {TAGS_FILE}"


def test_parquet_file_exists():
    """Verify generated Parquet file exists."""
    assert PARQUET_FILE.exists(), f"Missing: {PARQUET_FILE}"


# ============================================================
# USER COVERAGE
# ============================================================

def test_all_users_are_transferred(source_data, profiles):
    """
    Every user appearing in ratings.csv must appear in
    user_profiles.parquet.
    """
    ratings_users = set(source_data["ratings"]["userId"].unique())
    profile_users = set(profiles["user_id"].unique())

    assert profile_users == ratings_users


# ============================================================
# RATING COUNT
# ============================================================

def test_rating_count(source_data, profiles):
    """
    Verify rating_count in Parquet equals the number of
    rating rows for each user in ratings.csv.
    """
    ratings = source_data["ratings"]

    expected = ratings.groupby("userId").size()

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        expected_count = int(expected.loc[user_id])
        actual_count = int(row["rating_count"])

        assert actual_count == expected_count, (
            f"user_id={user_id}: "
            f"expected rating_count={expected_count}, "
            f"got {actual_count}"
        )


# ============================================================
# AVERAGE RATING
# ============================================================

def test_average_rating(source_data, profiles):
    """
    Verify avg_rating in Parquet equals the average rating
    calculated directly from ratings.csv.
    """
    ratings = source_data["ratings"]

    expected = ratings.groupby("userId")["rating"].mean()

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        expected_avg = round(float(expected.loc[user_id]), 3)
        actual_avg = round(float(row["avg_rating"]), 3)

        assert actual_avg == expected_avg, (
            f"user_id={user_id}: "
            f"expected avg_rating={expected_avg}, "
            f"got {actual_avg}"
        )


# ============================================================
# WATCHED MOVIE IDS
# ============================================================

def test_watched_movie_ids(source_data, profiles):
    """
    Verify watched_movie_ids comes from movieId values in
    ratings.csv for the same user.
    """
    ratings = source_data["ratings"]

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        source_user = ratings[ratings["userId"] == user_id]

        expected = unique_preserve_order(
            source_user["movieId"].tolist()
        )

        actual = as_list(row["watched_movie_ids"])

        assert actual == expected, (
            f"user_id={user_id}: watched_movie_ids mismatch"
        )


# ============================================================
# HIGH-RATED MOVIES
# ============================================================

def test_high_rated_movies(source_data, profiles):
    """
    Verify high_rated_movies is correctly generated from
    ratings >= 4.0 and movies.csv.
    """
    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left",
    )

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        user_data = merged[
            (merged["userId"] == user_id)
            & (merged["rating"] >= HIGH_RATING_THRESHOLD)
        ]

        expected = (
            user_data
            .sort_values("rating", ascending=False)
            ["title"]
            .drop_duplicates()
            .tolist()
        )

        actual = as_list(row["high_rated_movies"])

        assert actual == expected, (
            f"user_id={user_id}: high_rated_movies mismatch"
        )


# ============================================================
# HIGH-RATED GENRES
# ============================================================

def test_high_rated_genres(source_data, profiles):
    """
    Verify high_rated_genres is generated from genres of
    movies rated >= 4.0.
    """
    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[["movieId", "genres"]],
        on="movieId",
        how="left",
    )

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        user_data = merged[
            (merged["userId"] == user_id)
            & (merged["rating"] >= HIGH_RATING_THRESHOLD)
        ]

        expected_genres = []

        for genres in user_data["genres"]:
            expected_genres.extend(split_genres(genres))

        expected = unique_preserve_order(expected_genres)

        actual = as_list(row["high_rated_genres"])

        assert actual == expected, (
            f"user_id={user_id}: high_rated_genres mismatch"
        )


# ============================================================
# LOW-RATED GENRES
# ============================================================

def test_low_rated_genres(source_data, profiles):
    """
    Verify low_rated_genres is generated from genres of
    movies rated <= 3.0.
    """
    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[["movieId", "genres"]],
        on="movieId",
        how="left",
    )

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        user_data = merged[
            (merged["userId"] == user_id)
            & (merged["rating"] <= LOW_RATING_THRESHOLD)
        ]

        expected_genres = []

        for genres in user_data["genres"]:
            expected_genres.extend(split_genres(genres))

        expected = unique_preserve_order(expected_genres)

        actual = as_list(row["low_rated_genres"])

        assert actual == expected, (
            f"user_id={user_id}: low_rated_genres mismatch"
        )


# ============================================================
# USER TAGS
# ============================================================

def test_user_tags(source_data, profiles):
    """
    Verify user_tags in Parquet matches tags.csv for each user.
    """
    tags = source_data["tags"]

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        user_tags = tags[
            tags["userId"] == user_id
        ]["tag"].tolist()

        expected = unique_preserve_order(user_tags)

        actual = as_list(row["user_tags"])

        assert actual == expected, (
            f"user_id={user_id}: user_tags mismatch"
        )


# ============================================================
# TOP 2 GENRES
# ============================================================

def test_top_2_genres(source_data, profiles):
    """
    Verify top_2_genres is calculated from genre frequency
    in the user's watched movies.

    Ranking:
        1. Highest watch count
        2. Alphabetical order for ties
    """
    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[["movieId", "genres"]],
        on="movieId",
        how="left",
    )

    for _, row in profiles.iterrows():
        user_id = row["user_id"]

        user_data = merged[
            merged["userId"] == user_id
        ]

        genre_counts = {}

        for genres in user_data["genres"]:
            for genre in split_genres(genres):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        expected = [
            genre
            for genre, _ in sorted(
                genre_counts.items(),
                key=lambda x: (-x[1], x[0])
            )[:TOP_N_GENRES]
        ]

        actual = as_list(row["top_2_genres"])

        assert actual == expected, (
            f"user_id={user_id}: "
            f"expected top_2_genres={expected}, "
            f"got {actual}"
        )


# ============================================================
# COMPLETE PROFILE VALIDATION
# ============================================================

def test_profile_values_not_null(source_data, profiles):
    """
    Verify important scalar values are not missing after
    CSV -> transformation -> Parquet.
    """
    required_columns = [
        "user_id",
        "rating_count",
        "avg_rating",
    ]

    assert not profiles[required_columns].isnull().any().any()


def test_profile_count_matches_source(source_data, profiles):
    """
    Number of profiles must equal number of unique users
    in ratings.csv.
    """
    expected_users = source_data["ratings"]["userId"].nunique()
    actual_users = profiles["user_id"].nunique()

    assert actual_users == expected_users


# ============================================================
# FINAL SUMMARY
# ============================================================

def test_print_transfer_summary(source_data, profiles):
    """Print a short CSV -> Parquet validation summary."""
    ratings = source_data["ratings"]
    movies = source_data["movies"]
    tags = source_data["tags"]

    print("\n" + "=" * 70)
    print("CSV -> USER PROFILES PARQUET VALIDATION")
    print("=" * 70)

    print(f"Source ratings : {len(ratings):,}")
    print(f"Source movies  : {len(movies):,}")
    print(f"Source tags    : {len(tags):,}")
    print(f"Source users   : {ratings['userId'].nunique():,}")

    print("-" * 70)

    print(f"Parquet users  : {len(profiles):,}")
    print(f"Parquet columns: {len(profiles.columns):,}")

    print("-" * 70)

    print("Validation:")
    print("  User coverage       : PASSED")
    print("  Rating count        : PASSED")
    print("  Average rating      : PASSED")
    print("  Watched movies      : PASSED")
    print("  High-rated movies   : PASSED")
    print("  High-rated genres   : PASSED")
    print("  Low-rated genres    : PASSED")
    print("  User tags           : PASSED")
    print("  Top 2 genres        : PASSED")

    print("=" * 70)