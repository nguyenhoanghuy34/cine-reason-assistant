from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

PROFILE_FILE = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
    / "user_profiles.parquet"
)


# ============================================================
# EXPECTED SCHEMA
# ============================================================

EXPECTED_COLUMNS = {
    "user_id",
    "rating_count",
    "avg_rating",
    "watched_movie_ids",
    "high_rated_movies",
    "high_rated_genres",
    "low_rated_genres",
    "user_tags",
    "top_2_genres",
}


LIST_COLUMNS = [
    "watched_movie_ids",
    "high_rated_movies",
    "high_rated_genres",
    "low_rated_genres",
    "user_tags",
    "top_2_genres",
]


# ============================================================
# LOAD
# ============================================================

def load_profiles():
    assert PROFILE_FILE.exists(), (
        f"Profile file not found:\n{PROFILE_FILE}"
    )

    df = pd.read_parquet(
        PROFILE_FILE,
        engine="pyarrow",
    )

    assert not df.empty, (
        "user_profiles.parquet is empty"
    )

    return df


# ============================================================
# TEST 1 — FILE EXISTS
# ============================================================

def test_profile_file_exists():

    assert PROFILE_FILE.exists()


# ============================================================
# TEST 2 — NOT EMPTY
# ============================================================

def test_profiles_not_empty():

    df = load_profiles()

    assert len(df) == 610, (
        f"Expected 610 users, got {len(df)}"
    )


# ============================================================
# TEST 3 — SCHEMA
# ============================================================

def test_profile_schema():

    df = load_profiles()

    assert set(df.columns) == EXPECTED_COLUMNS


# ============================================================
# TEST 4 — USER ID
# ============================================================

def test_user_id():

    df = load_profiles()

    assert df["user_id"].notna().all()

    assert df["user_id"].is_unique

    assert (
        df["user_id"] > 0
    ).all()


# ============================================================
# TEST 5 — RATING COUNT
# ============================================================

def test_rating_count():

    df = load_profiles()

    assert df["rating_count"].notna().all()

    assert (
        df["rating_count"] > 0
    ).all()


# ============================================================
# TEST 6 — AVERAGE RATING
# ============================================================

def test_average_rating():

    df = load_profiles()

    assert df["avg_rating"].notna().all()

    assert (
        df["avg_rating"] >= 0
    ).all()

    assert (
        df["avg_rating"] <= 5
    ).all()


# ============================================================
# TEST 7 — WATCHED MOVIE IDS
# ============================================================

def test_watched_movie_ids():

    df = load_profiles()

    for _, row in df.iterrows():

        watched = row["watched_movie_ids"]

        # PyArrow/Pandas may return ndarray.
        assert isinstance(
            watched,
            (list, np.ndarray),
        )

        watched = np.asarray(
            watched
        )

        assert len(watched) == row[
            "rating_count"
        ]

        assert len(watched) == len(
            np.unique(watched)
        )

        assert np.issubdtype(
            watched.dtype,
            np.integer,
        )


# ============================================================
# TEST 8 — HIGH-RATED MOVIES
# ============================================================

def test_high_rated_movies():

    df = load_profiles()

    for _, row in df.iterrows():

        movies = row[
            "high_rated_movies"
        ]

        assert isinstance(
            movies,
            (list, np.ndarray),
        )

        movies = list(movies)

        assert len(movies) == len(
            set(movies)
        )

        assert all(
            isinstance(movie, str)
            for movie in movies
        )


# ============================================================
# TEST 9 — HIGH-RATED GENRES
# ============================================================

def test_high_rated_genres():

    df = load_profiles()

    for _, row in df.iterrows():

        genres = row[
            "high_rated_genres"
        ]

        assert isinstance(
            genres,
            (list, np.ndarray),
        )

        genres = list(genres)

        assert len(genres) == len(
            set(genres)
        )

        assert all(
            isinstance(genre, str)
            for genre in genres
        )


# ============================================================
# TEST 10 — LOW-RATED GENRES
# ============================================================

def test_low_rated_genres():

    df = load_profiles()

    for _, row in df.iterrows():

        genres = row[
            "low_rated_genres"
        ]

        assert isinstance(
            genres,
            (list, np.ndarray),
        )

        genres = list(genres)

        assert len(genres) == len(
            set(genres)
        )

        assert all(
            isinstance(genre, str)
            for genre in genres
        )


# ============================================================
# TEST 11 — USER TAGS
# ============================================================

def test_user_tags():

    df = load_profiles()

    for _, row in df.iterrows():

        tags = row["user_tags"]

        assert isinstance(
            tags,
            (list, np.ndarray),
        )

        tags = list(tags)

        assert len(tags) == len(
            set(tags)
        )

        assert all(
            isinstance(tag, str)
            for tag in tags
        )


# ============================================================
# TEST 12 — TOP 2 GENRES
# ============================================================

def test_top_2_genres():

    df = load_profiles()

    for _, row in df.iterrows():

        genres = row[
            "top_2_genres"
        ]

        assert isinstance(
            genres,
            (list, np.ndarray),
        )

        genres = list(genres)

        assert len(genres) <= 2

        assert len(genres) == len(
            set(genres)
        )


# ============================================================
# TEST 13 — PROFILE CONSISTENCY
# ============================================================

def test_profile_consistency():

    df = load_profiles()

    for _, row in df.iterrows():

        high_movies = list(
            row["high_rated_movies"]
        )

        high_genres = list(
            row["high_rated_genres"]
        )

        if len(high_movies) > 0:

            assert len(high_genres) > 0


# ============================================================
# TEST 14 — CORE DATA QUALITY
# ============================================================

def test_no_nan_values_in_core_columns():

    df = load_profiles()

    core_columns = [
        "user_id",
        "rating_count",
        "avg_rating",
    ]

    for column in core_columns:

        assert df[column].notna().all()


# ============================================================
# TEST 15 — CHECK ACTUAL PARQUET DATA
# ============================================================

def test_profile_summary():

    df = load_profiles()

    print()
    print("=" * 70)
    print("USER PROFILE PARQUET VALIDATION")
    print("=" * 70)

    print(
        f"File    : {PROFILE_FILE}"
    )

    print(
        f"Users   : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print()
    print("Column types after Parquet read:")

    for column in df.columns:

        value = df[column].iloc[0]

        print(
            f"  {column:<25} "
            f"{type(value).__name__}"
        )

    print()
    print("First profile:")

    first = df.iloc[0]

    for column in df.columns:

        print(
            f"  {column:<25}: "
            f"{first[column]}"
        )

    print("=" * 70)

    assert len(df) == 610