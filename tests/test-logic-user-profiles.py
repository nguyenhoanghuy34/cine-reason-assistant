from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

RAW_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
)

PARQUET_FILE = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
    / "user_profiles.parquet"
)

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies_with_plots.csv"
TAGS_FILE = RAW_DIR / "tags.csv"


# ============================================================
# CONFIG
# ============================================================

HIGH_RATING_THRESHOLD = 4.0
LOW_RATING_THRESHOLD = 3.0
TOP_N_GENRES = 2


# ============================================================
# EXPECTED SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "user_id",
    "high_rated_genres",
    "high_rated_movies",
    "user_tags",
    "low_rated_genres",
    "unwatched_matching_movie_ids",
    "top_2_genres",
]


# ============================================================
# HELPERS
# ============================================================

def split_genres(value):
    """Convert 'Action|Comedy|Drama' into a list."""

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        genre.strip()
        for genre in value.split("|")
        if genre.strip()
    ]


def unique_preserve_order(values):
    """Remove duplicates while preserving order."""

    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def as_list(value):
    """
    Convert Parquet list-like values to Python list
    ONLY for comparison.

    The original Parquet DataFrame is NOT modified.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if hasattr(value, "tolist"):
        converted = value.tolist()

        if isinstance(converted, list):
            return converted

    return list(value)


def progress(current, total, user_id):
    """Print validation progress."""

    percent = current / total * 100

    print(
        f"\r    Checking user "
        f"{current:>3}/{total:<3} "
        f"({percent:6.2f}%) "
        f"| user_id={user_id}",
        end="",
        flush=True,
    )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def source_data():

    ratings = pd.read_csv(
        RATINGS_FILE
    )

    movies = pd.read_csv(
        MOVIES_FILE
    )

    tags = pd.read_csv(
        TAGS_FILE
    )

    ratings["userId"] = (
        ratings["userId"]
        .astype(int)
    )

    ratings["movieId"] = (
        ratings["movieId"]
        .astype(int)
    )

    ratings["rating"] = (
        ratings["rating"]
        .astype(float)
    )

    movies["movieId"] = (
        movies["movieId"]
        .astype(int)
    )

    movies["genres_list"] = (
        movies["genres"]
        .apply(split_genres)
    )

    tags["userId"] = (
        tags["userId"]
        .astype(int)
    )

    tags["movieId"] = (
        tags["movieId"]
        .astype(int)
    )

    return {
        "ratings": ratings,
        "movies": movies,
        "tags": tags,
    }


@pytest.fixture(scope="module")
def profiles():

    return pd.read_parquet(
        PARQUET_FILE,
        engine="pyarrow",
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def test_source_files_exist():

    assert RATINGS_FILE.exists(), (
        f"Missing: {RATINGS_FILE}"
    )

    assert MOVIES_FILE.exists(), (
        f"Missing: {MOVIES_FILE}"
    )

    assert TAGS_FILE.exists(), (
        f"Missing: {TAGS_FILE}"
    )


def test_parquet_file_exists():

    assert PARQUET_FILE.exists(), (
        f"Missing: {PARQUET_FILE}"
    )


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def test_profile_schema(profiles):

    assert profiles.columns.tolist() == (
        EXPECTED_COLUMNS
    )


# ============================================================
# USER COVERAGE
# ============================================================

def test_all_users_transferred(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]

    source_users = set(
        ratings["userId"].unique()
    )

    profile_users = set(
        profiles["user_id"].unique()
    )

    assert profile_users == source_users

    assert len(profiles) == len(
        source_users
    )


# ============================================================
# HIGH-RATED GENRES
# ============================================================

def test_high_rated_genres(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[
            [
                "movieId",
                "genres_list",
            ]
        ],
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    total = len(profiles)

    print()
    print(
        "\n  HIGH-RATED GENRES"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        user_data = merged[
            merged["userId"] == user_id
        ]

        high_rated = user_data[
            user_data["rating"]
            >= HIGH_RATING_THRESHOLD
        ]

        expected_genres = []

        for genres in (
            high_rated["genres_list"]
        ):
            expected_genres.extend(
                genres
            )

        expected = (
            unique_preserve_order(
                expected_genres
            )
        )

        actual = as_list(
            profile[
                "high_rated_genres"
            ]
        )

        assert actual == expected, (
            f"\nuser_id={user_id}: "
            "high_rated_genres mismatch\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print()


# ============================================================
# HIGH-RATED MOVIES
# ============================================================

def test_high_rated_movies(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[
            [
                "movieId",
                "title",
            ]
        ],
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    total = len(profiles)

    print()
    print(
        "\n  HIGH-RATED MOVIES"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        user_data = merged[
            merged["userId"] == user_id
        ]

        high_rated = user_data[
            user_data["rating"]
            >= HIGH_RATING_THRESHOLD
        ]

        expected = (
            high_rated
            .sort_values(
                "rating",
                ascending=False,
            )
            ["title"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        actual = as_list(
            profile[
                "high_rated_movies"
            ]
        )

        assert actual == expected, (
            f"\nuser_id={user_id}: "
            "high_rated_movies mismatch\n"
            f"Expected: {expected[:10]}\n"
            f"Actual:   {actual[:10]}"
        )

    print()


# ============================================================
# USER TAGS
# ============================================================

def test_user_tags(
    source_data,
    profiles,
):

    tags = source_data["tags"]

    total = len(profiles)

    print()
    print(
        "\n  USER TAGS"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        source_tags = tags[
            tags["userId"] == user_id
        ]["tag"]

        source_tags = (
            source_tags
            .dropna()
            .astype(str)
            .str.strip()
        )

        source_tags = source_tags[
            source_tags != ""
        ]

        expected = (
            unique_preserve_order(
                source_tags.tolist()
            )
        )

        actual = as_list(
            profile["user_tags"]
        )

        assert actual == expected, (
            f"\nuser_id={user_id}: "
            "user_tags mismatch\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print()


# ============================================================
# LOW-RATED GENRES
# ============================================================

def test_low_rated_genres(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[
            [
                "movieId",
                "genres_list",
            ]
        ],
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    total = len(profiles)

    print()
    print(
        "\n  LOW-RATED GENRES"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        user_data = merged[
            merged["userId"] == user_id
        ]

        low_rated = user_data[
            user_data["rating"]
            <= LOW_RATING_THRESHOLD
        ]

        expected_genres = []

        for genres in (
            low_rated["genres_list"]
        ):
            expected_genres.extend(
                genres
            )

        expected = (
            unique_preserve_order(
                expected_genres
            )
        )

        actual = as_list(
            profile[
                "low_rated_genres"
            ]
        )

        assert actual == expected, (
            f"\nuser_id={user_id}: "
            "low_rated_genres mismatch\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print()


# ============================================================
# TOP 2 GENRES
# ============================================================

def test_top_2_genres(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]
    movies = source_data["movies"]

    merged = ratings.merge(
        movies[
            [
                "movieId",
                "genres_list",
            ]
        ],
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    total = len(profiles)

    print()
    print(
        "\n  TOP 2 GENRES"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        user_data = merged[
            merged["userId"] == user_id
        ]

        genre_counts = {}

        for genres in (
            user_data["genres_list"]
        ):

            for genre in genres:

                genre_counts[genre] = (
                    genre_counts.get(
                        genre,
                        0,
                    )
                    + 1
                )

        expected = [
            genre
            for genre, _ in sorted(
                genre_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )[:TOP_N_GENRES]
        ]

        actual = as_list(
            profile[
                "top_2_genres"
            ]
        )

        assert actual == expected, (
            f"\nuser_id={user_id}: "
            "top_2_genres mismatch\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print()


# ============================================================
# UNWATCHED MATCHING MOVIES
# ============================================================

def test_unwatched_matching_movie_ids(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]
    movies = source_data["movies"]

    total = len(profiles)

    print()
    print(
        "\n  UNWATCHED MATCHING MOVIES"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        # ----------------------------------------------------
        # Movies watched by this user
        # ----------------------------------------------------

        watched_ids = set(
            ratings.loc[
                ratings["userId"] == user_id,
                "movieId",
            ]
            .astype(int)
            .tolist()
        )

        # ----------------------------------------------------
        # High-rated genres
        # ----------------------------------------------------

        high_rated_genres = set(
            as_list(
                profile[
                    "high_rated_genres"
                ]
            )
        )

        # ----------------------------------------------------
        # Find unwatched movies
        # with matching genres
        # ----------------------------------------------------

        expected = []

        if high_rated_genres:

            unwatched_movies = movies[
                ~movies["movieId"].isin(
                    watched_ids
                )
            ]

            for _, movie in (
                unwatched_movies.iterrows()
            ):

                movie_genres = set(
                    movie["genres_list"]
                )

                if (
                    movie_genres
                    & high_rated_genres
                ):
                    expected.append(
                        int(movie["movieId"])
                    )

        expected = (
            unique_preserve_order(
                expected
            )
        )

        actual = as_list(
            profile[
                "unwatched_matching_movie_ids"
            ]
        )

        assert actual == expected, (
            f"\nuser_id={user_id}: "
            "unwatched_matching_movie_ids "
            "mismatch\n"
            f"Expected count: {len(expected)}\n"
            f"Actual count:   {len(actual)}\n"
            f"Expected first: {expected[:20]}\n"
            f"Actual first:   {actual[:20]}"
        )

        # ----------------------------------------------------
        # Additional logical checks
        # ----------------------------------------------------

        assert not (
            set(actual)
            & watched_ids
        ), (
            f"\nuser_id={user_id}: "
            "unwatched list contains "
            "a movie already watched"
        )

    print()


# ============================================================
# PROFILE LOGICAL CONSISTENCY
# ============================================================

def test_profile_consistency(
    profiles,
):

    total = len(profiles)

    print()
    print(
        "\n  PROFILE CONSISTENCY"
    )

    for position, (_, profile) in enumerate(
        profiles.iterrows(),
        start=1,
    ):

        user_id = int(
            profile["user_id"]
        )

        progress(
            position,
            total,
            user_id,
        )

        high_genres = set(
            as_list(
                profile[
                    "high_rated_genres"
                ]
            )
        )

        top_genres = as_list(
            profile[
                "top_2_genres"
            ]
        )

        matching_movies = as_list(
            profile[
                "unwatched_matching_movie_ids"
            ]
        )

        # Top genres must be <= 2
        assert len(top_genres) <= 2

        # Top genres must be unique
        assert len(top_genres) == len(
            set(top_genres)
        )

        # Matching movie IDs must be unique
        assert len(matching_movies) == len(
            set(matching_movies)
        )

        # Every matching movie must be
        # related to a high-rated genre.
        #
        # This is already tested against
        # the source data above.

        if matching_movies:
            assert high_genres

    print()


# ============================================================
# NO MISSING CORE VALUES
# ============================================================

def test_no_missing_core_values(
    profiles,
):

    core_columns = [
        "user_id",
        "high_rated_genres",
        "high_rated_movies",
        "user_tags",
        "low_rated_genres",
        "unwatched_matching_movie_ids",
        "top_2_genres",
    ]

    assert not (
        profiles[
            core_columns
        ]
        .isnull()
        .any()
        .any()
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def test_transfer_summary(
    source_data,
    profiles,
):

    ratings = source_data["ratings"]
    movies = source_data["movies"]
    tags = source_data["tags"]

    print()
    print()
    print("=" * 70)
    print(
        "CSV -> USER PROFILE PARQUET"
    )
    print(
        "DATA TRANSFER VALIDATION"
    )
    print("=" * 70)

    print()
    print("SOURCE DATA")
    print(
        f"  ratings.csv           : "
        f"{len(ratings):,}"
    )
    print(
        f"  movies_with_plots.csv : "
        f"{len(movies):,}"
    )
    print(
        f"  tags.csv              : "
        f"{len(tags):,}"
    )
    print(
        f"  unique users          : "
        f"{ratings['userId'].nunique():,}"
    )

    print()
    print("GENERATED PARQUET")
    print(
        f"  users                 : "
        f"{len(profiles):,}"
    )
    print(
        f"  columns               : "
        f"{len(profiles.columns)}"
    )

    print()
    print("FINAL SCHEMA")

    for column in profiles.columns:
        print(
            f"  - {column}"
        )

    print()
    print("=" * 70)
    print(
        "CSV -> PARQUET VALIDATION: PASSED"
    )
    print("=" * 70)