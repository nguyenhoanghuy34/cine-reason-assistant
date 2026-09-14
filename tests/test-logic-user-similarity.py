from pathlib import Path

import pandas as pd
import pytest


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
)

CLEAN_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
)

RATINGS_PATH = SOURCE_DIR / "ratings.csv"
MOVIES_PATH = SOURCE_DIR / "movies_with_plots.csv"

PARQUET_PATH = (
    CLEAN_DIR / "user_similarity.parquet"
)


# ============================================================
# EXPECTED SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "user_id",
    "related_user_ids",
]

HIGH_RATING_THRESHOLD = 4.0


# ============================================================
# HELPERS
# ============================================================

def split_genres(genres):
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


def progress(current, total, user_id):
    percentage = current / total * 100

    print(
        f"\r    Checking user "
        f"{current:,}/{total:,} "
        f"({percentage:6.2f}%) | user_id={user_id}",
        end="",
        flush=True,
    )

    if current == total:
        print()


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def source_data():

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    movies["movieId"] = movies["movieId"].astype(int)

    movies["genres_list"] = (
        movies["genres"].apply(split_genres)
    )

    return ratings, movies


@pytest.fixture(scope="module")
def similarity():

    df = pd.read_parquet(PARQUET_PATH)

    return df


@pytest.fixture(scope="module")
def expected_user_genres(source_data):

    ratings, movies = source_data

    high_rated = ratings[
        ratings["rating"] >= HIGH_RATING_THRESHOLD
    ].copy()

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

    user_genres = {}

    for row in high_rated.itertuples(index=False):

        user_id = int(row.userId)

        if user_id not in user_genres:
            user_genres[user_id] = set()

        for genre in row.genres_list:
            user_genres[user_id].add(genre)

    return user_genres


# ============================================================
# TEST 1
# ============================================================

def test_source_files_exist():

    print("\n\nSOURCE FILES")

    assert RATINGS_PATH.exists(), (
        f"Missing:\n{RATINGS_PATH}"
    )

    assert MOVIES_PATH.exists(), (
        f"Missing:\n{MOVIES_PATH}"
    )

    print("  ratings.csv              : OK")
    print("  movies_with_plots.csv    : OK")


# ============================================================
# TEST 2
# ============================================================

def test_parquet_exists():

    print("\nPARQUET")

    assert PARQUET_PATH.exists(), (
        f"Missing:\n{PARQUET_PATH}"
    )

    print(
        f"  user_similarity.parquet : OK"
    )


# ============================================================
# TEST 3
# ============================================================

def test_schema(similarity):

    print("\nSCHEMA")

    assert similarity.columns.tolist() == EXPECTED_COLUMNS

    print("  Columns                  : OK")
    print(
        f"  - user_id"
    )
    print(
        f"  - related_user_ids"
    )


# ============================================================
# TEST 4
# ============================================================

def test_one_row_per_user(similarity):

    print("\nUSER UNIQUENESS")

    assert not similarity["user_id"].duplicated().any()

    print(
        f"  Unique users             : "
        f"{similarity['user_id'].nunique():,}"
    )

    print("  One row per user         : OK")


# ============================================================
# TEST 5
# ============================================================

def test_all_users_transferred(
    source_data,
    similarity,
):

    print("\nUSER TRANSFER")

    ratings, _ = source_data

    source_users = set(
        ratings["userId"].unique()
    )

    output_users = set(
        similarity["user_id"].astype(int)
    )

    assert source_users == output_users

    print(
        f"  Source users             : "
        f"{len(source_users):,}"
    )

    print(
        f"  Parquet users            : "
        f"{len(output_users):,}"
    )

    print("  All users transferred    : OK")


# ============================================================
# TEST 6
# ============================================================

def test_no_self_relationship(
    similarity,
):

    print("\nSELF RELATIONSHIP")

    total = len(similarity)

    for index, row in enumerate(
        similarity.itertuples(index=False),
        start=1,
    ):

        user_id = int(row.user_id)

        progress(
            index,
            total,
            user_id,
        )

        related_users = [
            int(x)
            for x in row.related_user_ids
        ]

        assert user_id not in related_users, (
            f"User {user_id} is related to itself."
        )

    print(
        "  No self relationships    : OK"
    )


# ============================================================
# TEST 7
# ============================================================

def test_no_duplicate_related_users(
    similarity,
):

    print("\nDUPLICATE RELATIONSHIPS")

    total = len(similarity)

    for index, row in enumerate(
        similarity.itertuples(index=False),
        start=1,
    ):

        user_id = int(row.user_id)

        progress(
            index,
            total,
            user_id,
        )

        related_users = [
            int(x)
            for x in row.related_user_ids
        ]

        assert len(related_users) == len(
            set(related_users)
        ), (
            f"Duplicate related users "
            f"for user {user_id}."
        )

    print(
        "  No duplicate relationships : OK"
    )


# ============================================================
# TEST 8
# ============================================================

def test_related_users_exist(
    source_data,
    similarity,
):

    print("\nRELATED USER VALIDITY")

    ratings, _ = source_data

    source_users = set(
        ratings["userId"].unique()
    )

    total = len(similarity)

    for index, row in enumerate(
        similarity.itertuples(index=False),
        start=1,
    ):

        user_id = int(row.user_id)

        progress(
            index,
            total,
            user_id,
        )

        related_users = {
            int(x)
            for x in row.related_user_ids
        }

        invalid_users = (
            related_users - source_users
        )

        assert not invalid_users, (
            f"User {user_id} contains invalid "
            f"related users: "
            f"{sorted(invalid_users)}"
        )

    print(
        "\n  All related users exist  : OK"
    )


# ============================================================
# TEST 9
# ============================================================

def test_similarity_logic(
    similarity,
    expected_user_genres,
):

    print("\nSIMILARITY LOGIC")

    total = len(similarity)

    for index, row in enumerate(
        similarity.itertuples(index=False),
        start=1,
    ):

        user_id = int(row.user_id)

        progress(
            index,
            total,
            user_id,
        )

        actual_related = sorted(
            int(x)
            for x in row.related_user_ids
        )

        current_genres = (
            expected_user_genres.get(
                user_id,
                set(),
            )
        )

        expected_related = []

        for other_user_id, other_genres in (
            expected_user_genres.items()
        ):

            if other_user_id == user_id:
                continue

            shared_genres = (
                current_genres
                .intersection(other_genres)
            )

            if shared_genres:
                expected_related.append(
                    other_user_id
                )

        expected_related = sorted(
            expected_related
        )

        assert actual_related == expected_related, (
            f"\nSimilarity mismatch "
            f"for user {user_id}\n"
            f"Expected: {expected_related}\n"
            f"Actual:   {actual_related}"
        )

    print(
        "\n  Similarity logic         : OK"
    )


# ============================================================
# TEST 10
# ============================================================

def test_relationship_is_symmetric(
    similarity,
):

    print("\nSYMMETRIC RELATIONSHIPS")

    relationship_map = {}

    for row in similarity.itertuples(
        index=False
    ):

        user_id = int(row.user_id)

        relationship_map[user_id] = {
            int(x)
            for x in row.related_user_ids
        }

    total = len(relationship_map)

    for index, user_id in enumerate(
        sorted(relationship_map),
        start=1,
    ):

        progress(
            index,
            total,
            user_id,
        )

        for related_user in relationship_map[user_id]:

            assert user_id in relationship_map.get(
                related_user,
                set(),
            ), (
                f"Relationship is not symmetric: "
                f"{user_id} -> {related_user}"
            )

    print(
        "\n  Symmetric relationships  : OK"
    )


# ============================================================
# TEST 11
# ============================================================

def test_no_missing_values(similarity):

    print("\nMISSING VALUES")

    assert not similarity["user_id"].isna().any()
    assert not similarity["related_user_ids"].isna().any()

    print("  user_id                  : OK")
    print("  related_user_ids         : OK")


# ============================================================
# TEST 12
# ============================================================

def test_related_users_are_sorted(
    similarity,
):

    print("\nRELATIONSHIP ORDER")

    total = len(similarity)

    for index, row in enumerate(
        similarity.itertuples(index=False),
        start=1,
    ):

        user_id = int(row.user_id)

        progress(
            index,
            total,
            user_id,
        )

        related_users = [
            int(x)
            for x in row.related_user_ids
        ]

        assert related_users == sorted(
            related_users
        ), (
            f"related_user_ids is not sorted "
            f"for user {user_id}."
        )

    print(
        "\n  Related users sorted     : OK"
    )


# ============================================================
# TEST 13
# ============================================================

def test_summary(
    source_data,
    similarity,
):

    print("\n")
    print("=" * 70)
    print("USER SIMILARITY VALIDATION SUMMARY")
    print("=" * 70)

    ratings, movies = source_data

    relationship_counts = (
        similarity["related_user_ids"]
        .map(len)
    )

    print("\nSOURCE")

    print(
        f"  Ratings                 : "
        f"{len(ratings):,}"
    )

    print(
        f"  Movies                  : "
        f"{len(movies):,}"
    )

    print(
        f"  Users                   : "
        f"{ratings['userId'].nunique():,}"
    )

    print("\nOUTPUT")

    print(
        f"  Users                   : "
        f"{len(similarity):,}"
    )

    print(
        f"  Columns                 : "
        f"{len(similarity.columns)}"
    )

    print(
        f"  Total relationships     : "
        f"{relationship_counts.sum():,}"
    )

    print(
        f"  Average related users   : "
        f"{relationship_counts.mean():.2f}"
    )

    print(
        f"  Maximum related users   : "
        f"{relationship_counts.max():,}"
    )

    print(
        f"  Users with relations    : "
        f"{(relationship_counts > 0).sum():,}"
    )

    print("\nSCHEMA")

    for column in similarity.columns:
        print(f"  - {column}")

    print("\n" + "=" * 70)
    print("USER SIMILARITY VALIDATION: PASSED")
    print("=" * 70)