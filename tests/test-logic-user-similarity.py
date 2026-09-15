from pathlib import Path

import pandas as pd
import pytest


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
PARQUET_PATH = CLEAN_DIR / "user_similarity.parquet"

HIGH_RATING_THRESHOLD = 4.0


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


@pytest.fixture(scope="module")
def source_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    movies["movieId"] = movies["movieId"].astype(int)
    movies["genres_list"] = movies["genres"].apply(
        split_genres
    )

    return ratings, movies


@pytest.fixture(scope="module")
def similarity():
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def user_rating_stats(source_data):
    ratings, movies = source_data

    high_rated = ratings[
        ratings["rating"] >= HIGH_RATING_THRESHOLD
    ].copy()

    high_rated = high_rated.merge(
        movies[["movieId", "genres_list"]],
        on="movieId",
        how="inner",
        validate="many_to_one",
    )

    stats = {}

    for user_id in ratings["userId"].unique():
        user_id = int(user_id)

        user_ratings = high_rated[
            high_rated["userId"] == user_id
        ]

        genre_stats = {}

        for row in user_ratings.itertuples(
            index=False
        ):
            rating = float(row.rating)

            for genre in row.genres_list:
                if genre not in genre_stats:
                    genre_stats[genre] = {
                        "five_star": 0,
                        "four_star": 0,
                    }

                if rating == 5.0:
                    genre_stats[genre]["five_star"] += 1

                elif rating == 4.0:
                    genre_stats[genre]["four_star"] += 1

        stats[user_id] = genre_stats

    return stats


def calculate_common_score(
    user_id,
    related_user_id,
    user_rating_stats,
):
    current_stats = user_rating_stats.get(
        user_id,
        {},
    )

    related_stats = user_rating_stats.get(
        related_user_id,
        {},
    )

    shared_genres = (
        set(current_stats.keys())
        & set(related_stats.keys())
    )

    shared_five_star = 0
    shared_four_star = 0

    for genre in shared_genres:
        shared_five_star += min(
            current_stats[genre]["five_star"],
            related_stats[genre]["five_star"],
        )

        shared_four_star += min(
            current_stats[genre]["four_star"],
            related_stats[genre]["four_star"],
        )

    return (
        shared_five_star,
        shared_four_star,
    )


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


def test_parquet_exists():
    print("\nPARQUET")

    assert PARQUET_PATH.exists(), (
        f"Missing:\n{PARQUET_PATH}"
    )

    print("  user_similarity.parquet  : OK")


def test_schema(similarity):
    print("\nSCHEMA")

    assert similarity.columns.tolist() == [
        "user_id",
        "related_user_ids",
    ]

    print("  Columns                  : OK")


def test_one_row_per_user(similarity):
    print("\nUSER UNIQUENESS")

    assert not similarity[
        "user_id"
    ].duplicated().any()

    print(
        f"  Unique users             : "
        f"{similarity['user_id'].nunique():,}"
    )

    print("  One row per user         : OK")


def test_all_users_transferred(
    source_data,
    similarity,
):
    print("\nUSER TRANSFER")

    ratings, _ = source_data

    source_users = set(
        ratings["userId"]
        .astype(int)
        .unique()
    )

    output_users = set(
        similarity["user_id"]
        .astype(int)
    )

    missing_users = source_users - output_users
    extra_users = output_users - source_users

    assert not missing_users, (
        "Users missing from "
        "user_similarity.parquet: "
        f"{sorted(missing_users)}"
    )

    assert not extra_users, (
        "Users in parquet but not ratings.csv: "
        f"{sorted(extra_users)}"
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


def test_no_self_relationship(similarity):
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

    print("  No self relationships    : OK")


def test_no_duplicate_related_users(similarity):
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

    print("  No duplicate relationships : OK")


def test_related_users_exist(
    source_data,
    similarity,
):
    print("\nRELATED USER VALIDITY")

    ratings, _ = source_data

    source_users = set(
        ratings["userId"]
        .astype(int)
        .unique()
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

    print("\n  All related users exist   : OK")


def test_similarity_logic(
    source_data,
    similarity,
):
    print("\nSIMILARITY LOGIC")

    ratings, movies = source_data

    high_rated = ratings[
        ratings["rating"] >= HIGH_RATING_THRESHOLD
    ].copy()

    high_rated = high_rated.merge(
        movies[["movieId", "genres_list"]],
        on="movieId",
        how="inner",
        validate="many_to_one",
    )

    user_genres = {}

    for row in high_rated.itertuples(
        index=False
    ):
        user_id = int(row.userId)

        if user_id not in user_genres:
            user_genres[user_id] = set()

        for genre in row.genres_list:
            user_genres[user_id].add(genre)

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

        actual_related = {
            int(x)
            for x in row.related_user_ids
        }

        current_genres = user_genres.get(
            user_id,
            set(),
        )

        expected_related = set()

        for (
            other_user_id,
            other_genres,
        ) in user_genres.items():

            if other_user_id == user_id:
                continue

            if current_genres.intersection(
                other_genres
            ):
                expected_related.add(
                    other_user_id
                )

        assert actual_related == expected_related, (
            f"\nSimilarity mismatch "
            f"for user {user_id}\n"
            f"Expected: "
            f"{sorted(expected_related)}\n"
            f"Actual:   "
            f"{sorted(actual_related)}"
        )

    print("\n  Similarity logic          : OK")


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

        for related_user in (
            relationship_map[user_id]
        ):
            assert user_id in (
                relationship_map.get(
                    related_user,
                    set(),
                )
            ), (
                f"Relationship is not symmetric: "
                f"{user_id} -> {related_user}"
            )

    print("\n  Symmetric relationships  : OK")


def test_no_missing_values(similarity):
    print("\nMISSING VALUES")

    assert not similarity[
        "user_id"
    ].isna().any()

    assert not similarity[
        "related_user_ids"
    ].isna().any()

    print("  user_id                  : OK")
    print("  related_user_ids         : OK")


def test_related_users_are_sorted_by_5_and_4_star(
    similarity,
    user_rating_stats,
):
    print("\nRELATIONSHIP RANKING")

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

        scores = []

        for related_user_id in related_users:
            five_star, four_star = (
                calculate_common_score(
                    user_id,
                    related_user_id,
                    user_rating_stats,
                )
            )

            scores.append(
                (
                    related_user_id,
                    five_star,
                    four_star,
                )
            )

        expected_order = sorted(
            scores,
            key=lambda x: (
                -x[1],
                -x[2],
                x[0],
            ),
        )

        actual_order = [
            item[0]
            for item in scores
        ]

        expected_user_order = [
            item[0]
            for item in expected_order
        ]

        assert (
            actual_order
            == expected_user_order
        ), (
            f"\nWrong ranking "
            f"for user {user_id}\n"
            f"Expected: {expected_order}\n"
            f"Actual:   {scores}"
        )

    print(
        "\n  5-star score DESC         : OK"
    )

    print(
        "  4-star score DESC on tie : OK"
    )

    print(
        "  user_id ASC on full tie  : OK"
    )

    print(
        "  Ranking order             : OK"
    )


def test_ranking_examples(
    similarity,
    user_rating_stats,
):
    print("\nRANKING EXAMPLES")

    checked = 0

    for row in similarity.itertuples(
        index=False
    ):
        user_id = int(row.user_id)

        related_users = [
            int(x)
            for x in row.related_user_ids
        ]

        if len(related_users) < 2:
            continue

        for i in range(
            len(related_users) - 1
        ):
            current_user = related_users[i]
            next_user = related_users[i + 1]

            current_5, current_4 = (
                calculate_common_score(
                    user_id,
                    current_user,
                    user_rating_stats,
                )
            )

            next_5, next_4 = (
                calculate_common_score(
                    user_id,
                    next_user,
                    user_rating_stats,
                )
            )

            assert (
                current_5 > next_5
                or (
                    current_5 == next_5
                    and current_4 > next_4
                )
                or (
                    current_5 == next_5
                    and current_4 == next_4
                    and current_user < next_user
                )
            ), (
                f"\nInvalid order "
                f"for user {user_id}\n"
                f"Current: User {current_user} "
                f"(5★={current_5}, "
                f"4★={current_4})\n"
                f"Next: User {next_user} "
                f"(5★={next_5}, "
                f"4★={next_4})"
            )

            checked += 1

    print(
        f"  Consecutive relationships checked : "
        f"{checked:,}"
    )

    print(
        "  Ranking rule                        : OK"
    )


def test_summary(
    source_data,
    similarity,
):
    print("\n")
    print("=" * 70)
    print(
        "USER SIMILARITY VALIDATION SUMMARY"
    )
    print("=" * 70)

    ratings, movies = source_data

    relationship_counts = (
        similarity[
            "related_user_ids"
        ].map(len)
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

    print(
        "USER SIMILARITY VALIDATION: PASSED"
    )

    print("=" * 70)