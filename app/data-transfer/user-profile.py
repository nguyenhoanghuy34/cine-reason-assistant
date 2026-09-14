from pathlib import Path
import ast
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Subject\HOME_TEST\cine-reason-assistant")

RAW_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
OUTPUT_DIR = PROJECT_ROOT / "app" / "data" / "clean-data"

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies_with_plots.csv"
TAGS_FILE = RAW_DIR / "tags.csv"

OUTPUT_FILE = OUTPUT_DIR / "user_profiles.parquet"


# ============================================================
# CONFIG
# ============================================================

HIGH_RATING_THRESHOLD = 4.0
LOW_RATING_THRESHOLD = 3.0
TOP_GENRES = 2


# ============================================================
# HELPERS
# ============================================================

def split_genres(value):
    """Convert 'Action|Comedy|Drama' -> ['Action', 'Comedy', 'Drama']."""
    if pd.isna(value) or not str(value).strip():
        return []

    return [
        genre.strip()
        for genre in str(value).split("|")
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("Loading raw data")
    print("=" * 70)

    if not RATINGS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RATINGS_FILE}")

    if not MOVIES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {MOVIES_FILE}")

    if not TAGS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TAGS_FILE}")

    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    tags = pd.read_csv(TAGS_FILE)

    print(f"Ratings : {len(ratings):,}")
    print(f"Movies  : {len(movies):,}")
    print(f"Tags    : {len(tags):,}")

    return ratings, movies, tags


# ============================================================
# PREPARE MOVIE DATA
# ============================================================

def prepare_movies(movies):
    movies = movies.copy()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    # Keep only the columns needed for user-profile generation.
    movies = movies[
        [
            "movieId",
            "title",
            "genres_list",
        ]
    ]

    return movies


# ============================================================
# BUILD USER PROFILES
# ============================================================

def build_user_profiles(ratings, movies, tags):

    # --------------------------------------------------------
    # Join ratings with movie metadata
    # --------------------------------------------------------

    user_movies = ratings.merge(
        movies,
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    # Movies without metadata should not silently become valid
    # movie recommendations.
    missing_movies = user_movies["title"].isna().sum()

    if missing_movies > 0:
        print(
            f"Warning: {missing_movies:,} rating rows "
            f"have no matching movie metadata."
        )

    # --------------------------------------------------------
    # Prepare tag lookup
    # --------------------------------------------------------

    tags_by_user = (
        tags.groupby("userId")["tag"]
        .apply(lambda x: unique_preserve_order(
            [str(v).strip() for v in x if pd.notna(v) and str(v).strip()]
        ))
        .to_dict()
    )

    # --------------------------------------------------------
    # Build each user profile
    # --------------------------------------------------------

    profiles = []

    for user_id, user_df in user_movies.groupby("userId", sort=True):

        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        rating_count = len(user_df)

        avg_rating = round(
            user_df["rating"].mean(),
            3
        )

        watched_movie_ids = (
            user_df["movieId"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )

        # ====================================================
        # HIGH-RATED MOVIES >= 4
        # ====================================================

        high_rated = user_df[
            user_df["rating"] >= HIGH_RATING_THRESHOLD
        ]

        high_rated_movies = (
            high_rated
            .sort_values("rating", ascending=False)
            ["title"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        # ====================================================
        # HIGH-RATED GENRES
        # ====================================================

        high_genres = []

        for genres in high_rated["genres_list"]:
            high_genres.extend(genres)

        high_rated_genres = unique_preserve_order(high_genres)

        # ====================================================
        # LOW-RATED GENRES <= 3
        # ====================================================

        low_rated = user_df[
            user_df["rating"] <= LOW_RATING_THRESHOLD
        ]

        low_genres = []

        for genres in low_rated["genres_list"]:
            low_genres.extend(genres)

        low_rated_genres = unique_preserve_order(low_genres)

        # ====================================================
        # TOP 2 MOST WATCHED GENRES
        # ====================================================

        genre_counts = {}

        for genres in user_df["genres_list"]:
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        top_2_genres = [
            genre
            for genre, _ in sorted(
                genre_counts.items(),
                key=lambda x: (-x[1], x[0])
            )[:TOP_GENRES]
        ]

        # ====================================================
        # USER TAGS
        # ====================================================

        user_tags = tags_by_user.get(int(user_id), [])

        # ====================================================
        # PROFILE ROW
        # ====================================================

        profiles.append(
            {
                "user_id": int(user_id),

                "rating_count": int(rating_count),

                "avg_rating": avg_rating,

                "high_rated_movies": high_rated_movies,

                "high_rated_genres": high_rated_genres,

                "low_rated_genres": low_rated_genres,

                "user_tags": user_tags,

                "top_2_genres": top_2_genres,

                "watched_movie_ids": watched_movie_ids,
            }
        )

    return pd.DataFrame(profiles)


# ============================================================
# SAVE
# ============================================================

def save_profiles(profiles):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    print()
    print("=" * 70)
    print("Saved user profiles")
    print("=" * 70)

    print(f"File   : {OUTPUT_FILE}")
    print(f"Users  : {len(profiles):,}")
    print(f"Columns: {len(profiles.columns)}")

    print()
    print("Columns:")

    for column in profiles.columns:
        print(f"  - {column}")


# ============================================================
# MAIN
# ============================================================

def main():

    ratings, movies, tags = load_data()

    movies = prepare_movies(movies)

    profiles = build_user_profiles(
        ratings=ratings,
        movies=movies,
        tags=tags,
    )

    save_profiles(profiles)

    print()
    print("Done.")


if __name__ == "__main__":
    main()