from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

CLEAN_DATA_DIR = BASE_DIR / "data" / "clean-data"
TEMP_DATA_DIR = BASE_DIR / "data" / "temp-data"

USER_PROFILES_PATH = CLEAN_DATA_DIR / "user_profiles.parquet"
USER_SIMILARITY_PATH = CLEAN_DATA_DIR / "user_similarity.parquet"


def create_user_temp(user_id: int) -> Path:
    """
    Create a temporary parquet file for one user.

    The file is stored on disk and contains only the requested
    user's profile and similarity data.
    """

    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    user_profiles = pd.read_parquet(USER_PROFILES_PATH)
    user_similarity = pd.read_parquet(USER_SIMILARITY_PATH)

    # Filter requested user
    profile = user_profiles[
        user_profiles["user_id"] == user_id
    ].copy()

    similarity = user_similarity[
        user_similarity["user_id"] == user_id
    ].copy()

    if profile.empty:
        raise ValueError(f"User ID {user_id} not found.")

    # Rename similarity columns to avoid conflicts
    similarity = similarity.rename(
        columns={
            column: f"similarity_{column}"
            for column in similarity.columns
            if column != "user_id"
        }
    )

    # Merge profile + similarity
    temp_data = profile.merge(
        similarity,
        on="user_id",
        how="left",
    )

    # Save temporary file
    temp_path = TEMP_DATA_DIR / f"user_{user_id}.parquet"

    temp_data.to_parquet(
        temp_path,
        index=False,
    )

    return temp_path


def delete_user_temp(user_id: int) -> None:
    """
    Delete the temporary user file.
    """

    temp_path = TEMP_DATA_DIR / f"user_{user_id}.parquet"

    if temp_path.exists():
        temp_path.unlink()