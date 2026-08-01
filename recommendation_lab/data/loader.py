"""Load MovieLens files into DataFrames."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from recommendation_lab.config import DATA_DIR
from recommendation_lab.data.downloader import download_dataset



def load_ratings(dataset_dir: str | Path) -> pd.DataFrame:
    """Load ratings.dat into a DataFrame."""
    
    #defined the columns needed
    RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]

    df = pd.read_csv(
        Path(dataset_dir) / "ratings.dat",
        sep="::",
        engine="python",
        names=RATINGS_COLUMNS,
        encoding="latin-1",
    )
    df["rating"] = df["rating"].astype(float)
    return df


def load_users(dataset_dir: str | Path) -> pd.DataFrame:
    """Load users.dat into a DataFrame."""
    
    #defined the columns needed
    USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]

    return pd.read_csv(
        Path(dataset_dir) / "users.dat",
        sep="::",
        engine="python",
        names=USERS_COLUMNS,
        encoding="latin-1",
    )


def load_movies(dataset_dir: str | Path) -> pd.DataFrame:
    """Load movies.dat into a DataFrame with genres split into a list."""
        
    #defined the columns needed
    MOVIES_COLUMNS = ["movie_id", "title", "genres"]
    df = pd.read_csv(
        Path(dataset_dir) / "movies.dat",
        sep="::",
        engine="python",
        names=MOVIES_COLUMNS,
        encoding="latin-1",
    )
    df["genres"] = df["genres"].str.split("|")
    return df


def load_ml_1m(data_dir: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all MovieLens 1M tables, downloading the dataset if needed."""
    dataset_dir = download_dataset("ml-1m", data_dir=data_dir)
    return {
        "ratings": load_ratings(dataset_dir),
        "users": load_users(dataset_dir),
        "movies": load_movies(dataset_dir),
    }
