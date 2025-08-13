import sys
import joblib
import pandas as pd
from pathlib import Path


def inspect_movie_data(title):
    """
    Loads the final DataFrame and prints all details for a specific movie title.
    """
    print(f"--- Inspecting Data for: '{title}' ---")

    # --- Load Artifacts ---
    try:
        # Construct the path relative to this script's location
        MODEL_DIR = Path(__file__).resolve().parent / "movie_recommender" / "saved_models"
        api_df = joblib.load(MODEL_DIR / 'movies_df.joblib')
        indices_map = joblib.load(MODEL_DIR / 'indices_map.joblib')
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # --- Find Index and Print All Data ---
    try:
        # Use .get() for a safer lookup that returns None if not found
        idx = indices_map.get(title)

        if idx is None:
            print(f"Title '{title}' not found in indices_map.")
            return

        # Handle duplicate titles by taking the first one
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Retrieve the full row of data for this movie
        movie_data = api_df.iloc[idx]

        print("\n--- DATA ON RECORD ---")
        print(movie_data.to_string())
        print("\n----------------------")

    except KeyError:
        print(f"Could not find title '{title}' in the DataFrame.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_movie.py \"<Movie Title>\"")
    else:
        inspect_movie_data(sys.argv[1])

import sys
import joblib
import pandas as pd
from pathlib import Path

def inspect_movie_data(title):
    """
    Loads the final DataFrame and prints all details for a specific movie title.
    """
    print(f"--- Inspecting Data for: '{title}' ---")

    # --- Load Artifacts ---
    try:
        # Construct the path relative to this script's location
        MODEL_DIR = Path(__file__).resolve().parent / "movie_recommender" / "saved_models"
        api_df = joblib.load(MODEL_DIR / 'movies_df.joblib')
        indices_map = joblib.load(MODEL_DIR / 'indices_map.joblib')
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # --- Find Index and Print All Data ---
    try:
        # Use .get() for a safer lookup that returns None if not found
        idx = indices_map.get(title)

        if idx is None:
            print(f"Title '{title}' not found in indices_map.")
            return

        # Handle duplicate titles by taking the first one
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Retrieve the full row of data for this movie
        movie_data = api_df.iloc[idx]

        print("\n--- DATA ON RECORD ---")
        print(movie_data.to_string())
        print("\n----------------------")

    except KeyError:
        print(f"Could not find title '{title}' in the DataFrame.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_movie.py \"<Movie Title>\"")
    else:
        inspect_movie_data(sys.argv[1])