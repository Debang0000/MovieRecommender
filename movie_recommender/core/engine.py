# movie_recommender/core/engine.py (FINAL VERSION: Multi-Mode Logic + Optional SVD Hybrid)

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load All Artifacts at Startup ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "saved_models"
    
    api_df = joblib.load(MODEL_DIR / 'movies_df.joblib')
    embedding_matrix = np.load(MODEL_DIR / 'bge_embeddings.npy')
    indices_map = joblib.load(MODEL_DIR / 'indices_map.joblib')
    
    print(f"--- All artifacts for Multi-Mode Engine loaded successfully. ---")
except Exception as e:
    print(f"--- FATAL LOADING ERROR in engine.py: {e} ---")
    api_df = None
    embedding_matrix = None
    indices_map = None

# Try loading optional SVD model
svd_model = None
svd_trainset = None
svd_item_inner_map = None
try:
    SVD_PATH = MODEL_DIR / 'svd_model.joblib'
    if SVD_PATH.exists():
        svd_model = joblib.load(SVD_PATH)
        svd_trainset = getattr(svd_model, 'trainset', None)
        # In Surprise, mapping from raw item id -> inner id is via to_inner_iid
        if svd_trainset is not None:
            # Build a cached raw->inner mapping for quick lookup
            raw_item_ids = set()
            try:
                # Best effort: access the internal dict if available
                raw_item_ids = set(getattr(svd_trainset, '_raw2inner_id_items', {}).keys())
            except Exception:
                raw_item_ids = set()
            svd_item_inner_map = svd_trainset.to_inner_iid if hasattr(svd_trainset, 'to_inner_iid') else None
        print("--- Optional SVD model loaded for hybrid scoring. ---")
    else:
        print("--- SVD model not found; running without collaborative component. ---")
except Exception as e:
    print(f"--- WARNING: Failed to load SVD model: {e}. Proceeding without collaborative component. ---")
    svd_model = None
    svd_trainset = None
    svd_item_inner_map = None

# --- 2. Helper Functions ---

def _sanitize_movies(movies_df: pd.DataFrame):
    """Prepares movie list for JSON response."""
    results_list = movies_df.to_dict(orient='records')
    sanitized = []
    for movie in results_list:
        clean_movie = {k: v if not (pd.api.types.is_scalar(v) and pd.isna(v)) else None for k, v in movie.items()}
        if clean_movie.get('poster_path'):
            clean_movie['poster_url'] = f"https://image.tmdb.org/t/p/w500{clean_movie['poster_path']}"
        else:
            clean_movie['poster_url'] = "https://via.placeholder.com/500x750.png?text=No+Image"
        
        # Include overview (description) if available
        if 'overview' in clean_movie:
            clean_movie['overview'] = clean_movie.get('overview', 'No description available.')
        
        sanitized.append(clean_movie)
    return sanitized


def _get_paginated_results(df: pd.DataFrame, page: int = 1, page_size: int = 10) -> pd.DataFrame:
    """Slices a DataFrame for pagination."""
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    return df.iloc[start_index:end_index]


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D numpy array to [0, 1]. Returns zeros if degenerate."""
    if values is None or len(values) == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _compute_svd_scores_for_candidates(candidate_indices: np.ndarray, liked_indices: list) -> np.ndarray:
    """
    Compute SVD-based collaborative scores for candidate movies using a pseudo-user vector
    derived from liked movies. Returns an array aligned with candidate_indices.
    Falls back to zeros if SVD artifacts are unavailable.
    """
    if svd_model is None or svd_trainset is None or svd_item_inner_map is None:
        return np.zeros(len(candidate_indices), dtype=float)

    # Access SVD internals (best effort; compatible with Surprise SVD)
    global_mean = getattr(svd_model, 'global_mean', 0.0)
    bi = getattr(svd_model, 'bi', None)
    qi = getattr(svd_model, 'qi', None)
    if qi is None or bi is None:
        return np.zeros(len(candidate_indices), dtype=float)

    # Build pseudo user vector from liked items' factors if possible
    liked_item_vectors = []
    for idx in liked_indices:
        raw_item_id = api_df.iloc[idx]['id'] if 'id' in api_df.columns else None
        if raw_item_id is None:
            continue
        try:
            inner_iid = svd_trainset.to_inner_iid(str(raw_item_id)) if svd_item_inner_map is not None else None
        except Exception:
            inner_iid = None
        if inner_iid is not None and 0 <= inner_iid < qi.shape[0]:
            liked_item_vectors.append(qi[inner_iid])
    if len(liked_item_vectors) == 0:
        # No overlap with SVD item space; return zeros
        return np.zeros(len(candidate_indices), dtype=float)

    pseudo_user_vector = np.mean(np.stack(liked_item_vectors, axis=0), axis=0)

    # Score each candidate: approx prediction = global_mean + bi[i] + dot(pu*, qi[i])
    scores = np.zeros(len(candidate_indices), dtype=float)
    for pos, cand_idx in enumerate(candidate_indices):
        raw_item_id = api_df.iloc[cand_idx]['id'] if 'id' in api_df.columns else None
        inner_iid = None
        if raw_item_id is not None:
            try:
                inner_iid = svd_trainset.to_inner_iid(str(raw_item_id)) if svd_item_inner_map is not None else None
            except Exception:
                inner_iid = None
        if inner_iid is not None and 0 <= inner_iid < qi.shape[0]:
            scores[pos] = float(global_mean + bi[inner_iid] + np.dot(pseudo_user_vector, qi[inner_iid]))
        else:
            scores[pos] = np.nan

    # Replace NaNs with minimum finite value for stability
    if np.isnan(scores).all():
        return np.zeros_like(scores)
    min_finite = np.nanmin(scores)
    scores = np.where(np.isnan(scores), min_finite, scores)
    return scores

# --- 3. Core Recommendation Functions ---

def get_top_movies(page: int = 1, page_size: int = 10):
    """Returns a paginated list of top movies based on the WR score with metadata."""
    if api_df is None:
        return {"error": "DataFrame not loaded."}
    
    top_movies_df = api_df.sort_values('wr', ascending=False)
    total_count = len(top_movies_df)
    paginated_df = _get_paginated_results(top_movies_df, page, page_size)
    items = _sanitize_movies(paginated_df)
    
    return {
        "items": items,
        "total": int(total_count),
        "page": int(page),
        "page_size": int(page_size),
        "has_more": bool(page * page_size < total_count)
    }


def get_recommendations_from_multiple_titles(titles: list, page: int = 1, page_size: int = 10,
                                             semantic_top_k: int = 200,
                                             alpha_semantic: float = 0.6,
                                             beta_svd: float = 0.3,
                                             gamma_wr: float = 0.1):
    """
    Generates recommendations from multiple titles using a hybrid strategy:
    - Build a taste vector from BGE embeddings (semantic)
    - Retrieve top-K by semantic similarity
    - Re-rank using a weighted combination of normalized scores: semantic, SVD collaborative, WR quality
    Returns items with pagination metadata.
    """
    if api_df is None or embedding_matrix is None or indices_map is None:
        return {"error": "Models are not loaded."}

    valid_indices = []
    for title in titles:
        idx = indices_map.get(title)
        if idx is not None:
            if isinstance(idx, pd.Series):
                valid_indices.append(idx.iloc[0])
            else:
                valid_indices.append(idx)
    
    if not valid_indices:
        return {"error": "None of the provided movies were found."}

    # Create the "Taste Vector" by averaging the vectors of input movies
    taste_vector = np.mean(embedding_matrix[valid_indices], axis=0).reshape(1, -1)
    
    # Calculate cosine similarity of the taste vector against all movies
    cosine_sim_vector = cosine_similarity(taste_vector, embedding_matrix).flatten()

    # Exclude the selected movies themselves
    sim_scores_series = pd.Series(cosine_sim_vector, index=api_df.index).drop(valid_indices, errors='ignore')

    # Take top-K by semantic similarity as candidate pool
    top_k = max(semantic_top_k, page_size * 5)
    candidate_indices = sim_scores_series.sort_values(ascending=False).head(top_k).index.to_numpy()

    # Prepare scores
    semantic_scores = sim_scores_series.loc[candidate_indices].to_numpy()
    wr_scores = api_df.loc[candidate_indices, 'wr'].to_numpy() if 'wr' in api_df.columns else np.zeros(len(candidate_indices))

    # Optional SVD collaborative scores
    svd_scores = _compute_svd_scores_for_candidates(candidate_indices, valid_indices)

    # Normalize each component to [0,1]
    semantic_scores_n = _min_max_normalize(semantic_scores)
    svd_scores_n = _min_max_normalize(svd_scores)
    wr_scores_n = _min_max_normalize(wr_scores)

    # Combine
    combined = alpha_semantic * semantic_scores_n + beta_svd * svd_scores_n + gamma_wr * wr_scores_n

    # Sort candidates by combined score (desc)
    order = np.argsort(-combined)
    ranked_indices = candidate_indices[order]

    final_df = api_df.loc[ranked_indices]
    total_count = len(final_df)
    paginated_df = _get_paginated_results(final_df, page, page_size)
    items = _sanitize_movies(paginated_df)

    return {
        "items": items,
        "total": int(total_count),
        "page": int(page),
        "page_size": int(page_size),
        "has_more": bool(page * page_size < total_count)
    }
