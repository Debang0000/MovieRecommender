# movie_recommender/core/engine.py (FINAL VERSION: Multi-Mode Logic + Optional SVD Hybrid)

import joblib
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load All Artifacts at Startup ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "saved_models"
    
    api_df = joblib.load(MODEL_DIR / 'movies_df.joblib')
    # Memory-map embeddings to reduce RAM usage in constrained environments
    embedding_matrix = np.load(MODEL_DIR / 'bge_embeddings.npy', mmap_mode='r')
    indices_map = joblib.load(MODEL_DIR / 'indices_map.joblib')
    
    # Keep only essential columns to reduce memory footprint if available
    # Include multiple potential ID columns to preserve SVD mapping
    essential_columns = [
        'movieId', 'ml_movie_id', 'id', 'tmdb_id', 'imdb_id',
        'title', 'wr', 'poster_path', 'overview'
    ]
    try:
        available = [c for c in essential_columns if c in api_df.columns]
        if available:
            api_df = api_df[available]
    except Exception:
        pass

    # If MovieLens movieId is missing but TMDB id is present, attempt to enrich via links.csv
    try:
        if 'movieId' not in api_df.columns and 'id' in api_df.columns:
            # Allow override via env var; default to project data path
            project_root = Path(__file__).resolve().parents[2]
            default_links = project_root / 'data' / 'ml-latest-small' / 'links.csv'
            links_path_env = os.environ.get('LINKS_CSV_PATH')
            links_path = Path(links_path_env) if links_path_env else default_links
            if links_path.exists():
                links_df = pd.read_csv(links_path)
                # Normalize column names if needed
                cols_lower = {c.lower(): c for c in links_df.columns}
                tmdb_col = cols_lower.get('tmdbid') or cols_lower.get('tmdb_id') or 'tmdbId'
                ml_col = cols_lower.get('movieid') or cols_lower.get('movie_id') or 'movieId'
                if tmdb_col in links_df.columns and ml_col in links_df.columns:
                    # Cast to numeric, drop NaNs, deduplicate by TMDB id
                    links_small = links_df[[ml_col, tmdb_col]].copy()
                    links_small[tmdb_col] = pd.to_numeric(links_small[tmdb_col], errors='coerce')
                    links_small = links_small.dropna(subset=[tmdb_col])
                    links_small[tmdb_col] = links_small[tmdb_col].astype('int64')
                    # Keep the first mapping per TMDB id to avoid 1:N duplications
                    links_small = links_small.drop_duplicates(subset=[tmdb_col], keep='first')
                    id_to_ml = links_small.set_index(tmdb_col)[ml_col]
                    # Ensure api_df id is numeric-like where possible
                    if api_df['id'].dtype != 'int64' and api_df['id'].dtype != 'Int64':
                        id_numeric = pd.to_numeric(api_df['id'], errors='coerce').astype('Int64')
                    else:
                        id_numeric = api_df['id']
                    api_df['movieId'] = id_numeric.map(id_to_ml)
                    # Keep essential columns again (now with movieId if present)
                    available = [c for c in essential_columns if c in api_df.columns]
                    if available:
                        api_df = api_df[available]
                    print('--- Enriched api_df with MovieLens movieId via links.csv (no row duplication) ---')
                else:
                    print('--- links.csv missing required columns; skipping enrichment ---')
            else:
                print(f"--- links.csv not found at {links_path}; skipping enrichment ---")
    except Exception as e:
        print(f"--- Failed to enrich api_df with links.csv: {e} ---")

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
svd_raw_to_inner_canon_map = {}
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
            # Build canonicalized mapping to be robust to string/float/int representation
            try:
                for inner in range(svd_trainset.n_items):
                    try:
                        raw = svd_trainset.to_raw_iid(inner)
                        # Use the local canonizer defined below (declare placeholder and replace later)
                        pass
                    except Exception:
                        continue
                print(f"--- Built canonical SVD raw->inner map of size {len(svd_raw_to_inner_canon_map)} ---")
            except Exception as e:
                print(f"--- WARNING: Failed building canonical SVD map: {e} ---")
        print("--- Optional SVD model loaded for hybrid scoring. ---")
    else:
        print("--- SVD model not found; running without collaborative component. ---")
except Exception as e:
    print(f"--- WARNING: Failed to load SVD model: {e}. Proceeding without collaborative component. ---")
    svd_model = None
    svd_trainset = None
    svd_item_inner_map = None
    svd_raw_to_inner_canon_map = {}

# --- 2. Helper Functions ---

# Candidate columns that may hold the raw item id used during SVD training
POSSIBLE_ITEM_ID_COLUMNS = ['movieId', 'ml_movie_id', 'id', 'tmdb_id', 'imdb_id']

def _to_candidate_raw_ids(value) -> list[str]:
    """Generate common string variants for raw item id to maximize mapping hits.

    Handles cases where ids were saved as floats (e.g., '1234.0') vs ints ('1234').
    """
    variants: list[str] = []
    try:
        s = str(value)
        variants.append(s)
        # If numeric-like, try integer and float-fixed variants
        if isinstance(value, (int, np.integer)):
            variants.append(str(int(value)))
            variants.append(f"{float(value):.1f}")  # e.g., '1234.0'
        else:
            # Try to coerce to number
            vnum = float(value)
            if np.isfinite(vnum):
                variants.append(str(int(vnum)))
                variants.append(f"{vnum:.1f}")
    except Exception:
        pass
    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result

def _canonize_raw_id(value) -> str:
    """Canonical string for id comparisons: prefer integer string, strip trailing .0, trim spaces."""
    try:
        v = float(value)
        if np.isfinite(v):
            iv = int(round(v))
            if abs(v - iv) < 1e-9:
                return str(iv)
            return str(v)
    except Exception:
        pass
    s = str(value).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

def _get_svd_inner_iid_for_row(row) -> int:
    """Best-effort mapping from a DataFrame row to Surprise inner_iid.

    Tries multiple candidate ID columns in order. Returns None if no mapping.
    """
    if svd_trainset is None:
        return None
    for col in POSSIBLE_ITEM_ID_COLUMNS:
        if col in row and pd.notna(row[col]):
            # Prefer canonicalized lookup
            canon = _canonize_raw_id(row[col])
            inner = svd_raw_to_inner_canon_map.get(canon)
            if inner is not None:
                return inner
            # Fallback: try Surprise API with a few raw variants
            for raw in _to_candidate_raw_ids(row[col]):
                try:
                    inner_iid = svd_trainset.to_inner_iid(raw)
                    return inner_iid
                except Exception:
                    continue
    return None

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
        row = api_df.iloc[idx]
        inner_iid = _get_svd_inner_iid_for_row(row)
        if inner_iid is not None and 0 <= inner_iid < qi.shape[0]:
            liked_item_vectors.append(qi[inner_iid])
    if len(liked_item_vectors) == 0:
        # No overlap with SVD item space; return zeros
        return np.zeros(len(candidate_indices), dtype=float)

    pseudo_user_vector = np.mean(np.stack(liked_item_vectors, axis=0), axis=0)

    # Score each candidate: approx prediction = global_mean + bi[i] + dot(pu*, qi[i])
    scores = np.zeros(len(candidate_indices), dtype=float)
    for pos, cand_idx in enumerate(candidate_indices):
        row = api_df.iloc[cand_idx]
        inner_iid = _get_svd_inner_iid_for_row(row)
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

def _mmr_rerank(candidate_indices: np.ndarray,
                relevance_scores: np.ndarray,
                lambda_diversity: float = 0.7,
                select_k: int | None = None) -> np.ndarray:
    """Applies Maximal Marginal Relevance (MMR) to promote diversity.

    Greedy selection using embedding similarity as redundancy measure.
    """
    if select_k is None:
        select_k = len(candidate_indices)
    if len(candidate_indices) == 0:
        return candidate_indices

    # Prepare candidate vectors (L2 normalized) for cosine similarity
    cand_vecs = embedding_matrix[candidate_indices]
    # Normalize to unit vectors to use dot product as cosine similarity
    eps = 1e-12
    norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    cand_unit = cand_vecs / norms

    selected: list[int] = []
    selected_mask = np.zeros(len(candidate_indices), dtype=bool)

    # Precompute relevance scaled by lambda
    rel_scaled = lambda_diversity * relevance_scores
    div_scale = (1.0 - lambda_diversity)

    for _ in range(min(select_k, len(candidate_indices))):
        best_idx = -1
        best_score = -1e18
        for i in range(len(candidate_indices)):
            if selected_mask[i]:
                continue
            # Diversity penalty: max similarity to any already selected item
            if not selected:
                div_penalty = 0.0
            else:
                # cand_unit[selected] shape: (k, d); cand_unit[i] shape: (d,)
                sims = np.dot(cand_unit[selected], cand_unit[i])
                div_penalty = float(np.max(sims))
            mmr_score = float(rel_scaled[i] - div_scale * div_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_mask[best_idx] = True

    return candidate_indices[np.array(selected, dtype=int)]

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
                                             gamma_wr: float = 0.1,
                                             enable_mmr: bool = True,
                                             lambda_diversity: float = 0.7):
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

    # Optional diversity-promoting re-ranking (MMR)
    if enable_mmr:
        ranked_indices = _mmr_rerank(
            ranked_indices,
            combined[order],
            lambda_diversity=lambda_diversity,
            select_k=len(ranked_indices)
        )

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


def get_hybrid_status(sample_size: int = 200,
                      alpha_semantic: float = 0.6,
                      beta_svd: float = 0.3,
                      gamma_wr: float = 0.1) -> dict:
    """Returns a diagnostic snapshot of the hybrid recommender state."""
    status = {
        "has_dataframe": bool(api_df is not None),
        "dataframe_shape": None,
        "has_embeddings": bool(embedding_matrix is not None),
        "embedding_shape": None,
        "embedding_memmap": bool(isinstance(embedding_matrix, np.memmap)) if embedding_matrix is not None else False,
        "has_indices_map": bool(indices_map is not None),
        "has_svd": bool(svd_model is not None and svd_trainset is not None and svd_item_inner_map is not None),
        "svd_qi_shape": None,
        "svd_mapping_coverage": None,
        "weights": {
            "alpha_semantic": float(alpha_semantic),
            "beta_svd": float(beta_svd),
            "gamma_wr": float(gamma_wr),
        },
        "has_wr_column": False,
    }

    try:
        if api_df is not None:
            status["dataframe_shape"] = [int(api_df.shape[0]), int(api_df.shape[1])]
            status["has_wr_column"] = "wr" in api_df.columns
    except Exception:
        pass

    try:
        if embedding_matrix is not None:
            status["embedding_shape"] = [int(d) for d in list(embedding_matrix.shape)]
            status["embedding_memmap"] = bool(isinstance(embedding_matrix, np.memmap))
    except Exception:
        pass

    try:
        if svd_model is not None:
            qi = getattr(svd_model, 'qi', None)
            if qi is not None:
                status["svd_qi_shape"] = [int(d) for d in list(qi.shape)]
    except Exception:
        pass

    try:
        if status["has_svd"] and api_df is not None and len(api_df) > 0:
            # Random sample to avoid head-of-frame bias
            rng = np.random.default_rng(42)
            check_n = min(sample_size, len(api_df))
            sample_idx = rng.choice(len(api_df), size=check_n, replace=False)
            sample_df = api_df.iloc[sample_idx]
            qi = getattr(svd_model, 'qi', None)
            ok = 0
            total = 0
            if qi is not None:
                for _, row in sample_df.iterrows():
                    inner_iid = _get_svd_inner_iid_for_row(row)
                    total += 1
                    if inner_iid is not None and 0 <= inner_iid < qi.shape[0]:
                        ok += 1
            status["svd_mapping_coverage"] = float(ok) / float(total) if total > 0 else 0.0
            # Also compute set-based overlap coverage for robustness
            try:
                # DF ids (canonicalized)
                if 'id' in api_df.columns:
                    df_ids = set(_canonize_raw_id(x) for x in api_df['id'].dropna().tolist())
                else:
                    df_ids = set()
                # Build SVD canonical map if empty
                if not svd_raw_to_inner_canon_map and svd_trainset is not None:
                    for inner in range(svd_trainset.n_items):
                        try:
                            raw = svd_trainset.to_raw_iid(inner)
                            svd_raw_to_inner_canon_map[_canonize_raw_id(raw)] = inner
                        except Exception:
                            continue
                raw_ids_svd = set(svd_raw_to_inner_canon_map.keys())
                inter = df_ids.intersection(raw_ids_svd)
                status["svd_df_overlap_coverage"] = float(len(inter)) / float(len(df_ids)) if df_ids else 0.0
                status["svd_items_covered_in_df"] = int(len(inter))
                status["svd_items_total"] = int(len(raw_ids_svd))
            except Exception:
                pass
    except Exception:
        pass

    return status
