# F:/MovieRecommender/run.py (Final Version with Multi-Mode Endpoints)

import os
import sys
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

# --- 1. SETUP & ONE-TIME LOADING ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import the new, powerful recommendation functions
from movie_recommender.core.engine import get_recommendations_from_multiple_titles, get_top_movies, get_hybrid_status

# Load data for search/autocomplete
MODEL_DIR = Path(PROJECT_ROOT) / "movie_recommender" / "saved_models"
DF_PATH = MODEL_DIR / 'movies_df.joblib'
try:
    movies_df_for_search = joblib.load(DF_PATH)
    all_movie_titles = movies_df_for_search['title'].tolist()
    # Free heavy DataFrame reference as we only need titles for autocomplete
    del movies_df_for_search
    print("--- Movie DataFrame for search/autocomplete loaded successfully. ---")
except Exception as e:
    print(f"--- Error loading movies_df for search: {e} ---")
    movies_df_for_search, all_movie_titles = pd.DataFrame(), []

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

# --- 2. DEFINE ALL API ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top-movies', methods=['GET'])
def top_movies_endpoint():
    """Returns a list of top-rated movies with pagination metadata."""
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 10, type=int)
    result = get_top_movies(page=page, page_size=page_size)
    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 500
    response_payload = {
        'recommendations': result.get('items', []),
        'page': result.get('page', page),
        'page_size': result.get('page_size', page_size),
        'total': result.get('total', len(result.get('items', []))),
        'has_more': result.get('has_more', False)
    }
    return jsonify(response_payload)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles both single and multi-title recommendations with pagination metadata."""
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 10, type=int)

    def _get_float_arg(name: str, default: float) -> float:
        try:
            v = request.args.get(name, None)
            return float(v) if v is not None else default
        except Exception:
            return default

    def _get_int_arg(name: str, default: int) -> int:
        try:
            v = request.args.get(name, None)
            return int(v) if v is not None else default
        except Exception:
            return default

    def _get_bool_arg(name: str, default: bool) -> bool:
        v = request.args.get(name, None)
        if v is None:
            return default
        return str(v).lower() in ("1", "true", "yes", "y", "on")

    # Optional tuning params
    alpha = _get_float_arg('alpha', 0.6)
    beta = _get_float_arg('beta', 0.3)
    gamma = _get_float_arg('gamma', 0.1)
    semantic_top_k = _get_int_arg('semantic_top_k', 200)
    if semantic_top_k <= 0:
        semantic_top_k = 200
    enable_mmr = _get_bool_arg('enable_mmr', True)
    lambda_diversity = _get_float_arg('lambda_diversity', 0.7)
    lambda_diversity = min(max(lambda_diversity, 0.0), 1.0)
    debug_flag = _get_bool_arg('debug', False)
    data = request.get_json()
    movie_titles = data.get('titles', []) if isinstance(data, dict) else []
    
    if not movie_titles or not isinstance(movie_titles, list):
        return jsonify({'error': 'Please provide a list of movie titles.'}), 400
    
    result = get_recommendations_from_multiple_titles(
        movie_titles,
        page=page,
        page_size=page_size,
        semantic_top_k=semantic_top_k,
        alpha_semantic=alpha,
        beta_svd=beta,
        gamma_wr=gamma,
        enable_mmr=enable_mmr,
        lambda_diversity=lambda_diversity,
    )
    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 404
    response_payload = {
        'recommendations': result.get('items', []),
        'page': result.get('page', page),
        'page_size': result.get('page_size', page_size),
        'total': result.get('total', len(result.get('items', []))),
        'has_more': result.get('has_more', False)
    }
    if debug_flag:
        response_payload['debug'] = {
            'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
            'semantic_top_k': semantic_top_k,
            'enable_mmr': enable_mmr,
            'lambda_diversity': lambda_diversity,
            'hybrid_status': get_hybrid_status(),
        }
    return jsonify(response_payload)

@app.route('/search', methods=['GET'])
def search():
    """Handles autocomplete requests efficiently."""
    query = request.args.get('q', '').lower()
    if len(query) < 2 or not all_movie_titles:
        return jsonify([])
    matches = [title for title in all_movie_titles if query in title.lower()]
    return jsonify(matches[:10])

# --- 2.1. HEALTH CHECK ENDPOINTS ---
@app.route('/healthz', methods=['GET'])
@app.route('/health', methods=['GET'])
def healthz():
    """Lightweight readiness/liveness probe."""
    try:
        test = get_top_movies(page=1, page_size=1)
        if isinstance(test, dict) and 'error' in test:
            return "not ready", 500
        return "ok", 200
    except Exception:
        return "not ready", 500

@app.route('/debug/status', methods=['GET'])
def debug_status():
    """Return diagnostic info about hybrid pipeline status."""
    try:
        status = get_hybrid_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': str(e),
        }), 500

# --- 3. RUN THE APP ---
if __name__ == '__main__':
    print("Starting Final Multi-Mode Movie Recommender Web App...")
    host = os.environ.get('APP_HOST', '0.0.0.0')
    # Prefer platform-provided PORT (Render/Railway/Koyeb/etc.), fallback to APP_PORT or 5000
    port_str = os.environ.get('PORT') or os.environ.get('APP_PORT', '5000')
    port = int(port_str)
    debug = os.environ.get('APP_DEBUG', 'true').lower() in ('1', 'true', 'yes')
    app.run(debug=debug, host=host, port=port)
