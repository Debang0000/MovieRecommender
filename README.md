# Hybrid Movie Recommender

A lightweight Flask web app showcasing a hybrid movie recommendation engine. It combines semantic similarity (from precomputed text embeddings) with a quality re-ranking using a weighted rating (WR), and exposes a simple UI with autocomplete, multi-select input, and infinite scroll.

## Live Demo

- Deployed on Render: [movierecommender-v9dy.onrender.com](https://movierecommender-v9dy.onrender.com)

## Features

- Hybrid recommendations: semantic similarity + quality-based re-ranking (WR)
- "Taste Vector": average embeddings of multiple selected movies
- Clean SPA frontend: autocomplete, removable tags, hover descriptions, infinite scroll
- Fast responses: all heavy artifacts are precomputed and loaded once at startup

## Tech Stack

- Backend: Python, Flask, Pandas, NumPy, scikit-learn
- Embeddings: Sentence-Transformers (precomputed, loaded from disk); optional SVD collaborative component
- Frontend: HTML, CSS, Vanilla JavaScript
- Data: TMDb metadata, MovieLens (small) for experimentation

## Project Structure

```
MovieRecommender/
├── data/                                # Raw datasets (TMDb, MovieLens small)
├── movie_recommender/
│   ├── core/
│   │   └── engine.py                    # Core recommendation logic
│   └── saved_models/                    # Precomputed artifacts used by the API
│       ├── bge_embeddings.npy           # Embedding matrix (e.g., 768-dim vectors)
│       ├── indices_map.joblib           # Map: movie title -> row index in the DF/embeddings
│       └── movies_df.joblib             # Main DataFrame with metadata and WR
├── notebooks/
│   ├── exploration.ipynb                # Exploration and data cleaning
│   └── model_generator_v2.ipynb         # Builds artifacts in saved_models/
├── web/
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/script.js
│   └── templates/index.html
├── run.py                               # Flask app entrypoint
└── requirements.txt                     # Python dependencies
```

## How It Works

1. Offline
   - Data is cleaned and consolidated into a single narrative text per movie.
   - Embeddings are computed (e.g., using BAAI/bge-base-en-v1.5) and saved to `bge_embeddings.npy`.
   - A weighted rating (WR) is computed (IMDb-style) and stored in `movies_df.joblib`.
   - A title-to-index lookup `indices_map.joblib` is created.

2. Online (at request time)
   - For multi-title recommendations, vectors for the selected titles are averaged into a single "taste vector".
   - Cosine similarity ranks candidates; a candidate pool is re-ranked by a weighted combination of normalized scores: semantic, optional SVD collaborative, and WR.
   - A paginated, sanitized JSON list is returned to the frontend. Missing posters fall back to a placeholder URL.

## API

Base URL: `http://127.0.0.1:5000`

- GET `/healthz`
  - Liveness/readiness probe. Returns `ok` (200) when the service is healthy.

- GET `/debug/status`
  - Diagnostics for the hybrid engine. Returns JSON with keys such as `has_svd`, `svd_qi_shape`, `svd_mapping_coverage`, `svd_df_overlap_coverage`, `embedding_shape`, etc.
  - Useful for verifying that the collaborative (SVD) component is active and aligned with the dataset.

- GET `/top-movies?page=<int>&page_size=<int>`
  - Returns a paginated list of globally top movies by WR.
  - Response: `{ "recommendations": [ { ...movie fields... } ], "page", "page_size", "total", "has_more" }`

- POST `/recommend?page=<int>&page_size=<int>`
  - Request body: `{ "titles": ["Inception", "Interstellar"] }`
  - Returns recommendations based on a hybrid of semantic similarity (BGE), optional SVD collaborative score, and WR quality re-ranking.
  - Response: `{ "recommendations": [ { ...movie fields... } ], "page", "page_size", "total", "has_more" }`

  - Optional query params for tuning and diagnostics (defaults tuned for balanced hybrid):
    - `alpha` (float, default 0.6): weight for semantic component
    - `beta` (float, default 0.3): weight for SVD collaborative component (you may increase to 0.45–0.6 if coverage is healthy)
    - `gamma` (float, default 0.1): weight for WR quality component
    - `semantic_top_k` (int, default 200): size of semantic candidate pool
    - `enable_mmr` (bool, default true): whether to apply MMR re-ranking for diversity
    - `lambda_diversity` (float in [0,1], default 0.7): MMR tradeoff parameter
    - `debug` (bool, default false): if true, includes a `debug` block with current hybrid status

  - Example:
    ```bash
    curl -X POST "http://127.0.0.1:5000/recommend?page=1&page_size=10&alpha=0.45&beta=0.45&gamma=0.1&semantic_top_k=300&enable_mmr=true&lambda_diversity=0.7&debug=true" \
      -H "Content-Type: application/json" \
      -d '{"titles":["Inception","Interstellar"]}'
    ```

- GET `/search?q=<str>`
  - Autocomplete for titles (min length: 2 chars).
  - Response: `["Inception", "Inside Out", ...]`

Notes
- Returned movie records are derived from `movies_df.joblib`, sanitized for JSON, and include a computed `poster_url` (from TMDb `poster_path` if present) and `overview` when available.
- Pagination defaults: `page=1`, `page_size=10`.

## Configuration

Use environment variables to configure runtime:
- `APP_HOST` (default: `0.0.0.0`)
- `APP_PORT` (default: `5000`)
- `APP_DEBUG` (default: `true`)

Create a `.env` or export variables in your shell before running (Windows PowerShell example):
```powershell
$env:APP_HOST="0.0.0.0"; $env:APP_PORT="5000"; $env:APP_DEBUG="true"
```

## Running Locally

1. Python 3.9+ recommended
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Optional: enable SVD hybrid by installing scikit-surprise
   # pip install scikit-surprise
   ```
3. Ensure artifacts exist:
   - `movie_recommender/saved_models/movies_df.joblib`
   - `movie_recommender/saved_models/indices_map.joblib`
   - `movie_recommender/saved_models/bge_embeddings.npy`

   If you need to regenerate them, see `notebooks/model_generator_v2.ipynb` for the pipeline used to build these artifacts.

4. Start the app:
   ```bash
   python run.py
   ```
   Visit `http://127.0.0.1:5000`.

## Datasets

- MovieLens (ml-latest-small): place extracted CSVs under `data/ml-latest-small/`.
  - Download: `https://grouplens.org/datasets/movielens/` (choose "ml-latest-small")
  - Expected files: `ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`

- TMDb metadata (for `TMDB_all_movies.csv`):
  - Source: Kaggle dataset [TMDB Movies Dataset 2023 (930k movies)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
  - Path: `data/TMDB_all_movies.csv` (large file; excluded by `.gitignore`)
  - Note: During EDA, a large number of extremely niche or low-quality entries were filtered out.

## Docker

A minimal Dockerfile is provided to run the app with precomputed artifacts.

Build (from project root):
```bash
docker build -t movie-recommender .
```
Run:
```bash
docker run --rm -p 5000:5000 \
  -e APP_HOST=0.0.0.0 -e APP_PORT=5000 -e APP_DEBUG=false \
  -v %cd%/movie_recommender/saved_models:/app/movie_recommender/saved_models:ro \
  movie-recommender
```

## Developer Utilities

- Inspect movie data by title:
  ```bash
  python inspect_movie.py "Inception"
  ```

## Limitations

- Requires precomputed artifacts; embedding computation is not done at runtime.
- Title matching uses a simple lookup; ambiguous or duplicate titles are resolved to the first match.
- Datasets may have missing/partial metadata (e.g., `poster_path`, `overview`).

## Roadmap / Ideas for Improvement

- Add a robust search index (e.g., fuzzy matching) and support for IDs (TMDb/IMDb) to disambiguate titles.
- Paginated responses with total counts and cursors; page_size parameter (DONE)
- Introduce caching (e.g., LRU) for repeated queries and autocomplete.
- Improve error handling and validation (e.g., Pydantic models for request/response schemas).
- Prune unused dependencies; optionally provide a FastAPI alternative.
- CI workflow + minimal tests; better provenance/versioning for `saved_models/`.
- Accessibility, mobile refinements, and skeleton UI while loading. 