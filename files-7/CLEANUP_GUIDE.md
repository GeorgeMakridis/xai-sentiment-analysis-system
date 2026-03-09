# Codebase Cleanup Guide

## Summary

| Component | Before | After | Removed |
|-----------|--------|-------|---------|
| ai_outputs/app.py | 2,321 lines | 1,927 lines | 394 lines |
| dashboard/app.py | 523 lines | 312 lines | 211 lines |
| xai_service/app.py | 5,993 lines | 1,710 lines | 4,283 lines |
| ai_outputs/requirements.txt | 15 deps | 7 deps | 8 deps |
| xai_service/requirements.txt | 27 deps | 19 deps | 8 deps |
| **Dead files deleted** | — | — | **22 files** |
| **Total lines removed** | — | — | **~6,100+** |

## What Was Removed and Why

### Methodology

Traced every call from `index.html` JS → dashboard routes → xai_service endpoints
→ ai_outputs endpoints. Any code not in this chain is dead.

**Active data flow (the only flow the UI actually uses):**

```
index.html JS
  ├── fetch('/api/upload-data')        → dashboard → xai_service /ingest
  ├── fetch('/api/datasets')           → dashboard → ai_outputs /api/user-datasets
  ├── fetch('/api/select-dataset')     → dashboard → xai_service /ingest
  ├── fetch('/api/data-statistics')    → dashboard → xai_service /data-statistics
  │                                       └── dashboard → ai_outputs /api/store-plot-image (per plot)
  │                                       └── dashboard → ai_outputs /store-results (bulk)
  ├── fetch('/api/chat')               → dashboard → ai_outputs /chat (RAG)
  └── <img src="/api/plot-image/...">  → dashboard → ai_outputs /api/plots/{id}/image → RustFS
```

Everything outside this flow was removed.

### Dead Endpoints Removed

**ai_outputs/app.py** (8 endpoints removed):
- `POST /store-interactive-plot` — never called from dashboard
- `GET /api/plots` — list plots, never called
- `GET /api/plots/<id>` — get metadata, never called
- `GET /api/plots/<id>/html` — get HTML plot, never called
- `DELETE /api/plots/<id>` — delete plot, never called
- `GET /api/datasets/<user_id>` — duplicate of /api/user-datasets
- `POST /evaluate-faithfulness` — only used by test scripts
- `DELETE /clear-user-data/<user_id>` — never called

**dashboard/app.py** (8 proxy routes removed):
- `POST /api/upload-model` — no JS button
- `GET /api/get-results` — no JS call
- `POST /api/create-sp100-model` — no JS button
- `POST /api/train-model` — no JS button
- `POST /api/preprocess-data` — no JS button
- `POST /api/enhanced-xai` — no JS button
- `POST /api/download-finbert` — no JS button
- `POST /api/download-mnist` — no JS button

**xai_service/app.py** (12 endpoints + 15 helper functions removed):
- `POST /analyze` — dashboard proxy removed
- `POST /create-model` — dashboard proxy removed
- `POST /train-model` — dashboard proxy removed
- `POST /preprocess-data` — dashboard proxy removed
- `POST /direct-analyze` — never called
- `POST /enhanced-xai` — dashboard proxy removed
- `POST /download-finbert` — dashboard proxy removed
- `POST /get-examples` — never called
- `POST /run-xai` — never called (+ run_text_xai, run_image_xai helpers)
- `POST /download-mnist` — dashboard proxy removed
- `POST /generate-interactive-plot` — never called
- `POST /get-available-plots` — never called

Plus all helper functions only called by dead endpoints:
`validate_model_compatibility`, `load_model_with_metadata`, `load_news_sentiment_data`,
`load_model`, `create_sample_data`, `create_sample_model`,
`generate_comprehensive_xai_visualizations`, `create_finbert_sentiment_model`,
`generate_xai_visualizations_for_model`, `generate_timeseries_visualizations`,
`generate_text_visualizations`, `generate_enhanced_xai_visualizations`,
`generate_correlation_heatmap`, `generate_missing_values_heatmap`,
`generate_word_importance_heatmap`, `generate_topic_keywords_barchart`,
`generate_hierarchical_dendrogram`, `generate_intertopic_distance_map`,
`generate_title_based_sentiment_visualizations`, `generate_word_based_analysis`,
`generate_asset_specific_analysis`, `generate_enhanced_sentiment_visualizations`

### Dead Files to Delete

```
# ai_outputs modules (no longer imported)
ai_outputs/faithfulness_evaluator.py
ai_outputs/test_set_generator.py
ai_outputs/vector_store.py

# xai_service modules (no longer imported)
xai_service/ai_adapter.py
xai_service/download_finbert.py
xai_service/download_mnist.py
xai_service/download_model.py

# All faithfulness test scripts
scripts/evaluate_rag_faithfulness.py
scripts/run_quick_faithfulness_test.py
scripts/test_api_usage.py
scripts/test_faithfulness_direct.py
scripts/test_faithfulness_integration.py
scripts/test_faithfulness_with_mock_data.py
scripts/create_test_xai_data.py

# All test files (test removed/dead code)
tests/test_faithfulness_evaluator.py
tests/test_image_data_plots.py
tests/test_interactive_plots.py
tests/test_sentiment_visualizations.py

# Dead documentation
FAITHFULNESS_EVALUATION_FULL_REPORT.md
FAITHFULNESS_EVALUATION_RESULTS.md
API_AND_PROMPTS_STATUS.md
```

### Dead Dependencies Removed

**ai_outputs/requirements.txt** — removed:
- `seaborn` (no plots generated here)
- `pandas` (no DataFrame processing)
- `scikit-learn` (no ML models)
- `matplotlib` (no plots generated here)
- `Pillow` (images handled as raw bytes)
- `sentence-transformers` (embeddings via OpenAI, not local)
- `faiss-cpu` (using SimpleVectorDB, not FAISS)

**xai_service/requirements.txt** — removed:
- `seaborn` (all plots use plotly/matplotlib directly)
- `joblib` (model serialization removed with /analyze)
- `openpyxl` (Excel parsing removed)
- `pyarrow` (parquet support removed)
- `wordcloud` (wordcloud plots removed)
- `kneed` (elbow detection removed)
- `bertviz` (attention viz via custom code now)
- `accelerate` (not used in active code)

## Step-by-Step Merge Instructions

### 1. Replace cleaned files

```bash
cd /path/to/xai-sentiment-analysis-system

# Replace the 3 main app files
cp /path/to/xai-cleanup/ai_outputs/app.py       ai_outputs/app.py
cp /path/to/xai-cleanup/dashboard/app.py         dashboard/app.py
cp /path/to/xai-cleanup/xai_service/app.py       xai_service/app.py

# Replace requirements
cp /path/to/xai-cleanup/ai_outputs/requirements.txt   ai_outputs/requirements.txt
cp /path/to/xai-cleanup/xai_service/requirements.txt  xai_service/requirements.txt
```

### 2. Delete dead files

```bash
# Dead ai_outputs modules
git rm ai_outputs/faithfulness_evaluator.py
git rm ai_outputs/test_set_generator.py
git rm ai_outputs/vector_store.py

# Dead xai_service modules
git rm xai_service/ai_adapter.py
git rm xai_service/download_finbert.py
git rm xai_service/download_mnist.py
git rm xai_service/download_model.py

# Dead scripts
git rm scripts/evaluate_rag_faithfulness.py
git rm scripts/run_quick_faithfulness_test.py
git rm scripts/test_api_usage.py
git rm scripts/test_faithfulness_direct.py
git rm scripts/test_faithfulness_integration.py
git rm scripts/test_faithfulness_with_mock_data.py
git rm scripts/create_test_xai_data.py

# Dead tests
git rm tests/test_faithfulness_evaluator.py
git rm tests/test_image_data_plots.py
git rm tests/test_interactive_plots.py
git rm tests/test_sentiment_visualizations.py

# Dead docs
git rm FAITHFULNESS_EVALUATION_FULL_REPORT.md
git rm FAITHFULNESS_EVALUATION_RESULTS.md
git rm API_AND_PROMPTS_STATUS.md
```

### 3. Commit and deploy

```bash
git add -A
git commit -m "Cleanup: remove 6100+ dead lines, 22 dead files, 8 unused deps

- ai_outputs/app.py: 2321→1927 lines (removed 8 dead endpoints)
- dashboard/app.py: 523→312 lines (removed 8 unused proxy routes)
- xai_service/app.py: 5993→1710 lines (removed 12 dead endpoints + 15 helper fns)
- Deleted 22 dead files (faithfulness eval, dead scripts/tests/docs)
- Trimmed requirements.txt (removed 16 unused Python dependencies)

Active flow preserved: upload → ingest → data-statistics → RustFS → RAG chat"

git push
```

### 4. Rebuild Docker images

```bash
docker compose build --no-cache
docker compose up -d
```

## What's Left (Active Architecture)

```
auth/
├── __init__.py                     # Shared auth package
└── keycloak.py                     # Keycloak OIDC + legacy fallback

dashboard/
├── Dockerfile
├── requirements.txt
├── app.py                          # 312 lines — 8 routes
└── templates/
    ├── index.html                  # Main SPA
    └── login.html                  # Auth page

xai_service/
├── Dockerfile
├── requirements.txt
├── app.py                          # 1710 lines — 3 routes (/health, /ingest, /data-statistics)
└── plot_generators/                # Modular plot generation (sentiment, image)
    ├── __init__.py
    ├── base_plot_generator.py
    ├── registry.py
    ├── sentiment_plot_generator.py
    └── image_plot_generator.py

ai_outputs/
├── Dockerfile
├── requirements.txt
├── app.py                          # 1927 lines — 10 routes
├── plot_metadata_schema.py         # Metadata builder + vector text helpers
├── storage_abstraction.py          # Local/RustFS storage layer
└── restfs_client.py                # S3-compatible client for remote RustFS

docker-compose.yml
.env.example
scripts/
├── __init__.py
└── upload_data.py                  # Utility script
docs/                               # Architecture documentation
```

### Active Endpoints Summary

| Service | Endpoint | Purpose |
|---------|----------|---------|
| dashboard | `GET /` | Serve SPA |
| dashboard | `POST /api/upload-data` | Upload CSV/JSON → ingest |
| dashboard | `GET /api/datasets` | List datasets from RustFS |
| dashboard | `POST /api/select-dataset` | Select existing dataset → ingest |
| dashboard | `POST /api/data-statistics` | Generate plots → store to RustFS |
| dashboard | `GET /api/plot-image/<id>` | Proxy plot PNG from RustFS |
| dashboard | `POST /api/chat` | RAG chatbot |
| dashboard | `GET /health` | Health check |
| xai_service | `GET /health` | Health check |
| xai_service | `POST /ingest` | Parse + store data in memory |
| xai_service | `POST /data-statistics` | Generate XAI plots |
| ai_outputs | `GET /health` | Health check |
| ai_outputs | `POST /api/store-plot-image` | Save PNG + metadata to RustFS + Vector DB |
| ai_outputs | `GET /api/plots/<id>/image` | Serve PNG from RustFS |
| ai_outputs | `GET /api/user-datasets/<id>` | List user datasets in RustFS |
| ai_outputs | `POST /store-data` | Store data summary in Vector DB |
| ai_outputs | `POST /store-results` | Store analysis results in Vector DB |
| ai_outputs | `POST /store-attention-insights` | Store attention data |
| ai_outputs | `GET /results/<user_id>` | Get stored results |
| ai_outputs | `POST /chat` | RAG response (auto-rehydrates from RustFS) |
| ai_outputs | `POST /api/rehydrate/<id>` | Manual Vector DB reload from RustFS |
