# Merge Guide — Revisions 1–4

## Files Provided

```
NEW FILES (add to repo):
├── auth/
│   ├── __init__.py                  # shared auth package
│   └── keycloak.py                  # Keycloak OIDC + legacy fallback
├── xai_service/
│   └── ai_adapter.py               # AI service adapter contract
├── ai_outputs/
│   └── vector_store.py             # consolidated vector DB indexing

UPDATED FILES (replace in repo):
├── docker-compose.yml               # RustFS + Keycloak + env cleanup
├── .env.example                     # all new env vars documented
├── dashboard/
│   ├── app.py                       # Keycloak auth, RBAC, cleanup
│   ├── requirements.txt             # + authlib, python-jose
│   └── templates/login.html         # removed hardcoded password
└── ai_outputs/
    ├── plot_metadata_schema.py      # enhanced with summary_for_rag
    └── requirements.txt             # + python-jose
```

---

## Step-by-step Merge Instructions

### 1. Copy new files directly

These are brand new — just copy them in:

```bash
# Auth module (shared across all services)
cp -r auth/ /path/to/repo/auth/

# AI adapter
cp xai_service/ai_adapter.py /path/to/repo/xai_service/

# Vector store consolidation
cp ai_outputs/vector_store.py /path/to/repo/ai_outputs/
```

### 2. Replace complete files

These files are full rewrites — replace them entirely:

```bash
cp docker-compose.yml /path/to/repo/
cp .env.example /path/to/repo/
cp dashboard/app.py /path/to/repo/dashboard/
cp dashboard/requirements.txt /path/to/repo/dashboard/
cp dashboard/templates/login.html /path/to/repo/dashboard/templates/
cp ai_outputs/plot_metadata_schema.py /path/to/repo/ai_outputs/
cp ai_outputs/requirements.txt /path/to/repo/ai_outputs/
```

### 3. Delete dead files

```bash
rm /path/to/repo/ai_outputs/app.py.tmp
```

### 4. Manual patches to ai_outputs/app.py

The `ai_outputs/app.py` (2147 lines) is too large to fully rewrite in one
pass, but these targeted patches consolidate the store_plot_summary calls.

#### 4a. Add imports at the top (after existing imports, ~line 18)

```python
from vector_store import index_plot, index_plots_batch, rehydrate_from_restfs
```

#### 4b. Replace the `store_plot_summary` function (~line 332-368)

Delete the entire nested function `def store_plot_summary(...)` and replace
every call to it.  The old pattern was:

```python
# OLD (repeated 13 times with minor variations)
store_plot_summary(summary, source='results_plot_summaries')
```

Replace each call with:

```python
# NEW (unified)
index_plot(vector_db, user_id, summary, source='results_plot_summaries')
```

Search-and-replace targets (line numbers from the current file):

| Line | Old call | New call |
|------|----------|----------|
| 372 | `store_plot_summary(summary, source='results_plot_summaries')` | `index_plot(vector_db, user_id, summary, source='results_plot_summaries')` |
| 410 | `store_plot_summary({...}, source='results_images')` | `index_plot(vector_db, user_id, {...}, source='results_images')` |
| 425 | `store_plot_summary({...}, source='results_images')` | `index_plot(vector_db, user_id, {...}, source='results_images')` |
| 440 | `store_plot_summary(summary, source='data_statistics_plot_summaries')` | `index_plot(vector_db, user_id, summary, source='data_statistics_plot_summaries')` |
| 482 | `store_plot_summary(summary_payload, source='data_statistics_images')` | `index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')` |
| 492 | `store_plot_summary(summary_payload, source='data_statistics_images')` | `index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')` |
| 517 | `store_plot_summary(summary_payload, source='data_statistics_images')` | `index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')` |
| 527 | `store_plot_summary(summary_payload, source='data_statistics_images')` | `index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')` |
| 548 | `store_plot_summary({...}, source='attention_insights')` | `index_plot(vector_db, user_id, {...}, source='attention_insights')` |
| 559 | `store_plot_summary({...}, source='attention_insights')` | `index_plot(vector_db, user_id, {...}, source='attention_insights')` |
| 710 | `store_plot_summary(summary_payload, source='xai_visualizations')` | `index_plot(vector_db, user_id, summary_payload, source='xai_visualizations')` |
| 753 | `store_plot_summary(summary_payload, source='results_images')` | `index_plot(vector_db, user_id, summary_payload, source='results_images')` |

**Quick sed command to do most of them:**

```bash
cd /path/to/repo/ai_outputs
sed -i 's/store_plot_summary(\(.*\), source=\(.*\))/index_plot(vector_db, user_id, \1, source=\2)/g' app.py
```

Then delete the old `store_plot_summary` function definition (lines 332-368).

#### 4c. Add vector DB rehydration on first chat request

In the `/chat` endpoint (~line 1944), add rehydration before the first query:

```python
@app.route('/chat', methods=['POST'])
def chat():
    # ... existing code to get user_id and question ...
    
    # Rehydrate from RustFS if vector DB is empty for this user
    if not vector_db.has_documents(user_id):
        rehydrate_from_restfs(vector_db, user_id)
    
    # ... rest of existing chat logic ...
```

You may need to add a `has_documents()` method to your SimpleVectorDB class:

```python
def has_documents(self, user_id: str) -> bool:
    return user_id in self.user_docs and len(self.user_docs[user_id]) > 0
```

#### 4d. Clean up debug prints

```bash
# Convert debug prints to logger calls
cd /path/to/repo/ai_outputs
sed -i 's/print(f"DEBUG: /logger.debug(f"/g' app.py
sed -i 's/print(f"Error /logger.error(f"/g' app.py
sed -i 's/print(f"Warning: /logger.warning(f"/g' app.py
sed -i 's/print(f"Successfully /logger.info(f"/g' app.py
```

### 5. Manual patches to xai_service/app.py

#### 5a. Add AI adapter import (~line 30)

```python
from ai_adapter import (
    is_external_adapter_configured,
    get_predictions,
    get_explanations,
    get_model_info,
)
```

#### 5b. Fix hardcoded AI_OUTPUTS_SERVICE_URL

Search for any remaining hardcoded URLs:

```bash
grep -n "http://ai_outputs:8001" xai_service/app.py
```

Replace all with the env-var reference:

```python
AI_OUTPUTS_SERVICE_URL  # already defined at line 44
```

Specific lines to fix (based on current code):

| Line | Old | New |
|------|-----|-----|
| 4589 | `ai_outputs_url = 'http://ai_outputs:8001/store-results'` | `ai_outputs_url = f'{AI_OUTPUTS_SERVICE_URL}/store-results'` |
| 4778 | `response = requests.post(ai_outputs_url, ...)` | (fix the url var above) |
| 5642 | `ai_outputs_url = 'http://ai_outputs:8001/store-results'` | `ai_outputs_url = f'{AI_OUTPUTS_SERVICE_URL}/store-results'` |

#### 5c. Merge duplicate XAI visualisation functions

`generate_xai_visualizations()` (line 775) and
`generate_xai_visualizations_for_model()` (line 1663) are near-duplicates.

Keep `generate_xai_visualizations_for_model()` (the more complete one)
and replace all calls to the old one:

```bash
# Find calls to the old function
grep -n "generate_xai_visualizations(" xai_service/app.py | grep -v "for_model"
```

Replace those calls with `generate_xai_visualizations_for_model(...)` and
delete the old function (lines 775-885).

#### 5d. Clean up debug prints (same as ai_outputs)

```bash
cd /path/to/repo/xai_service
sed -i 's/print(f"DEBUG: /logger.debug(f"/g' app.py
sed -i 's/print(f"Error /logger.error(f"/g' app.py
```

### 6. Update Dockerfiles

Both `dashboard/Dockerfile` and `ai_outputs/Dockerfile` need the auth
module to be accessible.  The docker-compose.yml already mounts `./auth`
as a read-only volume.  Just ensure your Dockerfiles don't override the
`PYTHONPATH`:

Add to each Dockerfile (if not already present):

```dockerfile
ENV PYTHONPATH="/app:/app/auth:${PYTHONPATH}"
```

Alternatively, add to each service's `app.py`:

```python
import sys
sys.path.insert(0, '/app/auth')
```

(The dashboard/app.py provided already does this.)

### 7. Keycloak realm setup (when ready for production)

Keycloak starts with `--profile keycloak`:

```bash
docker compose --profile keycloak up -d
```

Then configure via the admin console (http://localhost:8080):

1. **Create realm**: `xai-platform`
2. **Create client**: `xai-dashboard`
   - Client type: OpenID Connect
   - Client authentication: On
   - Valid redirect URIs: `http://localhost:3001/*`
   - Web origins: `http://localhost:3001`
3. **Copy client secret** → set as `KC_CLIENT_SECRET` in `.env`
4. **Create roles**: `viewer`, `analyst`, `admin`
5. **Create users** and assign roles
6. **Set** `AUTH_MODE=keycloak` in `.env`
7. Restart: `docker compose up -d`

### 8. Test the legacy mode first

Before enabling Keycloak, verify everything works in legacy mode:

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, leave AUTH_MODE=legacy
docker compose up --build -d
# Open http://localhost:3001, login with admin / changeme
```

---

## Summary of Changes by Revision

### Revision 1 — AI Service Adapter
- **New**: `xai_service/ai_adapter.py`
- **Impact**: Non-breaking.  The adapter is opt-in via `AI_ADAPTER_URL` env var.
  If not set, the existing built-in mock model is used.

### Revision 2 — Keycloak Auth
- **New**: `auth/__init__.py`, `auth/keycloak.py`
- **Updated**: `dashboard/app.py` (full rewrite), `login.html`, `docker-compose.yml`,
  `dashboard/requirements.txt`
- **Impact**: Backward-compatible.  Default `AUTH_MODE=legacy` preserves the
  old login flow.  Switch to `keycloak` when ready.
- **Removed**: Hardcoded `password123`, hardcoded `secret_key`

### Revision 3 — Enhanced Metadata for RAG
- **New**: `ai_outputs/vector_store.py`
- **Updated**: `ai_outputs/plot_metadata_schema.py`
- **Impact**: Requires patching `ai_outputs/app.py` (see step 4 above).
  The new `summary_for_rag` fields ensure the vector DB has high-quality
  text for embedding, and the rehydration function prevents data loss
  on container restart.

### Revision 4 — Code Cleanup
- **Deleted**: `ai_outputs/app.py.tmp`
- **Updated**: `.env.example`, `ai_outputs/requirements.txt`
- **Impact**: Targeted patches to `ai_outputs/app.py` and `xai_service/app.py`
  (see steps 4-5).  Debug prints → logger calls.  Hardcoded URLs → env vars.
  Duplicate functions merged.
