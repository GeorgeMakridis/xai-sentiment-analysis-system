# Merge Guide — Vector DB Reads Plots from RustFS

## Problem Solved

Previously, the Vector DB (in-memory) and RustFS (persistent storage) were
disconnected. Plot images were saved to RustFS, but the chatbot's Vector DB
was populated only during the same session. On container restart, the
chatbot lost all knowledge of previously generated plots.

Now the system works like this:

```
WRITE PATH (during analysis):
  Plot generated → PNG saved to RustFS (xai-plots/)
                 → Metadata JSON saved to RustFS (xai-metadata/{user}/{plot_id}.json)
                 → Text summary embedded in Vector DB (RAM)

READ PATH (on chat, including after restart):
  User asks question → Vector DB empty? 
       YES → Scan RustFS xai-metadata/{user}/*.json
            → Download each metadata JSON
            → Re-embed text summaries into Vector DB
            → Then search normally
       NO  → Search Vector DB directly
```

## Files Changed

```
REPLACE in repo:
├── ai_outputs/app.py                  # 7 patches (see below)
└── ai_outputs/plot_metadata_schema.py  # enhanced with metadata_to_vector_text()
```

## Step-by-step

### 1. Replace both files

```bash
cp ai_outputs/app.py                 /path/to/repo/ai_outputs/app.py
cp ai_outputs/plot_metadata_schema.py /path/to/repo/ai_outputs/plot_metadata_schema.py
```

### 2. Deploy

```bash
docker compose up --build -d
```

### 3. Test

```bash
# Generate some plots
# (upload data, click "Generate Data Statistics" in dashboard)

# Restart the ai_outputs container (simulates crash/restart)
docker compose restart ai_outputs

# Ask the chatbot a question about the plots
# It should auto-rehydrate from RustFS and answer correctly

# Or manually trigger rehydration:
curl -X POST http://localhost:8002/api/rehydrate/admin
```

## What Changed in app.py (7 patches)

### Patch 1 — Import metadata helpers
Added `metadata_to_vector_text` and `metadata_to_vector_meta` to the 
`plot_metadata_schema` import. These convert structured metadata into
text suitable for embedding.

### Patch 2 — RustFS ↔ Vector DB bridge (3 new functions)
After `vector_db = VectorDatabase()`, added:

- `_persist_plot_to_restfs(user_id, plot_id, meta)` — saves metadata
  JSON sidecar to RustFS `xai-metadata/{user_id}/{plot_id}.json`
- `_rehydrate_user_from_restfs(user_id)` — scans RustFS for metadata
  JSONs, downloads each, re-embeds into Vector DB
- `_rehydrate_from_local_registry(user_id)` — fallback: reads from
  local `plots_registry.json` if RustFS is unavailable

### Patch 3 — store_plot_summary() persists to RustFS
Every time a plot summary is indexed in the Vector DB, its full metadata
is also saved as a JSON file to RustFS. This ensures rehydration has
data to work with.

### Patch 4 — store_plot_image_endpoint persists to RustFS
Same as Patch 3 but for the `/api/store-plot-image` endpoint (used when
the dashboard forwards individual plot images).

### Patch 5 — /chat auto-rehydrates
The `/chat` endpoint now checks if the Vector DB is empty for the user.
If so, it calls `_rehydrate_user_from_restfs()` before searching.
This means the chatbot automatically recovers after a container restart.

### Patch 6 — Startup logging
On app startup, logs whether RustFS is connected or not.

### Patch 7 — /api/rehydrate/<user_id> endpoint
New endpoint for manually triggering rehydration. Returns the count of
plots reloaded. Useful for debugging or force-refreshing.

## What Changed in plot_metadata_schema.py

Added two functions that the rehydration code uses:

- `metadata_to_vector_text(meta)` — converts a metadata dict into the
  single text string that gets embedded. Includes title, plot type,
  summary_for_rag.text, numeric facts, description, provenance.
- `metadata_to_vector_meta(meta)` — builds the metadata dict stored
  alongside the embedding (doc_type, plot_id, keywords, etc.)

Also added `summary_for_rag` section to `build_plot_metadata()` with
auto-generation of summary text, keywords, and numeric facts.

## RustFS Bucket Usage (after this patch)

| Bucket | Contents | Used By |
|--------|----------|---------|
| `xai-plots` | `{user}/{plot_id}.png` | Image serving to dashboard |
| `xai-metadata` | `{user}/{plot_id}.json` | **Vector DB rehydration** |
| `xai-metadata` | `{user}/plots_registry.json` | Plot listing, fallback rehydration |
| `xai-datasets` | `{user}/{filename}` | Dataset browser |
