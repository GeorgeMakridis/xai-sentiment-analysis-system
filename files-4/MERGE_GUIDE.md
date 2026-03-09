# Merge Guide — Remote RustFS (EXTRA-BRAIN)

## Overview

This update switches from a local Docker RustFS container to the shared
EXTRA-BRAIN project RustFS instance at `https://rustfs.extra-brain.unparallel.pt`.

Since RustFS is 100% S3-compatible, the Python `minio` SDK works unchanged.
Only configuration changes are needed — no logic changes anywhere.

---

## Files Changed

```
UPDATED (replace in repo):
├── ai_outputs/restfs_client.py    # new defaults: remote endpoint, HTTPS, no hardcoded creds
├── docker-compose.yml             # removed local rustfs service
└── .env.example                   # remote endpoint, credential placeholders
```

**No other files are affected.** The storage_abstraction.py, vector_store.py,
dashboard/app.py, and all other code use restfs_client.py through env vars,
so they work without any changes.

---

## Step-by-step Merge Instructions

### 1. Replace the 3 files

```bash
cp ai_outputs/restfs_client.py  /path/to/repo/ai_outputs/restfs_client.py
cp docker-compose.yml           /path/to/repo/docker-compose.yml
cp .env.example                 /path/to/repo/.env.example
```

### 2. Create your .env file

```bash
cd /path/to/repo
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
RESTFS_ENDPOINT=rustfs.extra-brain.unparallel.pt
RESTFS_ACCESS_KEY=<your-username>
RESTFS_SECRET_KEY=<your-password>
RESTFS_SECURE=true
```

**Important:** Never commit `.env` to git. Ensure `.gitignore` includes it.

### 3. Verify .gitignore

```bash
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
```

### 4. Deploy

```bash
docker compose up --build -d
```

Note: You no longer need to run a local RustFS/MinIO container.
The system connects directly to the remote instance over HTTPS.

---

## What Changed (Detail)

### ai_outputs/restfs_client.py

| Setting | Old (local) | New (remote) |
|---------|-------------|--------------|
| `RESTFS_ENDPOINT` default | `rustfs:9000` | `rustfs.extra-brain.unparallel.pt` |
| `RESTFS_SECURE` default | `false` (HTTP) | `true` (HTTPS) |
| `RESTFS_ACCESS_KEY` default | `rustfsadmin` | `""` (empty — must set in .env) |
| `RESTFS_SECRET_KEY` default | `rustfsadmin` | `""` (empty — must set in .env) |

Added safety guard: if credentials are empty, the client logs a warning
and falls back to local filesystem storage instead of crashing.

### docker-compose.yml

- **Removed**: `rustfs` service (was `rustfs/rustfs:latest` on ports 9000/9001)
- **Removed**: `rustfs_data` volume
- **Removed**: `depends_on: rustfs` from dashboard, xai_service, ai_outputs
- **Added**: `RESTFS_SECURE` env var to all three services
- **Changed**: `RESTFS_ENDPOINT` default from `rustfs:9000` to `rustfs.extra-brain.unparallel.pt`
- **Changed**: `RESTFS_ACCESS_KEY` / `RESTFS_SECRET_KEY` no longer have defaults
  (must be set in .env)

### .env.example

- Updated RustFS section to show remote endpoint
- Added `RESTFS_SECURE=true`
- Credential placeholders instead of defaults

---

## Buckets

The following buckets are auto-created on first use if they don't exist:

| Bucket | Contents |
|--------|----------|
| `xai-plots` | PNG plot images |
| `xai-datasets` | Uploaded CSV/JSON data files |
| `xai-metadata` | JSON sidecar metadata for RAG |

If bucket auto-creation fails (e.g. due to permissions), create them
manually via the RustFS console at https://rustfs.extra-brain.unparallel.pt.

---

## Fallback Behaviour

If the remote RustFS is unreachable or credentials are missing:

1. `restfs_client.py` returns `None` / `False` for all operations
2. `storage_abstraction.py` detects this and falls back to local
   `shared_volume/` filesystem storage
3. The dashboard, chat, and XAI analysis continue working — just
   without persistent object storage

This means the system is resilient to network issues with the remote
RustFS instance.

---

## Reverting to Local RustFS (if needed)

To switch back to a local Docker container, set these in `.env`:

```env
RESTFS_ENDPOINT=rustfs:9000
RESTFS_ACCESS_KEY=rustfsadmin
RESTFS_SECRET_KEY=rustfsadmin
RESTFS_SECURE=false
```

And restore the `rustfs` service in `docker-compose.yml` from the
previous commit (`2d2e1ab`).
