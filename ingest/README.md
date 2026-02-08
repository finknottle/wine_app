# KLWines → Pinecone reindex (v2)

This folder contains a **one-time (repeatable)** pipeline to rebuild the Pinecone index from your locally saved KLWines HTML.

## Why
The current recommender index mixes long review text and inconsistent fields in the embedding input.
This pipeline:
- parses **structured JSON-LD** from KLWines product pages
- builds a **compact canonical “sommelier profile”** string for embeddings
- re-embeds using **text-embedding-3-large (3072 dims)**
- upserts into a **new Pinecone index**

## Inputs
- A directory of KLWines HTML product pages (example from this workspace):
  - `/data/.openclaw/workspace/klwines/klwines_details_html/*.html`

## Required env
Set these in your shell (or `.env`, not committed):
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_ENV` (if your Pinecone setup needs it; modern Pinecone often does not)

## Run
Example (adjust paths/index name):

```bash
cd wine_app

export OPENAI_API_KEY=...
export PINECONE_API_KEY=...

python3 ingest/reindex_pinecone.py \
  --html-dir /data/.openclaw/workspace/klwines/klwines_details_html \
  --index-name aisomm-klwines-v2 \
  --embedding-model text-embedding-3-large \
  --batch-size 64 \
  --limit 0
```

Notes:
- `--limit 0` means no limit (process all files).
- The script writes a **checkpoint file** so you can resume if interrupted.

## Outputs
- `ingest/out/klwines_records.jsonl` (normalized records)
- `ingest/out/checkpoint.json` (resume state)

