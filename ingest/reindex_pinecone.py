#!/usr/bin/env python3
"""Rebuild a Pinecone index from locally saved KLWines HTML product pages.

- Parses JSON-LD Product blocks from HTML files.
- Filters to wine-only products.
- Builds a compact canonical embedding text.
- Embeds with OpenAI embeddings.
- Upserts into a NEW Pinecone index.

This script is designed to be resumable via a checkpoint file.

Usage:
  python3 ingest/reindex_pinecone.py --html-dir /path/to/klwines_details_html --index-name aisomm-klwines-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import openai
from pinecone import Pinecone, ServerlessSpec

# Ensure repo root is on sys.path so `ingest.*` imports work when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports
from ingest.klwines_parse import parse_klwines_product_html, is_wine_product
from ingest.embedding_text import build_embedding_text


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"done": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"done": {}}


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def ensure_index(pc: Pinecone, index_name: str, dimension: int, metric: str = "cosine") -> None:
    existing = [i.name for i in pc.list_indexes()]
    if index_name in existing:
        print(f"Index '{index_name}' already exists. Will upsert into it.")
        return

    # Default to serverless; region must be specified. Pinecone requires a cloud/region.
    # We'll try env vars, otherwise fail with a helpful message.
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION")
    if not region:
        raise SystemExit(
            "PINECONE_REGION is required to create a new serverless index. "
            "Set env var PINECONE_REGION (e.g., 'us-east-1') and rerun."
        )

    print(f"Creating new Pinecone index '{index_name}' (dim={dimension}, metric={metric}, {cloud}/{region})...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    # Wait until ready
    for _ in range(60):
        desc = pc.describe_index(index_name)
        if getattr(desc, "status", None) and desc.status.get("ready"):
            print("Index is ready.")
            return
        time.sleep(2)

    print("Index creation requested; continuing (it may still be initializing).")


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-dir", required=True, help="Directory containing KLWines product HTML files")
    ap.add_argument("--index-name", required=True, help="New Pinecone index name")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="ingest/out/klwines_records.jsonl")
    ap.add_argument("--checkpoint", default="ingest/out/checkpoint.json")
    args = ap.parse_args()

    def load_dotenv_loose(dotenv_path: Path) -> None:
        """Load a .env file without requiring python-dotenv.

        Supports lines like:
          KEY=value
          KEY = "value"
        Ignores blank lines and comments (# ...).
        """
        if not dotenv_path.exists():
            return
        for line in dotenv_path.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v

    # Load .env from repo root if present
    load_dotenv_loose(Path(".env"))

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY (set env var or add to .env)")
    if not pinecone_key:
        raise SystemExit("Missing PINECONE_API_KEY (set env var or add to .env)")

    # Embedding dims
    embedding_dim = 3072 if args.embedding_model == "text-embedding-3-large" else 1536

    client = openai.OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    ensure_index(pc, args.index_name, dimension=embedding_dim)
    index = pc.Index(args.index_name)

    html_dir = Path(args.html_dir)
    paths = sorted(html_dir.glob("*.html"))
    if not paths:
        raise SystemExit(f"No .html files found in {html_dir}")

    limit = args.limit
    if limit and limit > 0:
        paths = paths[:limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    done = ckpt.get("done", {})

    # We'll append to JSONL; resume-safe by skipping already-done file hashes.
    out_f = out_path.open("a", encoding="utf-8")

    to_process = []
    for p in paths:
        key = sha1(str(p))
        if key in done:
            continue
        to_process.append(p)

    print(f"Total HTML files: {len(paths)}")
    print(f"Already done: {len(done)}")
    print(f"To process now: {len(to_process)}")

    processed = 0
    skipped_not_wine = 0
    skipped_parse = 0

    for batch in batched(to_process, args.batch_size):
        recs = []
        vec_texts = []
        ids = []

        for p in batch:
            key = sha1(str(p))
            rec = parse_klwines_product_html(p)
            if not rec:
                skipped_parse += 1
                done[key] = {"status": "skip_parse"}
                continue
            if not is_wine_product(rec.category):
                skipped_not_wine += 1
                done[key] = {"status": "skip_not_wine", "sku": rec.sku}
                continue

            emb_text = build_embedding_text(rec)
            vec_id = f"klwines:{rec.sku}"

            # metadata: keep it small-ish
            md = rec.to_dict()
            md["source"] = "klwines"
            md["embedding_text"] = emb_text[:1000]  # for debugging; keep bounded

            recs.append(md)
            vec_texts.append(emb_text)
            ids.append(vec_id)

        if not ids:
            save_checkpoint(ckpt_path, {"done": done})
            continue

        # Embed
        try:
            emb = client.embeddings.create(model=args.embedding_model, input=vec_texts)
            vectors = []
            for i, item in enumerate(emb.data):
                vectors.append({
                    "id": ids[i],
                    "values": item.embedding,
                    "metadata": recs[i],
                })
        except Exception as e:
            print(f"Embedding batch failed: {e}")
            # do not mark as done; allow retry
            time.sleep(2)
            continue

        # Upsert
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Upsert failed: {e}")
            time.sleep(2)
            continue

        # Write normalized JSONL records for audit
        for md in recs:
            out_f.write(json.dumps(md, ensure_ascii=False) + "\n")

        # Mark batch files as done
        for p in batch:
            key = sha1(str(p))
            if key not in done:
                done[key] = {"status": "ok"}

        processed += len(ids)
        if processed % 1000 == 0:
            print(f"Processed vectors: {processed} (skipped_parse={skipped_parse}, skipped_not_wine={skipped_not_wine})")

        save_checkpoint(ckpt_path, {"done": done})

    out_f.close()
    save_checkpoint(ckpt_path, {"done": done})

    print("Done.")
    print(f"Vectors upserted: {processed}")
    print(f"Skipped (parse): {skipped_parse}")
    print(f"Skipped (not wine): {skipped_not_wine}")


if __name__ == "__main__":
    main()
