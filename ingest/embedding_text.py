"""Build canonical embedding text for wine retrieval.

Goal: improve similarity search by embedding a compact, structured profile
instead of long raw review dumps.

We keep:
- identity: producer/name/vintage (best-effort)
- origin/category
- price/size (optional)
- tasting descriptors (from description + *short* review snippets)

We avoid:
- huge review bodies
- noisy markup
"""

from __future__ import annotations

import re
from typing import Optional

from .klwines_parse import KlwinesRecord


_VINTAGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _extract_vintage(name: str) -> Optional[str]:
    m = _VINTAGE_RE.search(name or "")
    return m.group(1) if m else None


def _extract_producer(name: str) -> Optional[str]:
    # KLWines name often starts with vintage, then producer.
    # Example: "2011 Domaine Louis Jadot Marsannay"
    s = (name or "").strip()
    if not s:
        return None

    # remove leading vintage
    s2 = _VINTAGE_RE.sub("", s, count=1).strip()
    if not s2:
        return None

    # producer heuristic: first 2-4 title-cased tokens until a known region-ish token
    tokens = s2.split()
    if len(tokens) <= 1:
        return None

    stop = set(
        w.lower()
        for w in [
            "marsannay",
            "chablis",
            "sancerre",
            "barolo",
            "barbaresco",
            "rioja",
            "chianti",
            "brunello",
            "champagne",
        ]
    )
    prod_tokens = []
    for t in tokens[:5]:
        if t.lower() in stop:
            break
        prod_tokens.append(t)
        if len(prod_tokens) >= 4:
            break

    producer = " ".join(prod_tokens).strip()
    return producer or None


def build_embedding_text(rec: KlwinesRecord) -> str:
    name = rec.name
    vintage = _extract_vintage(name)
    producer = _extract_producer(name)

    parts = []
    parts.append(f"Name: {name}.")
    if producer:
        parts.append(f"Producer: {producer}.")
    if vintage:
        parts.append(f"Vintage: {vintage}.")

    if rec.category:
        parts.append(f"Category: {rec.category}.")
    if rec.country:
        parts.append(f"Country: {rec.country}.")
    if rec.size:
        parts.append(f"Bottle: {rec.size}.")
    if rec.price is not None:
        parts.append(f"Price: ${rec.price:.2f}.")

    # Use description as primary taste signal
    if rec.description:
        desc = re.sub(r"\s+", " ", rec.description).strip()
        parts.append(f"Tasting notes: {desc}")

    # Add up to 1 short critic snippet
    if rec.reviews:
        for r in rec.reviews[:1]:
            txt = (r.get("review") or "").strip()
            if txt:
                txt = re.sub(r"\s+", " ", txt)
                if len(txt) > 280:
                    txt = txt[:280].rstrip() + "…"
                who = (r.get("author") or "Critic").strip() or "Critic"
                rating = r.get("rating")
                if rating is not None:
                    parts.append(f"Critic ({who}) {rating}: {txt}")
                else:
                    parts.append(f"Critic ({who}): {txt}")
                break

    # Keywords sometimes include varietal/region
    if rec.keywords:
        # keep only the last few comma-separated keywords (often contains varietal/region)
        kws = [k.strip() for k in rec.keywords.split(",") if k.strip()]
        tail = ", ".join(kws[-6:])
        if tail:
            parts.append(f"Keywords: {tail}.")

    # Final cleanup
    out = "\n".join(parts)
    out = "\n".join([ln.strip() for ln in out.splitlines() if ln.strip()])
    return out
