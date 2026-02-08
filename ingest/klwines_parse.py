"""KLWines HTML parsing utilities.

Primary strategy: parse JSON-LD (schema.org Product) embedded in the HTML.
This is much more stable than scraping DOM.

The saved HTML often contains bot-protection JS; that's fine because we are not
fetching live pages, only parsing local files.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import json
import re


_JSONLD_RE = re.compile(
    r"<script[^>]*type=\"application/ld\+json\"[^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_jsonld_objects(html: str) -> list[Any]:
    objs: list[Any] = []
    for m in _JSONLD_RE.finditer(html):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        obj = _safe_json_loads(raw)
        if obj is None:
            continue
        if isinstance(obj, list):
            objs.extend(obj)
        else:
            objs.append(obj)
    return objs


def _first_product_jsonld(html: str) -> Optional[dict[str, Any]]:
    for obj in _extract_jsonld_objects(html):
        if isinstance(obj, dict) and obj.get("@type") == "Product":
            return obj
    return None


def _get_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return None


def _get_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _norm_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    return " ".join(cat.split())


def is_wine_product(category: Optional[str]) -> bool:
    """Heuristic filter: keep only wine products.

    KLWines categories look like:
      "Burgundy - Old and Rare / Wine - Red"
      "Spirits / Brandy"

    We'll keep anything that contains "Wine" and exclude common spirit terms.
    """

    if not category:
        return False
    c = category.lower()
    if "wine" not in c:
        return False
    # Exclude obvious non-wine categories even if they contain the word wine somewhere
    blocked = [
        "brandy",
        "cordial",
        "liqueur",
        "spirits",
        "whiskey",
        "whisky",
        "rum",
        "gin",
        "vodka",
        "tequila",
        "mezcal",
        "cognac",
        "armagnac",
    ]
    if any(b in c for b in blocked):
        return False
    return True


@dataclass
class KlwinesRecord:
    sku: str
    name: str
    category: Optional[str] = None
    country: Optional[str] = None
    size: Optional[str] = None
    image: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    availability: Optional[str] = None
    keywords: Optional[str] = None
    reviews: Optional[list[dict[str, Any]]] = None
    aggregate_rating: Optional[dict[str, Any]] = None
    source_url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)

        # Pinecone metadata constraints:
        # values must be string/number/bool or list[str]. Nested dict/list[dict] is not allowed.
        if d.get("aggregate_rating") is not None:
            try:
                d["aggregate_rating"] = json.dumps(d["aggregate_rating"], ensure_ascii=False)
            except Exception:
                d["aggregate_rating"] = str(d["aggregate_rating"])

        if d.get("reviews") is not None:
            reviews = d.get("reviews") or []
            out: list[str] = []
            if isinstance(reviews, list):
                for r in reviews[:3]:
                    if not isinstance(r, dict):
                        continue
                    author = (r.get("author") or "Critic").strip() if isinstance(r.get("author"), str) else "Critic"
                    rating = r.get("rating")
                    txt = (r.get("review") or "").strip() if isinstance(r.get("review"), str) else ""
                    if txt and len(txt) > 240:
                        txt = txt[:240].rstrip() + "…"
                    if rating is not None and txt:
                        out.append(f"{author} ({rating}): {txt}")
                    elif txt:
                        out.append(f"{author}: {txt}")
                    elif rating is not None:
                        out.append(f"{author} rating: {rating}")
            d["reviews"] = out or None

        # Remove empty fields for cleaner JSONL
        return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def parse_klwines_product_html(path: str | Path) -> Optional[KlwinesRecord]:
    p = Path(path)
    html = p.read_text(errors="ignore")

    product = _first_product_jsonld(html)
    if not product:
        return None

    sku = _get_text(product.get("sku"))
    name = _get_text(product.get("name"))
    if not sku or not name:
        return None

    category = _norm_category(_get_text(product.get("category")))

    # countryOfOrigin may be {"@type":"Country","name":"France"}
    country = None
    coo = product.get("countryOfOrigin")
    if isinstance(coo, dict):
        country = _get_text(coo.get("name"))
    else:
        country = _get_text(coo)

    size = _get_text(product.get("size"))
    image = _get_text(product.get("image"))
    description = _get_text(product.get("description"))
    keywords = _get_text(product.get("keywords"))

    # offers is usually a list with price info
    price = None
    availability = None
    offers = product.get("offers")
    if isinstance(offers, list) and offers:
        offer0 = offers[0]
        if isinstance(offer0, dict):
            price = _get_float(offer0.get("price"))
            availability = _get_text(offer0.get("availability"))

    reviews_out: list[dict[str, Any]] = []
    reviews = product.get("review")
    if isinstance(reviews, list):
        for r in reviews:
            if not isinstance(r, dict):
                continue
            author = r.get("author")
            author_name = None
            if isinstance(author, dict):
                author_name = _get_text(author.get("name"))
            else:
                author_name = _get_text(author)

            review_body = _get_text(r.get("reviewBody"))
            rating_val = None
            rr = r.get("reviewRating")
            if isinstance(rr, dict):
                rating_val = _get_float(rr.get("ratingValue"))
                best = _get_float(rr.get("bestRating"))
            else:
                best = None

            ro = {
                "author": author_name,
                "review": review_body,
            }
            if rating_val is not None:
                ro["rating"] = rating_val
            if best is not None:
                ro["best_rating"] = best

            # Keep only meaningful reviews
            if ro.get("review") or ro.get("rating") is not None:
                reviews_out.append(ro)

    aggregate_rating = product.get("aggregateRating")
    if not isinstance(aggregate_rating, dict):
        aggregate_rating = None

    # og:url meta exists in your sample, but easiest is also in product JSON-LD sometimes.
    # We'll try to recover it from HTML as fallback.
    source_url = None
    m = re.search(r"<meta\s+property=\"og:url\"\s+content=\"(.*?)\"", html, re.IGNORECASE)
    if m:
        source_url = m.group(1)

    rec = KlwinesRecord(
        sku=sku,
        name=name,
        category=category,
        country=country,
        size=size,
        image=image,
        description=description,
        price=price,
        availability=availability,
        keywords=keywords,
        reviews=reviews_out or None,
        aggregate_rating=aggregate_rating,
        source_url=source_url,
    )

    return rec
