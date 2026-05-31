"""
scraper.py
==========
Best-effort GENERIC review scraper for ReviewGuard.

Given any product URL, it tries to pull out individual customer reviews so the
app can analyze each one. It is intentionally site-agnostic — it does NOT
hard-code Amazon selectors. Instead it uses two general strategies:

    Strategy 1: structured data — schema.org "Review" objects embedded as
                JSON-LD (<script type="application/ld+json">). Many modern
                e-commerce / review sites publish reviews this way.

    Strategy 2: HTML heuristics — find elements whose class/id contains the
                word "review" and pull out reasonable-length text blocks.

If both fail (or the site blocks us), it returns an empty list and the caller
falls back to manual paste. Scraping live commercial sites is unreliable by
nature (CAPTCHAs, JavaScript, bot-blocking), so this is "best effort".
"""

import json
import re

import requests
from bs4 import BeautifulSoup

# Pretend to be a normal browser — many sites reject the default requests UA.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Reviews shorter/longer than these bounds are almost always noise.
MIN_LEN = 8
MAX_LEN = 3000
MAX_REVIEWS = 200  # cap so we don't analyze thousands of blocks


def _clean(text):
    """Collapse whitespace and strip a single review string."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _from_json_ld(soup):
    """
    Strategy 1: extract review bodies from schema.org JSON-LD blocks.

    Looks for objects with @type == 'Review' and grabs their 'reviewBody'
    (and 'author'/'reviewRating' if present). Handles reviews nested under a
    Product's 'review' field too.
    """
    found = []

    def harvest(obj):
        # A single Review node.
        if isinstance(obj, dict):
            otype = obj.get("@type", "")
            otype = otype if isinstance(otype, str) else ",".join(otype or [])
            if "Review" in otype:
                body = obj.get("reviewBody") or obj.get("description")
                if body:
                    rating = None
                    rr = obj.get("reviewRating")
                    if isinstance(rr, dict):
                        rating = rr.get("ratingValue")
                    found.append({"text": _clean(body), "rating": rating})
            # Recurse into nested review lists / product objects.
            for key in ("review", "reviews", "@graph", "itemListElement"):
                if key in obj:
                    harvest(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                harvest(item)

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        harvest(data)

    return found


def _from_html_heuristics(soup):
    """
    Strategy 2: find elements whose class or id mentions "review" and treat
    their text as candidate reviews. Generic enough to catch many custom sites.
    """
    found = []
    pattern = re.compile(r"review", re.I)

    # Match class OR id containing "review".
    candidates = soup.find_all(
        attrs={"class": pattern}) + soup.find_all(attrs={"id": pattern})

    seen = set()
    for el in candidates:
        text = _clean(el.get_text(" "))
        # Skip containers that are clearly headers/counts ("1,234 reviews").
        if MIN_LEN <= len(text) <= MAX_LEN and text.lower() not in seen:
            seen.add(text.lower())
            found.append({"text": text, "rating": None})
    return found


def _from_daraz(url):
    """
    Site-specific handler for Daraz (daraz.pk / daraz.com.* etc).

    Daraz loads reviews via JavaScript, so the raw product HTML has none. But
    it exposes a JSON reviews API. We pull the product's itemId out of the URL
    (the '-iXXXXXXXX-' segment) and call that API directly.

    Returns a list of {"text","rating"} dicts, or [] if it can't.
    """
    m = re.search(r"-i(\d+)", url)              # e.g. ...-i727023322-s... -> 727023322
    if not m:
        return []
    item_id = m.group(1)

    # Match the Daraz country domain so the API host lines up (pk/com.bd/etc).
    dom = re.search(r"daraz\.([a-z.]+)", url, re.I)
    tld = dom.group(1) if dom else "pk"

    headers = dict(HEADERS)
    headers["Accept"] = "application/json"
    headers["Referer"] = f"https://www.daraz.{tld}/"

    # Page through the review API until we run out of pages (or hit a safety
    # cap). Daraz returns ~10-50 reviews per page; we collect them all.
    found = []
    for page in range(1, 30):                   # up to ~30 pages safety cap
        api = (f"https://my.daraz.{tld}/pdp/review/getReviewList"
               f"?itemId={item_id}&pageSize=50&filter=0&sort=0&pageNo={page}")
        try:
            data = requests.get(api, headers=headers, timeout=15).json()
        except (requests.RequestException, ValueError):
            break

        # Daraz sometimes returns {"model": null} — guard against None at
        # every level so we never call .get() on a None.
        model = data.get("model") or {}
        items = model.get("items") or []
        if not items:                           # no more reviews -> stop
            break
        for it in items:
            if not isinstance(it, dict):
                continue
            text = _clean(it.get("reviewContent") or "")
            if len(text) >= MIN_LEN:            # skip rating-only reviews
                found.append({"text": text, "rating": it.get("rating")})

        # Use the API's own page count if present, else stop when a short page.
        paging = model.get("paging") or {}
        total_pages = paging.get("totalPage") or paging.get("pages")
        if total_pages and page >= int(total_pages):
            break
        if len(items) < 50:                     # last (partial) page reached
            break
    return found


def scrape_reviews(url):
    """
    Main entry point. Returns a dict:
        {"ok": bool, "reviews": [{"text":..., "rating":...}], "error": str}

    'reviews' is de-duplicated and capped at MAX_REVIEWS. On any failure
    (bad URL, blocked, nothing found) ok=False and the caller should fall back
    to manual paste.
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "reviews": [], "error": "Please enter a product URL."}
    # Be forgiving: prepend https:// if the user pasted a bare domain.
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url

    # Site-specific handler first: Daraz needs its JSON review API.
    if "daraz." in url.lower():
        reviews = _from_daraz(url)
    else:
        reviews = []

    # Generic path for every other site.
    if not reviews:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Structured data first (cleanest), then HTML heuristics.
            reviews = _from_json_ld(soup) or _from_html_heuristics(soup)
        except requests.RequestException as e:
            return {"ok": False, "reviews": [],
                    "error": f"Could not fetch the page ({e}). The site may block scrapers."}

    # De-duplicate by text and cap the count.
    unique, seen = [], set()
    for r in reviews:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) >= MAX_REVIEWS:
            break

    if not unique:
        return {"ok": False, "reviews": [],
                "error": "No reviews found on that page. Try pasting the review text manually."}

    return {"ok": True, "reviews": unique, "error": ""}
