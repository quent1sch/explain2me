import os, json
from datetime import datetime, timezone


def load_cache(CACHE_FILE):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_cache(cache, CACHE_FILE):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def merge_article_into_cache(cache, article_dict, issimple=True):
    """
    Store scraped article in JSON cache.
    Preserves and merges categories and category_urls from older entries.
    Updates timestamps intelligently.
    """
    url = article_dict["url"]
    now = datetime.now(timezone.utc).isoformat()

    old = next((item for item in cache if item["url"] == url), None)

    article_dict["last_scraped"] = now

    if old:
        # Preserve first scrape timestamp
        article_dict["first_scraped"] = old.get("first_scraped", now)

        if issimple:
            # Merge Simple Wiki categories
            article_dict["categories"] = list(
                set(old.get("categories", [])) |
                set(article_dict.get("categories", []))
            )
            article_dict["category_urls"] = list(
                set(old.get("category_urls", [])) |
                set(article_dict.get("category_urls", []))
            )

        cache.remove(old)

    else:
        # First time we see this URL
        article_dict["first_scraped"] = now
        if issimple:
            # Normalize fields to lists if missing
            article_dict["categories"] = article_dict.get("categories", [])
            article_dict["category_urls"] = article_dict.get("category_urls", [])

    cache.append(article_dict)
