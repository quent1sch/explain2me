# ------- IMPORT LIBRARIES -------

import json
from bs4 import BeautifulSoup, Tag
import requests
import re
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timezone
import os
from urllib.parse import urlparse, unquote
from typing import Union, List, Optional
import sqlite3
from random import sample, choices
import pandas as pd
from openai import OpenAI
import random
from tqdm import tqdm
from functools import partial
from typing import Tuple
import logging



# ------------- HELPERS -------------
# (try to) convert simple wiki page url to its classic wiki page url 
# counterpart
# Goal -> have a normal wikipedia page for each simple wikipedia page 
#         already stored

def simple2normalwiki_url(simple_url):
    page_url_segment = simple_url.split("/")[-1]
    normalwiki_base = "https://en.wikipedia.org/wiki/"
    normalwiki_url = normalwiki_base + page_url_segment
    return normalwiki_url


#-----------------------------
# And conversely....

def normal2simplewiki_url(normal_url):
    page_url_segment = normal_url.split("/")[-1]
    simplewiki_base = "https://simple.wikipedia.org/wiki/"
    simplewiki_url = simplewiki_base + page_url_segment
    return simplewiki_url





# -------- Get Wiki pages URLs from a Wiki Category URL --------------
# Used for scraping all pages in a category instead of scraping them individually.
# Works for both simple and normal wiki category pages.

def get_category_pages(category_url):

    headers = {
        "User-Agent": "YourBot/1.0 (https://example.com/contact)"
    }
    
    try:
         res = requests.get(category_url, headers=headers, timeout=10)
         # Raise an HTTPError for 4xx/5xx responses (e.g., 404, 500, 429), ensuring failed HTTP responses are treated as errors.
         res.raise_for_status()
		
    except requests.RequestException as e:
        # catches all request-related failures: connection errors, timeouts, invalid URLs, and HTTP errors raised by raise_for_status()
		# i.e. network/environment-level failures, not parsing or scraper-logic errors.
        raise RuntimeError(f"Failed to fetch category URL: {category_url}") from e
		
    
    soup = BeautifulSoup(res.text, "html.parser")

    # Extract category name from URL
    url_parse = urlparse(category_url)
    path = url_parse.path
    category_name = path.split(":")[-1]

    base = url_parse.scheme + "://" + url_parse.netloc # https://simple.wikipedia.org
    pages = []

    for li in soup.select("#mw-pages li a"):
        href = li.get("href")
        title = li.get_text(strip=True)
        if "Template:" in title:
            continue
        if href and href.startswith("/wiki/"):
            pages.append({
                "title": title,
                "url": base + href
            })
            
    if not pages:
            raise ValueError(f"No pages found in category {category_name} at {category_url}")

    return {
        "categories": category_name,
        "category_urls": category_url,
        "pages": pages
    }






# --------- Check if page is already in DB ----------------------------
# ------------- or should be scraped ----------------------------------


def page_needs_scraping(url: str, db_path: str) -> bool:
    """
    Returns True if the page (simple or technical) is NOT yet stored.
    Infers everything from the URL.
    """

    # Extract title
    path = urlparse(url).path
    if "/wiki/" not in path:
        return False  # Not a valid wiki page

    title = unquote(path.split("/wiki/")[-1])

    # Determine which indicator column to check
    if "simple.wikipedia.org" in url:
        indicator_col = "has_simple"
    elif "wikipedia.org" in url:
        indicator_col = "has_technical"
    else:
        return False  # Not supported domain

    # Open connection (thread-safe pattern)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(
            f"SELECT {indicator_col} FROM pages WHERE title = ?",
            (title,)
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        # Page not in DB at all → needs scraping
        return True

    # If indicator is 0 → needs scraping
    return row[0] == 0





# ------------ STORAGE FCT TO DB ---------------

def store_page(article_data: dict, conn, cur):
    """
    Stores a scraped Wikipedia article into DB.
    (Updates both tables 'pages' and 'definitions')
    
    article_data = {
        "url": str,
        "title": str,
        "sections": list,
        ...
    }
    """

    url = article_data["url"]
    title = article_data["title"]
    sections = article_data.get("sections", [])
    content = json.dumps(sections)  # store sections as JSON string

    # Determine kind from URL
    kind = "simple" if "simple.wikipedia.org" in url else "technical"

    source = url
    created_at = datetime.now(timezone.utc).isoformat()

    # Ensure page exists (redondance of 'OR IGNORE' as scrape_wikipedia already handles page uniqueness)
    cur.execute(
        "INSERT OR IGNORE INTO pages (title) VALUES (?);",
        (title,)
    )

    # Get page_id
    cur.execute("SELECT id FROM pages WHERE title = ?", (title,))
    page_id = cur.fetchone()[0]

    # Insert definition or update existing 
	# to correct: (scrape_wikipedia() prevents updates though as checks for existance of title first)
    cur.execute(
        """
        INSERT OR REPLACE INTO definitions
        (page_id, kind, content, source, created_at)
        VALUES (?, ?, ?, ?, ?);
        """,
        (page_id, kind, content, source, created_at)
    )

    # Update indicator & commit
    cur.execute(f"UPDATE pages SET has_{kind} = 1 WHERE id = ?", (page_id,))

    conn.commit()
