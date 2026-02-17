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

from data_pipeline.helpers import simple2normalwiki_url, normal2simplewiki_url





# ----------------------------------------
# 			  SCRAPING METHOD
# ----------------------------------------

# ---------------- CONFIG ----------------
IGNORE_CLASSES = {
    "sidebar-list", "navbar", "infobox", "toc",
    "thumb", "mw-default-size", "metadata"
}

STOP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading"
}



# ---------------- HELPERS ----------------
def clean_paragraph_tech(el: Tag) -> str:
    """Clean paragraph text, preserving math as LaTeX."""

    # Remove citation markers
    for sup in el.find_all("sup"):
        sup.decompose()

    # Preserve math
    for math in el.find_all("math"):
        latex = math.get("alttext") or math.get_text(strip=True)
        latex = latex.strip()

        is_block = el.get_text(strip=True) == math.get_text(strip=True)
        math.replace_with(
            f"\n$$\n{latex}\n$$\n" if is_block else f"${latex}$"
        )

    # Lists
    if el.name in {"ul", "ol"}:
        lines = []
        for i, li in enumerate(el.find_all("li", recursive=False), start=1):
            txt = clean_paragraph_tech(li)
            if txt:
                lines.append(f"- {txt}" if el.name == "ul" else f"{i}) {txt}")
        return "\n".join(lines)

    # Text cleanup
    text = el.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def is_ignored_tech(el: Tag) -> bool:
    """Ignore elements inside navboxes, infoboxes, thumbnails, TOC."""
    for parent in el.parents:
        classes = parent.get("class", [])
        if any(cls in IGNORE_CLASSES for cls in classes):
            return True
    return False





# ---------------- MAIN SCRAPER ----------------
def scrape_normal_wiki(url: str) -> dict:
    headers = {"User-Agent": "ReverseMentorBot/0.1"}
    
    try:
         res = requests.get(url, headers=headers, timeout=10)
         # Raise an HTTPError for 4xx/5xx responses (e.g., 404, 500, 429), ensuring failed HTTP responses are treated as errors.
         res.raise_for_status()
		
    except requests.RequestException as e:
        # catches all request-related failures: connection errors, timeouts, invalid URLs, and HTTP errors raised by raise_for_status()
		# i.e. network/environment-level failures, not parsing or scraper-logic errors.
        raise RuntimeError(f"Failed to fetch URL: {url}") from e


    soup = BeautifulSoup(res.text, "html.parser")


    content = soup.find("div", id="mw-content-text")
    
    if content is None:
        raise ValueError(f"Content div not found for {url}")

    sections = []
    intro = None
    current = None

    # Traverse in DOM order
    for el in content.find_all(
        ["p", "li", "dd", "ul", "ol", "h2", "h3", "h4", "h5"],
        recursive=True
    ):
        if is_ignored_tech(el):
            continue

        # ---------- HEADINGS ----------
        if el.name.startswith("h"):
            heading = el.get_text(" ", strip=True).replace("[edit]", "")
            if heading.lower() in STOP_SECTIONS:
                break

            current = {"heading": heading, "paragraphs": []}
            sections.append(current)
            continue

        # ---------- CONTENT ----------
        text = clean_paragraph_tech(el)
        if not text:
            continue

        if current is None:
            if intro is None:
                intro = {"heading": "Introduction", "paragraphs": []}
                sections.insert(0, intro)
            intro["paragraphs"].append(text)
        else:
            current["paragraphs"].append(text)

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None
    

	# --------- CATEGORY DATA ----------
    categories = []
    category_urls = []

    catlinks = soup.select("#mw-normal-catlinks ul li a")
    for cat in catlinks:
        categories.append(cat.get_text(strip=True))
        href = cat.get("href")
        if href and href.startswith("/wiki/"):
            category_urls.append("https://en.wikipedia.org" + href)



    return {
        "url": url,
        "title": title,
        "sections": sections,
        "categories": categories,
        "category_urls": category_urls,
    }

