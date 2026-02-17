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

from data_pipeline.helpers import get_category_pages, page_needs_scraping, store_page
from data_pipeline.scrape_simple_wiki import scrape_simple_wiki
from data_pipeline.scrape_technical_wiki import scrape_normal_wiki






# ---- FRAMEWORK FUNCTION - MAIN SCRAPING FUNCTION ----

def scrape_wikipedia(
        urls: Union[str, List[str]], 
        db_path: Optional[str] = None, 
        cat_workers: int = 5
    ) -> List[dict]:
    """
    General Wikipedia scraper framework.
    
    Parameters
    ----------
    urls : str or List[str]
        Wikipedia page(s) or category URL(s) to scrape.
    db_path : str, optional
        Path to SQLite DB to store results. If None, results are not stored.
    cat_workers : int
        Number of parallel threads for scraping.

    Returns
    -------
    List[dict]
        List of successfully scraped page results.
    
    Notes:
    - Accepts single URL or list of URLs (pages or categories)
    - Automatically detects:
        - Simple vs normal Wikipedia
        - Category vs single page
    - Expands categories to individual page URLs
    - Scrapes pages in parallel using ThreadPoolExecutor
    - Directs to the appropriate scraper function
    """

    if isinstance(urls, str):
        urls = [urls]

    # Expand all category URLs
    expanded_urls = []
    for url in urls:
        if "/wiki/Category:" in url:
            categories = get_category_pages(url)
            pages_urls = [page.get('url', None) for page in categories['pages']]
            expanded_urls.extend(pages_urls)
        else:
            expanded_urls.append(url)
    
    # Decide which need scraping (if db_path provided, don't scrape if already in db)     
    urls_to_scrape = []
    
    for url in expanded_urls:
        if db_path is None:
            urls_to_scrape.append(url)
        else:
            if page_needs_scraping(url, db_path=db_path):
                urls_to_scrape.append(url)
     
    if not urls_to_scrape:
        print("All pages already in DB. Nothing to scrape.")
        return []
    
    
	# Determine which scraper to use for each URL
    def scrape_dispatcher(url: str):
        conn = None
        cur = None
        if db_path:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

        try:
            if "simple.wikipedia.org" in url:
                result = scrape_simple_wiki(url)
            else:
                result = scrape_normal_wiki(url)

            if db_path and result:
                store_page(result, conn, cur)

            return result

        except RuntimeError:
            print(f"[WARNING] Skipping URL due to fetch error: {url}")
            return None

        finally: 
            if conn: conn.close()


    # Parallel scraping
    results = []
    with ThreadPoolExecutor(max_workers=cat_workers) as executor:
        futures = [executor.submit(scrape_dispatcher, u) for u in urls_to_scrape]
        for f in futures:
            res = f.result()
            results.append(res)

    return results