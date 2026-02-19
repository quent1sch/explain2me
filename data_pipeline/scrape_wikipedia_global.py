# ------- IMPORT LIBRARIES -------

from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Optional
import sqlite3
import logging

from data_pipeline.helpers import get_category_pages, page_needs_scraping, store_page
from data_pipeline.scrape_simple_wiki import scrape_simple_wiki
from data_pipeline.scrape_technical_wiki import scrape_normal_wiki


logger = logging.getLogger(__name__)



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
            try:
                categories = get_category_pages(url)
                pages_urls = [page.get('url') for page in categories.get('pages', []) if page.get('url')]
                expanded_urls.extend(pages_urls)
            except Exception as e:
                logger.warning("Failed to expand category URL '%s': %s", url, e)

        else:
            expanded_urls.append(url)
    
    # Decide which need scraping (if db_path provided, don't scrape if already in db)     
    urls_to_scrape = []
    
    for url in expanded_urls:
        if db_path is None:
            urls_to_scrape.append(url)
        else:
            try:
                if page_needs_scraping(url, db_path=db_path):
                    urls_to_scrape.append(url)
            except sqlite3.Error as e:
                logger.error("Failed to check page '%s' in DB: %s", url, e)
     
    if not urls_to_scrape:
        logger.info("All pages already in DB. Nothing to scrape.")
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
            logger.warning("Skipping URL due to fetch error: %s", url)
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