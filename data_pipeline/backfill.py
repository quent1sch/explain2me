"""
backfill.py

Checks the database for Wikipedia pages missing simple or technical counterparts
and attempts to construct and scrape the missing versions. Updates the DB with
newly retrieved content.

Purpose:
- Ensures as many concepts as possible have multiple difficulty levels (simple vs technical),
  which helps the model learn to explain the same concept to different audiences.
"""

from typing import List, Optional
import sqlite3
import logging
from data_pipeline.scrape_wikipedia_global import scrape_wikipedia


logger = logging.getLogger(__name__)



# ------------------------ BACKFILL DB ------------------------
# ----- with existing simple/technical page counterparts ------



def backfill_DB(
        db_path: str,
        cat_workers: int = 5
    ) -> List[Optional[dict]]:
    """
    Checks DB for pages where:
        - has_simple = 0
        - has_technical = 0
    
    Attempts to construct the missing URL version (works only if direct mapping between urls) and scrape it.
    
    Returns list of successfully scraped pages.
    """

    # ---------------------- Fetch missing pages ----------------------
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
		# Fetch all pages
        cur.execute("""
            SELECT id, title, has_simple, has_technical
            FROM pages
            WHERE has_simple = 0 OR has_technical = 0;
            """)
        
        rows = cur.fetchall()
        conn.close()
        
        logger.info("Found %d pages missing wiki versions.", len(rows))
        
    except sqlite3.Error as e:
        logger.exception("Database error during backfill_DB. db_path='%s'",
                         db_path)
        raise
    
    
    # -------------- Construct URL list ----------------------
    
    urls_to_scrape = []
    
    for _, title, has_simple, has_technical in rows:
        if has_simple == 0:
            # construct canonical technical or simple URLs and add to scraping list
            simple_url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"
            urls_to_scrape.append(simple_url)
        if has_technical == 0:
            normal_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            urls_to_scrape.append(normal_url)
      

    if not urls_to_scrape:
        logger.info("No missing wiki versions found in  db_path='%s'.", db_path)
        return []

    logger.info("Attempting to scrape %d missing versions...", len(urls_to_scrape))


    # ---------------------- Scrape ---------------------------

    try:
        results = scrape_wikipedia(
            urls=urls_to_scrape,
            db_path=db_path,
            cat_workers=cat_workers
        )

        success_count = len([r for r in results if r is not None])

        logger.info("Scraping completed: %d/%d pages successfully scraped.", 
                    success_count,
                    len(urls_to_scrape),
                    )
        return results # some elem in list results might be None but scrape_wikipedia() does not store them

    except Exception:
        logger.exception("Scraping failed in backfill_DB for db_path='%s'", db_path)
        raise


