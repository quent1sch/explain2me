# ------- IMPORT LIBRARIES -------

from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timezone
import sqlite3
from openai import OpenAI
import random
from tqdm import tqdm
from functools import partial
from typing import Tuple
import logging


logger = logging.getLogger(__name__) 




# ----------------- Definition for Kids - Generation ---------------


# generation of definitions for kids using a LLM, based on scraped wikipedia page content (use simple definition if available)
def generate_kids_definition(
    page_tuple: Tuple[int, str, str],
    client: OpenAI,
    max_input_tokens: int=16384,
    max_retries: int = 3,
    base_delay: float = 1.0,) -> Tuple[int, str] | None:
    
    """
    page_tuple: (page_id, title, content)
    """
    
    page_id, title, description = page_tuple
    description = (description or "")[:max_input_tokens]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct:novita",
                messages=[
                    {"role": "system",
                        "content": (
                            "Explain concepts clearly for children around 10 years old. "
                            "Use simple words, short sentences, and concrete examples. "
                            "Avoid technical terms unless explained. "
                            "No titles or meta commentary. Output only the explanation."
                        ),
                    },
                    {"role": "user",
                        "content": f"Explain this for a 10-year-old:\n\n"
                                   f"topic: {title}\n"
                                   f"description: {description}",
                    },
                ],
                
                max_completion_tokens=500,
            )

            kids_definition = completion.choices[0].message.content
            return (page_id, kids_definition)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(
                    "Kids generation FAILED for page '%s' after %d attempts: %s",
                    title,
                    max_retries,
                    e,
                )
                return None

            # exponential backoff
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_time)
            



# ----- fetching data from db for kid definition generation ------

# Fetching function to fetch all pages content to feed the llm 

def fetch4kidgen(db_path: str):

    logger.debug("Fetching pages for kids generation from DB.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # fetching logic: use first simple definitions (lower nb token + less complex text for a simple model)
    cur.execute("""
		SELECT
			p.id,
			p.title,
			CASE
				WHEN p.has_simple = 1 THEN s.content
				ELSE t.content
			END AS selected_content
		FROM pages p
		LEFT JOIN definitions s ON p.id = s.page_id AND s.kind = 'simple'
		LEFT JOIN definitions t ON p.id = t.page_id AND t.kind = 'technical'
		WHERE (p.has_simple = 1 OR p.has_technical = 1) AND p.has_kids = 0;
	""")

    data4gen = cur.fetchall()

    logger.info("Fetched %d pages for kids generation.", len(data4gen))

    conn.close()
    
    return data4gen



# STORING FUNCTION

def store_kids(kid_defs: list, db_path: str) -> None:
    """
    kid_defs: List of tuples (page_id: int, kids_definition: str)
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        for page_id, kid_def in kid_defs:
            if kid_def:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO definitions
                    (page_id, kind, content, source, created_at)
                    VALUES (?, 'kids', ?, 'LLM_generated', ?)
                    """,
                    (page_id, kid_def, datetime.now(timezone.utc).isoformat()),
                )
                cur.execute(
                    "UPDATE pages SET has_kids = 1 WHERE id=?",
                    (page_id,),
                )

        conn.commit()
        logger.debug("Storing %d kids definitions in DB.", len(kid_defs))

    except sqlite3.Error:
        conn.rollback()
        logger.exception("Failed storing kids definitions")
        raise

    finally: conn.close()
    
    


# -------------------- FULL FUNCTION --------------------


def generate_n_populate_kid_def(
    db_path: str,
    client,
    max_input_tokens: int,
    max_workers: int = 5,
    max_batch_retries: int = 10,
    batch_retry_wait: int = 20,  # seconds
) -> None:
    """
    Generate 'kids' definitions from existing wiki pages in DB using an LLM.
    Implements per-page exponential backoff AND batch-level retries for intermittent free-tier quota issues.

    Stores successes immediately, retries only remaining pages.
    """

    # Fetch pages that still need kid definitions
    remaining_pages = fetch4kidgen(db_path=db_path)
    total_pages = len(remaining_pages)

    if not remaining_pages:
        logger.info("No pages to generate kids definitions for - DB might be already fully populated")
        return

    logger.info("Starting kids definition generation for %d pages", total_pages)

    # Batch-level retries
    for batch_attempt in range(max_batch_retries):
        if not remaining_pages:
            logger.info("All pages generated successfully.")
            break

        logger.info("Batch attempt %d/%d — %d pages remaining", 
                    batch_attempt + 1, max_batch_retries, len(remaining_pages))

        worker = partial(
            generate_kids_definition,
            client=client,
            max_input_tokens=max_input_tokens,
        )

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(
                tqdm(
                    executor.map(worker, remaining_pages),
                    total=len(remaining_pages),
                    desc=f"Batch {batch_attempt + 1} generating"
                )
            )

        # Filter out failed generations (None)
        results = [r for r in batch_results if r is not None]

        # Store successful generations immediately
        if results:
            store_kids(results, db_path=db_path)
            logger.info("Stored %d kids definitions in DB this batch", len(results))

        # Determine remaining pages that still need generation
        successful_ids = {r[0] for r in results}
        remaining_pages = [p for p in remaining_pages if p[0] not in successful_ids]

        if remaining_pages:
            logger.warning(
                "%d pages failed this batch.",
                len(remaining_pages)
            )
            # Only sleep if another batch retry is coming
            if batch_attempt < max_batch_retries - 1:
                logger.info(
                    "Batch n°%d done, will retry after %ds",
                    batch_attempt + 1,
                    batch_retry_wait
                )
                time.sleep(batch_retry_wait)


    # Final report
    if remaining_pages:
        logger.error(
            "Kids definition generation incomplete: %d/%d pages failed after %d batch retries",
            len(remaining_pages),
            total_pages,
            max_batch_retries
        )
    else:
        logger.info("Kids definition generation complete: all %d pages generated successfully", total_pages)
