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
    client: OpenAI,
    max_input_tokens: int,
    max_workers: int = 5,
    ) -> None:

    data4gen = fetch4kidgen(db_path=db_path)
    
    if not data4gen:
        logger.info("No pages to generate kids definitions for - DB might be already fully populated")
        return

    worker = partial(
        generate_kids_definition,
        client=client,
        max_input_tokens=max_input_tokens,
    )

    total_pages = len(data4gen)
    logger.info("Generating kids definitions for %d pages.", total_pages)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(worker, data4gen),
                total=len(data4gen),
                desc="Generating"
            )
        )

    # Remove failed generations
    results = [r for r in results if r is not None]

    results_count = len(results)

    logger.info(
        "Kids generation complete — %d/%d successful.",
        results_count,
        total_pages,
    )

    store_kids(kid_defs=results, db_path=db_path)

    logger.info("Stored %d kids definitions in DB.", results_count)

