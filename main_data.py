# ------- IMPORT LIBRARIES -------

import logging
import time

from config import Settings
from data_pipeline.db import init_db
from data_pipeline.backfill import backfill_DB
from data_pipeline.generate_kids import generate_n_populate_kid_def
from data_pipeline.scrape_wikipedia_global import scrape_wikipedia
from data_pipeline.url_loader import load_seed_urls




# ------- LOG CONFIG ----------
def configure_logging():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("toy_test_log.log")
    ],
    force=True
)


# ----------- MAIN ------------
def main():

    # logs
    configure_logging()
    logger = logging.getLogger(__name__)

    start_time = time.time()
    logger.info("Starting dataset pipeline")


    try:
        # get database path from config.py file
        db_path = Settings.get_db_path()
        # initialize database if needed
        init_db(db_path=db_path)


        client = Settings.create_client()
        max_tokens = Settings.MAX_INPUT_TOKENS

    except Exception:
        logger.exception("Critical initialization failure")
        return  # stop program early
    


    # ---------------- SCRAPING ----------------
    try:
        # Scrape & Store wikipedia pages from the url list (from any simple/normal wiki page/category URL)
        urls = load_seed_urls("data_pipeline/wiki_urls")
        print(urls)

        if urls:
            scrape_wikipedia(
                urls=urls,
                db_path=db_path,
                cat_workers=5
                )
    
    except Exception:
        logger.exception("Scraping stage failed — continuing pipeline")
    


    # ---------------- BACKFILL ----------------
    try:
        # Backfill normal/simple wiki pages in db
        # i.e. try to scrape the normal (or simple) wiki page corresponding to the simple (or normal) wiki page in db.
        backfill_DB(db_path=db_path)
    
    except Exception:
        logger.exception("Backfill stage failed — continuing")
    


    # ----------- GENERATE DEFs FOR KIDS -------------
    try: 
        # Generate definition for 10yo kids from the existing wiki pages in db using an LLM
        generate_n_populate_kid_def(
            db_path=db_path,
            client=client,
            max_input_tokens=max_tokens,
            max_workers=5,
        )

    except Exception:
        logger.exception("Generation of definitions for kid audience stage failed")

    logger.info("Dataset pipeline completed successfully")

    duration = time.time() - start_time
    logger.info("Total runtime: %.2f seconds", duration)


if __name__ == "__main__":
    main()
