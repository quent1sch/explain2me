"""
main_data.py

Unified entry point for dataset generation in the Explain2Me project.

This script orchestrates the full data pipeline, from scraping raw Wikipedia
content to producing a structured LoRA training dataset with multi-level
explanations (simple, normal, technical).

STAGES
------
- scrape   : Collect Wikipedia pages (supports both single pages and categories)
- backfill : Enrich dataset by retrieving missing simple/normal counterparts
- generate : Use an LLM to create kid-friendly (age ~10) explanations
- format   : Build instruction-style dataset for LoRA fine-tuning
- save     : Store dataset locally and optionally push to Hugging Face Hub

WORKFLOW
--------
Typical data pipeline execution:

    1. Load seed URLs
        → configurable list of Wikipedia pages or categories

    2. Scrape content
        → expands category URLs into pages and scrapes in parallel

    3. Backfill missing variants
        → ensures coverage across difficulty levels

    4. Generate simplified definitions
        → LLM creates child-friendly explanations

    5. Build training dataset
        → formats data into instruction-style JSON (system/user/assistant)

OUTPUT
------
The final dataset is a JSON file for LoRA instruction tuning, where each sample:

- includes a system prompt defining assistant behavior
- includes a user prompt with contextual signals (e.g., age, knowledge level)
- includes an assistant response corresponding to a difficulty level:
    • simple     (child-friendly)
    • normal     (general audience)
    • technical  (expert-level)

DATA SOURCES
------------
- Seed URLs are defined in `data_pipeline/wiki_urls`
- Can be customized to target specific domains (e.g., law, science, history)
- Wikipedia category URLs are supported and automatically expanded into pages,
  enabling efficient large-scale data collection

RESULTS
-------
- SQLite database populated with scraped and generated content
- Final LoRA training dataset saved locally
- Optional upload to Hugging Face Hub via configuration

CONFIGURATION
-------------
- Paths, API keys, and parameters are managed via `Settings`
- Database location, model client, and token limits are configurable

PORTABILITY
-----------
- Modular and fault-tolerant pipeline (each stage logs errors and continues)
- Designed for scalability (parallel scraping + LLM workers)
- Can be adapted to other data sources or domains with minimal changes

EXAMPLE
-------
Run the full data pipeline from the repository root:

    python main_data.py
"""



import logging
import time

from data_pipeline.data_config import Settings
from data_pipeline.backfill import backfill_DB
from data_pipeline.db import init_db
from data_pipeline.generate_kids import generate_n_populate_kid_def
from data_pipeline.scrape_wikipedia_global import scrape_wikipedia
from data_pipeline.url_loader import load_seed_urls
from data_pipeline.lora_train_data_formatting import build_lora_training_dataset, save_data




# ------- LOG CONFIG ----------
def configure_logging():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_pipeline_logs.log")
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
        return
    


    # ---------------- SCRAPING ----------------

    start_scrape_time = time.time()

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

    start_backfill_time = time.time()

    try:
        # Backfill normal/simple wiki pages in db
        # i.e. try to scrape the normal (or simple) wiki page corresponding to the simple (or normal) wiki page in db.
        backfill_DB(db_path=db_path)
    
    except Exception:
        logger.exception("Backfill stage failed — continuing")
    
 

    # ----------- GENERATE DEFs FOR KIDS -------------

    start_llm4kid_time = time.time()

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
    
    


    # ----------- FORMAT & SAVE DATA FOR LORA ADAPTER TRAINING -------------

    start_lora_train_data_formatting_time = time.time()

    try:
        dataset = build_lora_training_dataset(db_path)
        if dataset:
            save_data(dataset, Settings.get_training_data_path())
            logging.info(f"LoRA training dataset saved successfully with {len(dataset)} items.")

            # ---------------- PUSH TO HUB -----------------
            try:
                Settings.push_dataset_to_hub()
            except Exception as e:
                logging.error(f"Failed to push dataset to Hugging Face Hub: {e}", exc_info=True)

        else:
            logging.warning("No dataset generated; nothing to save.")

    except Exception as e:
        logging.error(f"Failed to build or save LoRA training dataset: {e}", exc_info=True)




    # ----------- DATA PIPELINE COMPLETED -----------


    logger.info("Dataset pipeline completed successfully")

    end_time = time.time()

    duration = end_time - start_time
    logger.info("Total runtime: %.2f seconds", duration)
    logger.info("Scraping runtime: %.2f seconds", start_backfill_time - start_scrape_time)
    logger.info("Backfill runtime: %.2f seconds", start_llm4kid_time - start_backfill_time)
    logger.info("Kid defs generation runtime: %.2f seconds", start_lora_train_data_formatting_time - start_llm4kid_time)
    logger.info("Data formatting for LoRA training (+ saving locally and to HF hub) runtime: %.2f seconds", end_time - start_lora_train_data_formatting_time)



if __name__ == "__main__":
    main()

