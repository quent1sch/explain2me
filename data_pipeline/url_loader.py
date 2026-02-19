from pathlib import Path
from typing import List
import logging


logger = logging.getLogger(__name__)

def load_seed_urls(file_path: str) -> List[str]:
    path = Path(file_path)

    logger.debug("Loading seed URLs from %s", path.resolve())

    if not path.exists():
        logger.error("Seed URL file not found: %s", path.resolve())
        raise FileNotFoundError(f"Seed URL file not found: {path}")
    
    try:
        with path.open("r") as f:
            urls = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    
    except OSError:
        logger.exception("Failed reading seed URL file: %s", path.resolve())
        raise

    logger.info("Loaded %d seed URLs.", len(urls))

    return urls
