from pathlib import Path
from typing import List


def load_seed_urls(file_path: str) -> List[str]:
    path = Path(file_path)


    if not path.exists():
        return []

    with path.open("r") as f:
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    return urls
