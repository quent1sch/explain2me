# # ------- IMPORT LIBRARIES -------

# import json
# from bs4 import BeautifulSoup, Tag
# import requests
# import re
# from concurrent.futures import ThreadPoolExecutor
# import time
# from datetime import datetime, timezone
# import os
# from urllib.parse import urlparse, unquote
# from typing import Union, List, Optional
# import sqlite3
# from random import sample, choices
# import pandas as pd
# from openai import OpenAI
# import random
# from tqdm import tqdm
# from functools import partial
# from typing import Tuple
# import logging

# from config import DATA_PATH



# # --------------------- SETUP DATABASE ------------------

# # set working directory
# os.chdir(DATA_PATH)

# # Connect to DB (create if does not exist)
# conn = sqlite3.connect(database="WikipediaOne.db")
# cur = conn.cursor()

# # create tables 'pages' and 'definitions'.
# # 'definitions' contains the actual page content.
# # 'pages' serves as a lookup table for 'definitions'
# cur.execute("""
#             CREATE TABLE IF NOT EXISTS pages (
# 			id INTEGER PRIMARY KEY,
# 			title TEXT UNIQUE NOT NULL,
# 			has_simple INTEGER DEFAULT 0,
# 			has_technical INTEGER DEFAULT 0,
# 			has_kids INTEGER DEFAULT 0
#             );
#             """)

# cur.execute("""
#             CREATE TABLE IF NOT EXISTS definitions (
# 			id INTEGER PRIMARY KEY,
# 			page_id INTEGER NOT NULL,
# 			kind TEXT CHECK(kind IN ('simple', 'technical', 'kids')),
# 			content TEXT NOT NULL,
# 			source TEXT,
# 			created_at TEXT,
# 			FOREIGN KEY (page_id) REFERENCES pages(id),
# 			UNIQUE (page_id, kind)
#             );
# 			""")

# conn.commit()
# conn.close()

import sqlite3
import logging


logger = logging.getLogger(__name__)


def init_db(db_path: str) -> None:
    """
    Initialize the SQLite database.
    Creates required tables if they do not exist.
    """

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        logger.info("Initializing database at %s", db_path)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            title TEXT UNIQUE NOT NULL,
            has_simple INTEGER DEFAULT 0,
            has_technical INTEGER DEFAULT 0,
            has_kids INTEGER DEFAULT 0
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS definitions (
            id INTEGER PRIMARY KEY,
            page_id INTEGER NOT NULL,
            kind TEXT CHECK(kind IN ('simple', 'technical', 'kids')),
            content TEXT NOT NULL,
            source TEXT,
            created_at TEXT,
            FOREIGN KEY (page_id) REFERENCES pages(id),
            UNIQUE (page_id, kind)
            );
        """)

        conn.commit()
        logger.info("Database initialized successfully (if did not already exist).")

    except sqlite3.Error:
        logger.exception("Failed to initialize database.")
        raise

    finally:
        conn.close()
