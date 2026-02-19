# # ------- IMPORT LIBRARIES -------

import sqlite3
import logging



# # --------------------- SETUP DATABASE ------------------



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
        logger.exception("Failed to initialize database at %s", db_path)
        raise

    finally:
        conn.close()
