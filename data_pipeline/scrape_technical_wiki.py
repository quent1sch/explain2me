# ------- IMPORT LIBRARIES -------


from bs4 import BeautifulSoup, Tag
import requests
import re
import logging

logger = logging.getLogger(__name__)


# ----------------------------------------
# 			  SCRAPING METHOD
# ----------------------------------------

# ---------------- CONFIG ----------------
IGNORE_CLASSES = {
    "sidebar-list", "navbar", "infobox", "toc",
    "thumb", "mw-default-size", "metadata"
}

STOP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading"
}



# ---------------- HELPERS ----------------
def clean_paragraph_tech(el: Tag) -> str:
    """Clean paragraph text, preserving math as LaTeX."""

    # Remove citation markers
    for sup in el.find_all("sup"):
        sup.decompose()

    # Preserve math
    for math in el.find_all("math"):
        latex = math.get("alttext") or math.get_text(strip=True)
        latex = latex.strip()

        is_block = el.get_text(strip=True) == math.get_text(strip=True)
        math.replace_with(
            f"\n$$\n{latex}\n$$\n" if is_block else f"${latex}$"
        )

    # Lists
    if el.name in {"ul", "ol"}:
        lines = []
        for i, li in enumerate(el.find_all("li", recursive=False), start=1):
            txt = clean_paragraph_tech(li)
            if txt:
                lines.append(f"- {txt}" if el.name == "ul" else f"{i}) {txt}")
        return "\n".join(lines)

    # Text cleanup
    text = el.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def is_ignored_tech(el: Tag) -> bool:
    """Ignore elements inside navboxes, infoboxes, thumbnails, TOC."""
    for parent in el.parents:
        classes = parent.get("class", [])
        if any(cls in IGNORE_CLASSES for cls in classes):
            return True
    return False

 



# ---------------- MAIN SCRAPER ----------------
def scrape_normal_wiki(url: str) -> dict | None:
    headers = {"User-Agent": "ReverseMentorBot/0.1"}

    logger.info("Starting scrape_normal_wiki for URL: %s", url)
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 404:
            logger.info("Technical Wiki page not found (404): %s", url)
            return None

        # --- Other HTTP errors (500, 403, 429, etc.) ---
        if 400 <= res.status_code < 600:
            logger.warning(
                "HTTP error (%d) for URL: %s",
                res.status_code,
                url
            )
            return None         

        logger.info("Fetched URL successfully: %s", url)
		
    # catches all request-related failures: connection errors, timeouts, invalid URLs, and HTTP errors raised by raise_for_status()
	# i.e. network/environment-level failures, not parsing or scraper-logic errors.    
    except requests.Timeout:
        logger.warning("Timeout fetching URL: %s", url)
        return None
    except requests.ConnectionError:
        logger.warning("Connection error fetching URL: %s", url)
        return None
    except requests.RequestException as e:
        logger.exception("Unknown requests error for URL: %s", url)
        return None


    soup = BeautifulSoup(res.text, "html.parser")
    content = soup.find("div", id="mw-content-text")
    
    if content is None:
        logger.warning("Content div not found for URL: %s", url)
        return None

    sections = []
    intro = None
    current = None

    # Traverse in DOM order
    for el in content.find_all(
        ["p", "li", "dd", "ul", "ol", "h2", "h3", "h4", "h5"],
        recursive=True
    ):
        if is_ignored_tech(el):
            continue

        # ---------- HEADINGS ----------
        if el.name.startswith("h"):
            heading = el.get_text(" ", strip=True).replace("[edit]", "")
            if heading.lower() in STOP_SECTIONS:
                break

            current = {"heading": heading, "paragraphs": []}
            sections.append(current)
            continue

        # ---------- CONTENT ----------
        text = clean_paragraph_tech(el)
        if not text:
            logger.debug("Skipping empty element at %s", el)
            continue

        if current is None:
            if intro is None:
                intro = {"heading": "Introduction", "paragraphs": []}
                sections.insert(0, intro)
            intro["paragraphs"].append(text)
        else:
            current["paragraphs"].append(text)
     
    if not sections:
        logger.info("No sections extracted for URL: %s", url)
        return None

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None
    

	# --------- CATEGORY DATA ----------
    categories = []
    category_urls = []

    catlinks = soup.select("#mw-normal-catlinks ul li a")
    for cat in catlinks:
        categories.append(cat.get_text(strip=True))
        href = cat.get("href")
        if href and href.startswith("/wiki/"):
            category_urls.append("https://en.wikipedia.org" + href)
        else:
            logger.debug("Skipping non-standard category href: %s", href)
    

    logger.info("Scraped '%s' successfully: %d sections, %d categories",
                title or "Unknown",
                len(sections),
                len(categories)
            )


    return {
        "url": url,
        "title": title,
        "sections": sections,
        "categories": categories,
        "category_urls": category_urls,
    }



