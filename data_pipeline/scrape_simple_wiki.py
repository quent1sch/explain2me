# ------- IMPORT LIBRARIES -------

from bs4 import BeautifulSoup, Tag
import requests
import re
import logging

logger = logging.getLogger(__name__)



# -------------------------------------------------------------------------------------
#
# -------------------------------------- HELPERS --------------------------------------
#
# -------------------------------------------------------------------------------------



def clean_paragraph_simple(el: Tag):
    """
    Clean paragraph text, preserving formulas as LaTeX.
    Handles <p>, <li>, <dd>, <ul>, <ol> elements.
    """

    # Remove citation superscripts
    for sup in el.find_all("sup"):
        sup.decompose()

    # Replace <math> elements with LaTeX
    for math in el.find_all("math"):
        latex = math.get("alttext") or "".join(math.strings).strip()
        math.replace_with(f"${latex.strip()}$")

    # Handle lists
    if el.name in ["ul", "ol"]:
        items = []
        for i, li in enumerate(el.find_all("li", recursive=False), start=1):
            li_text = clean_paragraph_simple(li)
            if li_text:
                items.append(f"- {li_text}" if el.name == "ul" else f"{i}) {li_text}")
        return "\n".join(items)

    # Handle list items and description items
    if el.name in ["li", "dd"]:
        parts = []
        for child in el.children:
            if isinstance(child, Tag):
                parts.append(clean_paragraph_simple(child))
            else:
                parts.append(str(child))
        text = " ".join(filter(None, parts))
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        return text.strip()

    # Default text
    text = el.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


# --- Recursive content iterator ---
def iter_content_elements(el):
    """
    Yield all relevant content elements in document order.
    Skip navboxes, tables, scripts, styles.
    """
    for child in el.children:
        if not isinstance(child, Tag):
            continue

        if child.name in ["table", "script", "style"]:
            continue

        # Skip sideboxes, navboxes, metadata
        if child.name == "div":
            classes = child.get("class") or []
            if any(c in ["navbox", "vertical-navbox", "metadata", "mbox"] for c in classes):
                continue
            yield from iter_content_elements(child)
            continue
        
        # Skip geo/coordinates spans
        if child.name == "span" and any(c in ["geo", "coordinates"] for c in (child.get("class") or [])):
            continue

		# Recursively yield from spans (other inline containers)
        if child.name == "span":
            yield from iter_content_elements(child)
            continue

        # Yield headings and paragraph-like content
        if child.name in ["p", "ul", "ol", "dd"] + [f"h{i}" for i in range(2, 7)]:
            yield child
        else:
            yield from iter_content_elements(child)






# ------------------------------------------------------------------------------------
# 
# ----------------------------------- MAIN SCRAPER -----------------------------------
#
# ------------------------------------------------------------------------------------


def scrape_simple_wiki(url):
    """
    Scrapes a Simple Wikipedia page and returns structured article data.
    Handles redirects, missing content, and network errors.
    """
    
    headers = {
        "User-Agent": "ReverseMentorBot/0.1 (https://yourdomain.com/contact)"
    }

    stop_sections = {
        "references",
        "other websites",
        "related pages",
        "further reading",
        "external links",
        "see also",
    }

    # --- Fetch page with network error handling ---
    try:
        res = requests.get(url, headers=headers, timeout=10)
        # Raises for 4xx/5xx responses
        res.raise_for_status()

    except requests.Timeout as e:
        logger.warning("Timeout fetching URL: %s", url)
        raise
    except requests.RequestException as e:
        logger.error("HTTP/network error fetching URL: %s | %s", url, e)
        raise


    soup = BeautifulSoup(res.text, "html.parser")

    # --- Handle redirects ---
    redirect_div = soup.find("div", class_="redirectMsg")
    if redirect_div and redirect_div.find("a"):
        redirect_url = "https://simple.wikipedia.org" + redirect_div.find("a")["href"]
        logger.debug("Redirect detected: %s -> %s", url, redirect_url)
        return scrape_simple_wiki(redirect_url)


    # --- Main content ---
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        # Page layout changed or empty page
        logger.error("Main content div not found for URL: %s", url)
        raise ValueError(f"Main content not found for {url}")

    # --- Article title with fallback ---
    title_tag = soup.find("h1", id="firstHeading")
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        # fallback: use last segment of URL
        title = url.split("/")[-1].replace("_", " ")

    article_data = {
        "url": url,
        "title": title,
        "sections": [],
        "categories": [],
        "category_urls": [],
    }

    # --- Introduction section ---
    intro_section = {"heading": "Introduction", "paragraphs": []}
    current_section = intro_section

    # --- Walk content recursively ---
    for el in iter_content_elements(content):
        # Headings start new sections
        if el.name.startswith("h"):
            heading = el.get_text(" ", strip=True).replace("[edit]", "")
            if heading.lower() in stop_sections:
                break
            if intro_section["paragraphs"] and intro_section not in article_data["sections"]:
                article_data["sections"].append(intro_section)
            current_section = {"heading": heading, "paragraphs": []}
            article_data["sections"].append(current_section)
            continue

        # Paragraph-like content
        if el.name in ["p", "ul", "ol", "dd"]:
            text = clean_paragraph_simple(el)
            if text:
                current_section["paragraphs"].append(text)

    # --- Ensure intro is included if it has paragraphs ---
    if intro_section["paragraphs"] and intro_section not in article_data["sections"]:
        article_data["sections"].insert(0, intro_section)

    # --- Categories ---
    catlinks = soup.select("#mw-normal-catlinks ul li a")
    for cat in catlinks:
        article_data["categories"].append(cat.get_text(strip=True))
        href = cat.get("href")
        if href and href.startswith("/wiki/"):
            article_data["category_urls"].append("https://simple.wikipedia.org" + href)

    # --- Ensure at least one section exists ---
    if not article_data["sections"]:
        logger.warning("No sections extracted for URL: %s", url)
        raise ValueError(f"No sections found for {url}")

    return article_data
