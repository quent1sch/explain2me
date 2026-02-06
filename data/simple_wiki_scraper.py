import requests
import re
import json
import os
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from data_helpers import load_cache, save_cache, merge_article_into_cache





def get_category_pages(category_url):

    headers = {
        "User-Agent": "YourBot/1.0 (https://example.com/contact)"
    }
    
    res = requests.get(category_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    # Extract category name from URL
    url_parse = urlparse(category_url)
    path = url_parse.path
    category_name = path.split(":")[-1]

    base = url_parse.scheme + "://" + url_parse.netloc # https://simple.wikipedia.org
    pages = []

    for li in soup.select("#mw-pages li a"):
        href = li.get("href")
        title = li.get_text(strip=True)
        if "Template:" in title:
            continue
        if href and href.startswith("/wiki/"):
            pages.append({
                "title": title,
                "url": base + href
            })

    return {
        "categories": category_name,
        "category_urls": category_url,
        "pages": pages
    }


# ------------------------------------------------------------------------



# --- Utility functions ---
def clean_spaces(text):
    return " ".join(text.split())

def clean_paragraph(el: Tag):
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
            li_text = clean_paragraph(li)
            if li_text:
                items.append(f"- {li_text}" if el.name == "ul" else f"{i}) {li_text}")
        return "\n".join(items)

    # Handle list items and description items
    if el.name in ["li", "dd"]:
        parts = []
        for child in el.children:
            if isinstance(child, Tag):
                parts.append(clean_paragraph(child))
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


# --- Scraper ---
def scrape_simple_wiki(url):
    headers = {
        "User-Agent": "ReverseMentorBot/0.1 (https://yourdomain.com/contact)"
    }

    stop_sections = {
        "references",
        "other websites",
        "related pages",
        "further reading",
        "external links",
        "See also",
    }

    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    # --- Handle redirects ---
    redirect_div = soup.find("div", class_="redirectMsg")
    if redirect_div and redirect_div.find("a"):
        redirect_url = "https://simple.wikipedia.org" + redirect_div.find("a")["href"]
        return {
            "url": url,
            "title": None,
            "sections": [],
            "redirect": redirect_url,
        }

    # --- Main content ---
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return {
            "url": url,
            "title": None,
            "sections": [],
            "error": "Main content not found",
        }

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
            text = clean_paragraph(el)
            if text:
                current_section["paragraphs"].append(text)


    # --- Ensure intro is included if it has paragraphs ---
    if intro_section["paragraphs"] and intro_section not in article_data["sections"]:
        article_data["sections"].insert(0, intro_section)

    # --- Categories ---
    for cat in soup.select("#mw-normal-catlinks ul li a"):
        article_data["categories"].append(cat.get_text(strip=True))
        article_data["category_urls"].append(
            "https://simple.wikipedia.org" + cat.get("href")
        )

    return article_data


# ------------------------------------------------------------------------


def scrape_category_parallel(category_url, max_workers=10):
    """
    Scrape all pages in a Simple Wikipedia category in parallel
    and store each page in the JSON cache with category info.
    """
    result = []

    category_data = get_category_pages(category_url)
    cat_title = category_data["categories"]
    cat_url = category_data["category_urls"]

    page_urls = [p["url"] for p in category_data["pages"]]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        scraped_pages = list(executor.map(scrape_simple_wiki, page_urls))

    for scraped in scraped_pages:
        entry = {
            "title": scraped["title"],
            "url": scraped["url"],
            "sections": scraped["sections"],
            "categories": [cat_title],
            "category_urls": [cat_url]
        }
        result.append(entry)

    return result



def scrape_all_pages(urls, max_workers=10):
    
	result = []

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		scraped_pages = list(executor.map(scrape_simple_wiki, urls))
	
	for scraped in scraped_pages:
		entry = {
            "title": scraped["title"],
            "url": scraped["url"],
            "sections": scraped["sections"],
			"categories": [],
            "category_urls": []
        }
		result.append(entry)
	
	return result




def scrape_all_categories(category_urls, cat_workers=5, page_workers=10):

    results = []

    def scrape_single_category(url):
        return scrape_category_parallel(url, max_workers=page_workers)

    with ThreadPoolExecutor(max_workers=cat_workers) as executor:
        category_results = list(executor.map(scrape_single_category, category_urls))

    # Flatten the list of lists
    for r in category_results:
        results.extend(r)

    return results



# ------------------------------------------------------------------------



from data_helpers import load_cache, save_cache, merge_article_into_cache

CACHE_FILE = "simple_wiki_raw_data.json"


# helpers
def is_page_url(x: str) -> bool:
    return isinstance(x, str) \
        and x.startswith("https://simple.wikipedia.org/wiki/") \
        and "Category:" not in x

def is_category_url(x: str) -> bool:
    return isinstance(x, str) \
        and x.startswith("https://simple.wikipedia.org/wiki/Category:")
    
def is_page_url_list(x) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(is_page_url(i) for i in x)

def is_category_url_list(x) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(is_category_url(i) for i in x)


# handler functions
def handle_page_url(url):
    return [scrape_simple_wiki(url)]

def handle_category_url(category_url, max_workers=10):
    return scrape_category_parallel(category_url, max_workers)

def handle_page_url_list(urls, max_workers=10):
    return scrape_all_pages(urls, max_workers)

def handle_category_url_list(category_urls, cat_workers=5, page_workers=10):
    return scrape_all_categories(category_urls, cat_workers, page_workers)

# handlers
handlers = [
    (is_page_url, handle_page_url),
    (is_category_url, handle_category_url),
    (is_page_url_list, handle_page_url_list),
    (is_category_url_list, handle_category_url_list),
]

# Main scraping function
def scrape_all_simple(x, cache_file=None):
    for predicate, handler in handlers:
        if predicate(x):
            result_pages = handler(x)
            if cache_file:
                cache = load_cache(cache_file)
                for page in result_pages:
                    merge_article_into_cache(cache=cache, 
                                             article_dict=page, 
                                             issimple=True)
                save_cache(cache, cache_file)
            return result_pages
                
    raise ValueError(f"Unknown input type: {x!r}")




# ------------------------------------------------------------------------



category_urls = ["https://simple.wikipedia.org/wiki/Category:Statistics",
                 "https://simple.wikipedia.org/wiki/Category:Artificial_intelligence",
                 "https://simple.wikipedia.org/wiki/Category:Probability_theory",
                 "https://simple.wikipedia.org/wiki/Category:Probability_distributions",
                 "https://simple.wikipedia.org/wiki/Category:Graph_theory",
                 "https://simple.wikipedia.org/wiki/Category:Theoretical_computer_science",
                 "https://simple.wikipedia.org/wiki/Category:Algorithms",
                 "https://simple.wikipedia.org/wiki/Category:Data_compression",
                 "https://simple.wikipedia.org/wiki/Category:Greedy_algorithms",
                 "https://simple.wikipedia.org/wiki/Category:Numerical_analysis",
                 "https://simple.wikipedia.org/wiki/Category:Randomised_algorithms",
                 "https://simple.wikipedia.org/wiki/Category:Searching_and_sorting_algorithms",
                 "https://simple.wikipedia.org/wiki/Category:Cryptography",
                 "https://simple.wikipedia.org/wiki/Category:Algebra",
                 "https://simple.wikipedia.org/wiki/Category:Geometry",
                 "https://simple.wikipedia.org/wiki/Category:Logic",
                 "https://simple.wikipedia.org/wiki/Category:Number_theory",
                 "https://simple.wikipedia.org/wiki/Category:Mathematical_analysis",
                 "https://simple.wikipedia.org/wiki/Category:Calculus",
                 "https://simple.wikipedia.org/wiki/Category:Game_theory",
                 ]
