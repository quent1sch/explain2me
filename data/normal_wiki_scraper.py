import json
from bs4 import BeautifulSoup, Tag
import requests
import re
from display_helpers import pretty_print_page
from random import choice 
from concurrent.futures import ThreadPoolExecutor
from data_helpers import load_cache, save_cache, merge_article_into_cache
import time
import os
 
os.chdir('/home/schmi/projects/explain2me/data')
#----------------------------- 
# Load stored data
simple_wiki_data_path = "simple_wiki_raw_data.json"

with open(simple_wiki_data_path,"r") as file:
    simple_wiki_data = json.load(file)

len(simple_wiki_data), simple_wiki_data[0]


#-----------------------------
# retrieve simple wikipedia page titles & urls
page_titles = [page.get('title') for page in simple_wiki_data]
simple_page_urls = [page.get('url') for page in simple_wiki_data]


#-----------------------------
# (try to) convert simple wiki page url to its classic wiki page url 
# counterpart
# Goal -> have a normal wikipedia page for each simple wikipedia page 
#         already stored

def simple2normalwiki_url(simple_url):
    page_url_segment = simple_url.split("/")[-1]
    normalwiki_base = "https://en.wikipedia.org/wiki/"
    normalwiki_url = normalwiki_base + page_url_segment
    return normalwiki_url




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
def clean_paragraph(el: Tag) -> str:
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
            txt = clean_paragraph(li)
            if txt:
                lines.append(f"- {txt}" if el.name == "ul" else f"{i}) {txt}")
        return "\n".join(lines)

    # Text cleanup
    text = el.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def is_ignored(el: Tag) -> bool:
    """Ignore elements inside navboxes, infoboxes, thumbnails, TOC."""
    for parent in el.parents:
        classes = parent.get("class", [])
        if any(cls in IGNORE_CLASSES for cls in classes):
            return True
    return False


# ---------------- MAIN SCRAPER ----------------
def scrape_normal_wiki(url: str) -> dict:
    headers = {"User-Agent": "ReverseMentorBot/0.1"}
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        return {
            "url": url,
            "error": "page does not exist - no (normal) wiki page counterpart to this simple wiki page"
        }

    soup = BeautifulSoup(res.text, "html.parser")


    content = soup.find("div", id="mw-content-text")
    if not content:
        return {
            "url": url,
            "error": "content not found"
        }

    sections = []
    intro = None
    current = None

    # Traverse in DOM order
    for el in content.find_all(
        ["p", "li", "dd", "ul", "ol", "h2", "h3", "h4", "h5"],
        recursive=True
    ):
        if is_ignored(el):
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
        text = clean_paragraph(el)
        if not text:
            continue

        if current is None:
            if intro is None:
                intro = {"heading": "Introduction", "paragraphs": []}
                sections.insert(0, intro)
            intro["paragraphs"].append(text)
        else:
            current["paragraphs"].append(text)

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    return {
        "url": url,
        "title": title,
        "sections": sections
    }






# --------------------------------------------------------
# SCRAPING NORMAL WIKI PAGES FROM SIMPLE WIKI PAGES STORED
# --------------------------------------------------------

CACHE_FILE = "normal_wiki_raw_data.json"

def get_stored_simple_wiki():
    simple_wiki_data_path = "simple_wiki_raw_data.json"
    with open(simple_wiki_data_path,"r") as file:
            simple_wiki_data = json.load(file)
    return simple_wiki_data


# (try to) convert simple wiki page url to its classic wiki page url counterpart
def simple2normalwiki_url(simple_url):
    page_url_segment = simple_url.split("/")[-1]
    normalwiki_base = "https://en.wikipedia.org/wiki/"
    normalwiki_url = normalwiki_base + page_url_segment
    return normalwiki_url


def scrape_all_normal(max_workers=10, cache_file=None): 
    # get all stored simple wiki data
    simple_wiki_data = get_stored_simple_wiki()
    simple_urls = [page['url'] for page in simple_wiki_data]
    
	# (try to) convert simple wiki urls to normal wiki urls
    normal_urls = [simple2normalwiki_url(url) for url in simple_urls]
    
    t1 = time.perf_counter()
    
	# scrape normal wiki pages in parallel using normal wiki page urls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scraped_pages = list(executor.map(scrape_normal_wiki, normal_urls))
            t2 = time.perf_counter()

            if cache_file:
                cache = load_cache(cache_file)
                for page in scraped_pages:
                        merge_article_into_cache(
                            cache,
                            page,
                            issimple=False,
                            )
                save_cache(cache, cache_file)
            t3 = time.perf_counter()
            
    print(f"scraping time: {t2 - t1:.2f}s")
    print(f"storing time: {t3 - t2:.2f}s")
                    
    return scraped_pages


# normal_wiki_data = scrape_all_normal(max_workers=10, cache_file=CACHE_FILE)

# print(f"nb of scraped pages (+potentially stored): {len(normal_wiki_data)}")
# normal_wiki_data


# ------------------------------


# To add individual normal wiki page (not from simple wiki page store)

CACHE_FILE = "normal_wiki_raw_data.json"

def scrape_individual(url, CACHE_FILE=None):
    
	indiv_page = scrape_normal_wiki(url=url)
	if CACHE_FILE:
		with open("normal_wiki_raw_data.json","r") as file:
			stored_normal_wiki_data = json.load(file)

		issimple = 'simple.wikipedia.org' in url

		merge_article_into_cache(cache=stored_normal_wiki_data,
						   article_dict=indiv_page,
						   issimple=issimple)
		
		save_cache(cache=stored_normal_wiki_data, CACHE_FILE=CACHE_FILE)
	
	return indiv_page





# # Test from notebook





# # tony = scrape_individual(url='https://en.wikipedia.org/wiki/Tony_Stark_(Marvel_Cinematic_Universe)', CACHE_FILE=CACHE_FILE)
# # rob = scrape_individual('https://en.wikipedia.org/wiki/Robert_Downey_Jr.', CACHE_FILE=CACHE_FILE)


# # verify if new page stored correctly by doing lookup in the cache
# with open('normal_wiki_raw_data.json', 'r') as file:
# 	stored_normal_wiki_data = json.load(file)

# print(f'size of cache: {len(stored_normal_wiki_data)} pages')

# lookup_titles = ['Tony Stark', 'Robert Downey Jr']

# [page 
#  for page in stored_normal_wiki_data 
#  if any(title in page.get('title', '') for title in lookup_titles)]


# # ------------------------------



# with open("normal_wiki_raw_data.json","r") as file:
#     stored_normal_wiki_data = json.load(file)

# print(f"nb of stored pages: {len(stored_normal_wiki_data)}")

# [page for page in stored_normal_wiki_data if 'Robert Downey' in page.get('title', '')]


# # ------------------------------



# # Display page url that do not exist (inexistant mapping from simple wiki page url)

# content_not_found = [
#     page for page in stored_normal_wiki_data
#     if 'error' in page
# ]

# print(f"nb inexistant mapping from simple wiki pages: {len(content_not_found)}")
# content_not_found


# # ------------------------------


# rand_normal = choice(stored_normal_wiki_data)
# pretty_print_page(rand_normal)


# # ------------------------------


# from display_helpers import show_page_by_title

# # Pretty print page given a title

# show_page_by_title(stored_normal_wiki_data, "Bose–Einstein statistics")