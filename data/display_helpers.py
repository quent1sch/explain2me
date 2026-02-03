from IPython.display import display, Markdown

# --- Pretty Print Functionality ---
def pretty_print_page(data):
    md = f"# {data['title']}\n\n"
    
    for section in data['sections']:
        md += f"## {section['heading']}\n\n"
        for para in section["paragraphs"]:
            md += para + "\n\n"
    md += "\n------------------------------------------------------------------"
    display(Markdown(md))

# --- Pretty print page given a title ---
def show_page_by_title(pages, title):
    """
    Lookup a page by title from a list of pages and pretty-print its sections.
    """
    page = next((p for p in pages if p.get("title") == title), None)
    if page is None:
        print(f"Page with title '{title}' not found!")
        return
    pretty_print_page(page)