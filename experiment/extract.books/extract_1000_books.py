import gzip
import json
import random
import os
import sys
import pandas as pd

BOOKS_FILE = "/mnt/d/Data voor master vakken/GenAI/goodreads_books.json.gz"
AUTHORS_FILE = "/mnt/d/Data voor master vakken/GenAI/goodreads_book_authors.json.gz"
OUTPUT_XLSX = "goodreads_1000_books_clean.xlsx"
NEEDED = 1000

def sanitize_excel(t):
    if isinstance(t, str):
        return "".join(c for c in t if (ord(c) > 31 or ord(c) in (9, 10, 13)))
    return t

def list_to_str(v):
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x)
    return v

def to_int(v):
    try:
        if v is None:
            return None
        return int(float(v))
    except:
        return None

def to_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except:
        return None

def first_non_null(*vals):
    for v in vals:
        if v is not None and v != "" and v != [] and v != {}:
            return v
    return None

def extract_year(book: dict):
    y = first_non_null(
        book.get("publication_year"),
        book.get("original_publication_year"),
        book.get("first_publish_year"),
        book.get("published_year"),
        book.get("year"),
    )
    if y is None:
        p = book.get("published")
        if isinstance(p, str):
            import re
            m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", p)
            if m:
                y = m.group(1)
    return to_int(y)

def extract_genres(book: dict):
    g = book.get("genres")
    if isinstance(g, list) and g:
        return [str(x) for x in g if x]
    if isinstance(g, str) and g.strip():
        return [x.strip() for x in g.split(",") if x.strip()]

    shelves = book.get("popular_shelves") or book.get("shelves")
    if isinstance(shelves, list):
        names = []
        for s in shelves:
            if isinstance(s, dict):
                name = s.get("name") or s.get("shelf") or s.get("genre")
                if name:
                    names.append(str(name))
            elif isinstance(s, str) and s.strip():
                names.append(s.strip())

        noise = {"to-read", "currently-reading", "read", "owned", "default"}
        names = [n for n in names if n.lower() not in noise]

        seen = set()
        out = []
        for n in names:
            k = n.lower()
            if k not in seen:
                seen.add(k)
                out.append(n)
            if len(out) >= 8:
                break
        return out if out else None

    return None

def load_author_id_to_name(authors_file: str):
    mapping = {}
    with gzip.open(authors_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                a = json.loads(line)
            except:
                continue
            aid = a.get("author_id") or a.get("id")
            name = a.get("name") or a.get("author_name") or a.get("full_name") or a.get("fullname")
            if aid is not None and isinstance(name, str) and name.strip():
                mapping[str(aid)] = name.strip()
    return mapping

def extract_author_names_from_book(book: dict, author_map: dict):
    raw = book.get("authors")
    if not isinstance(raw, list) or not raw:
        return None

    names = []
    for entry in raw:
        if isinstance(entry, dict):
            aid = entry.get("author_id") or entry.get("id")
            if aid is None:
                continue
            aid = str(aid)
            nm = author_map.get(aid)
            names.append(nm if nm else f"author_id:{aid}")
        elif isinstance(entry, str) and entry.strip():
            aid = entry.strip()
            names.append(author_map.get(aid, f"author_id:{aid}"))

    seen = set()
    out = []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)

    return ", ".join(out) if out else None

def is_valid_row(row, required_columns):
    for col in required_columns:
        val = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        if isinstance(val, str) and val.strip() == "":
            return False
    return True

for path, label in [(BOOKS_FILE, "BOOKS_FILE"), (AUTHORS_FILE, "AUTHORS_FILE")]:
    if not os.path.exists(path):
        print(f"{label} niet gevonden: {path}")
        sys.exit(1)

author_map = load_author_id_to_name(AUTHORS_FILE)

selected = []
skip_lines = random.randint(0, 100_000)

with gzip.open(BOOKS_FILE, "rt", encoding="utf-8") as f:
    for _ in range(skip_lines):
        try:
            next(f)
        except StopIteration:
            break

    for line in f:
        if len(selected) >= NEEDED:
            break

        try:
            book = json.loads(line)
        except:
            continue

        desc = book.get("description")
        title = book.get("title")
        language = book.get("language_code")

        if (
            language == "eng"
            and title
            and desc
            and isinstance(desc, str)
            and len(desc) > 30
        ):
            author = extract_author_names_from_book(book, author_map)

            selected.append({
                "book_id": book.get("book_id") or book.get("id"),
                "title": title,
                "author": author,
                "publication_year": extract_year(book),
                "ratings_count": to_int(book.get("ratings_count")),
                "avg_rating": to_float(book.get("average_rating")),
                "summary_original": desc,
                "genres": extract_genres(book),
                "summary_enriched": None,
            })

if not selected:
    print("Geen boeken gevonden.")
    sys.exit(1)

df = pd.DataFrame(selected)

excel_df = pd.DataFrame({
    "source_id": df["book_id"],
    "item_type": "Book",
    "name": df["title"],
    "vote_count": df["ratings_count"],
    "vote_average": df["avg_rating"],
    "source_overview": df["summary_original"],
    "Year": df["publication_year"],
    "source_genres": df["genres"].apply(list_to_str),
    "Simplified genre": None,
    "created_by / director / author": df["author"],
    "enriched_description": df["summary_enriched"],
})

for col in excel_df.select_dtypes(include=["object"]).columns:
    excel_df[col] = excel_df[col].apply(sanitize_excel)

allowed_empty = {"Simplified genre", "enriched_description"}
required_columns = [c for c in excel_df.columns if c not in allowed_empty]

excel_df = excel_df[excel_df.apply(lambda r: is_valid_row(r, required_columns), axis=1)].reset_index(drop=True)
excel_df.to_excel(OUTPUT_XLSX, index=False)

print(f"Geschreven: {OUTPUT_XLSX} (rows={len(excel_df)})")