import gzip
import json
import random
import os
import sys
import pandas as pd

BOOKS_FILE = "/mnt/d/Data voor master vakken/GenAI/goodreads_books.json.gz"
AUTHORS_FILE = "/mnt/d/Data voor master vakken/GenAI/goodreads_book_authors.json.gz"
OUTPUT_XLSX = "goodreads_10_each_simplified_genre.xlsx"

TARGET_PER_GENRE = 10
GENRES_TARGET = ["Sci-Fi", "Fantasy", "Thriller", "Drama", "Horror", "Comedy", "Romance"]

SIMPLIFIED_GENRE_KEYWORDS = {
    "Sci-Fi": ["science fiction", "sci-fi", "scifi", "sf", "space", "alien", "time travel", "dystopia", "cyberpunk"],
    "Fantasy": ["fantasy", "magic", "magical", "dragon", "wizard", "witch", "epic fantasy", "high fantasy", "sword", "myth"],
    "Thriller": ["thriller", "suspense", "crime", "mystery", "detective", "noir", "psychological thriller"],
    "Drama": ["drama", "literary", "realistic", "family", "life", "relationship", "historical", "character driven"],
    "Horror": ["horror", "scary", "terror", "ghost", "haunted", "vampire", "zombie", "supernatural horror"],
    "Comedy": ["comedy", "humor", "humour", "funny", "satire", "parody", "comic"],
    "Romance": ["romance", "romantic", "love", "love story", "chick lit", "relationship romance"],
}

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

def simplify_genre(source_genres: str | None):
    if not source_genres or not isinstance(source_genres, str):
        return None
    text = source_genres.lower()
    for simplified, keywords in SIMPLIFIED_GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return simplified
    return None

def is_missing(val):
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

def row_ok(row_dict):
    allowed_empty = {"enriched_description"}
    for k, v in row_dict.items():
        if k in allowed_empty:
            continue
        if is_missing(v):
            return False
    return True

for path, label in [(BOOKS_FILE, "BOOKS_FILE"), (AUTHORS_FILE, "AUTHORS_FILE")]:
    if not os.path.exists(path):
        print(f"{label} niet gevonden: {path}")
        sys.exit(1)

author_map = load_author_id_to_name(AUTHORS_FILE)

picked = {g: [] for g in GENRES_TARGET}

def all_genres_filled():
    return all(len(picked[g]) >= TARGET_PER_GENRE for g in GENRES_TARGET)

def try_add_book_to_bucket(book: dict):
    desc = book.get("description")
    title = book.get("title")
    language = book.get("language_code")

    if not (language == "eng" and title and desc and isinstance(desc, str) and len(desc) > 30):
        return

    author = extract_author_names_from_book(book, author_map)
    year = extract_year(book)
    ratings_count = to_int(book.get("ratings_count"))
    avg_rating = to_float(book.get("average_rating"))
    genres_list = extract_genres(book)
    source_genres = list_to_str(genres_list)
    simplified = simplify_genre(source_genres)

    if simplified not in picked:
        return
    if len(picked[simplified]) >= TARGET_PER_GENRE:
        return

    row = {
        "source_id": book.get("book_id") or book.get("id"),
        "item_type": "Book",
        "name": title,
        "vote_count": ratings_count,
        "vote_average": avg_rating,
        "source_overview": desc,
        "Year": year,
        "source_genres": source_genres,
        "Simplified genre": simplified,
        "created_by / director / author": author,
        "enriched_description": None,
    }

    if row_ok(row):
        picked[simplified].append(row)

max_passes = 3
current_pass = 0

while current_pass < max_passes and not all_genres_filled():
    current_pass += 1
    random_start_offset = random.randint(0, 100_000)

    with gzip.open(BOOKS_FILE, "rt", encoding="utf-8") as file:
        for _ in range(random_start_offset):
            try:
                next(file)
            except StopIteration:
                break

        for line in file:
            if all_genres_filled():
                break

            try:
                book = json.loads(line)
            except json.JSONDecodeError:
                continue

            try_add_book_to_bucket(book)

rows = []
for genre in GENRES_TARGET:
    rows.extend(picked[genre][:TARGET_PER_GENRE])

if not rows:
    print("Geen rijen gevonden die aan alle eisen voldoen.")
    sys.exit(1)

excel_df = pd.DataFrame(rows)

for col in excel_df.select_dtypes(include=["object"]).columns:
    excel_df[col] = excel_df[col].apply(sanitize_excel)

excel_df.to_excel(OUTPUT_XLSX, index=False)

counts = excel_df["Simplified genre"].value_counts().to_dict()
print(f"Geschreven: {OUTPUT_XLSX} (rows={len(excel_df)})")
print("Per genre:", {g: counts.get(g, 0) for g in GENRES_TARGET})
missing = {g: TARGET_PER_GENRE - counts.get(g, 0) for g in GENRES_TARGET}
if any(v > 0 for v in missing.values()):
    print("Niet gehaald (tekort):", {g: v for g, v in missing.items() if v > 0})