import gzip
import json
import csv
import random
import os
import sys

INPUT_FILE = "goodreads_books.json.gz"
OUTPUT_JSON = "goodreads_1000_books_clean.json"
OUTPUT_CSV = "goodreads_1000_books_clean.csv"

# Check of het input bestand bestaat
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: Input file niet gevonden: {INPUT_FILE}")
    print("Zorg ervoor dat het pad naar goodreads_books.json.gz correct is.")
    sys.exit(1)

# Check of het een geldig gzip bestand is
if not INPUT_FILE.endswith('.gz'):
    print(f"ERROR: Input file moet een .gz bestand zijn: {INPUT_FILE}")
    sys.exit(1)

# Probeer het bestand te openen om te controleren of het geldig is
try:
    with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as test_file:
        # Probeer de eerste regel te lezen
        test_file.readline()
except Exception as e:
    print(f"ERROR: Kan input file niet openen of lezen: {INPUT_FILE}")
    print(f"Foutmelding: {e}")
    sys.exit(1)

print(f"Input file gevonden en geldig: {INPUT_FILE}")

needed = 1000
selected = []

print("Start: willekeurig deel van dataset overslaan...")

# Random skip tussen 0 en 2 miljoen regels
skip_lines = random.randint(0, 2_000_000)
print(f"Overslaan van {skip_lines} regels...")

with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as f:

    # Willekeurig skippen van regels
    for _ in range(skip_lines):
        try:
            next(f)
        except StopIteration:
            break

    print("Beginnen met verzamelen van bruikbare boeken...")

    for line in f:
        if len(selected) >= needed:
            break

        try:
            book = json.loads(line)
        except json.JSONDecodeError:
            continue

        desc = book.get("description")
        title = book.get("title")
        language = book.get("language_code")

        # Filter: bruikbare Engelse boeken met voldoende samenvatting
        if (
            language == "eng"
            and title
            and desc
            and isinstance(desc, str)
            and len(desc) > 50
        ):

            # ---- VEILIGE EXTRACTIE VAN AUTHORS ----
            authors_raw = book.get("authors")

            if isinstance(authors_raw, list):
                # Lijst kan strings of dicts bevatten
                if len(authors_raw) > 0 and isinstance(authors_raw[0], dict):
                    author = ", ".join([a.get("name") for a in authors_raw if a.get("name")])
                else:
                    author = ", ".join(authors_raw)
            else:
                author = authors_raw

            # ---- MAPPING NAAR JOUW DATABASESTRUCTUUR ----
            mapped = {

                # Identiteit
                "book_id": book.get("book_id"),
                "title": title,
                "author": author,
                "publication_year": book.get("publication_year"),
                "isbn": book.get("isbn"),
                "language": language,
                "series_name": book.get("series_name"),
                "series_number": book.get("series_number"),

                # Classificatie
                "genres": book.get("genres"),
                "categories": book.get("categories"),
                "audience": book.get("audience"),

                # Populariteit
                "avg_rating": book.get("average_rating"),
                "ratings_count": book.get("ratings_count"),
                "reviews_count": book.get("text_reviews_count"),
                "shelves": book.get("shelves"),

                # Tekstvelden
                "summary_original": desc,
                "summary_enriched": None,
                "reviews_selected": None,
                "full_text_combined": None,

                # Semantische verrijking
                "themes": None,
                "motifs": None,
                "tone": None,
                "writing_style": None,
                "character_types": None,
                "narrative_structure": None,

                # Embeddings
                "embedding_summary": None,
                "embedding_enriched": None,
                "embedding_final": None,
            }

            selected.append(mapped)

print(f"Gevonden bruikbare boeken: {len(selected)}")

# ------------------------
# Opslaan als JSON
# ------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

# ------------------------
# Opslaan als CSV
# ------------------------
headers = list(selected[0].keys())

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    for book in selected:
        row = []
        for key in headers:
            value = book.get(key)
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            row.append(value)
        writer.writerow(row)

print("Klaar! Je hebt nu een volledige dataset in de juiste structuur:")
print(f" - JSON: {OUTPUT_JSON}")
print(f" - CSV:  {OUTPUT_CSV}")
