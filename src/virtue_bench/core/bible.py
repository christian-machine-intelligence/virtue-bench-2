"""
Bible book injection for VirtueBench V2.

Fetches Bible text from the Free Use Bible API (bible.helloao.org)
for injection into system prompts. Supports multiple translations
and flexible book/chapter selection.

Translations are cached locally after first fetch to avoid repeated
API calls.

Usage:
    # CLI
    virtue-bench run --bible-book Romans --bible-translation eng_kjv
    virtue-bench run --bible-book "Matthew 5-7"  # Sermon on the Mount
    virtue-bench run --bible-book Proverbs --bible-translation BSB
    virtue-bench run --bible-books Romans,James,Proverbs

    # Programmatic
    from virtue_bench.core.bible import load_bible_text
    text = load_bible_text(books=["ROM"], translation="eng_kjv")
    text = load_bible_text(books=["MAT:5-7"], translation="BSB")
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Cache directory for downloaded Bible text
CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".bible_cache"

API_BASE = "https://bible.helloao.org/api"

# Supported translations with descriptions
TRANSLATIONS: Dict[str, str] = {
    "eng_kjv": "King James (Authorized) Version",
    "eng_kja": "King James Version + Apocrypha",
    "BSB": "Berean Standard Bible",
    "ENGWEBP": "World English Bible",
    "eng_web": "World English Bible Classic",
    "eng_webc": "World English Bible (Catholic)",
    "eng_asv": "American Standard Version (1901)",
    "eng_dra": "Douay-Rheims 1899",
    "eng_ylt": "Young's Literal Translation",
    "eng_lsv": "Literal Standard Version",
    "eng_msb": "Majority Standard Bible",
}

# Standard book ID mapping (common names -> API IDs)
BOOK_IDS: Dict[str, str] = {
    # Pentateuch
    "genesis": "GEN", "gen": "GEN",
    "exodus": "EXO", "exo": "EXO", "ex": "EXO",
    "leviticus": "LEV", "lev": "LEV",
    "numbers": "NUM", "num": "NUM",
    "deuteronomy": "DEU", "deu": "DEU", "deut": "DEU",
    # Historical
    "joshua": "JOS", "jos": "JOS", "josh": "JOS",
    "judges": "JDG", "jdg": "JDG", "judg": "JDG",
    "ruth": "RUT", "rut": "RUT",
    "1samuel": "1SA", "1sa": "1SA", "1sam": "1SA",
    "2samuel": "2SA", "2sa": "2SA", "2sam": "2SA",
    "1kings": "1KI", "1ki": "1KI",
    "2kings": "2KI", "2ki": "2KI",
    "1chronicles": "1CH", "1ch": "1CH", "1chr": "1CH",
    "2chronicles": "2CH", "2ch": "2CH", "2chr": "2CH",
    "ezra": "EZR", "ezr": "EZR",
    "nehemiah": "NEH", "neh": "NEH",
    "esther": "EST", "est": "EST",
    # Wisdom
    "job": "JOB",
    "psalms": "PSA", "psalm": "PSA", "psa": "PSA", "ps": "PSA",
    "proverbs": "PRO", "pro": "PRO", "prov": "PRO",
    "ecclesiastes": "ECC", "ecc": "ECC", "eccl": "ECC",
    "songofsolomon": "SNG", "sng": "SNG", "song": "SNG", "sos": "SNG",
    # Major Prophets
    "isaiah": "ISA", "isa": "ISA",
    "jeremiah": "JER", "jer": "JER",
    "lamentations": "LAM", "lam": "LAM",
    "ezekiel": "EZK", "ezk": "EZK", "eze": "EZK",
    "daniel": "DAN", "dan": "DAN",
    # Minor Prophets
    "hosea": "HOS", "hos": "HOS",
    "joel": "JOL", "jol": "JOL",
    "amos": "AMO", "amo": "AMO",
    "obadiah": "OBA", "oba": "OBA",
    "jonah": "JON", "jon": "JON",
    "micah": "MIC", "mic": "MIC",
    "nahum": "NAM", "nam": "NAM",
    "habakkuk": "HAB", "hab": "HAB",
    "zephaniah": "ZEP", "zep": "ZEP",
    "haggai": "HAG", "hag": "HAG",
    "zechariah": "ZEC", "zec": "ZEC",
    "malachi": "MAL", "mal": "MAL",
    # New Testament
    "matthew": "MAT", "mat": "MAT", "matt": "MAT",
    "mark": "MRK", "mrk": "MRK",
    "luke": "LUK", "luk": "LUK",
    "john": "JHN", "jhn": "JHN",
    "acts": "ACT", "act": "ACT",
    "romans": "ROM", "rom": "ROM",
    "1corinthians": "1CO", "1co": "1CO", "1cor": "1CO",
    "2corinthians": "2CO", "2co": "2CO", "2cor": "2CO",
    "galatians": "GAL", "gal": "GAL",
    "ephesians": "EPH", "eph": "EPH",
    "philippians": "PHP", "php": "PHP", "phil": "PHP",
    "colossians": "COL", "col": "COL",
    "1thessalonians": "1TH", "1th": "1TH", "1thess": "1TH",
    "2thessalonians": "2TH", "2th": "2TH", "2thess": "2TH",
    "1timothy": "1TI", "1ti": "1TI", "1tim": "1TI",
    "2timothy": "2TI", "2ti": "2TI", "2tim": "2TI",
    "titus": "TIT", "tit": "TIT",
    "philemon": "PHM", "phm": "PHM",
    "hebrews": "HEB", "heb": "HEB",
    "james": "JAS", "jas": "JAS",
    "1peter": "1PE", "1pe": "1PE", "1pet": "1PE",
    "2peter": "2PE", "2pe": "2PE", "2pet": "2PE",
    "1john": "1JN", "1jn": "1JN",
    "2john": "2JN", "2jn": "2JN",
    "3john": "3JN", "3jn": "3JN",
    "jude": "JUD", "jud": "JUD",
    "revelation": "REV", "rev": "REV",
}

# Named book collections for common injection patterns
BOOK_SETS: Dict[str, Dict] = {
    "gospels": {
        "books": ["MAT", "MRK", "LUK", "JHN"],
        "description": "The four Gospels (Matthew, Mark, Luke, John)",
    },
    "sermon_on_the_mount": {
        "books": ["MAT:5-7"],
        "description": "Sermon on the Mount (Matthew 5-7)",
    },
    "wisdom": {
        "books": ["PRO", "ECC", "JOB"],
        "description": "Wisdom literature (Proverbs, Ecclesiastes, Job)",
    },
    "proverbs": {
        "books": ["PRO"],
        "description": "Book of Proverbs",
    },
    "romans": {
        "books": ["ROM"],
        "description": "Paul's Epistle to the Romans",
    },
    "james": {
        "books": ["JAS"],
        "description": "Epistle of James (faith and works)",
    },
    "pastoral": {
        "books": ["1TI", "2TI", "TIT"],
        "description": "Pastoral epistles (1-2 Timothy, Titus)",
    },
    "johannine": {
        "books": ["JHN", "1JN", "2JN", "3JN"],
        "description": "Johannine writings (Gospel of John, 1-3 John)",
    },
    "torah": {
        "books": ["GEN", "EXO", "LEV", "NUM", "DEU"],
        "description": "The Torah / Pentateuch (Genesis through Deuteronomy)",
    },
    "prophets_major": {
        "books": ["ISA", "JER", "EZK", "DAN"],
        "description": "Major prophets (Isaiah, Jeremiah, Ezekiel, Daniel)",
    },
}


def resolve_book_id(name: str) -> str:
    """Resolve a book name or abbreviation to an API book ID."""
    clean = name.lower().replace(" ", "").replace(".", "")
    if clean in BOOK_IDS:
        return BOOK_IDS[clean]
    # Try as-is (might already be an API ID like "GEN")
    if name.upper() in {v for v in BOOK_IDS.values()}:
        return name.upper()
    raise ValueError(
        f"Unknown Bible book '{name}'. Use standard names (e.g., 'Genesis', 'Romans') "
        f"or abbreviations (e.g., 'GEN', 'ROM')."
    )


def parse_book_spec(spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Parse a book specification like 'ROM', 'MAT:5-7', 'Proverbs', 'Matthew 5-7'.

    Returns (book_id, start_chapter_or_None, end_chapter_or_None).
    """
    # Handle "Book:start-end" or "Book start-end"
    parts = spec.replace(":", " ").split()
    book_name = parts[0]
    book_id = resolve_book_id(book_name)

    start_ch = None
    end_ch = None

    if len(parts) > 1:
        ch_spec = parts[1]
        if "-" in ch_spec:
            start_str, end_str = ch_spec.split("-", 1)
            start_ch = int(start_str)
            end_ch = int(end_str)
        else:
            start_ch = int(ch_spec)
            end_ch = start_ch

    return book_id, start_ch, end_ch


def fetch_chapter(
    translation: str,
    book_id: str,
    chapter: int,
    cache: bool = True,
) -> List[str]:
    """Fetch a single chapter's verse texts from the API.

    Returns list of verse strings. Caches locally after first fetch.
    """
    cache_path = CACHE_DIR / translation / book_id / f"{chapter}.json"

    if cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    url = f"{API_BASE}/{translation}/{book_id}/{chapter}.json"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ValueError(f"API error fetching {url}: {e.code} {e.reason}")
    except Exception as e:
        raise ValueError(f"Error fetching {url}: {e}")

    verses = []
    for item in data.get("chapter", {}).get("content", []):
        if item.get("type") == "verse":
            # Content is a list of strings and formatting objects
            text_parts = [p for p in item.get("content", []) if isinstance(p, str)]
            verses.append(" ".join(text_parts).strip())

    # Cache
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(verses, f)

    return verses


def fetch_book_info(translation: str) -> Dict[str, Dict]:
    """Fetch book metadata (number of chapters, etc.) for a translation."""
    cache_path = CACHE_DIR / translation / "_books.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    url = f"{API_BASE}/{translation}/books.json"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Error fetching {url}: {e}")

    books = {}
    for book in data.get("books", []):
        books[book["id"]] = {
            "name": book["name"],
            "chapters": book["numberOfChapters"],
            "first_chapter": book.get("firstChapterNumber", 1),
        }

    if books:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(books, f)

    return books


def load_bible_text(
    books: Optional[List[str]] = None,
    book_set: Optional[str] = None,
    translation: str = "eng_kjv",
    cache: bool = True,
) -> str:
    """Load Bible text for injection into system prompts.

    Args:
        books: List of book specs (e.g., ["ROM", "MAT:5-7", "Proverbs"])
        book_set: Named collection (e.g., "sermon_on_the_mount", "wisdom")
        translation: Translation ID (default: eng_kjv)

    Returns:
        Formatted Bible text ready for system prompt injection.
    """
    specs = []

    if book_set:
        if book_set not in BOOK_SETS:
            raise ValueError(
                f"Unknown book set '{book_set}'. "
                f"Choose from: {list(BOOK_SETS.keys())}"
            )
        specs.extend(BOOK_SETS[book_set]["books"])

    if books:
        specs.extend(books)

    if not specs:
        raise ValueError("No books specified. Use books or book_set parameter.")

    # Fetch book info for chapter counts
    book_info = fetch_book_info(translation)

    parts = []
    for spec in specs:
        book_id, start_ch, end_ch = parse_book_spec(spec)

        if book_id not in book_info:
            raise ValueError(
                f"Book '{book_id}' not found in translation '{translation}'. "
                f"Available: {list(book_info.keys())}"
            )

        info = book_info[book_id]
        book_name = info["name"]
        total_chapters = info["chapters"]
        first_ch = info.get("first_chapter", 1)

        if start_ch is None:
            start_ch = first_ch
            end_ch = total_chapters

        for ch in range(start_ch, end_ch + 1):
            verses = fetch_chapter(translation, book_id, ch, cache=cache)
            if verses:
                header = f"{book_name} {ch}"
                text = " ".join(f"{i+1}. {v}" for i, v in enumerate(verses))
                parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def list_translations() -> Dict[str, str]:
    """Return available translations with descriptions."""
    return dict(TRANSLATIONS)


def list_book_sets() -> Dict[str, str]:
    """Return available named book collections."""
    return {name: info["description"] for name, info in BOOK_SETS.items()}
