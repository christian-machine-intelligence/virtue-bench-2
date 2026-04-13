"""
Bible book injection for VirtueBench V2.

Loads KJV Bible text from a bundled local JSON file for injection into
system prompts. No network calls required.

Usage:
    # CLI
    virtue-bench run --bible Romans
    virtue-bench run --bible "Matthew 5-7"  # Sermon on the Mount
    virtue-bench run --bible Romans --bible James  # Multiple books
    virtue-bench run --bible-set sermon_on_the_mount

    # Programmatic
    from virtue_bench.core.bible import load_bible_text
    text = load_bible_text(books=["ROM"])
    text = load_bible_text(books=["MAT:5-7"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Bundled KJV data file
_DATA_FILE = Path(__file__).parent.parent.parent.parent / "data" / "bible_kjv.json"

# Lazy-loaded Bible data
_bible_data: Optional[dict] = None


def _load_data() -> dict:
    """Load and cache the bundled Bible JSON."""
    global _bible_data
    if _bible_data is None:
        with open(_DATA_FILE, encoding="utf-8") as f:
            _bible_data = json.load(f)
    return _bible_data


# Standard book ID mapping (common names -> 3-letter IDs)
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
    """Resolve a book name or abbreviation to a 3-letter book ID."""
    clean = name.lower().replace(" ", "").replace(".", "")
    if clean in BOOK_IDS:
        return BOOK_IDS[clean]
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


def load_bible_text(
    books: Optional[List[str]] = None,
    book_set: Optional[str] = None,
) -> str:
    """Load Bible text for injection into system prompts.

    Args:
        books: List of book specs (e.g., ["ROM", "MAT:5-7", "Proverbs"])
        book_set: Named collection (e.g., "sermon_on_the_mount", "wisdom")

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

    data = _load_data()
    books_by_id = {b["id"]: b for b in data["books"]}

    parts = []
    for spec in specs:
        book_id, start_ch, end_ch = parse_book_spec(spec)

        if book_id not in books_by_id:
            raise ValueError(
                f"Book '{book_id}' not found in bundled KJV data. "
                f"Available: {list(books_by_id.keys())}"
            )

        book = books_by_id[book_id]
        book_name = book["book"]
        chapters = book["chapters"]

        if start_ch is None:
            start_ch = chapters[0]["chapter"]
            end_ch = chapters[-1]["chapter"]

        for ch_data in chapters:
            ch_num = ch_data["chapter"]
            if ch_num < start_ch or ch_num > end_ch:
                continue
            verses = ch_data["verses"]
            if verses:
                header = f"{book_name} {ch_num}"
                text = " ".join(
                    f"{v['verse']}. {v['text']}" for v in verses
                )
                parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def list_book_sets() -> Dict[str, str]:
    """Return available named book collections."""
    return {name: info["description"] for name, info in BOOK_SETS.items()}
