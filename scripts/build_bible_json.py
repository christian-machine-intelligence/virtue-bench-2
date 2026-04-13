#!/usr/bin/env python3
"""
One-time script to convert KJV text files (from kjv_books.zip) into a single
bundled JSON file for virtue-bench Bible injection.

Input format (per file):
    BOOKNAME
    1:1: In the beginning God created...
    1:2: And the earth was without form...

Output: data/bible_kjv.json matching the psalms_kjv.json structure.

Usage:
    # Unzip kjv_books.zip first, then:
    python scripts/build_bible_json.py <directory_of_txt_files>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Map filename stems to the 3-letter API IDs used by bible.py
FILENAME_TO_ID = {
    "01-genesis": "GEN", "02-exodus": "EXO", "03-leviticus": "LEV",
    "04-numbers": "NUM", "05-deuteronomy": "DEU", "06-joshua": "JOS",
    "07-judges": "JDG", "08-ruth": "RUT", "09-1samuel": "1SA",
    "10-2samuel": "2SA", "11-1kings": "1KI", "12-2kings": "2KI",
    "13-1chronicles": "1CH", "14-2chronicles": "2CH", "15-ezra": "EZR",
    "16-nehemiah": "NEH", "17-esther": "EST", "18-job": "JOB",
    "19-psalms": "PSA", "20-proverbs": "PRO", "21-ecclesiastes": "ECC",
    "22-songofsolomon": "SNG", "23-isaiah": "ISA", "24-jeremiah": "JER",
    "25-lamentations": "LAM", "26-ezekiel": "EZK", "27-daniel": "DAN",
    "28-hosea": "HOS", "29-joel": "JOL", "30-amos": "AMO",
    "31-obadiah": "OBA", "32-jonah": "JON", "33-micah": "MIC",
    "34-nahum": "NAM", "35-habakkuk": "HAB", "36-zephaniah": "ZEP",
    "37-haggai": "HAG", "38-zecharaiah": "ZEC", "39-malachi": "MAL",
    "40-matthew": "MAT", "41-mark": "MRK", "42-luke": "LUK",
    "43-john": "JHN", "44-acts": "ACT", "45-romans": "ROM",
    "46-1corinthians": "1CO", "47-2corinthians": "2CO",
    "48-galatians": "GAL", "49-ephesians": "EPH", "50-philippians": "PHP",
    "51-colossians": "COL", "52-1thessalonians": "1TH",
    "53-2thessalonians": "2TH", "54-1timothy": "1TI", "55-2timothy": "2TI",
    "56-titus": "TIT", "57-philemon": "PHM", "58-hebrews": "HEB",
    "59-james": "JAS", "60-1peter": "1PE", "61-2peter": "2PE",
    "62-1john": "1JN", "63-2john": "2JN", "64-3john": "3JN",
    "65-jude": "JUD", "66-revelation": "REV",
}

VERSE_RE = re.compile(r"^(\d+):(\d+):\s*(.+)$")


def parse_book_file(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    book_name = lines[0].strip()

    chapters: dict[int, list[dict]] = {}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        m = VERSE_RE.match(line)
        if not m:
            continue
        ch, vs, text = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        chapters.setdefault(ch, []).append({"verse": vs, "text": text})

    stem = path.stem
    book_id = FILENAME_TO_ID.get(stem, stem.upper())

    return {
        "id": book_id,
        "book": book_name.title(),
        "chapters": [
            {"chapter": ch, "verses": chapters[ch]}
            for ch in sorted(chapters)
        ],
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory_of_txt_files>")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    if not src_dir.is_dir():
        print(f"Error: {src_dir} is not a directory")
        sys.exit(1)

    txt_files = sorted(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {src_dir}")
        sys.exit(1)

    books = [parse_book_file(f) for f in txt_files]
    result = {"translation": "KJV", "books": books}

    out_path = Path(__file__).parent.parent / "data" / "bible_kjv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    total_verses = sum(
        len(ch["verses"]) for b in books for ch in b["chapters"]
    )
    print(f"Wrote {len(books)} books, {total_verses} verses to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
