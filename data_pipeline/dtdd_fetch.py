"""
dtdd_fetch.py - DTDD content warning fetcher for Film Helix
------------------------------------------------------------------
Fetches trigger/warning data from DoesTheDogDie.com for all valid
films in movies.db, matched by title + year with TMDB ID verification.

Strategy per film:
  1. Search DTDD by title → find result matching release year
  2. Fetch /media/{dtdd_id} → verify tmdbId, extract topic triggers
  3. Save as wide CSV (one column per trigger category)

Output: data/dtdd_warnings.csv (same format as dtdd_test_with_columns.csv)

Usage:
  python dtdd_fetch.py            #fetch all valid films in movies.db
  python dtdd_fetch.py --limit 500  #test run with first 500 films

Requires: pip install aiohttp
"""

import asyncio
import aiohttp
import csv
import json
import os
import sqlite3
import ssl
import certifi
import argparse
import time
from pathlib import Path

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

API_KEY       = "7320374a8bb9314662e669a073861f47"
BASE_URL      = "https://www.doesthedogdie.com"
HEADERS       = {"Accept": "application/json", "X-API-KEY": API_KEY}
OUTPUT_CSV    = "data/dtdd_warnings.csv"
PROGRESS_FILE = "data/dtdd_processed_tmdb_ids.txt"

CONCURRENT    = 5
DELAY         = 0.3
SAVE_EVERY    = 200
DB_PATH       = "movies.db"

#warning categories stored as (dtdd_topic_name_lowercase, csv_column_name) dtdd_topic_name must match what DTDD returns in topic["topic"]["name"].lower()
PRIORITY_WARNINGS = {
    "a dog dies",
    "a pet dies",
    "an animal dies",
    "animals are abused",
    "a kid dies",
    "a major character dies",
    "someone dies by suicide",
    "someone attempts suicide",
    "someone self harms",
    "someone is sexually assaulted",
    "someone is raped onscreen",
    "there's child abuse",
    "there's domestic violence",
    "there's pedophilia",
    "there's excessive gore",
    "there's blood/gore",
    "there are jump scares",
    "there's torture",
    "there's flashing lights or images",
    "someone uses drugs",
    "there's addiction",
    "the ending is sad",
    "there's body horror",
}

def load_valid_films(limit: int | None = None) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT id, title, SUBSTR(release_date, 1, 4) as year
        FROM movies
        WHERE is_valid = 1
          AND title IS NOT NULL
          AND release_date IS NOT NULL
        ORDER BY vote_average DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [{"tmdb_id": str(r[0]), "title": r[1], "year": str(r[2])} for r in rows]


def load_processed_ids() -> set[str]:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def mark_processed(tmdb_id: str):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{tmdb_id}\n")

#API
async def search_dtdd(session: aiohttp.ClientSession, title: str) -> list[dict]:
    """Search DTDD by title, return list of candidate items."""
    url = f"{BASE_URL}/dddsearch"
    params = {"q": title}
    for attempt in range(3):
        try:
            async with session.get(url, params=params, headers=HEADERS,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
                return data.get("items", [])
        except Exception:
            await asyncio.sleep(2 ** attempt)
    return []


async def fetch_dtdd_detail(session: aiohttp.ClientSession,
                            dtdd_id: int) -> dict | None:
    """Fetch full topic stats for a DTDD item."""
    url = f"{BASE_URL}/media/{dtdd_id}"
    for attempt in range(3):
        try:
            async with session.get(url, headers=HEADERS,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception:
            await asyncio.sleep(2 ** attempt)
    return None


#match, extract
async def process_film(session: aiohttp.ClientSession,
                       semaphore: asyncio.Semaphore,
                       film: dict) -> dict | None:
    """
    Search for a film, verify match, extract trigger data.
    Returns a flat dict with tmdb_id + all topic columns. Returns None if no match.
    """
    tmdb_id = film["tmdb_id"]
    title   = film["title"]
    year    = film["year"]

    async with semaphore:
        candidates = await search_dtdd(session, title)
        await asyncio.sleep(DELAY)

        if not candidates:
            return None

        #find best candidate: year must match, prefer exact title match
        year_matches = [c for c in candidates
                        if str(c.get("releaseYear", "")) == year]
        if not year_matches:
            #fallback within 1 year for release date discrepancies
            try:
                year_int = int(year)
                year_matches = [c for c in candidates
                                if abs(int(c.get("releaseYear", 0) or 0) - year_int) <= 1]
            except ValueError:
                pass

        if not year_matches:
            return None

        #pick best title: exact title match wins, otherwise first year match
        title_lower = title.lower()
        exact = [c for c in year_matches
                 if c.get("name", "").lower() == title_lower]
        candidate = exact[0] if exact else year_matches[0]
        dtdd_id = candidate["id"]

        detail = await fetch_dtdd_detail(session, dtdd_id)
        await asyncio.sleep(DELAY)

        if not detail:
            return None

        item = detail.get("item", {})

        #verify TMDB ID if available to prevent film mismatches
        detail_tmdb = str(item.get("tmdbId") or "")
        if detail_tmdb and detail_tmdb != tmdb_id:
            return None  # Wrong film — skip

        #extract topics: yesSum > noSum → "yes", else "no"
        topics = detail.get("topicItemStats", [])
        triggers = {}
        for t in topics:
            name = t.get("topic", {}).get("name", "").strip()
            if not name:
                continue
            col = name.lower().replace(" ", "_").replace("/", "_").replace("'", "")
            value = "yes" if t.get("yesSum", 0) > t.get("noSum", 0) else "no"
            triggers[col] = value

        return {
            "tmdb_id":     tmdb_id,
            "dtdd_id":     dtdd_id,
            "dtdd_name":   item.get("name", ""),
            "release_year": item.get("releaseYear", ""),
            **triggers,
        }


#write CSVs
_csv_columns_written = False
_all_columns: list[str] = []


def save_batch(rows: list[dict], first_write: bool):
    global _csv_columns_written, _all_columns

    if not rows:
        return

    for row in rows:
        for k in row:
            if k not in _all_columns:
                _all_columns.append(k)

    mode = "w" if first_write else "a"
    write_header = first_write

    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_all_columns,
                                extrasaction="ignore", restval="no")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


#main
async def main(limit: int | None = None):
    Path("data").mkdir(exist_ok=True)

    films     = load_valid_films(limit)
    processed = load_processed_ids()
    to_fetch  = [f for f in films if f["tmdb_id"] not in processed]

    total     = len(to_fetch)
    first_write = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0

    print(f"Valid films in DB:      {len(films):,}")
    print(f"Already processed:      {len(processed):,}")
    print(f"Left to fetch:          {total:,}")
    if total == 0:
        print("Nothing to do.")
        return

    semaphore  = asyncio.Semaphore(CONCURRENT)
    matched    = 0
    no_match   = 0
    batch: list[dict] = []
    start_time = time.monotonic()

    connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT, limit=CONCURRENT + 2)
    async with aiohttp.ClientSession(connector=connector) as session:

        CHUNK = 50
        for chunk_start in range(0, total, CHUNK):
            chunk = to_fetch[chunk_start: chunk_start + CHUNK]
            tasks = [process_film(session, semaphore, f) for f in chunk]
            results = await asyncio.gather(*tasks)

            for film, result in zip(chunk, results):
                mark_processed(film["tmdb_id"])
                if result:
                    batch.append(result)
                    matched += 1
                else:
                    no_match += 1

            if len(batch) >= SAVE_EVERY:
                save_batch(batch, first_write)
                first_write = False
                batch = []

            done     = chunk_start + len(chunk)
            elapsed  = time.monotonic() - start_time
            rate     = done / elapsed if elapsed > 0 else 0
            pct      = done / total * 100
            eta_sec  = (total - done) / rate if rate > 0 else 0
            print(f"  {pct:5.1f}%  {matched:,} matched  {no_match:,} no match  "
                  f"{rate:.1f} films/sec  ETA {int(eta_sec//3600)}h {int((eta_sec%3600)//60)}m")

    if batch:
        save_batch(batch, first_write)

    elapsed = time.monotonic() - start_time
    match_rate = matched / (matched + no_match) * 100 if (matched + no_match) > 0 else 0
    print(f"\n🎉 Done.")
    print(f"   {matched:,} films matched ({match_rate:.1f}% match rate)")
    print(f"   {no_match:,} films not found on DTDD")
    print(f"   Output: {OUTPUT_CSV}")
    print(f"   Time: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m")
    print(f"\nNext step: python merge_layers.py  (to merge warnings into movies.db)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N films (for testing)")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit))