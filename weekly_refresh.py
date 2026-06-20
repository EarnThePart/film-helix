"""
weekly_refresh.py       Automated data pipeline
-------------------------------------------------------
Runs in sequence:
  1. TMDB enrichment:   update keywords/genres/metadata for recent or low-vote films
  2. IMDb updates:      refresh vote_count / vote_average from latest TSV files
  3. Wikipedia plots:   fetch missing plots for valid films
  4. Posters & scores:  TMDB posters + OMDb RT scores for new valid films
  5. Cache rebuild:     regenerate .npy embedding caches if any content changed

Usage:
  python weekly_refresh.py                  # full run
  python weekly_refresh.py --dry-run        # preview only, no writes
  python weekly_refresh.py --skip-tmdb      # skip TMDB enrichment (slow)
  python weekly_refresh.py --skip-wiki      # skip Wikipedia fetch
  python weekly_refresh.py --skip-cache     # skip cache rebuild
"""

import argparse
import gzip
import io
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ssl
import certifi
import requests

from dotenv import load_dotenv
load_dotenv()

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

DB_PATH        = "movies.db"
LOG_PATH       = "data/weekly_refresh.log"
TMDB_API_KEY   = os.environ.get("TMDB_API_KEY", "")
OMDB_API_KEY   = os.environ.get("OMDB_API_KEY", "")
TMDB_BASE      = "https://api.themoviedb.org/3"
WIKI_API       = "https://en.wikipedia.org/w/api.php"
IMDB_BASICS_URL   = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL  = "https://datasets.imdbws.com/title.ratings.tsv.gz"

#TMDB rate limit
TMDB_WORKERS   = 10
TMDB_CHUNK     = 38
TMDB_SLEEP     = 10.0

WIKI_SLEEP     = 1.0
VOTE_THRESHOLD = 1000   #min votes for valid film
IMDB_CHANGE_PCT = 0.05  #only update if vote_count changed by >5%


#logging
Path("data").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("refresh")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    existing = {r[1] for r in conn.execute("PRAGMA table_info(movies)").fetchall()}
    if "validated_at" not in existing:
        conn.execute("ALTER TABLE movies ADD COLUMN validated_at TEXT")
        conn.commit()
    conn.execute("""
        UPDATE movies SET validated_at='2000-01-01'
        WHERE is_valid=1 AND validated_at IS NULL
    """)
    conn.commit()
    return conn


def is_valid(vote_count):
    try:
        return float(vote_count or 0) >= VOTE_THRESHOLD
    except (ValueError, TypeError):
        return False


#TMDB
def fetch_tmdb_metadata(tmdb_id):
    """Single TMDB call with append_to_response=keywords,credits."""
    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    params = {
        "api_key":            TMDB_API_KEY,
        "language":           "en-US",
        "append_to_response": "keywords,credits",
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 429:
            return None, "rate_limited"
        if r.status_code == 404:
            return None, "not_found"
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def _keywords_str(data):
    kws = data.get("keywords", {}).get("keywords", [])
    return " ".join(k["name"].replace(" ", "").lower() for k in kws)


def _genres_str(data):
    return " ".join(g["name"] for g in data.get("genres", []))


_VALID_DATE_RE = re.compile(r'^(19|20|21)\d{2}-\d{2}-\d{2}$')
_SLASH_DATE_RE = re.compile(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$')


def _normalize_date(raw):
    """Convert MM/DD/YY or MM/DD/YYYY to YYYY-MM-DD. Returns None if unparseable."""
    if not raw:
        return None
    s = str(raw).strip()
    if _VALID_DATE_RE.match(s):
        return s
    m = _SLASH_DATE_RE.match(s)
    if m:
        mo, day, yr = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(m.group(3)) == 2:
            yr = int(yr)
            this_century_yr = datetime.now().year % 100
            yr = (1900 + yr) if yr > this_century_yr else (2000 + yr)
        else:
            yr = int(yr)
        if 1900 <= yr <= 2100 and 1 <= mo <= 12 and 1 <= day <= 31:
            return f"{yr:04d}-{mo:02d}-{day:02d}"
    return None


def _normalize_title_for_match(title):
    """Lowercase, strip articles/possessives/punctuation for IMDb title matching."""
    import unicodedata
    t = unicodedata.normalize('NFKD', str(title)).encode('ascii', 'ignore').decode('ascii')
    t = t.lower()
    t = re.sub(r"'s\b", '', t)          # possessives
    t = re.sub(r"[^a-z0-9\s]", '', t)   # punctuation
    t = re.sub(r'^(the|a|an)\s+', '', t) # leading articles
    return t.strip()


def _normalize_dates_in_db(conn, dry_run):
    """Find all non-ISO release_dates in movies and normalize them to YYYY-MM-DD in-place."""
    rows = conn.execute("""
        SELECT rowid, id, title, release_date FROM movies
        WHERE release_date IS NOT NULL AND release_date != ''
          AND release_date NOT LIKE '____-__-__'
    """).fetchall()
    fixed = 0
    for row in rows:
        normalized = _normalize_date(row['release_date'])
        if normalized and normalized != str(row['release_date']).strip():
            log.info(f"  [DATE] '{row['title']}' {row['release_date']!r} → {normalized}")
            if not dry_run:
                conn.execute("UPDATE movies SET release_date=? WHERE rowid=?",
                             (normalized, row['rowid']))
            fixed += 1
    if not dry_run and fixed:
        conn.commit()
    log.info(f"  [DATE] {fixed} dates normalized")
    return fixed


def _dedup_valid_films(conn, dry_run):
    import re as _re
    _ISO = _re.compile(r'^\d{4}-\d{2}-\d{2}$')

    def _date_score(rd):
        if not rd or not str(rd).strip():
            return 0
        return 2 if _ISO.match(str(rd).strip()) else 1

    def _completeness(row):
        return sum(1 for v in row if v is not None and str(v).strip() not in ('', 'nan', 'None'))

    rows = conn.execute("SELECT rowid, * FROM movies WHERE is_valid=1 AND id IS NOT NULL").fetchall()

    bad_date_rowids = []
    groups = {}
    for row in rows:
        rd = str(row["release_date"] or "").strip()
        if not _VALID_DATE_RE.match(rd):
            bad_date_rowids.append(row["rowid"])
            log.warning(f"  [DEDUP] bad date rowid={row['rowid']} id={row['id']} "
                        f"'{row['title']}' release_date={rd!r} — demoting")
        else:
            groups.setdefault(row["id"], []).append(row)

    to_demote = list(bad_date_rowids)
    for tmdb_id, group in groups.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: (-_date_score(r["release_date"]), -_completeness(r), r["rowid"]))
        for row in ordered[1:]:
            to_demote.append(row["rowid"])
            log.warning(f"  [DEDUP] duplicate rowid={row['rowid']} id={tmdb_id} '{row['title']}' "
                        f"(keeping rowid={ordered[0]['rowid']})")

    if to_demote:
        log.info(f"  [DEDUP] demoting {len(to_demote)} rows ({len(bad_date_rowids)} bad-date, "
                 f"{len(to_demote)-len(bad_date_rowids)} duplicates)")
        if not dry_run:
            conn.executemany("UPDATE movies SET is_valid=0 WHERE rowid=?", [(r,) for r in to_demote])
            conn.commit()
    else:
        log.info("  [DEDUP] no bad-date or duplicate rows found")


def run_tmdb_enrichment(conn, dry_run):
    log.info("PHASE 1: TMDB enrichment")
    rows = conn.execute("""
        SELECT id, title, release_date, vote_count, dna_keywords, dna_genres, runtime
        FROM movies
        WHERE (release_date LIKE '2024%' OR release_date LIKE '2025%' OR release_date LIKE '2026%') AND CAST(vote_count AS REAL) >= 1000
           OR (
               CAST(vote_count AS REAL) >= 1000
               AND CAST(vote_count AS REAL) < 50000
               AND (dna_keywords IS NULL OR TRIM(dna_keywords) = '')
           )
        ORDER BY CAST(vote_count AS REAL) DESC
    """).fetchall()
    log.info(f"  {len(rows)} films targeted for TMDB enrichment")

    updated = 0
    errors  = 0
    chunks  = [rows[i:i+TMDB_CHUNK] for i in range(0, len(rows), TMDB_CHUNK)]

    for chunk_idx, chunk in enumerate(chunks):
        results = {}
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            futs = {pool.submit(fetch_tmdb_metadata, row["id"]): row for row in chunk}
            for fut in as_completed(futs):
                row  = futs[fut]
                data, err = fut.result()
                if err == "rate_limited":
                    log.warning(f"    rate limited on {row['title']} — will retry next run")
                    errors += 1
                elif err:
                    errors += 1
                elif data:
                    results[row["id"]] = (row, data)

        for tmdb_id, (row, data) in results.items():
            new_kw   = _keywords_str(data)
            new_gen  = _genres_str(data)
            new_vc   = float(data.get("vote_count") or 0)
            new_va   = float(data.get("vote_average") or 0)
            raw_rt   = data.get("runtime")
            if raw_rt is not None:
                if raw_rt <= 1 or raw_rt > 600:
                    log.warning(f"  [RUNTIME] Rejected bad runtime {raw_rt} for '{row['title']}'")
                    raw_rt = None
            new_rt = raw_rt

            changed = (
                new_kw  != (row["dna_keywords"] or "") or
                new_gen != (row["dna_genres"]   or "") or
                abs(new_vc - float(row["vote_count"] or 0)) > 1 or
                (new_rt is not None and new_rt != row["runtime"])
            )
            if changed:
                if not dry_run:
                    conn.execute("""
                        UPDATE movies
                        SET dna_keywords=?, dna_genres=?, vote_count=?, vote_average=?, runtime=?
                        WHERE id=?
                    """, (new_kw or row["dna_keywords"],
                          new_gen or row["dna_genres"],
                          new_vc, new_va, new_rt, tmdb_id))
                updated += 1

        if not dry_run:
            conn.commit()

        #TMDB rate limit check
        if chunk_idx < len(chunks) - 1:
            time.sleep(TMDB_SLEEP)

    log.info(f"  TMDB: {updated} films updated, {errors} errors")
    return updated


#IMDB updates
def _download_tsv_gz(url):
    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    log.info(f"  Downloading {url}...")
    with urllib.request.urlopen(url, timeout=120, context=_SSL_CTX) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        chunks = []
        if _has_tqdm:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                      desc=f"  {url.split('/')[-1]}", leave=False) as bar:
                while True:
                    chunk = resp.read(1 << 16)  # 64 KB
                    if not chunk:
                        break
                    chunks.append(chunk)
                    bar.update(len(chunk))
        else:
            chunks = [resp.read()]
    compressed = b"".join(chunks)
    with gzip.open(io.BytesIO(compressed), "rt", encoding="utf-8") as f:
        return f.read()


def run_imdb_updates(conn, dry_run):
    log.info("PHASE 2: IMDb updates")
    try:
        ratings_raw = _download_tsv_gz(IMDB_RATINGS_URL)
    except Exception as e:
        log.error(f"  Failed to download IMDb ratings: {e}")
        return 0

    #parse ratings into dict: tconst_y → (avg_rating, num_votes)
    imdb_ratings = {}
    for line in ratings_raw.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            imdb_ratings[parts[0]] = (float(parts[1]), int(parts[2]))

    log.info(f"  Loaded {len(imdb_ratings):,} IMDb ratings")

    rows = conn.execute("""
        SELECT id, tconst_y, vote_count, vote_average
        FROM movies
        WHERE tconst_y IS NOT NULL AND tconst_y != ''
    """).fetchall()

    updated = 0
    for row in rows:
        tconst_y = str(row["tconst_y"]).strip()
        if tconst_y not in imdb_ratings:
            continue
        new_avg, new_votes = imdb_ratings[tconst_y]
        old_votes = float(row["vote_count"] or 0)
        if old_votes > 0 and abs(new_votes - old_votes) / old_votes < IMDB_CHANGE_PCT:
            continue
        if not dry_run:
            conn.execute(
                "UPDATE movies SET vote_count=?, vote_average=? WHERE id=?",
                (new_votes, new_avg, row["id"])
            )
        updated += 1

    if not dry_run:
        conn.commit()
    log.info(f"  IMDb: {updated} films updated")
    return updated




#wiki plots
def _extract_plot_section(text):
    """Return the Plot/Synopsis section from plain-text Wikipedia extract."""
    if not text:
        return None
    #find plot sections with header == Plot == or == Synopsis == (case-insensitive, flexible spacing)
    m = re.search(r'==\s*(?:Plot|Synopsis)\s*==\s*\n(.*?)(?:\n==\s|\Z)', text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    #fallback: if "Plot" appears as a standalone line heading, grab everything after it
    m = re.search(r'(?:^|\n)Plot\n[-=]+\n(.*?)(?:\n[A-Z][^\n]{0,40}\n[-=]+|\Z)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _strip_wiki_markup(text):
    """Remove common wikitext markup from raw wikitext."""
    #removes [[File:...]] and [[Image:...]] blocks
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    #removes inline citation markers like [[a]], [[b]], [[1]]
    text = re.sub(r'\[\[[a-z0-9]\]\]', '', text, flags=re.IGNORECASE)
    #unwraps [[link|display]] → display, [[link]] → link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    #removes {{...}} templates iteratively to handle nesting
    for _ in range(10):
        new = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if new == text:
            break
        text = new
    #removes HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    #removes bold/italic markup
    text = re.sub(r"'{2,}", '', text)
    #trims whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _titles_loosely_match(film_title, wiki_title):
    """Return True if film_title and wiki_title share at least one meaningful word."""
    import unicodedata as _ud
    def _normalize(s):
        return _ud.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    _stop = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for",
             "is", "it", "be", "or", "by", "as", "up", "do"}
    def _words(s):
        return {w for w in re.sub(r"[^a-z0-9 ]", "", _normalize(s).replace('_', ' ').lower()).split() if w not in _stop}
    film_words = _words(film_title)
    wiki_words = _words(wiki_title)
    return bool(film_words & wiki_words)


def _wikidata_title_from_tconst(tconst, film_title):
    """Query Wikidata SPARQL for the enwiki article title using an IMDb tconst.

    Verifies the returned Wikipedia title loosely matches the expected film
    title before returning it, to guard against wrong tconst → wrong article.
    """
    if not tconst:
        return None
    headers = {"User-Agent": "FilmHelixApp/1.0 (admin@filmhelix.local) python-requests/2.31"}
    try:
        query = f"""SELECT ?article WHERE {{
  ?film wdt:P345 '{tconst}'.
  ?article schema:about ?film;
           schema:isPartOf <https://en.wikipedia.org/>.
}}"""
        r = requests.get("https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        results = r.json().get("results", {}).get("bindings", [])
        if not results:
            return None
        url = results[0].get("article", {}).get("value", "")
        wiki_title = urllib.parse.unquote(url.split("/wiki/")[-1]) if "/wiki/" in url else None
        if not wiki_title:
            return None
        if _titles_loosely_match(film_title, wiki_title):
            return wiki_title
        print(f"    [Wikidata mismatch] tconst={tconst} returned {wiki_title!r} "
              f"for film {film_title!r} — falling back to title search")
        return None
    except Exception:
        pass
    return None


def _fetch_section1(query, headers):
    """Fetch section 1 of a Wikipedia article (almost always Plot) via rvsection=1."""
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "format": "json",
            "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "rvsection": "1",
            "titles": query, "redirects": 1,
        }, headers=headers, timeout=12)
        if r.status_code == 429:
            print(f"    [Wiki 429] Rate limited on {query!r}. Cooling down 60s...")
            time.sleep(60)
            return None
        if r.status_code != 200:
            return None
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("pageid", -1) == -1:
                continue
            revs = page.get("revisions", [])
            if not revs:
                continue
            content = revs[0].get("slots", {}).get("main", {}).get("*", "") or revs[0].get("*", "")
            content = _strip_wiki_markup(content.strip())
            #strip leading == Plot == / == Synopsis == header line
            content = re.sub(r'^\s*==\s*(?:Plot|Synopsis|Story)\s*==\s*\n?', '', content, flags=re.IGNORECASE).strip()
            if content and len(content) >= 500:
                return content[:15000]
    except Exception as e:
        print(f"    [Wiki section1 crash] {e} for {query!r}")
    return None


def _fetch_wiki_extract(query, headers):
    """Fetch plot text for a Wikipedia page title.

    Primary:  rvsection=1 (section 1 of film articles is almost always Plot)
    Fallback: prop=extracts full article text, then regex plot extraction
    """
    #primary
    result = _fetch_section1(query, headers)
    if result:
        return result

    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "format": "json", "prop": "extracts",
            "exintro": False, "explaintext": True,
            "titles": query, "redirects": 1,
        }, headers=headers, timeout=10)

        if r.status_code == 429:
            print(f"    [Wiki 429] Rate limited on {query!r}. Cooling down 60s...")
            time.sleep(60)
            return None

        if r.status_code != 200:
            print(f"    [Wiki Error] {r.status_code} for {query!r}")
            return None

        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("pageid", -1) == -1:
                continue
            extract = page.get("extract", "").strip()
            if not extract:
                continue

            plot_section = _extract_plot_section(extract)
            if plot_section and len(plot_section) >= 500:
                return plot_section[:15000]

            if len(extract) >= 500:
                return extract[:15000]

    except Exception as e:
        print(f"    [Wiki Crash] {e} for {query!r}")

    return None


def _fetch_wiki_plot(title, year, tconst=None):
    """Fetch a Wikipedia plot section for a film.

    Step 1: if tconst is available, resolve the exact Wikipedia title via
            Wikidata's IMDb sitelink — eliminates title-matching ambiguity.
    Step 2: fall back to title-based search queries.
    """
    headers = {"User-Agent": "FilmHelixApp/1.0 (admin@filmhelix.local) python-requests/2.31"}

    #Wikidata → exact Wikipedia title
    if tconst:
        wiki_title = _wikidata_title_from_tconst(tconst, title)
        if wiki_title:
            print(f"    [Wikidata] {tconst} → {wiki_title!r}")
            result = _fetch_wiki_extract(wiki_title, headers)
            if result:
                return result
            time.sleep(WIKI_SLEEP)

    #title-based fallback
    for query in [f"{title} ({year} film)", f"{title} (film)", title]:
        result = _fetch_wiki_extract(query, headers)
        if result:
            return result
        time.sleep(WIKI_SLEEP)

    return None


def run_wiki_plots(conn, dry_run, min_votes=None, since_cutoff=None):
    import csv
    log.info("PHASE 3: Wikipedia plots")
    threshold = min_votes if min_votes is not None else VOTE_THRESHOLD
    if since_cutoff is not None:
        rows = conn.execute("""
            SELECT id, title, release_date, tconst_y, vote_count
            FROM movies
            WHERE wiki_plot IS NULL
              AND is_valid = 1
              AND validated_at IS NOT NULL
              AND validated_at >= ?
            ORDER BY CAST(vote_count AS REAL) DESC
        """, (since_cutoff,)).fetchall()
        log.info(f"  {len(rows)} newly-promoted films missing wiki plots (validated_at >= {since_cutoff[:10]})")
        if not rows:
            log.info("  0 newly promoted films need wiki plots — skipping")
            return 0
    else:
        rows = conn.execute("""
            SELECT id, title, release_date, tconst_y, vote_count
            FROM movies
            WHERE wiki_plot IS NULL
              AND CAST(vote_count AS REAL) >= ?
              AND is_valid = 1
            ORDER BY CAST(vote_count AS REAL) DESC
        """, (threshold,)).fetchall()
        log.info(f"  {len(rows)} films missing wiki plots (min_votes={threshold:,})")

    failures_path = "wiki_fetch_failures.csv"
    failure_rows = []

    fetched = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        _year_match = re.search(r'\d{4}', str(row["release_date"] or ""))
        year = _year_match.group(0) if _year_match else ""
        tconst = (row["tconst_y"] or "").strip() or None
        plot = _fetch_wiki_plot(row["title"], year, tconst=tconst)
        if plot:
            print(f"[{i}/{total}] {row['title']} ({year}) — ok ({len(plot):,} chars)")
            if not dry_run:
                conn.execute(
                    "UPDATE movies SET wiki_plot=?, wiki_plot_status='ok', wiki_plot_fetched_at=? WHERE id=?",
                    (plot, datetime.now(timezone.utc).isoformat(), row["id"])
                )
            fetched += 1
            if fetched % 20 == 0:
                if not dry_run:
                    conn.commit()
                log.info(f"    {fetched} plots fetched so far.")
        else:
            print(f"[{i}/{total}] {row['title']} ({year}) — FAILED")
            failure_rows.append({
                "id":           row["id"],
                "title":        row["title"],
                "release_date": row["release_date"],
                "tconst_y":     row["tconst_y"] or "",
                "vote_count":   row["vote_count"],
            })
        time.sleep(WIKI_SLEEP)

    if not dry_run:
        conn.commit()

    if failure_rows:
        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "title", "release_date", "tconst_y", "vote_count"])
            writer.writeheader()
            writer.writerows(failure_rows)
        log.info(f"  {len(failure_rows)} failures logged to {failures_path}")

    log.info(f"  Wiki: {fetched} plots fetched, {len(failure_rows)} failed")
    return fetched


#posters/RT
def _fetch_poster(row):
    tmdb_id = row["id"]
    url = f"{TMDB_BASE}/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url, timeout=6).json()
        path = data.get("poster_path")
        if path:
            return (row["id"], f"https://image.tmdb.org/t/p/w500{path}", None)
    except Exception:
        pass
    return (row["id"], None, None)


def _fetch_rt(row):
    tconst_y = str(row["tconst_y"] or "").strip()
    if not tconst_y.startswith("tt"):
        return (row["id"], None)
    try:
        data = requests.get(
            f"http://www.omdbapi.com/?i={tconst_y}&apikey={OMDB_API_KEY}", timeout=6
        ).json()
        for r in data.get("Ratings", []):
            if r["Source"] == "Rotten Tomatoes":
                score = int(r["Value"].replace("%", ""))
                return (row["id"], score)
    except Exception:
        pass
    return (row["id"], None)


def _rt_slug(title, year=None):
    import unicodedata
    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = "".join(c for c in normalized if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9\s]", "", ascii_title.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return f"{slug}_{year}" if year else slug


def _fetch_rt_scrape(row):
    """RT direct scrape fallback for films OMDb didn't cover."""
    import json as _json
    year = str(row["release_date"] or "")[:4]
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    for slug in [_rt_slug(row["title"], year), _rt_slug(row["title"])]:
        try:
            r = requests.get(f"https://www.rottentomatoes.com/m/{slug}", headers=headers, timeout=6)
            if r.status_code != 200:
                continue
            import re as _re
            jld = _re.search(r'<script type="application/ld\+json">(.*?)</script>', r.text, _re.DOTALL)
            if jld:
                val = _json.loads(jld.group(1)).get("aggregateRating", {}).get("ratingValue")
                if val:
                    return (row["id"], int(val))
        except Exception:
            pass
    return (row["id"], None)


def run_posters_scores(conn, dry_run, since_cutoff=None):
    log.info("PHASE 4: Posters & RT scores")
    if since_cutoff is not None:
        poster_rows = conn.execute("""
            SELECT id FROM movies
            WHERE poster IS NULL AND is_valid=1
              AND validated_at IS NOT NULL AND validated_at >= ?
            ORDER BY vote_count DESC
        """, (since_cutoff,)).fetchall()
        rt_rows = conn.execute("""
            SELECT id, tconst_y FROM movies
            WHERE rt_score IS NULL AND is_valid=1 AND tconst_y IS NOT NULL
              AND validated_at IS NOT NULL AND validated_at >= ?
            ORDER BY vote_count DESC
        """, (since_cutoff,)).fetchall()
        log.info(f"  {len(poster_rows)} posters missing, {len(rt_rows)} RT scores missing (validated_at >= {since_cutoff[:10]})")
        if not poster_rows and not rt_rows:
            log.info("  0 newly promoted films need posters or RT scores — skipping")
            return 0, 0
    else:
        poster_rows = conn.execute(
            "SELECT id FROM movies WHERE poster IS NULL AND is_valid=1 ORDER BY vote_count DESC"
        ).fetchall()
        rt_rows = conn.execute(
            "SELECT id, tconst_y FROM movies WHERE rt_score IS NULL AND is_valid=1 AND tconst_y IS NOT NULL ORDER BY vote_count DESC"
        ).fetchall()
        log.info(f"  {len(poster_rows)} posters missing, {len(rt_rows)} RT scores missing")

    posters_fetched = 0
    chunks = [poster_rows[i:i+TMDB_CHUNK] for i in range(0, len(poster_rows), TMDB_CHUNK)]
    for chunk in chunks:
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            for film_id, poster_url, _ in pool.map(_fetch_poster, chunk):
                if poster_url:
                    if not dry_run:
                        conn.execute("UPDATE movies SET poster=? WHERE id=?", (poster_url, film_id))
                    posters_fetched += 1
        if not dry_run:
            conn.commit()
        time.sleep(1)

    rt_fetched = 0
    for film_id, score in map(_fetch_rt, rt_rows):
        if score is not None:
            if not dry_run:
                conn.execute("UPDATE movies SET rt_score=? WHERE id=?", (score, film_id))
            rt_fetched += 1
        time.sleep(0.1)
    if not dry_run:
        conn.commit()

    if since_cutoff is not None:
        rt_scrape_rows = conn.execute("""
            SELECT id, title, release_date FROM movies
            WHERE is_valid=1 AND (rt_score IS NULL OR rt_score=0)
              AND validated_at IS NOT NULL AND validated_at >= ?
            ORDER BY vote_count DESC
        """, (since_cutoff,)).fetchall()
    else:
        rt_scrape_rows = conn.execute(
            """SELECT id, title, release_date FROM movies
               WHERE is_valid=1 AND (rt_score IS NULL OR rt_score=0)
               ORDER BY vote_count DESC"""
        ).fetchall()

    if not rt_scrape_rows:
        log.info("  0 newly promoted films need RT scores — skipping scrape fallback")
    else:
        log.info(f"  RT scrape fallback: {len(rt_scrape_rows)} films still missing scores")
    rt_scraped = 0
    for row in rt_scrape_rows:
        film_id, score = _fetch_rt_scrape(row)
        if score is not None:
            if not dry_run:
                conn.execute("UPDATE movies SET rt_score=? WHERE id=?", (score, film_id))
            rt_scraped += 1
        time.sleep(0.5)
    if not dry_run:
        conn.commit()

    log.info(f"  Posters: {posters_fetched} fetched, RT via OMDb: {rt_fetched}, RT via scrape: {rt_scraped}")
    return posters_fetched, rt_fetched + rt_scraped


def _word_overlap(a, b, chars=500):
    """Jaccard word overlap over first `chars` characters of each string."""
    def _words(s):
        return set(re.sub(r"[^a-z0-9]", " ", s[:chars].lower()).split())
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def run_verify_ids(conn, film_ids, fix_mismatches=False):
    """Verify wiki_plot for specific film IDs — prints MATCH or MISMATCH for each."""
    headers = {"User-Agent": "FilmHelixApp/1.0 (admin@filmhelix.local) python-requests/2.31"}

    placeholders = ",".join("?" * len(film_ids))
    rows = conn.execute(f"""
        SELECT id, title, release_date, tconst_y, wiki_plot
        FROM movies
        WHERE id IN ({placeholders})
        ORDER BY id
    """, film_ids).fetchall()

    found_ids = {row["id"] for row in rows}
    for fid in film_ids:
        if int(fid) not in found_ids:
            print(f"  id={fid} — not found in DB")

    for row in rows:
        title = row["title"]
        year_m = re.search(r"\d{4}", str(row["release_date"] or ""))
        year = year_m.group(0) if year_m else ""
        tconst = (row["tconst_y"] or "").strip() or None
        stored = row["wiki_plot"] or ""

        wiki_title = _wikidata_title_from_tconst(tconst, title) if tconst else None
        queries = [wiki_title] if wiki_title else [
            f"{title} ({year} film)", f"{title} (film)", title
        ]

        fetched_plot = None
        for query in queries:
            result = _fetch_section1(query, headers)
            if result:
                fetched_plot = result
                break
            time.sleep(WIKI_SLEEP)

        if not fetched_plot:
            print(f"  id={row['id']} {title!r} — could not fetch, skipping")
            continue

        sim = _word_overlap(stored, fetched_plot)
        status = "MISMATCH" if sim < 0.7 else "MATCH"
        print(f"  id={row['id']} {title!r}  sim={sim:.2f}  {status}")
        if status == "MISMATCH":
            print(f"    stored[:200]:  {stored[:200]!r}")
            print(f"    fetched[:200]: {fetched_plot[:200]!r}")
            if fix_mismatches and fetched_plot:
                conn.execute("UPDATE movies SET wiki_plot=? WHERE id=?", (fetched_plot, row["id"]))
                conn.commit()
                print(f"    → fixed in DB ({len(fetched_plot):,} chars)")

        time.sleep(WIKI_SLEEP)


def run_verify_plots(conn, min_votes, max_votes=None, fix_mismatches=False, output_csv="wiki_mismatches.csv"):
    import csv
    range_str = f"min_votes={min_votes:,}" + (f", max_votes={max_votes:,}" if max_votes else "")
    if fix_mismatches:
        range_str += ", fix_mismatches=on"
    log.info(f"VERIFY PLOTS ({range_str})")
    headers = {"User-Agent": "FilmHelixApp/1.0 (admin@filmhelix.local) python-requests/2.31"}

    if max_votes:
        rows = conn.execute("""
            SELECT id, title, release_date, tconst_y, wiki_plot, vote_count
            FROM movies
            WHERE is_valid = 1
              AND wiki_plot IS NOT NULL AND wiki_plot != ''
              AND CAST(vote_count AS REAL) >= ?
              AND CAST(vote_count AS REAL) <= ?
            ORDER BY CAST(vote_count AS REAL) DESC
        """, (min_votes, max_votes)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, title, release_date, tconst_y, wiki_plot, vote_count
            FROM movies
            WHERE is_valid = 1
              AND wiki_plot IS NOT NULL AND wiki_plot != ''
              AND CAST(vote_count AS REAL) >= ?
            ORDER BY CAST(vote_count AS REAL) DESC
        """, (min_votes,)).fetchall()

    total = len(rows)
    log.info(f"  {total:,} films to verify")

    mismatches = []
    fetch_failures = []
    fixed = 0

    for i, row in enumerate(rows, 1):
        title = row["title"]
        year_m = re.search(r"\d{4}", str(row["release_date"] or ""))
        year = year_m.group(0) if year_m else ""
        tconst = (row["tconst_y"] or "").strip() or None

        wiki_title = _wikidata_title_from_tconst(tconst, title) if tconst else None
        queries = [wiki_title] if wiki_title else [
            f"{title} ({year} film)", f"{title} (film)", title
        ]

        fetched_plot = None
        for query in queries:
            result = _fetch_section1(query, headers)
            if result:
                fetched_plot = result
                break
            time.sleep(WIKI_SLEEP)

        if not fetched_plot:
            print(f"[{i}/{total}] {title!r} — no fetch result, skipping")
            continue

        sim = _word_overlap(row["wiki_plot"], fetched_plot)
        is_mismatch = sim < 0.7

        if not is_mismatch:
            print(f"[{i}/{total}] {title!r}  sim={sim:.2f}  ok")
        elif fix_mismatches:
            #re-fetch using full _fetch_wiki_plot pipeline
            new_plot = _fetch_wiki_plot(title, year, tconst=tconst)
            if new_plot:
                conn.execute(
                    "UPDATE movies SET wiki_plot=?, wiki_plot_status='ok', wiki_plot_fetched_at=? WHERE id=?",
                    (new_plot, datetime.now(timezone.utc).isoformat(), row["id"])
                )
                conn.commit()
                fixed += 1
                print(f"[{i}/{total}] {title!r}  sim={sim:.2f}  MISMATCH → fixed ({len(new_plot):,} chars)")
            else:
                fetch_failures.append({
                    "id": row["id"], "title": title,
                    "release_date": row["release_date"],
                    "tconst_y": row["tconst_y"] or "",
                    "vote_count": row["vote_count"],
                })
                print(f"[{i}/{total}] {title!r}  sim={sim:.2f}  MISMATCH → fetch failed")
        else:
            print(f"[{i}/{total}] {title!r}  sim={sim:.2f}  MISMATCH")
            mismatches.append({
                "id":             row["id"],
                "title":          title,
                "release_date":   row["release_date"],
                "tconst_y":       row["tconst_y"] or "",
                "stored_length":  len(row["wiki_plot"]),
                "fetched_length": len(fetched_plot),
                "similarity":     round(sim, 4),
            })

        time.sleep(WIKI_SLEEP)

    #writes mismatch CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "title", "release_date", "tconst_y",
            "stored_length", "fetched_length", "similarity"
        ])
        writer.writeheader()
        writer.writerows(mismatches)

    #writes fetch failures CSV
    if fetch_failures:
        with open("wiki_fetch_failures.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "title", "release_date", "tconst_y", "vote_count"])
            writer.writeheader()
            writer.writerows(fetch_failures)

    log.info(f"\n{'─'*50}")
    log.info(f"Verify summary:")
    log.info(f"  Checked:    {total:,}")
    log.info(f"  Mismatches: {len(mismatches) + len(fetch_failures) + fixed:,}")
    if fix_mismatches:
        log.info(f"  Fixed:      {fixed:,}")
        log.info(f"  Fetch failures: {len(fetch_failures):,}")
    log.info(f"  Written to: {output_csv}")


#cache rebuild
CACHE_FILES = {
    "semantic":      "semantic_embeddings_cache.npy",
    "wiki_semantic": "wiki_semantic_embeddings_cache.npy",
}
MODEL_NAME   = "all-MiniLM-L6-v2"
CHUNK_WORDS  = 180
EMB_DIM      = 384


def _load_sentence_transformer():
    """Import SentenceTransformer or raise a clear fatal error."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except ImportError:
        log.error("─" * 60)
        log.error("FATAL: sentence-transformers is not installed locally.")
        log.error("The production app cannot rebuild caches. This must be")
        log.error("done locally, then the .npy files pushed to GitHub.")
        log.error("")
        log.error("Install it:  pip install sentence-transformers")
        log.error("Then re-run: python weekly_refresh.py")
        log.error("─" * 60)
        sys.exit(1)


def run_cache_rebuild(dry_run):
    import numpy as np
    import pandas as pd

    log.info("PHASE 5: Rebuilding embedding caches...")

    if dry_run:
        log.info("  [dry-run] would rebuild all 4 .npy cache files locally")
        return True

    model = _load_sentence_transformer()
    log.info(f"  Loaded model: {MODEL_NAME}")

    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM movies WHERE overview IS NOT NULL AND is_valid = 1",
        conn
    ).fillna("")
    conn.close()
    n = len(df)
    log.info(f"  {n:,} valid films loaded")

    #semantic overview
    log.info("  Encoding overview embeddings...")
    overviews = df["overview"].tolist()
    sem_embs = model.encode(
        overviews, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    np.save(CACHE_FILES["semantic"], sem_embs)
    log.info(f"  Saved {CACHE_FILES['semantic']}  shape={sem_embs.shape}")

    #wiki semantic
    log.info("  Encoding wiki_plot embeddings (chunked)...")
    wiki_texts  = df["wiki_plot"].tolist()
    film_chunks = []
    no_wiki_idx = []
    for i, text in enumerate(wiki_texts):
        t = str(text).strip()
        if not t or t == "nan":
            no_wiki_idx.append(i)
            continue
        words = t.split()
        for start in range(0, len(words), CHUNK_WORDS):
            film_chunks.append((i, " ".join(words[start:start + CHUNK_WORDS])))

    log.info(f"    {len(film_chunks)} chunks for {n - len(no_wiki_idx)} films with wiki text")
    wiki_sem = np.zeros((n, EMB_DIM), dtype=np.float32)
    if film_chunks:
        texts_only = [c[1] for c in film_chunks]
        all_embs = model.encode(
            texts_only, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        counts = np.zeros(n, dtype=np.int32)
        for (film_idx, _), emb in zip(film_chunks, all_embs):
            wiki_sem[film_idx] += emb
            counts[film_idx]   += 1
        has_wiki = counts > 0
        wiki_sem[has_wiki] /= counts[has_wiki, np.newaxis]
        norms = np.linalg.norm(wiki_sem, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        wiki_sem /= norms
    np.save(CACHE_FILES["wiki_semantic"], wiki_sem)
    log.info(f"  Saved {CACHE_FILES['wiki_semantic']}  shape={wiki_sem.shape}")

    log.info("  ✓ Both cache files rebuilt. Push .npy files to GitHub.")
    return True

#main
def main():
    ap = argparse.ArgumentParser(description="Film Helix weekly data pipeline")
    ap.add_argument("--dry-run",    action="store_true", help="Preview changes without writing")
    ap.add_argument("--skip-tmdb",  action="store_true", help="Skip TMDB enrichment")
    ap.add_argument("--skip-imdb",  action="store_true", help="Skip IMDb vote updates")
    ap.add_argument("--skip-wiki",  action="store_true", help="Skip Wikipedia plot fetch")
    ap.add_argument("--skip-posters", action="store_true", help="Skip poster/RT fetch")
    ap.add_argument("--skip-cache",    action="store_true", help="Skip cache rebuild")
    ap.add_argument("--force-rebuild",   action="store_true", help="Force cache rebuild even if no content changes detected")
    ap.add_argument("--force-refresh",   action="store_true", help="Ignore IMDb index pickle and re-download fresh TSVs")
    ap.add_argument("--since",          type=str, default=None, help="Limit Phase 3/4 to films promoted to is_valid=1 on or after DATE (YYYY-MM-DD)")
    ap.add_argument("--verify-plots",    action="store_true", help="Verify stored wiki_plots against fresh Wikipedia fetch")
    ap.add_argument("--fix-mismatches", action="store_true", help="When used with --verify-plots, re-fetch and overwrite mismatched plots in the DB")
    ap.add_argument("--min-votes",       type=int, default=200000, help="Min vote_count for --verify-plots and wiki fetch phase (default: 200000)")
    ap.add_argument("--max-votes",       type=int, default=None,   help="Max vote_count for --verify-plots (optional upper bound)")
    ap.add_argument("--verify-ids",      type=str, default="", help="Comma-separated film IDs to verify (read-only)")
    ap.add_argument("--verify-ids-file", type=str, default="", help="File of IDs to verify: plain text (one per line) or CSV with an 'id' column (e.g. wiki_mismatches.csv)")
    args = ap.parse_args()

    if args.verify_plots:
        conn = get_conn()
        run_verify_plots(conn, args.min_votes, max_votes=args.max_votes, fix_mismatches=args.fix_mismatches)
        conn.close()
        return

    if args.verify_ids or args.verify_ids_file:
        film_ids = []
        if args.verify_ids:
            film_ids = [x.strip() for x in args.verify_ids.split(",") if x.strip()]
        if args.verify_ids_file:
            import csv as _csv
            with open(args.verify_ids_file, newline="", encoding="utf-8") as f:
                sample = f.read(1024)
                f.seek(0)
                if "," in sample.splitlines()[0] if sample else False:
                    reader = _csv.DictReader(f)
                    if reader.fieldnames and "id" in reader.fieldnames:
                        film_ids += [str(row["id"]).strip() for row in reader if str(row["id"]).strip()]
                    else:
                        #fallback to first column if ID column not found
                        f.seek(0)
                        reader = _csv.reader(f)
                        next(reader, None)
                        film_ids += [row[0].strip() for row in reader if row and row[0].strip()]
                else:
                    #one ID per line
                    film_ids += [line.strip() for line in f if line.strip()]
        film_ids = list(dict.fromkeys(film_ids))  #deduplicate, preserve order
        conn = get_conn()
        run_verify_ids(conn, film_ids, fix_mismatches=args.fix_mismatches)
        conn.close()
        return

    mode = "[DRY RUN] " if args.dry_run else ""
    log.info(f"{'='*60}")
    log.info(f"{mode}Film Helix weekly refresh — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'='*60}")

    conn = get_conn()

    _normalize_dates_in_db(conn, args.dry_run)
    _dedup_valid_films(conn, args.dry_run)
    content_changed = False

    #TMDB
    tmdb_updated = 0
    if not args.skip_tmdb:
        tmdb_updated = run_tmdb_enrichment(conn, args.dry_run)
        if tmdb_updated > 0:
            content_changed = True
    else:
        log.info("PHASE 1: TMDB enrichment skipped.")

    #IMDB
    imdb_updated = 0
    if not args.skip_imdb:
        imdb_updated = run_imdb_updates(conn, args.dry_run)
        if imdb_updated > 0:
            content_changed = True
    else:
        log.info("PHASE 2: IMDb updates skipped.")

    #valid film check
    new_valid = 0
    if not args.dry_run and (tmdb_updated > 0 or imdb_updated > 0):
        _date_filter = """
              AND (release_date LIKE '19__-__-__'
                OR release_date LIKE '20__-__-__'
                OR release_date LIKE '21__-__-__')
        """
        new_valid = conn.execute(f"""
            SELECT COUNT(*) FROM movies
            WHERE is_valid = 0
              AND CAST(vote_count AS REAL) >= ?
              {_date_filter}
        """, (VOTE_THRESHOLD,)).fetchone()[0]
        if new_valid > 0:
            conn.execute(f"""
                UPDATE movies SET is_valid=1, validated_at=?
                WHERE is_valid=0
                  AND CAST(vote_count AS REAL) >= {VOTE_THRESHOLD}
                  {_date_filter}
            """, (datetime.now(timezone.utc).isoformat(),))
            conn.commit()
            log.info(f"  Promoted {new_valid} films to is_valid=1")
            content_changed = True

    _since_cutoff = args.since if args.since else None

    #wikipedia
    wiki_fetched = 0
    if not args.skip_wiki:
        wiki_fetched = run_wiki_plots(
            conn, args.dry_run,
            min_votes=args.min_votes if args.min_votes != 200000 else None,
            since_cutoff=_since_cutoff,
        )
        if wiki_fetched > 0:
            content_changed = True
    else:
        log.info("PHASE 3: Wikipedia plots skipped.")

    #posters/scores
    posters_fetched, rt_fetched = 0, 0
    if not args.skip_posters:
        posters_fetched, rt_fetched = run_posters_scores(conn, args.dry_run, since_cutoff=_since_cutoff)
    else:
        log.info("PHASE 4: Posters & RT scores skipped.")

    conn.close()

    #cache rebuild
    cache_rebuilt = False
    if args.skip_cache:
        log.info("PHASE 5: Cache rebuild skipped (--skip-cache)")
    elif content_changed or args.force_rebuild:
        if args.force_rebuild and not content_changed:
            log.info("PHASE 5: Cache rebuild forced (--force-rebuild)")
        cache_rebuilt = run_cache_rebuild(args.dry_run)
    else:
        log.info("PHASE 5: No content changes, cache rebuild skipped.")

    #summary
    log.info(f"{'='*60}")
    log.info(f"{mode}SUMMARY")
    log.info(f"  TMDB films updated:    {tmdb_updated}")
    log.info(f"  IMDb films updated:    {imdb_updated}")
    log.info(f"  New films validated:   {new_valid}")
    log.info(f"  Wiki plots fetched:    {wiki_fetched}")
    log.info(f"  Posters fetched:       {posters_fetched}")
    log.info(f"  RT scores fetched:     {rt_fetched}")
    log.info(f"  Caches rebuilt:        {cache_rebuilt}")
    log.info(f"{'='*60}")

    print(f"\n{'─'*50}")
    print(f"  Films updated (TMDB):  {tmdb_updated}")
    print(f"  Films updated (IMDb):  {imdb_updated}")
    print(f"  New films validated:   {new_valid}")
    print(f"  Wiki plots fetched:    {wiki_fetched}")
    print(f"  Posters fetched:       {posters_fetched}")
    print(f"  RT scores fetched:     {rt_fetched}")
    print(f"  Caches rebuilt:        {cache_rebuilt}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
