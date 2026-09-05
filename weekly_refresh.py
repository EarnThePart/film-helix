"""
weekly_refresh.py       Automated data pipeline — the single entry point
-------------------------------------------------------
Runs in sequence:
  0. Integrity gate:    assert invariants, normalize dates, dedup, resolve tconst
                        collisions, enforce is_valid in both directions
  1a. Blank-date refetch: recover is_valid=0 stubs whose release_date is empty
  1. TMDB enrichment:   update keywords/genres/metadata for recent or low-vote films
  2. IMDb updates:      refresh vote_count / vote_average, link missing tconsts
  2a. DTDD warnings:    backfill content warnings from the DTDD export
  2b. Validity recheck: re-settle is_valid after vote counts moved
  3. Wikipedia plots:   fetch missing plots for valid films
  4. Posters & scores:  TMDB posters + OMDb RT scores for new valid films
  5. Cache rebuild:     regenerate .npy embedding caches if any content changed

This supersedes merge_layers.py (retired to archive/). That script rebuilt the
whole table with to_sql(if_exists='replace'), which is how the tconst column got
silently blanked; every write here is a targeted UPDATE instead.

Usage:
  python weekly_refresh.py                  # full run — no other script needed
  python weekly_refresh.py --dry-run        # preview only, no writes
  python weekly_refresh.py --skip-tmdb      # skip TMDB enrichment (slow)
  python weekly_refresh.py --skip-wiki      # skip Wikipedia fetch
  python weekly_refresh.py --skip-cache     # skip cache rebuild
  python weekly_refresh.py --skip-integrity # skip the Phase 0 gate
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
from datetime import datetime, timezone, timedelta
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
    for col in ("rt_status", "rt_checked_at", "last_tmdb_check",
                "poster_status", "poster_checked_at"):
        if col not in existing:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {col} TEXT")
            conn.commit()
    #attempt counters. A flat 30-day cooldown still retries a hopeless film forever;
    #these let a fetch give up for good after MAX_FETCH_ATTEMPTS.
    for col in ("dtdd_status", "dtdd_checked_at"):
        if col not in existing:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {col} TEXT")
            conn.commit()
    for col in ("rt_attempts", "poster_attempts", "wiki_attempts", "dtdd_attempts"):
        if col not in existing:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {col} INTEGER DEFAULT 0")
            conn.commit()
    #these indexes turn the Phase 3/4 eligibility scans from full-table reads into
    #index lookups; the table shipped with no indexes at all.
    for name, ddl in (
        ("idx_movies_id",       "CREATE INDEX IF NOT EXISTS idx_movies_id ON movies(id)"),
        ("idx_movies_tconst",   "CREATE INDEX IF NOT EXISTS idx_movies_tconst ON movies(tconst)"),
        ("idx_movies_is_valid", "CREATE INDEX IF NOT EXISTS idx_movies_is_valid ON movies(is_valid)"),
    ):
        conn.execute(ddl)
    conn.commit()
    conn.execute("""
        UPDATE movies SET validated_at='2000-01-01'
        WHERE is_valid=1 AND validated_at IS NULL
    """)
    conn.commit()
    return conn


#protected films — high-profile collision-prone films that dedup/demotion/tconst logic
#has mishandled before. Not blocked from operations, just logged loudly when touched.
PROTECTED_TCONSTS = {
    'tt2316411': 'Enemy',
    'tt0375679': 'Crash',
    'tt33764258': 'The Odyssey',
    'tt4972582': 'Split',
    'tt2798920': 'Annihilation',
    'tt0075314': 'Taxi Driver',
    'tt1396484': 'It',
}


def _warn_if_protected(tconst, title, action):
    name = PROTECTED_TCONSTS.get(str(tconst or '').strip())
    if name:
        log.warning(f"  [PROTECTED FILM] {action}: tconst={tconst} '{title}' "
                    f"(protected list entry: {name}) — verify this is correct")


RECHECK_DAYS = 30  # cooldown before retrying a failed wiki/RT fetch
TMDB_RECHECK_DAYS = 6  # cooldown before re-enriching an already-checked film

#Retry policy. The weekly refresh is about CURRENT films; without these bounds
#Phase 4 re-scraped ~16,000 RT pages every run (88% of them pre-2023, 3,953
#pre-2000) at 0.5s each — over two hours of guaranteed misses, every week.
MAX_FETCH_ATTEMPTS = 3   # give up on a film after this many failed tries
RECENT_YEARS = 2         # "current" window for routine re-attempts
#Release date is a poor predictor of whether a fetch will succeed; vote count is a
#much better one. 12 Monkeys (1995, 683K votes) certainly has an RT page, a 1943
#obscurity does not. Without this override, recency-gating silently abandoned
#Star Wars (1.58M votes, no wiki plot) and 12 Monkeys forever. Films at or above
#this threshold stay eligible at any age, still bounded by MAX_FETCH_ATTEMPTS.
HIGH_VALUE_VOTES = 50000
#Old films are NOT auto-attempted. A miss on a pre-2000 obscurity costs ~10s
#(Wikidata lookup + three title-variant queries, each with its own sleep) and
#almost never succeeds — draining that backlog automatically added hours per run
#for nearly nothing. Use --all-years for a deliberate historical sweep.
BACKLOG_LIMIT = 0        # never-before-tried OLD films to drain per run


def _fetch_eligibility_sql(status_col, checked_col, attempts_col, recent_years,
                           retry_exhausted=False):
    """SQL fragment limiting a fetch phase to films actually worth attempting.

    Three rules, in order of importance:
      1. Never exceed MAX_FETCH_ATTEMPTS — a film that has failed this many times
         is treated as permanently unavailable, not retried monthly forever.
      2. Respect the cooldown after a recorded failure.
      3. Only re-attempt films released inside the recent window. Older films get
         exactly ONE attempt ever (when attempts=0), so nothing is skipped
         silently, but nothing is ground over repeatedly either.

    Returns (sql_fragment, params).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECHECK_DAYS)).isoformat()
    attempts = f"COALESCE({attempts_col}, 0)"
    frag = f"""
        AND (
            {attempts} = 0
            OR (
                {'1=1' if retry_exhausted else f'{attempts} < {MAX_FETCH_ATTEMPTS}'}
                AND (
                    release_date >= date('now', '-{int(recent_years)} years')
                    OR CAST(vote_count AS REAL) >= {HIGH_VALUE_VOTES}
                )
                AND NOT ({status_col} = 'failed' AND {checked_col} >= ?)
            )
        )
    """
    return frag, [cutoff]


def _record_fetch_result(conn, film_id, ok, status_col, checked_col, attempts_col,
                         extra_sql="", extra_params=()):
    """Mark a fetch attempt so the next run can skip or back off appropriately.

    Every attempt increments the counter, success or failure — that is what makes
    the attempt cap meaningful. Previously a failed poster fetch recorded nothing
    at all, so the same 618 films were retried on every single run.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(f"""
        UPDATE movies
        SET {status_col}=?, {checked_col}=?, {attempts_col}=COALESCE({attempts_col},0)+1
            {extra_sql}
        WHERE id=?
    """, ('ok' if ok else 'failed', now, *extra_params, film_id))


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


def fetch_tmdb_by_imdb_id(tconst):
    """Resolve an IMDb tconst to a TMDB movie id via /find, then pull full metadata.

    Going tconst-first is what makes acquisition safe: the IMDb ID is unambiguous,
    so a new film arrives already correctly linked and can never enter the
    (title, year) fuzzy-match lottery that made 15 different films named "Alone"
    claim the same identity.
    """
    try:
        r = requests.get(f"{TMDB_BASE}/find/{tconst}",
                         params={"api_key": TMDB_API_KEY, "external_source": "imdb_id"},
                         timeout=8)
        if r.status_code != 200:
            return None, f"find_{r.status_code}"
        results = r.json().get("movie_results") or []
        if not results:
            return None, "no_tmdb_match"
        return fetch_tmdb_metadata(results[0]["id"])
    except Exception as e:
        return None, str(e)


def _people_str(credits, jobs=None, dept=None, limit=None):
    """Space-joined CamelCase names, matching how the DB stores dna_cast/crew."""
    out = []
    if jobs is None and dept is None:
        for p in (credits.get("cast") or [])[:limit or 15]:
            out.append(str(p.get("name", "")).replace(" ", ""))
    else:
        for p in credits.get("crew") or []:
            if (jobs and p.get("job") in jobs) or (dept and p.get("department") == dept):
                out.append(str(p.get("name", "")).replace(" ", ""))
    return " ".join(n for n in out if n)


def run_acquire_new_films(conn, dry_run, imdb_ratings, basics_by_tconst,
                          min_votes, limit):
    """PHASE 1b — insert films that exist on IMDb with real audiences but are
    missing from movies.db entirely.

    Every other phase only UPDATEs existing rows, so a film released last week was
    invisible to the refresh no matter how popular it got — you had to remember to
    run tmdb_fetch.py / import_new_movies.py / fetch_missing_by_imdb_id.py by hand.

    Driving off the IMDb vote threshold (rather than TMDB's "discover recent")
    means only films that already clear the validity bar get inserted, so this
    cannot flood the DB with the unreleased/zero-vote stubs that produced the
    ~75K blank-date backlog.
    """
    log.info(f"PHASE 1b: Reconciling IMDb films (votes >= {min_votes:,})")

    have = {r[0] for r in conn.execute(
        "SELECT tconst FROM movies WHERE tconst IS NOT NULL AND tconst != ''").fetchall()}
    candidates = [
        (tc, v[1]) for tc, v in imdb_ratings.items()
        if v[1] >= min_votes and tc not in have and tc in basics_by_tconst
    ]
    candidates.sort(key=lambda x: -x[1])
    log.info(f"  {len(candidates):,} IMDb films above the threshold are unlinked in the DB")
    if not candidates:
        return 0
    batch = candidates[:limit]
    if len(candidates) > limit:
        log.info(f"  taking the top {limit:,} by vote count this run "
                 f"(raise with --acquire-limit)")

    if dry_run:
        for tc, votes in batch[:10]:
            log.info(f"    [dry-run] would reconcile {basics_by_tconst[tc][0]!r} "
                     f"({basics_by_tconst[tc][1]}) {tc} — {votes:,} votes")
        return 0

    existing_ids = {r[0] for r in conn.execute(
        "SELECT id FROM movies WHERE id IS NOT NULL").fetchall()}

    inserted = linked = skipped = errors = 0
    chunks = [batch[i:i + TMDB_CHUNK] for i in range(0, len(batch), TMDB_CHUNK)]
    for ci, chunk in enumerate(chunks, 1):
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            futs = {pool.submit(fetch_tmdb_by_imdb_id, tc): (tc, v) for tc, v in chunk}
            for fut in as_completed(futs):
                tconst, imdb_votes = futs[fut]
                data, err = fut.result()
                if err or not data:
                    errors += 1
                    continue
                tmdb_id = data.get("id")
                if tmdb_id is None or tconst in have:
                    skipped += 1
                    continue

                #The film is usually already in the DB, just unlinked — Seven, the
                #original Star Wars trilogy and Dune all showed up as "missing"
                #simply because they had no tconst. Inserting would duplicate them.
                #TMDB's /find gave us an EXACT imdb->tmdb mapping, so link the
                #existing row instead. This is strictly safer than the title+year
                #fuzzy match, which is what created the collisions in the first place.
                if tmdb_id in existing_ids:
                    _warn_if_protected(tconst, data.get("title"), "exact-linking existing row")
                    conn.execute("""
                        UPDATE movies SET tconst=?, vote_count=?, vote_average=?
                        WHERE id=? AND (tconst IS NULL OR TRIM(tconst)='')
                    """, (tconst, imdb_votes, imdb_ratings[tconst][0], tmdb_id))
                    have.add(tconst)
                    linked += 1
                    if linked <= 10:
                        log.info(f"    [EXACT LINK] {data.get('title')!r} → {tconst} "
                                 f"({imdb_votes:,} votes) — was already in DB, unlinked")
                    continue

                rel = _normalize_date(data.get("release_date"))
                if not rel:
                    skipped += 1
                    continue
                runtime = data.get("runtime")
                if runtime is not None and (runtime <= 1 or runtime > 600):
                    runtime = None
                credits = data.get("credits") or {}
                avg = imdb_ratings[tconst][0]
                conn.execute("""
                    INSERT INTO movies
                      (id, title, original_title, release_date, runtime, overview,
                       dna_keywords, dna_genres, dna_cast, dna_director, dna_writer,
                       vote_average, vote_count, tconst, is_valid, validated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    tmdb_id, data.get("title") or basics_by_tconst[tconst][0],
                    data.get("original_title") or "", rel, runtime,
                    data.get("overview") or "",
                    _keywords_str(data), _genres_str(data),
                    _people_str(credits),
                    _people_str(credits, jobs={"Director"}),
                    _people_str(credits, jobs={"Writer", "Screenplay", "Story"}),
                    avg, imdb_votes, tconst,
                    1 if imdb_votes >= VOTE_THRESHOLD else 0,
                    datetime.now(timezone.utc).isoformat(),
                ))
                existing_ids.add(tmdb_id)
                have.add(tconst)
                inserted += 1
                if inserted <= 20:
                    log.info(f"    [NEW] {data.get('title')!r} ({rel[:4]}) "
                             f"{tconst} — {imdb_votes:,} votes")
        conn.commit()
        if ci % 5 == 0 or ci == len(chunks):
            log.info(f"    [ACQUIRE] chunk {ci}/{len(chunks)} — "
                     f"{inserted:,} inserted, {linked:,} exact-linked")
        if ci < len(chunks):
            time.sleep(TMDB_SLEEP)

    log.info(f"  [ACQUIRE] {inserted:,} new films inserted, {linked:,} existing rows "
             f"exact-linked, {skipped:,} skipped, {errors:,} errors")
    return inserted + linked


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
            _warn_if_protected(row["tconst"], row["title"], "demoting for bad release_date")
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
            _warn_if_protected(row["tconst"], row["title"], "demoting duplicate row")
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


def _resolve_tconst_collisions(conn, dry_run):
    """Resolve one-tconst-claimed-by-many-TMDB-ids collisions.

    These are NOT duplicate rows. IMDb linking matches on (clean_title, year),
    so every distinct film sharing a generic title with a real one ("Alone",
    "Home", "Silence") gets the same tconst, inherits its IMDb vote_count, and
    becomes spuriously is_valid=1. Fifteen unrelated 2020 films named "Alone"
    all carried the real film's 36,386 votes.

    Demoting by rowid (the old behavior) was wrong — it silently killed
    legitimately distinct films while leaving the stolen tconst and fabricated
    vote_count in place on the survivor's rivals. Instead, ask TMDB for each
    member's OWN vote_count: the real film has thousands, the 1-minute short
    has ~zero. Winner keeps the tconst; the rest are released with their true
    votes restored so the validity pass can re-judge them honestly.
    """
    rows = conn.execute("""
        SELECT id, title, release_date, tconst, vote_count
        FROM movies
        WHERE tconst IS NOT NULL AND tconst != ''
          AND tconst IN (
            SELECT tconst FROM movies
            WHERE is_valid=1 AND tconst IS NOT NULL AND tconst != ''
            GROUP BY tconst HAVING COUNT(*) > 1)
    """).fetchall()

    if not rows:
        log.info("  [TCONST] no collisions — every tconst claimed by one film")
        return 0

    groups = {}
    for row in rows:
        groups.setdefault(row["tconst"], []).append(row)
    log.info(f"  [TCONST] {len(groups):,} collisions covering {len(rows):,} rows "
             f"— querying TMDB for each film's own vote_count")

    if dry_run:
        for tconst, members in list(groups.items())[:5]:
            log.info(f"    [dry-run] {tconst} claimed by {len(members)} films: "
                     f"{', '.join(str(m['id']) for m in members[:6])}")
        log.info(f"  [dry-run] would resolve {len(groups):,} collisions")
        return 0

    all_ids = [r["id"] for r in rows]
    tmdb_votes = {}
    gone = set()   # 404 — TMDB deleted/merged this entry; permanent, not a blip
    chunks = [all_ids[i:i + TMDB_CHUNK] for i in range(0, len(all_ids), TMDB_CHUNK)]
    for i, chunk in enumerate(chunks, 1):
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            futs = {pool.submit(fetch_tmdb_metadata, fid): fid for fid in chunk}
            for fut in as_completed(futs):
                data, err = fut.result()
                if err == "not_found":
                    #TMDB removed this id, almost always because it was a duplicate
                    #entry merged into the real film. Treating it as "0 votes" is
                    #correct: it loses the tconst to the surviving row. Deferring it
                    #as if transient would strand the group forever.
                    gone.add(futs[fut])
                    tmdb_votes[futs[fut]] = (0.0, 0.0)
                elif not err and data:
                    tmdb_votes[futs[fut]] = (float(data.get("vote_count") or 0),
                                             float(data.get("vote_average") or 0))
        if i % 10 == 0 or i == len(chunks):
            log.info(f"    [TCONST] chunk {i}/{len(chunks)} — {len(tmdb_votes):,} resolved "
                     f"({len(gone):,} deleted on TMDB)")
        if i < len(chunks):
            time.sleep(TMDB_SLEEP)

    released = 0
    deferred = 0
    for tconst, members in groups.items():
        #a film TMDB didn't answer for has no usable vote signal this run. Treating
        #a timeout as "zero votes" would strip a real film's votes and demote it,
        #so the whole group is left untouched and retried next run.
        if any(m["id"] not in tmdb_votes for m in members):
            deferred += 1
            continue
        scored = sorted(members, key=lambda m: -tmdb_votes[m["id"]][0])
        winner = scored[0]
        _warn_if_protected(tconst, winner["title"], "awarding tconst after collision")
        for loser in scored[1:]:
            votes, avg = tmdb_votes[loser["id"]]
            conn.execute("""
                UPDATE movies SET tconst=NULL, vote_count=?, vote_average=?,
                       is_valid=CASE WHEN ? >= ? THEN 1 ELSE 0 END
                WHERE id=?
            """, (votes, avg, votes, VOTE_THRESHOLD, loser["id"]))
            released += 1
        log.info(f"    [TCONST] {tconst} -> id={winner['id']} '{winner['title']}' "
                 f"({tmdb_votes[winner['id']][0]:,.0f} TMDB votes); "
                 f"released {len(scored)-1} impostor(s)"
                 + (f", {sum(1 for m in scored[1:] if m['id'] in gone)} deleted on TMDB"
                    if any(m["id"] in gone for m in scored[1:]) else ""))
    conn.commit()
    log.info(f"  [TCONST] {len(groups)-deferred:,} collisions resolved, "
             f"{released:,} impostor rows released "
             f"({len(gone):,} of them deleted on TMDB), "
             f"{deferred:,} deferred (transient error only, retried next run)")
    return released


def _build_linked_vote_index(conn):
    """vote_count -> [linked live rows], for variant-duplicate detection."""
    idx = {}
    for r in conn.execute("""
        SELECT title, release_date, vote_count, tconst FROM movies
        WHERE is_valid=1 AND tconst IS NOT NULL AND TRIM(tconst)!=''
    """):
        try:
            idx.setdefault(round(float(r["vote_count"] or 0)), []).append(r)
        except (TypeError, ValueError):
            continue
    return idx


def _variant_twin(row, vote_index):
    """Return the linked live film `row` duplicates under a punctuation variant.

    Requires identical vote_count, release year within 1, and a loose title match.
    """
    try:
        v = round(float(row["vote_count"] or 0))
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    oy = str(row["release_date"] or "")[:4]
    if not oy.isdigit():
        return None
    for cand in vote_index.get(v, []):
        cy = str(cand["release_date"] or "")[:4]
        if not cy.isdigit() or abs(int(oy) - int(cy)) > 1:
            continue
        if _titles_loosely_match(row["title"], cand["title"]):
            return cand
    return None


def _refetch_fabricated_votes(conn, dry_run):
    """Recover true vote counts for unlinked rows carrying a linked film's numbers.

    The old title+year matcher gave unlinked rows the vote_count of whatever film
    it matched, so an unlinked row whose vote_count is byte-identical to a live
    linked film's is showing a fabricated number.

    Crucially these are usually NOT duplicates. "Mother" (2017, Rodrigo Sorogoyen,
    original_title "Madre") sat at exactly mother!'s 269,567 votes, and "(M)Other"
    (Antonia Hungerland) at exactly Mother's 2,928 — different directors, different
    films, stolen numbers. Demoting on the title+votes match would have deleted
    real films, so this asks TMDB for each row's OWN vote_count instead and lets
    the validity pass judge honestly. Genuine junk rows fall below the threshold on
    their own; real films keep their place with correct numbers.
    """
    vote_index = _build_linked_vote_index(conn)
    orphans = conn.execute("""
        SELECT id, title, release_date, vote_count FROM movies
        WHERE is_valid=1 AND (tconst IS NULL OR TRIM(tconst)='')
    """).fetchall()
    suspect = [(o, _variant_twin(o, vote_index)) for o in orphans]
    suspect = [(o, t) for o, t in suspect if t is not None]

    if not suspect:
        log.info("  [VOTES] no unlinked rows carrying a linked film's vote count")
        return 0

    log.info(f"  [VOTES] {len(suspect)} of {len(orphans)} unlinked valid rows show a vote "
             f"count identical to a linked film — re-fetching their real numbers from TMDB")
    if dry_run:
        for o, t in suspect[:8]:
            log.info(f"    [dry-run] '{o['title']}' ({str(o['release_date'])[:4]}) "
                     f"shows {float(o['vote_count'] or 0):,.0f} — same as '{t['title']}' {t['tconst']}")
        return 0

    ids = [o["id"] for o, _ in suspect]
    fixed = 0
    chunks = [ids[i:i + TMDB_CHUNK] for i in range(0, len(ids), TMDB_CHUNK)]
    for ci, chunk in enumerate(chunks, 1):
        results = {}
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            futs = {pool.submit(fetch_tmdb_metadata, fid): fid for fid in chunk}
            for fut in as_completed(futs):
                data, err = fut.result()
                if err == "not_found":
                    results[futs[fut]] = (0.0, 0.0)   # gone from TMDB — genuinely junk
                elif not err and data:
                    results[futs[fut]] = (float(data.get("vote_count") or 0),
                                          float(data.get("vote_average") or 0))
        for o, t in suspect:
            if o["id"] not in results:
                continue   # transient failure: leave the row untouched, retry next run
            votes, avg = results[o["id"]]
            conn.execute("""
                UPDATE movies SET vote_count=?, vote_average=?,
                       is_valid=CASE WHEN ? >= ? THEN 1 ELSE 0 END
                WHERE id=?
            """, (votes, avg, votes, VOTE_THRESHOLD, o["id"]))
            if fixed < 15:
                verdict = "stays valid" if votes >= VOTE_THRESHOLD else "demoted"
                log.info(f"    [VOTES] '{o['title']}' {float(o['vote_count'] or 0):,.0f} "
                         f"-> {votes:,.0f} real TMDB votes ({verdict})")
            fixed += 1
        conn.commit()
        if ci < len(chunks):
            time.sleep(TMDB_SLEEP)

    log.info(f"  [VOTES] {fixed} rows corrected to their real TMDB vote counts")
    return fixed


WIKI_AUDIT_CSV = "data/wiki_title_mismatches.csv"


def _audit_wiki_titles(conn, dry_run, null_mismatches=False):
    """Flag films whose stored wiki_plot came from a suspiciously-titled article.

    Title-based Wikipedia lookup is a guess, and a wrong guess is silent: the film
    keeps a plausible-looking plot that actually belongs to another movie, which
    then poisons the wiki, wiki_semantic and keyword channels with no error
    anywhere. This compares the DB title against the article the plot was taken
    from (wiki_title) — pure string work, no network, so it can run every week.

    Reports by default; --fix-wiki-mismatches nulls the plot so the next run
    re-fetches it through the title-guarded path.
    """
    import csv as _csv
    rows = conn.execute("""
        SELECT id, title, original_title, release_date, wiki_title, vote_count
        FROM movies
        WHERE is_valid=1
          AND wiki_plot IS NOT NULL AND TRIM(wiki_plot) != ''
          AND wiki_title IS NOT NULL AND TRIM(wiki_title) != ''
    """).fetchall()

    #Check original_title too. A film legitimately released under two names —
    #Pirate Radio / "The Boat That Rocked", or The X Files / "The X-Files (film)" —
    #has a correct plot from an article whose name matches only the other title.
    #Flagging those as corruption would bury the real hits in noise.
    def _title_ok(r):
        return (_titles_loosely_match(r["title"], r["wiki_title"])
                or (r["original_title"]
                    and _titles_loosely_match(r["original_title"], r["wiki_title"])))

    def _year_conflict(r):
        """True when wiki_title names a year that contradicts the film's own.

        Title matching alone is far too permissive: "Arrival" (2016) shares the
        word "arrival" with "The Arrival (1991 film)", so a loose match accepted a
        plot from a different film 25 years apart — on an 871K-vote title. When the
        article name carries a year, it is decisive evidence.
        """
        wm = re.search(r'\((\d{4})\b', str(r["wiki_title"] or ""))
        dm = re.match(r'^(\d{4})', str(r["release_date"] or ""))
        if not wm or not dm:
            return False
        return abs(int(wm.group(1)) - int(dm.group(1))) > 1

    #A stored plot taken from a disambiguation/name/novel page is wrong no matter
    #how well the titles match, so this is checked independently of _title_ok.
    #Without it the audit could never see them and they were never re-fetched.
    bad = [r for r in rows
           if _non_film_article_type(r["wiki_title"])
           or (not _title_ok(r))
           or _year_conflict(r)]
    checked = len(rows)
    log.info(f"  [WIKI AUDIT] {checked:,} plots have a recorded source article; "
             f"{len(bad):,} look mismatched")

    if not bad:
        return 0

    for r in sorted(bad, key=lambda r: -float(r["vote_count"] or 0))[:10]:
        log.warning(f"    [WIKI AUDIT] '{r['title']}' ({str(r['release_date'])[:4]}) "
                    f"has plot from article {r['wiki_title']!r} "
                    f"({float(r['vote_count'] or 0):,.0f} votes)")

    Path("data").mkdir(exist_ok=True)
    with open(WIKI_AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "title", "release_date", "wiki_title", "vote_count"])
        w.writeheader()
        for r in sorted(bad, key=lambda r: -float(r["vote_count"] or 0)):
            w.writerow({k: r[k] for k in ("id", "title", "release_date", "wiki_title", "vote_count")})
    log.info(f"  [WIKI AUDIT] full list written to {WIKI_AUDIT_CSV}")

    if null_mismatches and not dry_run:
        conn.executemany("""UPDATE movies SET wiki_plot=NULL, wiki_title=NULL,
                            wiki_plot_status=NULL, wiki_attempts=0 WHERE id=?""",
                         [(r["id"],) for r in bad])
        conn.commit()
        log.info(f"  [WIKI AUDIT] nulled {len(bad):,} suspect plots — they will be re-fetched")
    elif not null_mismatches:
        log.info("  [WIKI AUDIT] reporting only; re-run with --fix-wiki-mismatches to null them")
    return len(bad)


def _enforce_validity(conn, dry_run):
    """Make is_valid agree with the vote threshold, in both directions.

    Promotion used to run only when Phase 1/2 reported updates, so a film that
    crossed the threshold during a skipped or failed phase could sit invisible
    indefinitely. Demotion never ran at all, which is how 484 films with fewer
    than 1,000 votes (429 of them with zero) stayed in the dropdown.

    A film is only promoted if it has a parseable ISO release_date and its
    tconst isn't already claimed by a live row — otherwise promoting would
    reintroduce the duplicate the collision resolver just cleaned up.
    """
    demote = conn.execute("""
        SELECT rowid, title, vote_count FROM movies
        WHERE is_valid=1 AND CAST(vote_count AS REAL) < ?
    """, (VOTE_THRESHOLD,)).fetchall()

    claimed = {r[0] for r in conn.execute(
        "SELECT DISTINCT tconst FROM movies WHERE is_valid=1 AND tconst IS NOT NULL AND tconst != ''"
    ).fetchall()}
    #a TMDB id may already have a live row, and several is_valid=0 rows can share
    #one id. Guarding only on tconst promoted three copies of "What's in a Name".
    live_ids = {r[0] for r in conn.execute(
        "SELECT DISTINCT id FROM movies WHERE is_valid=1 AND id IS NOT NULL"
    ).fetchall()}

    promote = []
    skipped_claimed = 0
    skipped_dup_id = 0
    for row in conn.execute("""
        SELECT rowid, id, title, tconst, vote_count FROM movies
        WHERE is_valid=0 AND CAST(vote_count AS REAL) >= ?
          AND (release_date LIKE '19__-__-__'
            OR release_date LIKE '20__-__-__'
            OR release_date LIKE '21__-__-__')
        ORDER BY CAST(vote_count AS REAL) DESC, rowid
    """, (VOTE_THRESHOLD,)).fetchall():
        tc = (row["tconst"] or "").strip()
        if tc and tc in claimed:
            skipped_claimed += 1
            continue
        if row["id"] is not None and row["id"] in live_ids:
            skipped_dup_id += 1
            continue
        if tc:
            claimed.add(tc)
        if row["id"] is not None:
            live_ids.add(row["id"])
        promote.append(row)

    log.info(f"  [VALIDITY] {len(demote):,} to demote (votes < {VOTE_THRESHOLD:,}), "
             f"{len(promote):,} to promote (votes >= {VOTE_THRESHOLD:,}), "
             f"{skipped_claimed:,} skipped — tconst already live, "
             f"{skipped_dup_id:,} skipped — TMDB id already live")

    if dry_run:
        for r in promote[:5]:
            log.info(f"    [dry-run] promote '{r['title']}' ({float(r['vote_count'] or 0):,.0f} votes)")
        return 0

    if demote:
        conn.executemany("UPDATE movies SET is_valid=0 WHERE rowid=?", [(r["rowid"],) for r in demote])
    if promote:
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany("UPDATE movies SET is_valid=1, validated_at=? WHERE rowid=?",
                         [(now, r["rowid"]) for r in promote])
        for r in promote[:10]:
            log.info(f"    [VALIDITY] promoted '{r['title']}' ({float(r['vote_count'] or 0):,.0f} votes)")
    conn.commit()
    return len(demote) + len(promote)


def run_integrity_gate(conn, dry_run, fix_wiki_mismatches=False):
    """PHASE 0 — assert and repair DB invariants before any network work.

    Everything downstream keys off tconst. When it is empty, Phase 2, the title
    sync, RT fetch, and the collision guard all quietly match zero rows and the
    refresh 'succeeds' while doing nothing. That failure is silent by nature, so
    it gets asserted here and aborts the run rather than wasting hours.
    """
    log.info("PHASE 0: Integrity gate")

    valid_total = conn.execute("SELECT COUNT(*) FROM movies WHERE is_valid=1").fetchone()[0]
    linked = conn.execute(
        "SELECT COUNT(*) FROM movies WHERE tconst IS NOT NULL AND TRIM(tconst) != ''"
    ).fetchone()[0]
    log.info(f"  [CHECK] {valid_total:,} valid films, {linked:,} rows carry a tconst")

    if valid_total > 1000 and linked == 0:
        log.error("─" * 60)
        log.error("FATAL: the tconst column is empty. IMDb vote sync, title sync,")
        log.error("RT fetch, and collision detection would all silently no-op.")
        log.error("")
        log.error("Known cause: merge_layers.py's pandas merge suffixes a")
        log.error("pre-existing tconst column to tconst_x/tconst_y, then writes an")
        log.error("all-NaN 'tconst' over the table. Restore from a backup that")
        log.error("still has it and do not re-run merge_layers.py.")
        log.error("─" * 60)
        sys.exit(1)

    _normalize_dates_in_db(conn, dry_run)
    _dedup_valid_films(conn, dry_run)
    _resolve_tconst_collisions(conn, dry_run)
    #must precede _enforce_validity: it rewrites vote_count, and validity
    #decisions should be made on the corrected numbers
    _refetch_fabricated_votes(conn, dry_run)
    _enforce_validity(conn, dry_run)
    #promotion can surface a duplicate that did not exist when the first dedup ran,
    #so sweep again afterward. Cheap (indexed) and makes the gate self-consistent.
    _dedup_valid_films(conn, dry_run)

    #A row claiming wiki_plot_status='ok' with no actual plot is stale metadata —
    #an older cleanup nulled bad plots without resetting status. Left alone it is
    #merely confusing, but it also misreports coverage. Reset so the film is
    #treated honestly as "no plot yet".
    stale = conn.execute("""
        SELECT COUNT(*) FROM movies
        WHERE is_valid=1 AND (wiki_plot IS NULL OR TRIM(wiki_plot)='')
          AND wiki_plot_status='ok'
    """).fetchone()[0]
    if stale:
        log.info(f"  [WIKI] {stale} rows claim status='ok' but hold no plot — resetting")
        if not dry_run:
            conn.execute("""
                UPDATE movies SET wiki_plot_status=NULL, wiki_title=NULL
                WHERE is_valid=1 AND (wiki_plot IS NULL OR TRIM(wiki_plot)='')
                  AND wiki_plot_status='ok'
            """)
            conn.commit()

    _audit_wiki_titles(conn, dry_run, null_mismatches=fix_wiki_mismatches)

    log.info("  [CHECK] post-gate state:")
    for label, sql in [
        ("valid films",                "SELECT COUNT(*) FROM movies WHERE is_valid=1"),
        ("valid w/ votes < threshold", f"SELECT COUNT(*) FROM movies WHERE is_valid=1 AND CAST(vote_count AS REAL) < {VOTE_THRESHOLD}"),
        ("invalid w/ votes >= thresh", f"SELECT COUNT(*) FROM movies WHERE is_valid=0 AND CAST(vote_count AS REAL) >= {VOTE_THRESHOLD} AND (release_date LIKE '19__-__-__' OR release_date LIKE '20__-__-__' OR release_date LIKE '21__-__-__')"),
        ("valid w/ blank date",        "SELECT COUNT(*) FROM movies WHERE is_valid=1 AND (release_date IS NULL OR release_date='')"),
        ("duplicate-id groups",        "SELECT COUNT(*) FROM (SELECT id FROM movies WHERE is_valid=1 AND id IS NOT NULL GROUP BY id HAVING COUNT(*)>1)"),
        ("duplicate-tconst groups",    "SELECT COUNT(*) FROM (SELECT tconst FROM movies WHERE is_valid=1 AND tconst IS NOT NULL AND tconst!='' GROUP BY tconst HAVING COUNT(*)>1)"),
    ]:
        log.info(f"    {label:28} {conn.execute(sql).fetchone()[0]:,}")


def run_blank_date_refetch(conn, dry_run, limit):
    """Re-fetch TMDB metadata for is_valid=0 stubs with a blank release_date.

    These fall through every other phase: Phase 1's query requires a 2024-2026
    date match (blank fails it), and the stub-linker requires a parseable year
    to fuzzy-match against IMDb (blank fails that too) — so a real, released
    film can sit here forever if TMDB's own snapshot was taken pre-release.
    ~75K rows in the DB have this shape, almost all genuinely unreleased/junk
    TMDB entries, so this can't run unbounded — it prioritizes by whatever
    vote_count the row already has (the strongest available "this might be
    real" signal) and takes the top `limit` rows per run.
    """
    log.info(f"PHASE 1a: Blank-date stub re-fetch (top {limit} by vote_count)")
    rows = conn.execute("""
        SELECT id, title, vote_count
        FROM movies
        WHERE is_valid = 0 AND (release_date IS NULL OR release_date = '')
        ORDER BY CAST(vote_count AS REAL) DESC
        LIMIT ?
    """, (limit,)).fetchall()
    log.info(f"  {len(rows)} blank-date stubs targeted (of ~75K total in this shape)")

    updated = 0
    errors = 0
    chunks = [rows[i:i+TMDB_CHUNK] for i in range(0, len(rows), TMDB_CHUNK)]
    for chunk_idx, chunk in enumerate(chunks):
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            futs = {pool.submit(fetch_tmdb_metadata, row["id"]): row for row in chunk}
            for fut in as_completed(futs):
                row = futs[fut]
                data, err = fut.result()
                if err:
                    errors += 1
                    continue
                new_date = _normalize_date(data.get("release_date"))
                if not new_date:
                    continue  # still unreleased/no date on TMDB's side either
                new_kw  = _keywords_str(data)
                new_gen = _genres_str(data)
                new_vc  = float(data.get("vote_count") or 0)
                new_va  = float(data.get("vote_average") or 0)
                raw_rt  = data.get("runtime")
                if raw_rt is not None and (raw_rt <= 1 or raw_rt > 600):
                    raw_rt = None
                log.info(f"    [BLANK DATE] '{row['title']}' → release_date={new_date}, "
                         f"votes={new_vc:,.0f}")
                if not dry_run:
                    conn.execute("""
                        UPDATE movies
                        SET title=?, original_title=?, release_date=?, runtime=?, overview=?,
                            dna_keywords=?, dna_genres=?, vote_count=?, vote_average=?
                        WHERE id=?
                    """, (
                        data.get("title") or row["title"], data.get("original_title") or row["title"],
                        new_date, raw_rt, data.get("overview") or "",
                        new_kw, new_gen, new_vc, new_va, row["id"],
                    ))
                updated += 1
        if not dry_run:
            conn.commit()
        if chunk_idx < len(chunks) - 1:
            time.sleep(TMDB_SLEEP)

    log.info(f"  [BLANK DATE] {updated} stubs gained a real release_date, {errors} errors")
    return updated


def run_tmdb_enrichment(conn, dry_run, tmdb_min_votes=None):
    log.info("PHASE 1: TMDB enrichment")
    recheck_cutoff = (datetime.now(timezone.utc) - timedelta(days=TMDB_RECHECK_DAYS)).isoformat()
    params = [recheck_cutoff]
    votes_clause = ""
    if tmdb_min_votes is not None:
        votes_clause = "AND CAST(vote_count AS REAL) >= ?"
        params.append(tmdb_min_votes)
    rows = conn.execute(f"""
        SELECT id, title, release_date, vote_count, dna_keywords, dna_genres, runtime, tconst
        FROM movies
        WHERE (
            (release_date LIKE '2024%' OR release_date LIKE '2025%' OR release_date LIKE '2026%') AND CAST(vote_count AS REAL) >= 1000
            OR (
                CAST(vote_count AS REAL) >= 1000
                AND CAST(vote_count AS REAL) < 50000
                AND (dna_keywords IS NULL OR TRIM(dna_keywords) = '')
            )
            --valid films still missing a runtime. Folded in from fix_runtime.py; this
            --phase already writes runtime, it just never targeted these rows.
            OR (is_valid = 1 AND (runtime IS NULL OR runtime = 0))
        )
        AND (last_tmdb_check IS NULL OR last_tmdb_check < ?)
        {votes_clause}
        ORDER BY CAST(vote_count AS REAL) DESC
    """, params).fetchall()
    log.info(f"  {len(rows)} films targeted for TMDB enrichment "
             f"(delta: last_tmdb_check older than {TMDB_RECHECK_DAYS}d or unset"
             + (f", tmdb_min_votes={tmdb_min_votes:,})" if tmdb_min_votes is not None else ")"))

    updated = 0
    errors  = 0
    chunks  = [rows[i:i+TMDB_CHUNK] for i in range(0, len(rows), TMDB_CHUNK)]

    for chunk_idx, chunk in enumerate(chunks):
        results = {}
        checked_ids = []
        with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
            futs = {pool.submit(fetch_tmdb_metadata, row["id"]): row for row in chunk}
            for fut in as_completed(futs):
                row  = futs[fut]
                data, err = fut.result()
                if err == "rate_limited":
                    log.warning(f"    rate limited on {row['title']} — will retry next run")
                    errors += 1
                elif err == "not_found":
                    errors += 1
                    checked_ids.append(row["id"])
                elif err:
                    errors += 1
                elif data:
                    results[row["id"]] = (row, data)
                    checked_ids.append(row["id"])

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

            #IMDb is the authoritative vote source once a tconst is linked (Phase 2
            #syncs it from the real IMDb TSV). TMDB's own vote_count is typically far
            #smaller and must not clobber it here — only use TMDB's vote_count for
            #rows that aren't linked to IMDb yet, where it's the only signal available.
            has_tconst = bool((row["tconst"] or "").strip())

            changed = (
                new_kw  != (row["dna_keywords"] or "") or
                new_gen != (row["dna_genres"]   or "") or
                (not has_tconst and abs(new_vc - float(row["vote_count"] or 0)) > 1) or
                (new_rt is not None and new_rt != row["runtime"])
            )
            if changed:
                if not dry_run:
                    if has_tconst:
                        conn.execute("""
                            UPDATE movies
                            SET dna_keywords=?, dna_genres=?, runtime=?
                            WHERE id=?
                        """, (new_kw or row["dna_keywords"],
                              new_gen or row["dna_genres"],
                              new_rt, tmdb_id))
                    else:
                        conn.execute("""
                            UPDATE movies
                            SET dna_keywords=?, dna_genres=?, vote_count=?, vote_average=?, runtime=?
                            WHERE id=?
                        """, (new_kw or row["dna_keywords"],
                              new_gen or row["dna_genres"],
                              new_vc, new_va, new_rt, tmdb_id))
                updated += 1

        if not dry_run and checked_ids:
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                "UPDATE movies SET last_tmdb_check=? WHERE id=?",
                [(now_iso, cid) for cid in checked_ids]
            )

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


def _parse_imdb_basics(basics_raw):
    """Parse title.basics.tsv text.

    Returns:
      basics_by_tconst: tconst -> (primaryTitle, startYear)
      basics_index:      (clean_title, startYear) -> tconst   (titleType='movie' only)
    """
    import html
    basics_by_tconst = {}
    basics_index = {}
    for line in basics_raw.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        tconst, title_type, primary_title = parts[0], parts[1], html.unescape(parts[2])
        start_year = parts[5]
        if title_type != "movie":
            continue
        if start_year == r"\N" or not start_year.isdigit():
            continue
        basics_by_tconst[tconst] = (primary_title, start_year)
        key = (_normalize_title_for_match(primary_title), start_year)
        if key not in basics_index:  # first hit wins; collisions are rare enough to ignore
            basics_index[key] = tconst
    return basics_by_tconst, basics_index


def _correct_dates_against_imdb(conn, dry_run, basics_by_tconst, min_gap=90):
    """Fix CENTURY-FLIPPED release dates using IMDb's startYear as the authority.

    _normalize_date resolves a 2-digit year as `yy > current_yy ? 1900+yy : 2000+yy`,
    so silent-era films land exactly a century late: The Kid (1921) -> 2021,
    Nosferatu (1922) -> 2022, Sunrise (1927) -> 2027, a date in the future. Beyond
    being wrong in the UI, these films look brand-new to the Phase 3/4 recency
    window and get re-fetched every single run.

    Deliberately narrow: only a gap of >= `min_gap` years is treated as corruption,
    because that is the unmistakable signature of the century flip. Smaller
    disagreements are usually production-year vs release-year and are NOT bugs —
    Cannibal! The Musical (shot 1993, released 1996) and Kill Bill: The Whole Bloody
    Affair (2004 cut, 2011 screening) are both legitimately dated in the DB, and
    a looser tolerance would silently rewrite them.
    """
    rows = conn.execute("""
        SELECT rowid, title, release_date, tconst FROM movies
        WHERE is_valid=1 AND tconst IS NOT NULL AND tconst != ''
          AND release_date IS NOT NULL AND release_date != ''
    """).fetchall()

    fixed = 0
    for row in rows:
        entry = basics_by_tconst.get(str(row["tconst"]).strip())
        if not entry:
            continue
        imdb_year = entry[1]
        if not str(imdb_year).isdigit():
            continue
        db_year_m = re.match(r'^(\d{4})', str(row["release_date"]))
        if not db_year_m:
            continue
        db_year = int(db_year_m.group(1))
        if abs(db_year - int(imdb_year)) < min_gap:
            continue
        corrected = f"{int(imdb_year):04d}{str(row['release_date'])[4:]}"
        log.info(f"    [DATE FIX] '{row['title']}' {row['release_date']} -> {corrected} "
                 f"(IMDb startYear={imdb_year}, tconst={row['tconst']})")
        if not dry_run:
            conn.execute("UPDATE movies SET release_date=? WHERE rowid=?",
                         (corrected, row["rowid"]))
        fixed += 1

    if not dry_run and fixed:
        conn.commit()
    log.info(f"  [DATE FIX] {fixed} release dates corrected against IMDb")
    return fixed


def _link_missing_tconsts(conn, dry_run, basics_index, imdb_ratings):
    """Link any film missing a tconst to IMDb by title+year, and pull in votes.

    Supersedes both the old 2023-2026-only stub linker and merge_layers.py's
    bulk linking pass. Two differences from merge_layers, which is why that
    script is retired:

      1. One tconst, one film. merge_layers let every film sharing a
         (clean_title, year) key claim the same tconst, so unrelated films
         inherited a real film's votes and went live. A tconst already held by
         another row is refused here.
      2. Targeted UPDATEs instead of to_sql(if_exists='replace'), so a partial
         failure can't rewrite the whole table or blank a column.
    """
    #Scoped deliberately. Linking EVERY unlinked row means 587K title+year guesses
    #on films like "Colorado Trail" (1938, 19 votes) — pure collision risk for rows
    #the app never shows. Only films that are live, or close enough to the threshold
    #to become live, are worth a fuzzy link at all.
    rows = conn.execute("""
        SELECT rowid, id, title, release_date, tconst, vote_count
        FROM movies
        WHERE (tconst IS NULL OR TRIM(tconst) = '')
          AND release_date IS NOT NULL AND release_date != ''
          AND title IS NOT NULL AND title != ''
          AND (is_valid = 1 OR CAST(vote_count AS REAL) >= ?)
    """, (VOTE_THRESHOLD,)).fetchall()
    log.info(f"  [LINK] {len(rows):,} live/near-threshold films missing a tconst "
             f"(unlinked obscure rows are left alone)")

    claimed = {r[0] for r in conn.execute(
        "SELECT DISTINCT tconst FROM movies WHERE tconst IS NOT NULL AND tconst != ''"
    ).fetchall()}

    linked = 0
    contested = 0
    for row in rows:
        year_m = re.search(r'\d{4}', str(row["release_date"] or ""))
        if not year_m:
            continue
        year = year_m.group(0)
        tconst = basics_index.get((_normalize_title_for_match(row["title"]), year))
        if not tconst or tconst not in imdb_ratings:
            continue
        if tconst in claimed:
            contested += 1
            continue
        new_avg, new_votes = imdb_ratings[tconst]
        _warn_if_protected(tconst, row["title"], "linking to tconst")
        if not dry_run:
            conn.execute(
                "UPDATE movies SET tconst=?, vote_count=?, vote_average=? WHERE rowid=?",
                (tconst, new_votes, new_avg, row["rowid"])
            )
        claimed.add(tconst)
        linked += 1
        if linked <= 15:
            log.info(f"    [LINK] '{row['title']}' ({year}) → {tconst} (votes={new_votes:,})")

    if not dry_run and linked:
        conn.commit()
    log.info(f"  [LINK] {linked:,} films linked, {contested:,} refused "
             f"(tconst already held by another film)")
    return linked


def run_imdb_updates(conn, dry_run, acquire=True, acquire_min_votes=10000,
                     acquire_limit=1200):
    log.info("PHASE 2: IMDb updates")
    try:
        ratings_raw = _download_tsv_gz(IMDB_RATINGS_URL)
    except Exception as e:
        log.error(f"  Failed to download IMDb ratings: {e}")
        return 0

    #parse ratings into dict: tconst → (avg_rating, num_votes)
    imdb_ratings = {}
    for line in ratings_raw.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            imdb_ratings[parts[0]] = (float(parts[1]), int(parts[2]))

    log.info(f"  Loaded {len(imdb_ratings):,} IMDb ratings")

    try:
        basics_raw = _download_tsv_gz(IMDB_BASICS_URL)
        basics_by_tconst, basics_index = _parse_imdb_basics(basics_raw)
        log.info(f"  Loaded {len(basics_by_tconst):,} IMDb movie titles")
    except Exception as e:
        log.error(f"  Failed to download IMDb basics: {e} — skipping stub-link and title-sync")
        basics_by_tconst, basics_index = {}, {}

    #correct century-flipped dates before anything keys off release_date
    if basics_by_tconst:
        _correct_dates_against_imdb(conn, dry_run, basics_by_tconst)

    #acquisition runs here because the IMDb tables are already parsed and in memory.
    #It must come BEFORE linking/vote-sync so newly inserted films are picked up by
    #the same run rather than waiting a week.
    acquired = 0
    if acquire and basics_by_tconst:
        acquired = run_acquire_new_films(conn, dry_run, imdb_ratings, basics_by_tconst,
                                         acquire_min_votes, acquire_limit)

    #link any film still missing a tconst (title+year); folded in from merge_layers.py
    linked = 0
    if basics_index:
        linked = _link_missing_tconsts(conn, dry_run, basics_index, imdb_ratings)

    rows = conn.execute("""
        SELECT id, title, tconst, vote_count, vote_average, is_valid
        FROM movies
        WHERE tconst IS NOT NULL AND tconst != ''
    """).fetchall()

    updated = 0
    title_synced = 0
    for row in rows:
        tconst = str(row["tconst"]).strip()

        #title sync: DB title should match IMDb's primaryTitle for the linked tconst.
        #Scoped to is_valid=1 only — the ~33K films the app actually shows. Applying
        #this to the full tconst-linked set (700K+ rows, mostly obscure/mismatched)
        #produced thousands of noisy or outright wrong syncs from bad title/year
        #collisions that don't matter for invisible rows but would for valid ones.
        basics_entry = basics_by_tconst.get(tconst)
        if basics_entry and row["is_valid"] == 1:
            imdb_title = basics_entry[0]
            if imdb_title and imdb_title != row["title"]:
                _warn_if_protected(tconst, row["title"], "title sync (TMDB title → IMDb title)")
                log.info(f"    [TITLE SYNC] '{row['title']}' → '{imdb_title}' (tconst={tconst})")
                if not dry_run:
                    conn.execute("UPDATE movies SET title=? WHERE id=?", (imdb_title, row["id"]))
                title_synced += 1

        if tconst not in imdb_ratings:
            continue
        new_avg, new_votes = imdb_ratings[tconst]
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
    log.info(f"  IMDb: {updated} films updated, {title_synced} titles synced, "
             f"{linked} linked, {acquired} newly acquired")
    return updated + linked + title_synced + acquired




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
    """Return True if film_title and wiki_title plausibly name the same film."""
    import unicodedata as _ud
    def _normalize(s):
        return _ud.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    _stop = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for",
             "is", "it", "be", "or", "by", "as", "up", "do"}
    def _words(s):
        return {w for w in re.sub(r"[^a-z0-9 ]", "", _normalize(s).replace('_', ' ').lower()).split() if w not in _stop}
    if _words(film_title) & _words(wiki_title):
        return True

    #Token overlap alone misses typographic variants. NFKD turns the superscript in
    #"Alien³" into "Alien3" — one token — while Wikipedia's "Alien 3" is two, so they
    #share nothing and a correct article gets rejected. Compare the squashed
    #alphanumeric forms as a second chance; "alien3" == "alien3" resolves it.
    def _squash(s):
        base = re.sub(r'\s*\((?:[^()]*)\)\s*$', '', str(s))   # drop a trailing "(1999 film)"
        return re.sub(r'[^a-z0-9]', '', _normalize(base).lower())
    a, b = _squash(film_title), _squash(wiki_title)
    if a and b and len(a) >= 4 and len(b) >= 4:
        if a == b or a.startswith(b) or b.startswith(a):
            return True
    return False


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


#Article types that cannot be a film's plot. A redirect does not make them right:
#"Müslüm" redirects to "Muslim (name)" and "The Candidate" to
#"Candidate (disambiguation)", whose bodies are bullet lists
#("== Film == * The Candidate (1959 film)"), not plots. The adaptation qualifiers
#matter too — a novel or TV series shares a title with the film but tells a
#different version of the story.
#Shared by the fetch guard AND the weekly audit: detecting this only at fetch time
#left already-stored bad plots invisible to the audit, so they were never re-fetched.
NON_FILM_ARTICLE_PATTERNS = (
    "(disambiguation)", "(name)", "(surname)", "(given name)",
    "(novel)", "(book)", "(tv series)", "(video game)",
    "(album)", "(song)", "(band)", "(magazine)", "(play)",
)


def _non_film_article_type(article_title):
    """Return the offending qualifier if the article is not about a film."""
    lower = str(article_title or "").lower()
    for pat in NON_FILM_ARTICLE_PATTERNS:
        if pat in lower:
            return pat.strip("()")
    return None


def _wiki_article_mismatch(film_title, film_year, article_title, via_redirect=False):
    """Reason the resolved article cannot belong to this film, or None if it can.

    Two independent checks, because the old fetcher's scoring failed on each:

      * TITLE — "Identity (2003 film)" shares no words with "X2: X-Men United",
        but its film-formatting bonuses (+8.5) outweighed the zero-overlap
        penalty (-6.0) and it cleared a 1.0 acceptance floor.
      * YEAR  — "The Arrival (1991 film)" earned +4.5 just for LOOKING like a film
        page; being the correct year was worth only +1.5. So a wrong-year article
        outscored the right one. Same story for Sing (1989 vs 2016), Anna (1951 vs
        2019), Doctor Dolittle (1967 vs 1998).

    An explicit year in the article name is decisive evidence, so it is checked
    independently of the title rather than traded off against it.
    """
    if not article_title:
        return None

    _bad_type = _non_film_article_type(article_title)
    if _bad_type:
        return f"non-film article ({article_title!r} is a {_bad_type} page)"

    #A Wikipedia redirect is Wikipedia asserting the two names denote the same
    #subject, which is exactly how alternate and international titles work:
    #"Forbidden Empire" -> "Viy (2014 film)", "Leap!" -> "Ballerina (2016 film)",
    #"A Cry in the Dark" -> "Evil Angels (film)". Title comparison cannot know
    #that, so the redirect is better evidence than our own string matching and
    #the title check is skipped. The year check still applies.
    if not via_redirect and not _titles_loosely_match(film_title, article_title):
        return f"title mismatch ({article_title!r} vs {film_title!r})"
    m = re.search(r'\((\d{4})\b', str(article_title))
    if m and film_year and str(film_year).isdigit():
        if abs(int(m.group(1)) - int(film_year)) > 1:
            return (f"year mismatch (article says {m.group(1)}, "
                    f"film is {film_year})")
    return None


#Records which Wikipedia article the last successful fetch actually resolved to,
#so run_wiki_plots can persist it into wiki_title. That column is what makes the
#Phase 0 title audit possible offline — without it, checking whether a stored plot
#belongs to the right film would mean re-fetching every article.
#Safe as module state: the wiki fetch loop is strictly sequential.
_LAST_WIKI_TITLE = {"title": None}


def _fetch_section1(query, headers, expect_title=None, expect_year=None):
    """Fetch section 1 of a Wikipedia article (almost always Plot) via rvsection=1.

    `expect_title`: when set, the page Wikipedia actually resolved to must share a
    meaningful word with it, or the result is rejected. Wikipedia follows redirects
    and resolves loose titles, so a bare-title query can land on a completely
    unrelated article — this is how "Tan Lines" once stored the plot of
    "Superman (2025 film)". Injecting a wrong plot silently poisons every TF-IDF
    and semantic channel for that film, so a miss is far cheaper than a bad hit.
    """
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
        _payload = r.json().get("query", {})
        _redirected = {d.get("to", "") for d in (_payload.get("redirects") or [])}
        pages = _payload.get("pages", {})
        for page in pages.values():
            if page.get("pageid", -1) == -1:
                continue
            resolved = page.get("title", "") or ""
            _bad = _wiki_article_mismatch(expect_title, expect_year, resolved,
                                          via_redirect=resolved in _redirected) if expect_title else None
            if _bad:
                print(f"    [Wiki guard] {query!r} resolved to {resolved!r} — rejecting: {_bad}")
                continue
            revs = page.get("revisions", [])
            if not revs:
                continue
            content = revs[0].get("slots", {}).get("main", {}).get("*", "") or revs[0].get("*", "")
            content = _strip_wiki_markup(content.strip())
            #strip leading == Plot == / == Synopsis == header line
            content = re.sub(r'^\s*==\s*(?:Plot|Synopsis|Story)\s*==\s*\n?', '', content, flags=re.IGNORECASE).strip()
            if content and len(content) >= 500:
                _LAST_WIKI_TITLE["title"] = resolved
                return content[:15000]
    except Exception as e:
        print(f"    [Wiki section1 crash] {e} for {query!r}")
    return None


def _fetch_wiki_extract(query, headers, expect_title=None, expect_year=None):
    """Fetch plot text for a Wikipedia page title.

    Primary:  rvsection=1 (section 1 of film articles is almost always Plot)
    Fallback: prop=extracts full article text, then regex plot extraction

    `expect_title` is passed through to both paths as a wrong-article guard.
    """
    #primary
    result = _fetch_section1(query, headers, expect_title=expect_title, expect_year=expect_year)
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

        _payload2 = r.json().get("query", {})
        _redirected2 = {d.get("to", "") for d in (_payload2.get("redirects") or [])}
        pages = _payload2.get("pages", {})
        for page in pages.values():
            if page.get("pageid", -1) == -1:
                continue
            resolved = page.get("title", "") or ""
            _bad = _wiki_article_mismatch(expect_title, expect_year, resolved,
                                          via_redirect=resolved in _redirected2) if expect_title else None
            if _bad:
                print(f"    [Wiki guard] {query!r} resolved to {resolved!r} — rejecting: {_bad}")
                continue
            extract = page.get("extract", "").strip()
            if not extract:
                continue
            _LAST_WIKI_TITLE["title"] = resolved

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
    _LAST_WIKI_TITLE["title"] = None  # clear so a miss can't inherit the previous film's page

    #Wikidata → exact Wikipedia title
    if tconst:
        wiki_title = _wikidata_title_from_tconst(tconst, title)
        if wiki_title:
            print(f"    [Wikidata] {tconst} → {wiki_title!r}")
            result = _fetch_wiki_extract(wiki_title, headers)
            if result:
                return result
            time.sleep(WIKI_SLEEP)

    #title-based fallback. Every variant is title-verified: these queries are
    #guesses, and an unverified guess is how a film ends up storing another
    #film's plot.
    for query in [f"{title} ({year} film)", f"{title} (film)", title]:
        result = _fetch_wiki_extract(query, headers, expect_title=title, expect_year=year)
        if result:
            return result
        time.sleep(WIKI_SLEEP)

    return None


def run_wiki_plots(conn, dry_run, min_votes=None, since_cutoff=None,
                   recent_years=RECENT_YEARS, backlog_limit=BACKLOG_LIMIT,
                   retry_exhausted=False):
    import csv
    log.info("PHASE 3: Wikipedia plots")
    threshold = min_votes if min_votes is not None else VOTE_THRESHOLD
    w_frag, w_params = _fetch_eligibility_sql(
        "wiki_plot_status", "wiki_plot_fetched_at", "wiki_attempts",
        recent_years, retry_exhausted)
    since_clause = "AND validated_at IS NOT NULL AND validated_at >= ?" if since_cutoff else ""
    since_params = [since_cutoff] if since_cutoff else []

    rows = conn.execute(f"""
        SELECT id, title, release_date, tconst, vote_count
        FROM movies
        WHERE wiki_plot IS NULL
          AND is_valid = 1
          AND CAST(vote_count AS REAL) >= ?
          {since_clause}
          {w_frag}
        ORDER BY CAST(vote_count AS REAL) DESC
    """, [threshold] + since_params + w_params).fetchall()
    rows = _bounded_backlog(rows, recent_years, backlog_limit, "WIKI")
    log.info(f"  {len(rows):,} films eligible for a wiki plot fetch "
             f"(min_votes={threshold:,}, attempts < {MAX_FETCH_ATTEMPTS}, "
             f"recent window {recent_years}y)")
    if not rows:
        log.info("  nothing eligible — skipping")
        return 0

    failures_path = "wiki_fetch_failures.csv"
    failure_rows = []

    fetched = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        _year_match = re.search(r'\d{4}', str(row["release_date"] or ""))
        year = _year_match.group(0) if _year_match else ""
        tconst = (row["tconst"] or "").strip() or None
        plot = _fetch_wiki_plot(row["title"], year, tconst=tconst)
        if plot:
            print(f"[{i}/{total}] {row['title']} ({year}) — ok ({len(plot):,} chars)")
            if not dry_run:
                #persist which article this came from — the Phase 0 audit reads it
                _record_fetch_result(conn, row["id"], True, "wiki_plot_status",
                                     "wiki_plot_fetched_at", "wiki_attempts",
                                     extra_sql=", wiki_plot=?, wiki_title=?",
                                     extra_params=(plot, _LAST_WIKI_TITLE["title"]))
            fetched += 1
            if fetched % 20 == 0:
                if not dry_run:
                    conn.commit()
                log.info(f"    {fetched} plots fetched so far.")
        else:
            print(f"[{i}/{total}] {row['title']} ({year}) — FAILED")
            if not dry_run:
                _record_fetch_result(conn, row["id"], False, "wiki_plot_status",
                                     "wiki_plot_fetched_at", "wiki_attempts")
            failure_rows.append({
                "id":           row["id"],
                "title":        row["title"],
                "release_date": row["release_date"],
                "tconst":     row["tconst"] or "",
                "vote_count":   row["vote_count"],
            })
        time.sleep(WIKI_SLEEP)

    if not dry_run:
        conn.commit()

    if failure_rows:
        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "title", "release_date", "tconst", "vote_count"])
            writer.writeheader()
            writer.writerows(failure_rows)
        log.info(f"  {len(failure_rows)} failures logged to {failures_path}")

    log.info(f"  Wiki: {fetched} plots fetched, {len(failure_rows)} failed")
    return fetched


def run_wiki_categories(conn, dry_run, min_votes=10000,
                        backfill_limit=60, skip_backfill=False):
    """PHASE 3a — Wikipedia categories -> category_tags (the `cattags` channel).

    Delegates to wiki_category_pipeline.py rather than duplicating its ~500 lines
    of strip rules and umbrella mappings. That module is import-safe (all work is
    behind `if __name__ == "__main__"`), and its three phases each take a conn.

    Free and deterministic — Wikipedia's public API only, no paid calls — so it
    belongs in the automatic weekly run. The Claude-API helix taggers deliberately
    do NOT live here; those stay manual so their spend can be watched.
    """
    log.info("PHASE 3a: Wikipedia categories -> category_tags")
    try:
        import wiki_category_pipeline as wcp
    except Exception as e:
        log.warning(f"  could not import wiki_category_pipeline ({e}) — skipping")
        return 0

    missing = conn.execute("""
        SELECT COUNT(*) FROM movies
        WHERE is_valid=1 AND CAST(vote_count AS REAL) >= ?
          AND wiki_categories IS NULL
    """, (min_votes,)).fetchone()[0]
    log.info(f"  {missing:,} valid films (>= {min_votes:,} votes) missing wiki_categories")

    if dry_run:
        log.info("  [dry-run] would run fetch -> backfill -> build")
        return 0

    try:
        if hasattr(wcp, "add_columns_if_missing"):
            wcp.add_columns_if_missing(conn)
        if missing:
            #Phase A batches 50 films per request — fast and safe to run in full.
            wcp.run_fetch(conn, min_votes)
            #Phase B is the opposite: several sequential lookups per film, heavily
            #rate-limited by Wikipedia (~2.6 films/min observed). Unbounded it ran
            #661 films = ~4.2 hours, which is exactly the kind of tail this
            #consolidation exists to eliminate. It is resumable — each run chips
            #away at the remainder — so cap it and let it drain across weeks.
            if skip_backfill or backfill_limit == 0:
                log.info("  Phase B (disambiguation backfill) skipped")
            else:
                wcp.run_backfill(conn, min_votes, limit=backfill_limit)
        #build is cheap local string processing and resumable, so always run it —
        #it also picks up rows whose categories arrived in an earlier run
        wcp.run_build(conn, False)
    except Exception as e:
        log.error(f"  wiki category pipeline failed: {e}")
        return 0

    built = conn.execute("""
        SELECT COUNT(*) FROM movies
        WHERE is_valid=1 AND category_tags IS NOT NULL AND TRIM(category_tags) != ''
    """).fetchone()[0]
    log.info(f"  {built:,} valid films now have category_tags")
    return built


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
    tconst = str(row["tconst"] or "").strip()
    if not tconst.startswith("tt"):
        return (row["id"], None)
    try:
        data = requests.get(
            f"http://www.omdbapi.com/?i={tconst}&apikey={OMDB_API_KEY}", timeout=6
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


def _bounded_backlog(rows, recent_years, limit, label):
    """Split candidates into always-do films and a capped old-film backlog.

    Always-do = released within `recent_years` OR carrying >= HIGH_VALUE_VOTES.
    The vote clause matters: gating purely on release date meant Star Wars
    (1.58M votes, no wiki plot) and 12 Monkeys (683K votes, no RT score) were
    never attempted, while the cap still keeps genuinely obscure old films from
    consuming the run. Remaining old films drain by vote_count order.
    """
    cutoff_year = datetime.now().year - int(recent_years)
    keep, old = [], []
    for r in rows:
        year_m = re.search(r'\d{4}', str(r["release_date"] or ""))
        is_recent = year_m and int(year_m.group(0)) >= cutoff_year
        try:
            high_value = float(r["vote_count"] or 0) >= HIGH_VALUE_VOTES
        except (KeyError, IndexError, TypeError, ValueError):
            high_value = False
        (keep if (is_recent or high_value) else old).append(r)
    kept_old = old[:limit]
    if old or keep:
        log.info(f"    [{label}] {len(keep):,} recent-or-popular + "
                 f"{len(kept_old):,} of {len(old):,} other older films this run")
    return keep + kept_old


def run_posters_scores(conn, dry_run, since_cutoff=None, skip_rt_backlog=False,
                       rt_min_votes=None, recent_years=RECENT_YEARS,
                       backlog_limit=BACKLOG_LIMIT, retry_exhausted=False):
    log.info("PHASE 4: Posters & RT scores")
    rt_votes_clause = "AND CAST(vote_count AS REAL) >= ?" if rt_min_votes is not None else ""
    rt_vote_params = [rt_min_votes] if rt_min_votes is not None else []
    since_clause = "AND validated_at IS NOT NULL AND validated_at >= ?" if since_cutoff else ""
    since_params = [since_cutoff] if since_cutoff else []

    #posters
    p_frag, p_params = _fetch_eligibility_sql(
        "poster_status", "poster_checked_at", "poster_attempts", recent_years, retry_exhausted)
    poster_rows = conn.execute(f"""
        SELECT id, release_date, vote_count FROM movies
        WHERE (poster IS NULL OR poster='') AND is_valid=1
          {since_clause} {p_frag}
        ORDER BY CAST(vote_count AS REAL) DESC
    """, since_params + p_params).fetchall()
    poster_rows = _bounded_backlog(poster_rows, recent_years, backlog_limit, "POSTER")

    #RT via OMDb (needs a tconst)
    r_frag, r_params = _fetch_eligibility_sql(
        "rt_status", "rt_checked_at", "rt_attempts", recent_years, retry_exhausted)
    rt_rows = conn.execute(f"""
        SELECT id, tconst, release_date, vote_count FROM movies
        WHERE rt_score IS NULL AND is_valid=1 AND tconst IS NOT NULL AND tconst != ''
          {since_clause} {rt_votes_clause} {r_frag}
        ORDER BY CAST(vote_count AS REAL) DESC
    """, since_params + rt_vote_params + r_params).fetchall()
    rt_rows = _bounded_backlog(rt_rows, recent_years, backlog_limit, "RT-OMDb")

    log.info(f"  {len(poster_rows):,} posters and {len(rt_rows):,} RT scores eligible "
             f"(attempts < {MAX_FETCH_ATTEMPTS}, recent window {recent_years}y, "
             f"backlog cap {backlog_limit})")

    posters_fetched = 0
    if not dry_run:
        chunks = [poster_rows[i:i+TMDB_CHUNK] for i in range(0, len(poster_rows), TMDB_CHUNK)]
        for chunk in chunks:
            with ThreadPoolExecutor(max_workers=TMDB_WORKERS) as pool:
                for film_id, poster_url, _ in pool.map(_fetch_poster, chunk):
                    #record every attempt, hit or miss. Only recording successes is
                    #what left 618 films with a NULL flag being retried forever.
                    if poster_url:
                        _record_fetch_result(conn, film_id, True, "poster_status",
                                             "poster_checked_at", "poster_attempts",
                                             extra_sql=", poster=?", extra_params=(poster_url,))
                        posters_fetched += 1
                    else:
                        _record_fetch_result(conn, film_id, False, "poster_status",
                                             "poster_checked_at", "poster_attempts")
            conn.commit()
            time.sleep(1)

    rt_fetched = 0
    if not dry_run:
        for row in rt_rows:
            film_id, score = _fetch_rt(row)
            if score is not None:
                _record_fetch_result(conn, film_id, True, "rt_status", "rt_checked_at",
                                     "rt_attempts", extra_sql=", rt_score=?",
                                     extra_params=(score,))
                rt_fetched += 1
            else:
                _record_fetch_result(conn, film_id, False, "rt_status",
                                     "rt_checked_at", "rt_attempts")
            time.sleep(0.1)
        conn.commit()

    #RT scrape fallback
    rt_scraped = 0
    if skip_rt_backlog:
        log.info("  RT scrape fallback skipped (--skip-rt-backlog)")
    else:
        s_frag, s_params = _fetch_eligibility_sql(
            "rt_status", "rt_checked_at", "rt_attempts", recent_years, retry_exhausted)
        rt_scrape_rows = conn.execute(f"""
            SELECT id, title, release_date, vote_count FROM movies
            WHERE is_valid=1 AND (rt_score IS NULL OR rt_score=0)
              {since_clause} {rt_votes_clause} {s_frag}
            ORDER BY CAST(vote_count AS REAL) DESC
        """, since_params + rt_vote_params + s_params).fetchall()
        rt_scrape_rows = _bounded_backlog(rt_scrape_rows, recent_years, backlog_limit, "RT-scrape")
        log.info(f"  RT scrape fallback: {len(rt_scrape_rows):,} films eligible")

        if not dry_run:
            for i, row in enumerate(rt_scrape_rows, 1):
                film_id, score = _fetch_rt_scrape(row)
                if score is not None:
                    _record_fetch_result(conn, film_id, True, "rt_status", "rt_checked_at",
                                         "rt_attempts", extra_sql=", rt_score=?",
                                         extra_params=(score,))
                    rt_scraped += 1
                else:
                    _record_fetch_result(conn, film_id, False, "rt_status",
                                         "rt_checked_at", "rt_attempts")
                if i % 100 == 0:
                    conn.commit()
                    log.info(f"    RT scrape: {i}/{len(rt_scrape_rows)} checked, {rt_scraped} found")
                time.sleep(0.5)
            conn.commit()

    log.info(f"  Posters: {posters_fetched} fetched, RT via OMDb: {rt_fetched}, "
             f"RT via scrape: {rt_scraped}")
    return posters_fetched, rt_fetched + rt_scraped


DTDD_CSV = "data/dtdd_test_with_columns.csv"
DTDD_BASE = "https://www.doesthedogdie.com"
DTDD_API_KEY = os.environ.get("DTDD_API_KEY", "")

#DTDD free tier (as of 2026-09): 30 requests/minute, 5,000 requests/MONTH.
#Each film costs 2 requests (search, then detail to verify tmdbId), so the ceiling
#is ~2,500 films/month and the ~25.7K film backlog needs roughly 10 months. The
#monthly cap is the real constraint, not speed — burn it in one run and warnings
#stop for the rest of the month, so spend is budgeted and tracked across runs.
DTDD_RATE_PER_MIN = 30
DTDD_REQUESTS_PER_FILM = 2
DTDD_MONTHLY_BUDGET = 4500          # headroom under the 5,000 hard limit
DTDD_BUDGET_FILE = "data/dtdd_budget.json"


def _dtdd_budget_load():
    """Requests already spent this calendar month (resets automatically)."""
    import json
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    try:
        with open(DTDD_BUDGET_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("month") == month:
            return month, int(data.get("requests", 0))
    except (OSError, ValueError, TypeError):
        pass
    return month, 0


def _dtdd_budget_save(month, requests_used):
    import json
    Path("data").mkdir(exist_ok=True)
    with open(DTDD_BUDGET_FILE, "w", encoding="utf-8") as f:
        json.dump({"month": month, "requests": requests_used}, f)


def _dtdd_normalize_topic(name):
    """Match the exact casing the existing 16,022 warning rows use.

    The original CSV pipeline lowercased, replaced "/" and "_" with spaces, and
    DROPPED apostrophes before title-casing — so DTDD's "there's torture" became
    "Theres Torture". Calling .title() on the raw name instead yields "There'S
    Torture" (Python capitalises after an apostrophe), which would render
    inconsistently beside every pre-existing row.
    """
    s = str(name).lower().replace("'", "").replace("/", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _dtdd_warnings_for(title, year, tmdb_id, headers, status=None):
    """`status` (optional dict) receives {'http_error': bool}.

    A film DTDD simply does not cover and a DTDD outage both used to return None,
    so a run of obscure films looked identical to the API going down — the
    circuit breaker tripped after 40 legitimate misses and halted the whole queue.
    Separating transport failure from "no match" is what lets the breaker fire
    only on real trouble.
    """
    if status is not None:
        status["http_error"] = False
    return _dtdd_warnings_impl(title, year, tmdb_id, headers, status)


def _dtdd_warnings_impl(title, year, tmdb_id, headers, status):
    """Look up one film on DTDD and return its warning string, or None.

    The tmdbId check is the safety rail: DTDD search is title-based, and without
    verifying the id back against our own row a same-titled film would attach the
    wrong sensitivity warnings — worse than having none.
    """
    try:
        r = requests.get(f"{DTDD_BASE}/dddsearch", params={"q": title},
                         headers=headers, timeout=15)
        if r.status_code != 200:
            if status is not None:
                status["http_error"] = True
            return None
        items = (r.json() or {}).get("items") or []
    except Exception:
        if status is not None:
            status["http_error"] = True
        return None

    candidates = [it for it in items if str(it.get("tmdbId") or "") == str(tmdb_id)]
    if not candidates and year:
        candidates = [it for it in items
                      if str(it.get("releaseYear") or "") == str(year)
                      and str(it.get("name", "")).strip().lower() == str(title).strip().lower()]
    if not candidates:
        return None

    try:
        d = requests.get(f"{DTDD_BASE}/media/{candidates[0]['id']}",
                         headers=headers, timeout=15)
        if d.status_code != 200:
            if status is not None:
                status["http_error"] = True
            return None
        detail = d.json() or {}
    except Exception:
        if status is not None:
            status["http_error"] = True
        return None

    item = detail.get("item", {})
    detail_tmdb = str(item.get("tmdbId") or "")
    if detail_tmdb and detail_tmdb != str(tmdb_id):
        return None  # wrong film

    triggers = []
    for t in detail.get("topicItemStats", []) or []:
        name = (t.get("topic", {}) or {}).get("name", "").strip()
        if name and (t.get("yesSum") or 0) > (t.get("noSum") or 0):
            triggers.append(_dtdd_normalize_topic(name))
    #the CSV pipeline emitted topics in source order, not sorted; keep insertion
    #order so new rows read like the 16,022 existing ones
    seen, ordered = set(), []
    for t in triggers:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ", ".join(ordered)


def run_dtdd_fetch(conn, dry_run, recent_years=RECENT_YEARS,
                   backlog_limit=BACKLOG_LIMIT, retry_exhausted=False,
                   film_limit=None, monthly_budget=DTDD_MONTHLY_BUDGET,
                   fetch_all=False):
    """Fetch DTDD sensitivity warnings live for valid films that have none.

    The CSV export is a fixed snapshot, so newly acquired or promoted films would
    otherwise never get warnings at all.

    Free-tier quota is the binding constraint (5,000 requests/month, 2 per film),
    so this spends deliberately: highest vote counts first, capped per run, and
    tracked across runs in a month-stamped budget file. Running out of quota mid-
    month would leave the newest films — the ones users actually search for —
    without warnings until the reset.
    """
    if not DTDD_API_KEY:
        log.info("  DTDD_API_KEY not set — skipping live warning fetch "
                 "(add it to .env to enable)")
        return 0

    month, spent = _dtdd_budget_load()
    headers = {"Accept": "application/json", "X-API-KEY": DTDD_API_KEY}

    if fetch_all:
        #Drain everything in pure vote order, ignoring the monthly budget and the
        #recency/backlog windows. The server-side quota is still the real ceiling —
        #this just stops US from stopping early. The 30 req/min pacing is kept
        #because that is an API limit, not a policy choice; exceeding it earns 429s.
        log.info(f"  [DTDD] --dtdd-all: ignoring local budget "
                 f"({spent:,} requests already logged this month)")
        rows = conn.execute("""
            SELECT id, title, release_date, vote_count FROM movies
            WHERE is_valid=1 AND (warnings IS NULL OR TRIM(warnings)='')
              AND COALESCE(dtdd_attempts,0) < ?
            ORDER BY CAST(vote_count AS REAL) DESC
        """, (MAX_FETCH_ATTEMPTS if not retry_exhausted else 10**6,)).fetchall()
        total_eligible = len(rows)
        if film_limit is not None:
            rows = rows[:film_limit]
    else:
        remaining_requests = max(0, monthly_budget - spent)
        affordable = remaining_requests // DTDD_REQUESTS_PER_FILM
        log.info(f"  DTDD quota {month}: {spent:,}/{monthly_budget:,} requests used, "
                 f"{remaining_requests:,} left (~{affordable:,} films)")
        if affordable <= 0:
            log.warning(f"  [DTDD] monthly budget exhausted — resumes next month")
            return 0
        frag, params = _fetch_eligibility_sql(
            "dtdd_status", "dtdd_checked_at", "dtdd_attempts", recent_years, retry_exhausted)
        rows = conn.execute(f"""
            SELECT id, title, release_date, vote_count FROM movies
            WHERE is_valid=1 AND (warnings IS NULL OR TRIM(warnings)='')
              {frag}
            ORDER BY CAST(vote_count AS REAL) DESC
        """, params).fetchall()
        rows = _bounded_backlog(rows, recent_years, backlog_limit, "DTDD")
        cap = min(affordable, film_limit if film_limit is not None else affordable)
        total_eligible = len(rows)
        rows = rows[:cap]
    #30 req/min across 2 requests per film -> one film per 4s, plus a little slack
    per_film_sleep = (60.0 / DTDD_RATE_PER_MIN) * DTDD_REQUESTS_PER_FILM + 0.2
    log.info(f"  {total_eligible:,} eligible, taking {len(rows):,} this run "
             f"(~{len(rows)*per_film_sleep/60:.0f} min at {DTDD_RATE_PER_MIN} req/min)")
    if dry_run or not rows:
        return 0

    found = used = 0
    consecutive_misses = 0
    for i, row in enumerate(rows, 1):
        year_m = re.search(r'\d{4}', str(row["release_date"] or ""))
        _st = {}
        warnings = _dtdd_warnings_for(row["title"], year_m.group(0) if year_m else "",
                                      row["id"], headers, status=_st)
        used += DTDD_REQUESTS_PER_FILM
        if warnings:
            _record_fetch_result(conn, row["id"], True, "dtdd_status", "dtdd_checked_at",
                                 "dtdd_attempts", extra_sql=", warnings=?",
                                 extra_params=(warnings,))
            found += 1
            consecutive_misses = 0
        else:
            _record_fetch_result(conn, row["id"], False, "dtdd_status",
                                 "dtdd_checked_at", "dtdd_attempts")
        #Count only TRANSPORT failures. DTDD legitimately has no entry for many
        #obscure films, and a run of those is normal — not a reason to stop.
        consecutive_misses = consecutive_misses + 1 if _st.get("http_error") else 0

        if consecutive_misses >= 25:
            log.warning(f"  [DTDD] {consecutive_misses} consecutive HTTP failures at "
                        f"film {i}/{len(rows)} — quota exhausted or API down; "
                        f"stopping cleanly. Re-run later to resume.")
            break

        if i % 25 == 0:
            conn.commit()
            _dtdd_budget_save(month, spent + used)
            log.info(f"    DTDD: {i}/{len(rows)} checked, {found} matched, "
                     f"{spent+used:,} requests used this month")
        time.sleep(per_film_sleep)
    conn.commit()
    _dtdd_budget_save(month, spent + used)
    log.info(f"  [DTDD] {found:,} films gained warnings; {used:,} requests spent, "
             f"{spent+used:,} used this month (of {total_eligible:,} eligible)")
    return found


def run_dtdd_warnings(conn, dry_run):
    """Backfill DoesTheDogDie content warnings from the DTDD export.

    The last job merge_layers.py did that nothing else covered. It rebuilt the
    whole table to attach these; this only writes rows whose warnings actually
    change, so it is safe to run every week and cheap when there is nothing new.
    """
    import csv as _csv
    log.info("PHASE 2a: DTDD content warnings")
    path = Path(DTDD_CSV)
    if not path.exists():
        log.info(f"  {DTDD_CSV} not found — skipping")
        return 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 5:
            log.warning(f"  {DTDD_CSV} has no warning columns — skipping")
            return 0
        warning_cols = header[4:]
        incoming = {}
        for parts in reader:
            if len(parts) < 5:
                continue
            try:
                film_id = int(float(parts[0]))
            except (ValueError, TypeError):
                continue
            triggers = [warning_cols[i].replace("_", " ").title()
                        for i, v in enumerate(parts[4:])
                        if i < len(warning_cols) and str(v).strip().lower() == "yes"]
            incoming[film_id] = ", ".join(triggers)

    log.info(f"  {len(incoming):,} films in the DTDD export")

    updates = []
    blanked = 0
    for row in conn.execute("SELECT id, warnings FROM movies WHERE id IS NOT NULL"):
        new = incoming.get(row["id"])
        if new is None:
            continue
        existing = (row["warnings"] or "").strip()
        #10,560 of the 26,703 CSV rows list no triggers at all, and writing those
        #back as "" erases warnings the live API already found — Terminator 2 lost
        #a full set that way. A snapshot with nothing to say must not overwrite
        #data that does.
        if not new.strip() and existing:
            blanked += 1
            continue
        if existing != new:
            updates.append((new, row["id"]))
    if blanked:
        log.info(f"  {blanked:,} films kept their existing warnings "
                 f"(CSV had no triggers for them)")

    log.info(f"  {len(updates):,} films need a warnings update from the CSV")
    if not dry_run and updates:
        conn.executemany("UPDATE movies SET warnings=? WHERE id=?", updates)
        conn.commit()
    return len(updates)


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
        SELECT id, title, release_date, tconst, wiki_plot
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
        tconst = (row["tconst"] or "").strip() or None
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
            SELECT id, title, release_date, tconst, wiki_plot, vote_count
            FROM movies
            WHERE is_valid = 1
              AND wiki_plot IS NOT NULL AND wiki_plot != ''
              AND CAST(vote_count AS REAL) >= ?
              AND CAST(vote_count AS REAL) <= ?
            ORDER BY CAST(vote_count AS REAL) DESC
        """, (min_votes, max_votes)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, title, release_date, tconst, wiki_plot, vote_count
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
        tconst = (row["tconst"] or "").strip() or None

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
                    "tconst": row["tconst"] or "",
                    "vote_count": row["vote_count"],
                })
                print(f"[{i}/{total}] {title!r}  sim={sim:.2f}  MISMATCH → fetch failed")
        else:
            print(f"[{i}/{total}] {title!r}  sim={sim:.2f}  MISMATCH")
            mismatches.append({
                "id":             row["id"],
                "title":          title,
                "release_date":   row["release_date"],
                "tconst":       row["tconst"] or "",
                "stored_length":  len(row["wiki_plot"]),
                "fetched_length": len(fetched_plot),
                "similarity":     round(sim, 4),
            })

        time.sleep(WIKI_SLEEP)

    #writes mismatch CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "title", "release_date", "tconst",
            "stored_length", "fetched_length", "similarity"
        ])
        writer.writeheader()
        writer.writerows(mismatches)

    #writes fetch failures CSV
    if fetch_failures:
        with open("wiki_fetch_failures.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "title", "release_date", "tconst", "vote_count"])
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
    ap.add_argument("--skip-integrity", action="store_true", help="Skip the Phase 0 integrity gate (date normalization, dedup, tconst collision resolution, validity enforcement)")
    ap.add_argument("--skip-warnings",  action="store_true", help="Skip the DTDD content-warnings backfill")
    ap.add_argument("--dtdd-limit", type=int, default=250, help="Max films for the Phase 2a live DTDD fetch per run (default: 250 = 500 requests, ~17 min). Free tier allows 5,000 requests/month total.")
    ap.add_argument("--dtdd-all", action="store_true", help="Fetch DTDD warnings for EVERY film missing them, highest vote_count first, ignoring the monthly budget and recency windows. Still paced at 30 req/min (an API limit). Stops cleanly if the server-side quota runs out.")
    ap.add_argument("--dtdd-monthly-budget", type=int, default=DTDD_MONTHLY_BUDGET, help=f"Monthly DTDD request ceiling, tracked across runs in {DTDD_BUDGET_FILE} (default: {DTDD_MONTHLY_BUDGET}, under the 5,000 free-tier limit)")
    ap.add_argument("--recent-years",   type=int, default=RECENT_YEARS, help=f"How many years back counts as 'current' for wiki/poster/RT re-attempts (default: {RECENT_YEARS}). Older films get exactly one attempt ever.")
    ap.add_argument("--backlog-limit",  type=int, default=BACKLOG_LIMIT, help=f"Max never-before-attempted OLD films to drain per run, per phase (default: {BACKLOG_LIMIT}). Keeps a run bounded.")
    ap.add_argument("--all-years",      action="store_true", help="Ignore the recent-years window and backlog cap — full historical sweep. Slow; use deliberately.")
    ap.add_argument("--retry-exhausted", action="store_true", help=f"Also retry films that already failed {MAX_FETCH_ATTEMPTS}+ times (normally given up on permanently)")
    ap.add_argument("--skip-acquire",   action="store_true", help="Skip Phase 1b — do not insert new films missing from the DB")
    ap.add_argument("--fix-wiki-mismatches", action="store_true", help="In Phase 0, NULL any wiki_plot whose source article title does not match the film, so it is re-fetched (default: report only)")
    ap.add_argument("--skip-categories", action="store_true", help="Skip Phase 3a — Wikipedia categories -> category_tags")
    ap.add_argument("--category-backfill-limit", type=int, default=60, help="Max films for the Phase 3a disambiguation backfill per run (default: 60). It sleeps 4-5s per film, so this bounds it to ~5 min; it is resumable and drains over successive runs.")
    ap.add_argument("--skip-category-backfill", action="store_true", help="Skip Phase 3a's Phase B disambiguation backfill entirely")
    ap.add_argument("--category-min-votes", type=int, default=10000, help="Min vote_count for the Phase 3a Wikipedia category fetch (default: 10000)")
    ap.add_argument("--acquire-min-votes", type=int, default=10000, help="Min IMDb votes for a missing film to be acquired (default: 10000)")
    ap.add_argument("--acquire-limit",  type=int, default=1200, help="Max films to reconcile (insert or exact-link) per run, highest IMDb vote count first (default: 1200 — enough to clear the current ~894 backlog in one run)")
    ap.add_argument("--skip-tmdb",  action="store_true", help="Skip TMDB enrichment")
    ap.add_argument("--skip-blank-date-refetch", action="store_true", help="Skip the blank-release_date stub re-fetch pre-pass")
    ap.add_argument("--blank-date-refetch-limit", type=int, default=500, help="Max is_valid=0 blank-release_date stubs to re-fetch per run, prioritized by existing vote_count (default: 500)")
    ap.add_argument("--skip-imdb",  action="store_true", help="Skip IMDb vote updates")
    ap.add_argument("--skip-wiki",  action="store_true", help="Skip Wikipedia plot fetch")
    ap.add_argument("--skip-posters", action="store_true", help="Skip poster/RT fetch")
    ap.add_argument("--skip-rt-backlog", action="store_true", help="Skip the RT scrape fallback for the full missing-score backlog (still runs OMDb-based RT fetch)")
    ap.add_argument("--skip-cache",    action="store_true", help="Skip cache rebuild")
    ap.add_argument("--force-rebuild",   action="store_true", help="Force cache rebuild even if no content changes detected")
    ap.add_argument("--force-refresh",   action="store_true", help="Ignore IMDb index pickle and re-download fresh TSVs")
    ap.add_argument("--since",          type=str, default=None, help="Limit Phase 3/4 to films promoted to is_valid=1 on or after DATE (YYYY-MM-DD)")
    ap.add_argument("--verify-plots",    action="store_true", help="Verify stored wiki_plots against fresh Wikipedia fetch")
    ap.add_argument("--fix-mismatches", action="store_true", help="When used with --verify-plots, re-fetch and overwrite mismatched plots in the DB")
    ap.add_argument("--min-votes",       type=int, default=200000, help="Min vote_count for --verify-plots (default: 200000)")
    ap.add_argument("--max-votes",       type=int, default=None,   help="Max vote_count for --verify-plots (optional upper bound)")
    ap.add_argument("--tmdb-min-votes",  type=int, default=None,   help="Min vote_count for Phase 1 TMDB enrichment (default: no floor). Use for a one-off scoped run.")
    ap.add_argument("--wiki-min-votes",  type=int, default=None,   help="Min vote_count for Phase 3 wiki plot fetch (default: 1000). Use for a one-off scoped run.")
    ap.add_argument("--rt-min-votes",    type=int, default=None,   help="Min vote_count for Phase 4 RT fetch, OMDb + scrape fallback (default: no floor). Use for a one-off scoped run.")
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

    if args.skip_integrity:
        log.info("PHASE 0: Integrity gate skipped (--skip-integrity)")
    else:
        run_integrity_gate(conn, args.dry_run,
                           fix_wiki_mismatches=args.fix_wiki_mismatches)
    content_changed = False

    #blank-date stub re-fetch (must run before Phase 1/stub-linker, which both
    #require a real release_date to target/match a film)
    blank_date_updated = 0
    if not args.skip_tmdb and not args.skip_blank_date_refetch:
        blank_date_updated = run_blank_date_refetch(conn, args.dry_run, args.blank_date_refetch_limit)
        if blank_date_updated > 0:
            content_changed = True
    else:
        log.info("PHASE 1a: Blank-date stub re-fetch skipped.")

    #TMDB
    tmdb_updated = 0
    if not args.skip_tmdb:
        tmdb_updated = run_tmdb_enrichment(conn, args.dry_run, tmdb_min_votes=args.tmdb_min_votes)
        if tmdb_updated > 0:
            content_changed = True
    else:
        log.info("PHASE 1: TMDB enrichment skipped.")

    #IMDB
    imdb_updated = 0
    if not args.skip_imdb:
        imdb_updated = run_imdb_updates(
            conn, args.dry_run,
            acquire=not args.skip_acquire,
            acquire_min_votes=args.acquire_min_votes,
            acquire_limit=args.acquire_limit,
        )
        if imdb_updated > 0:
            content_changed = True
    else:
        log.info("PHASE 2: IMDb updates skipped.")

    #DTDD content warnings (folded in from the retired merge_layers.py)
    warnings_updated = 0
    if not args.skip_warnings:
        warnings_updated = run_dtdd_warnings(conn, args.dry_run)
        #the CSV export is a fixed snapshot, so newly acquired/promoted films need
        #a live lookup or they would never get sensitivity warnings at all
        warnings_updated += run_dtdd_fetch(
            conn, args.dry_run,
            recent_years=(200 if args.all_years else args.recent_years),
            backlog_limit=(10**9 if args.all_years else args.backlog_limit),
            retry_exhausted=args.retry_exhausted,
            film_limit=(None if args.dtdd_all else args.dtdd_limit),
            monthly_budget=args.dtdd_monthly_budget,
            fetch_all=args.dtdd_all,
        )
        if warnings_updated > 0:
            content_changed = True
    else:
        log.info("PHASE 2a: DTDD content warnings skipped.")

    #re-settle is_valid now that Phase 1/2 have moved vote counts around. Runs
    #unconditionally: gating it on "did an earlier phase report updates" is what
    #let films that crossed the threshold during a skipped phase stay invisible.
    new_valid = 0
    if not args.dry_run:
        log.info("PHASE 2b: Re-settling validity after acquisition and vote updates")
        new_valid = _enforce_validity(conn, args.dry_run)
        #newly acquired/promoted rows are the ones most likely to introduce a
        #duplicate, so sweep again here rather than leaving it for next week.
        _dedup_valid_films(conn, args.dry_run)
        if new_valid > 0:
            content_changed = True

    _since_cutoff = args.since if args.since else None

    #wikipedia
    #--all-years means "no recency window, no backlog cap": a very wide window and
    #an effectively unlimited drain reproduce the old full-sweep behavior.
    _recent_years   = 200 if args.all_years else args.recent_years
    _backlog_limit  = 10**9 if args.all_years else args.backlog_limit

    wiki_fetched = 0
    if not args.skip_wiki:
        wiki_fetched = run_wiki_plots(
            conn, args.dry_run,
            min_votes=args.wiki_min_votes,
            since_cutoff=_since_cutoff,
            recent_years=_recent_years,
            backlog_limit=_backlog_limit,
            retry_exhausted=args.retry_exhausted,
        )
        if wiki_fetched > 0:
            content_changed = True
    else:
        log.info("PHASE 3: Wikipedia plots skipped.")

    #wikipedia categories -> category_tags (free; helix tagging stays manual)
    if not args.skip_categories:
        cats = run_wiki_categories(conn, args.dry_run, min_votes=args.category_min_votes,
                                   backfill_limit=(10**9 if args.all_years else args.category_backfill_limit),
                                   skip_backfill=args.skip_category_backfill)
        if cats > 0:
            content_changed = True
    else:
        log.info("PHASE 3a: Wikipedia categories skipped.")

    #posters/scores
    posters_fetched, rt_fetched = 0, 0
    if not args.skip_posters:
        posters_fetched, rt_fetched = run_posters_scores(
            conn, args.dry_run, since_cutoff=_since_cutoff,
            skip_rt_backlog=args.skip_rt_backlog, rt_min_votes=args.rt_min_votes,
            recent_years=_recent_years, backlog_limit=_backlog_limit,
            retry_exhausted=args.retry_exhausted,
        )
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
    log.info(f"  Blank-date stubs fixed: {blank_date_updated}")
    log.info(f"  TMDB films updated:    {tmdb_updated}")
    log.info(f"  IMDb films updated:    {imdb_updated}")
    log.info(f"  Warnings updated:      {warnings_updated}")
    log.info(f"  Validity changes:      {new_valid}")
    log.info(f"  Wiki plots fetched:    {wiki_fetched}")
    log.info(f"  Posters fetched:       {posters_fetched}")
    log.info(f"  RT scores fetched:     {rt_fetched}")
    log.info(f"  Caches rebuilt:        {cache_rebuilt}")
    log.info(f"{'='*60}")

    print(f"\n{'─'*50}")
    print(f"  Blank-date stubs fixed: {blank_date_updated}")
    print(f"  Films updated (TMDB):  {tmdb_updated}")
    print(f"  Films updated (IMDb):  {imdb_updated}")
    print(f"  Warnings updated:      {warnings_updated}")
    print(f"  Validity changes:      {new_valid}")
    print(f"  Wiki plots fetched:    {wiki_fetched}")
    print(f"  Posters fetched:       {posters_fetched}")
    print(f"  RT scores fetched:     {rt_fetched}")
    print(f"  Caches rebuilt:        {cache_rebuilt}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
