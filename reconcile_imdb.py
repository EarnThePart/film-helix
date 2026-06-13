"""
reconcile_imdb.py
-----------------
Finds valid films in movies.db missing a tconst_y ID, matches them against
IMDb title.basics + title.ratings, using the same clean_title logic as
merge_layers.py, and patches tconst_y + vote_count for confident matches.

Usage:
  python reconcile_imdb.py            # dry-run (default)
  python reconcile_imdb.py --commit   # write to DB
"""

import argparse
import gc
import os
import re
import sqlite3
import unicodedata

import pandas as pd

DB_PATH = "movies.db"
BASICS_CANDIDATES = ["data/title.basics.tsv", "data/title.basics.tsv.gz", "title.basics.tsv"]
RATINGS_CANDIDATES = ["data/title.ratings.tsv", "data/title.ratings.tsv.gz", "title.ratings.tsv"]


def _find(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _clean_title(s):
    s = str(s).lower().strip()
    # unicode normalization: é → e, ö → o, etc.
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # remove leading parenthetical groups e.g. "(500) Days" → "500 days"
    s = re.sub(r'^\(([^)]+)\)', r'\1', s)
    # roman numeral normalizations
    s = re.sub(r'\bpart\s+ii\b',       'part 2',    s)
    s = re.sub(r'\bpart\s+iii\b',      'part 3',    s)
    s = re.sub(r'\bpart\s+iv\b',       'part 4',    s)
    s = re.sub(r'\bpart\s+v\b',        'part 5',    s)
    s = re.sub(r'\bchapter\s+two\b',   'chapter 2', s)
    s = re.sub(r'\bchapter\s+three\b', 'chapter 3', s)
    s = re.sub(r'\bchapter\s+four\b',  'chapter 4', s)
    # trailing year parentheticals e.g. "Dune (2021)" → "dune"
    s = re.sub(r'\s*\(\d{4}\)\s*$', '', s)
    # ampersand → and
    s = re.sub(r'&', 'and', s)
    # apostrophes / possessives: snicket's → snickets
    s = re.sub(r"'s\b", 's', s)
    s = re.sub(r"'", '', s)
    # strip all non-alphanumeric except spaces
    s = re.sub(r'[^a-z0-9 ]', '', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def main():
    ap = argparse.ArgumentParser(description="Reconcile missing IMDb tconst_y values")
    ap.add_argument("--commit", action="store_true", help="Write patches to DB (default: dry-run)")
    args = ap.parse_args()
    dry_run = not args.commit

    basics_path = _find(BASICS_CANDIDATES)
    ratings_path = _find(RATINGS_CANDIDATES)
    if not basics_path or not ratings_path:
        print("ERROR: IMDb TSV files not found. Expected in data/ directory.")
        return

    #load unmatched valid films
    conn = sqlite3.connect(DB_PATH)
    films = pd.read_sql("""
        SELECT id, title, release_date, vote_count
        FROM movies
        WHERE is_valid = 1
          AND (tconst_y IS NULL OR tconst_y = '')
    """, conn)
    print(f"Unmatched valid films: {len(films):,}")

    films["clean_title"] = films["title"].map(_clean_title)
    films["year_key"] = films["release_date"].astype(str).str[:4]

    #load IMDb
    print("Loading IMDb basics...")
    basics = pd.read_csv(basics_path, sep="\t", usecols=["tconst", "titleType", "primaryTitle", "startYear"], dtype=str)
    basics = basics[basics["titleType"] == "movie"]

    print("Loading IMDb ratings...")
    ratings = pd.read_csv(ratings_path, sep="\t", dtype={"tconst": str, "numVotes": float})

    imdb = pd.merge(basics, ratings[["tconst", "numVotes"]], on="tconst", how="inner")
    del basics, ratings
    gc.collect()

    imdb["clean_title"] = imdb["primaryTitle"].map(_clean_title)
    imdb["year_key"] = imdb["startYear"].astype(str)
    #keep highest-vote entry per (clean_title, year_key) to avoid ambiguous matches
    imdb = imdb.sort_values("numVotes", ascending=False).drop_duplicates(subset=["clean_title", "year_key"])
    print(f"IMDb movies loaded: {len(imdb):,}")

    #pull exact clean_title + exact year
    imdb_lookup = imdb.set_index(["clean_title", "year_key"])[["tconst", "numVotes"]]

    def lookup_exact(row):
        key = (row["clean_title"], row["year_key"])
        if key in imdb_lookup.index:
            return imdb_lookup.loc[key, "tconst"], imdb_lookup.loc[key, "numVotes"]
        return None, None

    films[["tconst", "numVotes"]] = films.apply(
        lambda r: pd.Series(lookup_exact(r)), axis=1
    )

    pass1 = films[films["tconst"].notna()].copy()
    remaining = films[films["tconst"].isna()].copy()
    print(f"\nPass 1 (exact year):  {len(pass1):,} matched")

    #build year-agnostic lookup: clean_title → list of (year, tconst, numVotes)
    imdb_by_title = {}
    for _, row in imdb.iterrows():
        imdb_by_title.setdefault(row["clean_title"], []).append(
            (row["year_key"], row["tconst"], row["numVotes"])
        )

    def lookup_fuzzy_year(row):
        candidates = imdb_by_title.get(row["clean_title"], [])
        try:
            film_year = int(row["year_key"])
        except (ValueError, TypeError):
            return None, None
        for imdb_year_str, tconst, votes in candidates:
            try:
                if abs(int(imdb_year_str) - film_year) <= 1:
                    return tconst, votes
            except (ValueError, TypeError):
                continue
        return None, None

    remaining[["tconst", "numVotes"]] = remaining.apply(
        lambda r: pd.Series(lookup_fuzzy_year(r)), axis=1
    )

    pass2 = remaining[remaining["tconst"].notna()].copy()
    unmatched = remaining[remaining["tconst"].isna()].copy()
    print(f"Pass 2 (year ±1):     {len(pass2):,} matched")
    print(f"Still unmatched:      {len(unmatched):,}")

    all_matched = pd.concat([pass1, pass2], ignore_index=True)

    if all_matched.empty:
        print("Nothing to patch.")
        conn.close()
        return

    #dry run/preview
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Sample patches Pass 1 (first 10):")
    for _, row in pass1.head(10).iterrows():
        print(f"  id={int(row['id']):>7}  votes → {int(row['numVotes']):>7}  "
              f"{row['tconst']}  {row['title']!r}")

    if not pass2.empty:
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Sample patches Pass 2 year ±1 (first 10):")
        for _, row in pass2.head(10).iterrows():
            print(f"  id={int(row['id']):>7}  year={row['year_key']}  votes → {int(row['numVotes']):>7}  "
                  f"{row['tconst']}  {row['title']!r}")

    #commit
    if not dry_run:
        cur = conn.cursor()
        for _, row in all_matched.iterrows():
            cur.execute(
                "UPDATE movies SET tconst_y = ?, vote_count = ? WHERE id = ?",
                (row["tconst"], int(row["numVotes"]), int(row["id"])),
            )
        conn.commit()
        print(f"\n✓ Patched {len(all_matched):,} films ({len(pass1):,} exact + {len(pass2):,} year±1).")
    else:
        print(f"\nRun with --commit to apply {len(all_matched):,} patches.")

    conn.close()

    if not unmatched.empty:
        print(f"\nStill unmatched (first 20):")
        for _, row in unmatched.head(20).iterrows():
            print(f"  {row['title']!r}  ({row['year_key']})  clean={row['clean_title']!r}")


if __name__ == "__main__":
    main()
