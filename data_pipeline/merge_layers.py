import pandas as pd
import sqlite3
import gc
import os
import re
import unicodedata

def _clean_title(s):
    s = str(s).lower().strip()
    #unicode normalization: é → e, ö → o, etc.
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    #remove leading parenthetical groups only e.g. "(500) Days" → "500 days"
    s = re.sub(r'^\(([^)]+)\)', r'\1', s)
    #roman numeral normalizations
    s = re.sub(r'\bpart\s+ii\b',       'part 2',    s)
    s = re.sub(r'\bpart\s+iii\b',      'part 3',    s)
    s = re.sub(r'\bpart\s+iv\b',       'part 4',    s)
    s = re.sub(r'\bpart\s+v\b',        'part 5',    s)
    s = re.sub(r'\bchapter\s+two\b',   'chapter 2', s)
    s = re.sub(r'\bchapter\s+three\b', 'chapter 3', s)
    s = re.sub(r'\bchapter\s+four\b',  'chapter 4', s)
    #trailing year parentheticals e.g. "Dune (2021)" → "dune"
    s = re.sub(r'\s*\(\d{4}\)\s*$', '', s)
    #ampersand → and
    s = re.sub(r'&', 'and', s)
    #apostrophes / possessives: snicket's → snickets
    s = re.sub(r"'s\b", 's', s)
    s = re.sub(r"'", '', s)
    #strip all non-alphanumeric except spaces
    s = re.sub(r'[^a-z0-9 ]', '', s)
    #cut whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

DB_PATH = 'movies.db'
IMDB_BASICS_FILENAME = 'title.basics.tsv.gz' 
IMDB_RATINGS_FILENAME = 'title.ratings.tsv'
DTDD_FILENAME = 'dtdd_test_with_columns.csv'

def get_path(filename):
    if os.path.exists(f"data/{filename}"): return f"data/{filename}"
    if os.path.exists(filename): return filename
    return None

def run_merge():
    print("⏳ Loading Database...")
    conn = sqlite3.connect(DB_PATH)
    try:
        movies = pd.read_sql("SELECT * FROM movies", conn)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    movies = movies.drop(columns=[col for col in ['tconst_x', 'tconst_y', 'vote_count_x', 'vote_count_y', 'vote_average_x', 'vote_average_y', 'warnings_x', 'warnings_y'] if col in movies.columns], errors='ignore')
    movies['clean_title'] = movies['title'].astype(str).map(_clean_title)
    movies['year_key'] = movies['release_date'].astype(str).str[:4]
    print(f"   DB Size: {len(movies)} movies")

    basics_path = get_path(IMDB_BASICS_FILENAME)
    ratings_path = get_path(IMDB_RATINGS_FILENAME)

    if basics_path and ratings_path:
        print("🏗️ Building IMDb Bridge...")
        
        ratings = pd.read_csv(ratings_path, sep='\t')
        basics = pd.read_csv(
            basics_path, 
            sep='\t', 
            usecols=['tconst', 'titleType', 'primaryTitle', 'startYear'],
            dtype=str
        )
        basics = basics[basics['titleType'] == 'movie']
        imdb_merged = pd.merge(basics, ratings, on='tconst', how='inner')
        
        del basics
        del ratings
        gc.collect()
        
        imdb_merged['clean_title'] = imdb_merged['primaryTitle'].astype(str).map(_clean_title)
        imdb_merged['year_key'] = imdb_merged['startYear'].astype(str)
        imdb_merged = imdb_merged.sort_values('numVotes', ascending=False).drop_duplicates(subset=['clean_title', 'year_key'])
        
        print(f"   IMDb Movie Count: {len(imdb_merged)}")

        print("🔗 Linking IMDb to TMDB...")
        master = pd.merge(
            movies, 
            imdb_merged[['clean_title', 'year_key', 'tconst', 'averageRating', 'numVotes']], 
            on=['clean_title', 'year_key'], 
            how='left'
        )
        
        master['vote_average'] = master['averageRating'].fillna(0)
        master['vote_count'] = master['numVotes'].fillna(0)
    
    else:
        print("⚠️ Skipping IMDb: Files not found.")
        master = movies
        if 'vote_count' not in master.columns: master['vote_count'] = 0
        if 'tconst' not in master.columns: master['tconst'] = None

    print("🧹 Pruning Junk...")
    
    if 'is_valid' not in master.columns:
        master['is_valid'] = 0

    def update_validity(row):
        if row['vote_count'] > 1000: return 1 
        if row['is_valid'] == 1: return 1     
        return 0

    master['is_valid'] = master.apply(update_validity, axis=1)

    dtdd_path = get_path(DTDD_FILENAME)
    
    if dtdd_path:
        print(f"Merging Content Warnings from {dtdd_path}...")
        try:
            dtdd = pd.read_csv(dtdd_path)
            warning_cols = dtdd.columns[4:] 
            
            def extract_warnings(row):
                triggers = []
                for col in warning_cols:
                    if str(row[col]).lower() == 'yes':
                        triggers.append(col.replace('_', ' ').title())
                return ", ".join(triggers)

            dtdd['warnings'] = dtdd.apply(extract_warnings, axis=1)
            
            master['id'] = pd.to_numeric(master['id'], errors='coerce')
            dtdd['id'] = pd.to_numeric(dtdd['id'], errors='coerce')
            
            master = master.drop(columns=[col for col in ['warnings_x', 'warnings_y', 'warnings'] if col in master.columns], errors='ignore')
            master = pd.merge(master, dtdd[['id', 'warnings']], on='id', how='left')
            master['warnings'] = master['warnings'].fillna("")
            
        except Exception as e:
            print(f"DTDD Error: {e}")
    else:
        print(f"DTDD Skipped.")
        if 'warnings' not in master.columns: master['warnings'] = ""

    print(f"Saving Enriched Database...")
    
    cols_to_drop = ['clean_title', 'year_key', 'averageRating', 'numVotes']
    keep_cols = [c for c in master.columns if c not in cols_to_drop]
    master = master[keep_cols]
    
    master.to_sql('movies', conn, if_exists='replace', index=False)
    conn.close()
    
    valid_count = len(master[master['is_valid'] == 1])
    print(f"✅ SUCCESS! {valid_count} Valid movies in dropdown. Full dataset retained for model training.")

if __name__ == "__main__":
    run_merge()