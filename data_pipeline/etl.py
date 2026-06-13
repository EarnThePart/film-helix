import pandas as pd
import sqlite3
import re

DATA_PATH = 'data/tmdb_data.csv'
DB_PATH = 'movies.db'

#cleaning
def get_names_by_job(text, target_jobs):
    if pd.isna(text): return ""
    text = str(text)
    parts = text.split(',')
    found_names = []
    for part in parts:
        for job in target_jobs:
            if f"({job})" in part:
                name = part.replace(f"({job})", "").strip().replace(" ", "")
                found_names.append(name)
                break 
    return " ".join(found_names)

def clean_simple_list(text):
    if pd.isna(text): return ""
    items = str(text).split(',')
    cleaned = [i.strip().replace(" ", "") for i in items]
    return " ".join(cleaned)

def clean_cast(text):
    if pd.isna(text): return ""
    actors = str(text).split(',')
    names = []
    for actor in actors[:5]:
        name = re.sub(r'\([^)]*\)', '', actor).strip().replace(" ", "")
        if name: names.append(name)
    return " ".join(names)

def clean_keywords(text):
    if pd.isna(text): return ""
    text = str(text).replace("[", "").replace("]", "").replace("'", "")
    words = text.split(',')
    cleaned = [w.strip().replace(" ", "").lower() for w in words]
    return " ".join(cleaned)

def run_etl():
    print(f"🧬 Loading Raw Data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 failed. Switching to ISO-8859-1...")
        df = pd.read_csv(DATA_PATH, low_memory=False, encoding='ISO-8859-1', on_bad_lines='skip')

    df = df.dropna(subset=['Overview', 'Keywords'])

    #column check
    for c in ['Lighting', 'Visual Effects', 'Production', 'Original Language', 'Original Title']:
        if c not in df.columns: df[c] = ""

    print("🧬 Extracting Full Genome...")

    #story
    df['dna_writer'] = df['Writing'].apply(lambda x: get_names_by_job(x, ['Writer', 'Screenplay', 'Author', 'Story']))

    #directors
    df['dna_director'] = df['Directing'].apply(lambda x: get_names_by_job(x, ['Director']))

    #visuals (currently inactive, reserved for v2 crew/production matching)
    df['dna_camera'] = df['Camera'].apply(lambda x: get_names_by_job(x, ['Director of Photography']))
    df['dna_art'] = df['Art'].apply(lambda x: get_names_by_job(x, ['Production Design', 'Art Direction']))
    df['dna_costume'] = df['Costume & Make-Up'].apply(lambda x: get_names_by_job(x, ['Costume Design']))
    df['dna_lighting'] = df['Lighting'].apply(lambda x: get_names_by_job(x, ['Gaffer']))
    df['dna_vfx'] = df['Visual Effects'].apply(lambda x: get_names_by_job(x, ['Visual Effects Supervisor']))

    #composer
    df['dna_sound'] = df['Sound'].apply(lambda x: get_names_by_job(x, ['Original Music Composer']))

    #editor
    df['dna_editor'] = df['Editing'].apply(lambda x: get_names_by_job(x, ['Editor']))

    #production info
    df['dna_country'] = df['Production Countries'].apply(clean_simple_list)
    df['dna_company'] = df['Production Companies'].apply(clean_simple_list)
    df['dna_producer'] = df['Production'].apply(lambda x: get_names_by_job(x, ['Producer', 'Executive Producer']))
    df['dna_lang'] = df['Original Language'].fillna("")
    
    #cast, genre, keywords
    df['dna_cast'] = df['Cast'].apply(clean_cast)
    df['dna_keywords'] = df['Keywords'].apply(clean_keywords)
    df['dna_genres'] = df['Genres'].fillna("").astype(str).apply(clean_simple_list)

    print(f"💾 Saving to {DB_PATH}...")
    
    df_final = df[['TMDb ID', 'Title', 'Original Title', 'Release Date', 'Runtime (Minutes)', 'Budget', 'Revenue', 'Overview',
                   'dna_keywords', 'dna_cast', 'dna_director', 'dna_writer', 
                   'dna_camera', 'dna_art', 'dna_costume', 'dna_lighting', 'dna_vfx',
                   'dna_sound', 'dna_editor', 'dna_country', 'dna_company', 'dna_producer', 'dna_lang', 
                   'dna_genres']].copy()
    
    df_final.columns = ['id', 'title', 'original_title', 'release_date', 'runtime', 'budget', 'revenue', 'overview',
                        'dna_keywords', 'dna_cast', 'dna_director', 'dna_writer', 
                        'dna_camera', 'dna_art', 'dna_costume', 'dna_lighting', 'dna_vfx',
                        'dna_sound', 'dna_editor', 'dna_country', 'dna_company', 'dna_producer', 'dna_lang', 
                        'dna_genres']

    conn = sqlite3.connect(DB_PATH)
    df_final.to_sql('movies', conn, if_exists='replace', index=False)
    conn.close()
    print("✅ SUCCESS: Database Updated with Original Titles.")

if __name__ == "__main__":
    run_etl()