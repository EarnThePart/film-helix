-- Pulls every field recommender.py actually scores films on, for one film.
-- Usage: sqlite3 movies.db < data_pipeline/lookup_film_dna.sql
-- Edit the WHERE clause below to target a different title/year.

.mode line
SELECT
    id, title, release_date, vote_count, is_valid,
    dna_genres,
    dna_keywords,       -- feeds 'keywords' + 'mood' channels (mood subset filtered at runtime)
    dna_cast,
    dna_director,
    dna_writer,
    overview,            -- feeds 'overview' TF-IDF + 'semantic' embedding channels
    wiki_plot,           -- feeds 'wiki' TF-IDF + 'wiki_semantic' embedding channels
    category_tags,       -- feeds 'cattags' channel
    helix_dom,
    helix_sty,
    helix_pro,
    helix_dyn,
    helix_thm,
    helix_str,
    helix_ton
FROM movies
WHERE title = 'Whiplash' AND release_date LIKE '2014%';
