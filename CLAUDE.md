# Film Helix — Project Brief for Claude Code

## What This Is
A content-based movie recommendation engine (portfolio MVP) that matches films
on narrative DNA rather than user behavior or collaborative filtering.
Think "Pandora for movies." Built to demonstrate data engineering, NLP, and
algorithm design skills for a Data Analytics/Data Science role in entertainment.

## Core Concept
- Multi-channel TF-IDF + semantic scoring across keywords, wiki plots, overviews, cast, crew, and Wikipedia category tags
- Hard genre gate prevents cross-genre contamination: 0.20 standard / 0.35 if source is a strict genre
  (Comedy/Animation/Documentary/Romance) / 0.45 if source is a true musical — PLUS separate, stricter
  cross-contamination floors a candidate must clear if it carries a genre the source doesn't: Comedy 0.60,
  Animation/Family 0.70, Musical 0.50, Documentary excluded outright. See "Genre Gate" in README for the
  full breakdown (recommender.py lines ~771-847) — horror has no elevated threshold anywhere in this logic.
- Keyword burn list (META_KEYWORD_STOPWORDS) eliminates meta/production keywords
- Mood/atmosphere keywords separated into their own TF-IDF channel (MOOD_KEYWORDS)
- Wiki plots encoded as chunked mean-pooled sentence embeddings (separate from overview semantic)
- Wikipedia category tags normalized into narrative tokens (cattags channel)
- Keyword and category diversity multipliers dampen matches based on sparse token overlap
- Keyword floor uses pre-multiplier raw similarity to avoid incorrectly excluding films with real but sparse overlap
- Explainability layer shows users exactly WHY a match was made (shared keywords/cast/director)
- Score normalization scales raw cosine similarity to human-readable percentages
- is_valid filter restricts dropdown and results to films with 1000+ IMDb votes

## Tech Stack
Python, SQLite (movies.db), Pandas, scikit-learn (TF-IDF, cosine similarity),
sentence-transformers (all-MiniLM-L6-v2), Streamlit, OMDb API (posters + RT scores),
IMDb TSV files (ratings)

## Key Files
NOTE: paths below are repo-root-relative. Several pipeline scripts live in `data_pipeline/`,
not the root — check there before concluding a file is missing.

- app.py                               — Streamlit frontend, UI, pagination, API calls
- recommender.py                       — TF-IDF engine, scoring, normalization, burn list
- weekly_refresh.py                    — THE data pipeline. One command does everything (see below)
- data_pipeline/etl.py                 — ingests tmdb_data.csv, outputs movies.db (STABLE - DO NOT MODIFY)
- archive/retired_by_weekly_refresh/merge_layers.py — RETIRED 2026-09-04, do not run (see below)
- import_new_movies.py                 — incremental import of new TMDB films into movies.db
- fetch_missing_by_imdb_id.py          — finds IMDb films missing from DB, fetches via TMDB /find endpoint
- tmdb_fetch.py                        — async TMDB scraper; supports --since DATE and --ids CSV
- wiki_plot_fetch_v3_db_patched.py     — Wikipedia plot fetcher (writes directly to movies.db wiki_plot column)
- wiki_audit.py                        — audits fetched wiki plots for title/year mismatches
- null_mismatches.py                   — nulls flagged mismatch rows so they get re-fetched
- fix_corrupted_genres.py              — one-time fix: re-fetches genres for films with numeric dna_genres
- test_engine.py                       — headless CLI tester; runs control films and prints top-N matches
- wiki_category_pipeline.py            — consolidates the old fetch_wiki_categories.py +
                                         backfill_wiki_pageids.py + build_category_tags.py into
                                         Phases A/B/C. Those three separate scripts no longer exist.
- movies.db                            — enriched SQLite database (source of truth)
- semantic_embeddings_cache.npy        — cached overview semantic embeddings (delete to rebuild)
- wiki_semantic_embeddings_cache.npy   — cached wiki chunked semantic embeddings (delete to rebuild)

## Secrets
- All API keys belong in `.env` (gitignored), read via `os.environ`. NEVER hardcode.
- `.gitignore` uses a `*` + allowlist pattern, so most scripts are untracked and safe —
  but `data_pipeline/**` IS allowlisted and therefore published to the PUBLIC GitHub repo.
  Anything placed there is world-readable. A DTDD key was leaked this way (rotated 2026-09-04).
- Keys used: TMDB_API_KEY, OMDB_API_KEY, DTDD_API_KEY (DTDD_API_KEY is required for the
  Phase 2a live warning fetch; without it that step logs a notice and skips).
- Keep the GitHub token in the osxkeychain credential helper, never inline in the
  remote URL — `git remote set-url origin https://github.com/EarnThePart/film-helix.git`.

## Constraints — DO NOT touch without explicit permission
- Do NOT modify etl.py — it is stable
- Do NOT modify merge_layers.py — it is stable
- Do NOT revert the is_valid filter — intentional fix for UI performance
- Do NOT revert ngram_range=(1,3) on keywords vectorizer — intentional fix for multi-word keyword phrases
- Do NOT revert the genre gate to a simple two-tier 0.2/0.35 system — it's actually a multi-tier gate
  (0.20/0.35/0.45 base thresholds by source genre, plus separate 0.50/0.60/0.70/exclude cross-contamination
  floors by candidate genre — see Core Concept above and README's Genre Gate section). This line used to
  say "prevents Drama/Horror bleed," which was never accurate — horror isn't in STRICT_GENRES and has no
  elevated threshold anywhere in the code.
- Do NOT revert score normalization — intentional, makes scores human-readable
- Do NOT revert mood/plot keyword split — intentional, prevents tone tags from polluting plot matching
- Do NOT revert keyword floor to use post-multiplier s_keywords — intentional fix (see Keyword Floor Bug below)
- Always test recommendation changes using: python test_engine.py

## Current Status (verified against recommender.py, 2026-08-18)
Engine has 17 live scored channels (see PRIORITY_WEIGHTS in recommender.py): keywords, semantic,
wiki, wiki_semantic, mood, overview, cattags, cast, director, writer, and 7 helix sub-channels
(helix_pro/dyn/thm/str/ton/dom/sty). PRIORITY_WEIGHTS also carries `logline`, `tagline`, and
`helix_spl` keys, but all three are dead weight — pinned at 0.00 in every priority mode, and
`s_logline`/`s_tagline`/`helix_spl`'s matrix are hardcoded zero vectors in recommender.py
(see "helix_spl is no longer scored" comment ~line 592). Don't count these three as live channels,
and don't wire real weight into them without first checking why they were zeroed out.
TÁR correctly surfaces as #1 for Whiplash (keyword floor bug fix, 2026-04-09).
Black Swan surfaces at #3 for Whiplash (Smell Test helix escape hatch fix, 2026-04-27) — this
superseded the older "Black Swan absent" expectation still listed further down in this file's
control-film table; that older line is stale, trust this one.
Helix taxonomy has 8 tagged dimensions in the DB (helix_dom/dyn/pro/spl/str/sty/thm/ton), but only
7 feed scoring — helix_spl (narrative resolution) is tagged and populated, just not wired in yet.
Run app: streamlit run app.py
Run headless tests: python test_engine.py

## Data Pipeline State (verified via filmhelix_stats.py, 2026-08-18)
- movies.db: 863,169 total films, 43,357 valid (is_valid=1, 1000+ IMDb votes)
- Total films grew from 260K → 863K after relaxing import filter (now requires Overview only, not Keywords)
- DTDD content warnings: complete for original valid set
- Wikipedia plots: 42,562 films have wiki_plot
- Wikipedia categories: 11,720 films have wiki_categories; 11,683 have category_tags
- Helix taxonomy: 34,140 films tagged on at least one of the 8 real dimensions (excludes the
  helix_low_confidence QA flag column, which inflates this count if included); 18,231 tagged on
  all 8; 20,343 tagged on all 7 dimensions actually used in scoring (helix_spl excluded)
- IMDb TSV files: current as of 2026-03-28 (used for vote counts + tconst linking)
- "Obscure" popularity filter (recommender.py exclude_obscure) gates at 20,000 votes, not 25K/50K — verify against code before quoting elsewhere

## merge_layers.py is RETIRED (2026-09-04) — do not run it
It destroyed the `tconst` column and the old "drop tconst_x/tconst_y afterward" ritual
below then deleted the only surviving copies. The exact chain:

1. `movies.db` already had a `tconst` column from the previous merge.
2. merge_layers line 79 dropped `tconst_x`/`tconst_y` but NOT `tconst`, so it survived.
3. The line-111 merge hit `imdb_merged`, which also has `tconst` — pandas renamed both
   sides to `tconst_x`/`tconst_y`, so no plain `tconst` column existed anymore.
4. Line 127 (`master.loc[excluded_mask, 'tconst'] = None`) then MATERIALIZED a brand-new
   all-NaN `tconst` column, because .loc-assigning a missing column creates it.
5. `to_sql(if_exists='replace')` wrote that all-NULL column over the table.
6. Dropping tconst_x/tconst_y per the old instructions destroyed the real IDs.

Result: 202,798 IMDb links → 0, which silently no-op'd IMDb vote sync, title sync, RT
fetch, and duplicate detection. Restored from movies.db.bak_20260805_151920 on 2026-09-04.
Everything merge_layers did now lives in weekly_refresh.py using targeted UPDATEs, which
cannot blank a column. If you ever need it, fix line 79 to drop `tconst` too — but prefer
weekly_refresh.

## Database Notes
- movies.db has NO tconst_x / tconst_y columns anymore. The IMDb ID column is `tconst`.
  Never re-introduce the _x/_y split — that is what caused the data loss above.
- Indexes added 2026-09-04: idx_movies_id, idx_movies_tconst, idx_movies_is_valid.
  The table had ZERO indexes before, so every `WHERE id=?` was a full 863K-row scan.
  Do not drop these — a bulk UPDATE pass went from hours to ~60s.
- Ringu (TMDB id 2671) was manually patched: tconst='tt0178868', vote_count=116000

## The Weekly Refresh — ONE command
    python weekly_refresh.py

That is the whole routine. No repair or fetch scripts afterward. Phases:
  0.  Integrity gate — aborts if tconst is empty; normalizes dates; dedups by TMDB id;
      resolves tconst collisions; enforces is_valid in BOTH directions (demote AND promote)
  1a. Blank-date stub re-fetch
  1.  TMDB enrichment (also backfills valid films missing a runtime)
  1b. IMDb reconciliation — exact-links unlinked films and INSERTS genuinely missing
      ones (see below). This is the only phase that adds rows.
  2.  IMDb vote sync + title+year linking for live/near-threshold films
  2a. DTDD content warnings — CSV backfill + LIVE API fetch for films the CSV
      snapshot doesn't cover (newly acquired/promoted films). Needs DTDD_API_KEY
      in .env; skips gracefully with a log line if unset.
  2b. Validity re-settle + dedup after acquisition
  3.  Wikipedia plots
  3a. Wikipedia categories -> category_tags (delegates to wiki_category_pipeline.py)
  4.  Posters + RT scores
  5.  Embedding cache rebuild (only if content changed)

STILL MANUAL (by design): helix_tagger.py — costs money, run it yourself and watch it.
Everything else the app needs is in the one command.

Useful flags: --dry-run, --skip-integrity, --skip-acquire, --skip-tmdb, --skip-wiki,
--skip-cache, --all-years, --retry-exhausted, --acquire-min-votes N, --acquire-limit N

### Phase 1b acquisition — why it is tconst-first
Resolves each missing IMDb id through TMDB `/find/{imdb_id}`, giving an EXACT
imdb->tmdb mapping, then:
  - if that TMDB id already exists in the DB -> UPDATE its tconst (exact link)
  - otherwise -> INSERT a new row
On the first run this exact-linked 881 films and inserted only 13. An insert-only
design would have created 881 duplicates, because films like Se7en (TMDB id 807) and
Star Wars (id 11) were already present — just unlinked and sitting at 0 votes.
Never "acquire" by title+year; that is what created the collisions.

### Fetch retry policy (Phases 3/4) — do not remove
Every fetch attempt records `<thing>_status`, `<thing>_checked_at`, and increments
`<thing>_attempts`, on failure as well as success. Previously only successes were
recorded, so `rt_status` was set on ZERO rows and the "skip failed" guard never fired —
Phase 4 re-scraped ~16,000 RT pages every run (88% pre-2023) for ~2.2 hours of
guaranteed misses.
Rules: give up after MAX_FETCH_ATTEMPTS (3); only re-attempt films released within
RECENT_YEARS (2); older films get exactly ONE attempt ever; per-run backlog drain is
capped at BACKLOG_LIMIT (400) per phase. Use --all-years for a deliberate full sweep.

### Cache alignment (IMPORTANT)
The .npy caches are POSITIONAL — row i is the i-th film of
`SELECT * FROM movies WHERE overview IS NOT NULL AND is_valid = 1`.
Any change to the valid set invalidates them. recommender.py:633 checks
`cached.shape[0] == n` and raises rather than silently mis-scoring, so a stale cache
means the app refuses to start. Phase 5 rebuilds automatically whenever content
changed; after any manual is_valid edit, re-run with --force-rebuild.

### Adding New Films to the DB
1. python fetch_missing_by_imdb_id.py   — finds films with 10K+ IMDb votes missing from DB, fetches via TMDB
2. python import_new_movies.py          — imports data/tmdb_data.csv rows not already in DB
3. python weekly_refresh.py             — links tconsts, syncs votes, promotes, enriches. Do NOT run merge_layers.py.

### One tconst = one film (invariant)
IMDb linking matches on (normalized title, year), so generic titles collide: 15 unrelated
2020 films named "Alone" all claimed tt7711170, inherited its 36,386 votes, and went live.
4,031 of 43,733 "valid" films (9.2%) were spuriously valid this way.
Phase 0 resolves collisions by asking TMDB for each film's OWN vote_count — the real film
has thousands, the 1-minute short has ~zero — awards the tconst to the winner, and releases
the rest with true votes restored. `_link_missing_tconsts` refuses an already-claimed tconst.
NEVER "fix" a tconst collision by demoting rows: they are distinct films, not duplicates.
Demoting silently deletes real films while leaving the stolen identity in place.

## Wikipedia Fetch — Key Facts
- Script: wiki_plot_fetch_v3_db_patched.py (writes DIRECTLY to movies.db wiki_plot column)
- wiki_merge.py is OBSOLETE — do not use (v3 script bypasses CSV entirely)
- Skips films where wiki_plot IS NOT NULL (checkpoint-safe, resume anytime)
- WAL mode enabled — can run alongside Streamlit without DB lock conflicts
- After fetch: run wiki_audit.py → null_mismatches.py → restart fetch for remaining mismatches
- Fetch command (background):
  python wiki_plot_fetch_v3_db_patched.py --db movies.db --limit 20000 --min-votes 1000 --sleep 0.5 --jitter 0.15 --progress-every 20 --heartbeat data/wiki_fetch_heartbeat.json > data/wiki_fetch.log 2>&1 &

## Wiki Audit Workflow
1. python3 wiki_audit.py           — dry run, shows counts
2. python3 wiki_audit.py --apply   — nulls BOTH-flagged rows only
3. python3 null_mismatches.py      — nulls YEAR_MISMATCH + TITLE_MISMATCH rows
4. Restart fetch to re-fetch nulled rows

## Wikipedia Category Tags Pipeline

### Overview
Wikipedia categories are noisy and encyclopedic (maintenance tags, crew credits, studio names,
award categories, year/decade buckets). We fetch them, strip the noise, and normalize the
remainder into narrative tokens stored in the `category_tags` column. These feed a dedicated
TF-IDF channel (`cattags`) in the recommender.

### Step 1 — fetch_wiki_categories.py
- Targets: is_valid=1 films with 10K+ votes and wiki_categories IS NULL
- Strategy: batch 50 films per Wikipedia API request using pageids; falls back to title-based
  lookup for films without a wiki_pageid
- Critical implementation detail: wiki_pageid is stored as REAL in SQLite (e.g. 57279206.0).
  Must cast with str(int(pid)) before passing to API — bare str() produces "57279206.0" which
  the API silently rejects, returning empty results for the entire batch.
- Handles Wikipedia API pagination via continuation tokens (films with many categories hit
  per-request result limits; loop until no "continue" key in response).
- Result (as of that initial fetch run): 93.8% coverage (11,149 / 11,885 films matched). The
  corpus has grown since — see "Data Pipeline State" above for the current live wiki_categories
  count (11,720); don't quote this run's number as the current total.

### Step 2 — backfill_wiki_pageids.py
- Targets: films with wiki_categories='' (confirmed no-hit) that have wiki_plot, meaning
  Wikipedia definitely has a page for them — they failed due to disambiguation or naming issues.
- resolve_film_page() tries in order:
  1. "{title} (film)"
  2. "{title} ({year} film)"
  3. "{title} ({year} American film)"
  4. Raw "{title}" — only accepted if not a disambiguation page
  5. Wikipedia search API fallback
- is_disambiguation() checks page categories for disambiguation markers before accepting a match
- Result: 427/475 targeted no-hits fixed; 48 remaining are genuinely obscure regional films

### Step 3 — build_category_tags.py
- Reads wiki_categories (pipe-separated), applies STRIP_PREFIXES and STRIP_PATTERNS to discard:
  - Maintenance/meta categories (CS1, Wikipedia templates, stub/cleanup tags)
  - Crew credits ("Films directed by...", "Films produced by...")
  - Studio/distributor categories (Warner Bros, A24, Netflix Original, etc.)
  - Language/nationality tags ("English-language films", "American films", etc.)
  - Year/decade buckets ("2008 films", "1990s action films")
  - Award categories (Academy Award, BAFTA, Golden Globe, etc.)
  - Location tags ("Films set in...", "Films shot in...")
- Remaining categories are tokenized to snake_case:
  "Films about stalking" → "about_stalking"
  "Psychological thrillers" → "psychological_thrillers"
- UMBRELLA_RULES add broader tokens alongside specific ones:
  e.g. any category matching "stalking|kidnapping|heist|mafia|..." → adds "crime" token
  Covers: crime, psychological, supernatural, survival, family_drama, romance,
  sci_fi, war, coming_of_age, political_social, biopic_historical, lgbtq, thriller_action
- Result (as of that initial run): 11,795 films have category_tags written to movies.db. Current
  live count is 11,683 — see "Data Pipeline State" above.

### Limitations / Known Gaps
- Wikipedia categories capture *subject matter*, not *themes*. "Obsessive perfectionism" is
  not a Wikipedia category — so thematically similar films (Whiplash/TÁR) still rely on
  wiki_semantic and keyword channels for that connection.
- Category vocabulary is sparse: many specific tokens appear in only 1-2 films and get
  dropped by min_df=2 in the TF-IDF vectorizer.
- The category diversity multiplier (1 shared tag=0.20, 2=0.45, 3=0.75, 4+=1.0) prevents
  a single shared broad tag (e.g. "independent") from over-driving scores.

## Recommender — Current Architecture
17 live scored channels, all blended via PRIORITY_WEIGHTS per selected priority. (PRIORITY_WEIGHTS
also has `logline`, `tagline`, and `helix_spl` keys — all three are pinned at 0.00 in every mode
and their similarity vectors are hardcoded zeros in recommender.py, so they contribute nothing.
Don't count them, and don't quote "20 channels" from a raw len(PRIORITY_WEIGHTS) count.)

| Channel        | Method                                    | Notes |
|----------------|-------------------------------------------|-------|
| keywords       | TF-IDF, ngram (1,3), plot keywords only   | Mood keywords excluded from this channel |
| mood           | TF-IDF, atmosphere/tone keywords only     | Feeds "Style & Tone" priority |
| wiki           | TF-IDF, Wikipedia plot text               | 42,562 films have coverage |
| overview       | TF-IDF, TMDB overview text                | Lightweight backstop |
| semantic       | Sentence-transformer, TMDB overview       | Cached in semantic_embeddings_cache.npy |
| wiki_semantic  | Sentence-transformer, wiki plots chunked  | 180-word chunks, mean-pooled, cached separately |
| cast           | CountVectorizer, lowercase=False          | CamelCase names preserved |
| director       | CountVectorizer, lowercase=False          | |
| writer         | CountVectorizer, lowercase=False          | |
| cattags        | TF-IDF, Wikipedia category tokens         | min_df=2, max_df=0.4; diversity multiplier applied |
| helix_pro/dyn/thm/str/ton/dom/sty | IDF-weighted cosine (7 channels) | helix_spl exists in the DB and is tagged, but is NOT wired into scoring — weight is 0.00 everywhere |

## PRIORITY_WEIGHTS (balanced, verbatim from recommender.py — re-copy this after any weight tuning so it never goes stale again)
keywords=0.10, semantic=0.22, wiki=0.05, wiki_semantic=0.16, logline=0.00, tagline=0.00, mood=0.04, overview=0.02, cast=0.02, director=0.01, writer=0.00, cattags=0.07, helix_pro=0.05, helix_dyn=0.05, helix_thm=0.05, helix_str=0.04, helix_ton=0.05, helix_spl=0.00, helix_dom=0.08, helix_sty=0.03

Verified 2026-08-18: these sum to 1.04, not 1.00 (helix channels alone contribute 0.35). There is
no downstream normalization step — `final_scores` gets multiplied straight into the displayed
percentage (`f"{int(final_scores[i] * 100)}%"` in get_recommendations) with no division by the
weight sum. This does NOT affect relative ranking within a priority mode (the same weights apply
uniformly to every candidate, so ordering is untouched) — it only means the theoretical score
ceiling is 104%, not 100%, and displayed percentages read a hair high. Harmless in practice, not
yet worth a renormalization pass; flagging here so nobody re-discovers it as a "bug." 16 of the 20
keys are nonzero in balanced mode (writer=0.00 here, but nonzero in Writer mode) — so "17 live
channels" elsewhere in this doc means "17 across all modes combined," not "17 nonzero in balanced."

## Match Priority Options (UI pills)
Balanced | Plot & Story | Genre | Style & Tone | Cast | Director | Writer
(exact UI labels, from app.py's _PRIORITY_MAP — "Style & Tone" maps to the internal "vibe" priority key)

## Keyword Burn Lists
- META_KEYWORD_STOPWORDS: production/meta tags (basedon*, remake, sequel, city names, ethnic descriptors, sensitive content)
- MOOD_KEYWORDS: atmosphere/tone descriptors (tense, atmospheric, awestruck, excited, etc.) — these go to mood channel only, removed from plot keywords

## Smell Test Helix Escape Hatch (2026-04-27)
The Smell Test (`final_scores[(s_semantic < 0.15) & (s_cattags < 0.10)] *= 0.10`) was
crushing Black Swan's score against Whiplash despite 5 shared helix tags (total_helix_sim=2.38).
Black Swan's TMDB overview is thematically distant from Whiplash's (ballet vs jazz drumming),
so s_semantic=0.065 < 0.15, and with only "independent" as a shared category tag, s_cattags
post-multiplier was 0.0035 < 0.10 — both conditions firing, 10× crush applied.

Fix: added `& (total_helix_sim < 0.50)` to the Smell Test condition. Films sharing strong
helix DNA (>= 0.50 combined similarity) escape the penalty regardless of semantic/cattags.
Caddyshack-on-Parasite still gets crushed (near-zero helix overlap). Black Swan now #3 for Whiplash.

total_helix_sim is now computed before the Smell Test and reused by the helix bouncer below,
eliminating a redundant calculation.

## Keyword Floor Bug Fix (2026-04-09)
The hard keyword floor (`final_scores[s_keywords < 0.02] = 0.0`) was applied to the
post-diversity-multiplier `s_keywords` value. The diversity multiplier reduces scores for
films sharing only 1 keyword token (0.35x multiplier). This meant a film with real but sparse
keyword overlap (e.g. TÁR sharing "musician" with Whiplash: raw sim=0.045, post-mult=0.016)
would fall below the 0.02 floor and get incorrectly excluded — despite having strong
semantic/wiki_semantic similarity (both ~0.40).

Fix: `s_keywords_raw = s_keywords.copy()` before the diversity multiplier is applied;
the floor check uses `s_keywords_raw < 0.02`. The multiplier still dampens the channel's
weighted contribution, but no longer causes false exclusions. TÁR now correctly appears
as #1 for Whiplash.

## helix_sty Modal-Tag Fix (2026-08-18)
`style_classical_invisible` (the LLM tagger's default/modal helix_sty value) covers 75.7% of the
25,496 films tagged on that dimension — a GROUP BY per helix column surfaced this while writing up
the README's Known Limitations. It wasn't signaling "classical, invisible style" so much as "the
tagger had nothing distinctive to say." Separately, and independently, `app.py`'s explainability
layer had already been filtering `style_classical_invisible` out of the display as a "generic noise
tag" — the two observations (useless for display, useless for scoring) had never been connected.

Fix: `HELIX_STY_NOISE_TAGS = {'style_classical_invisible'}` (module-level constant in recommender.py,
near `STRICT_GENRES`). Stripped in `load_data()` before `vec_str_helix_sty` is built (so it's never
vectorized, never enters IDF `df_counts`, and no longer drives cosine similarity), and stripped again
in `get_recommendations()`'s shared-tag computation (so it can't show up as a "shared tag" in raw
debug output either — `app.py`'s display filter is now redundant for this specific tag, but harmless
left in place). 15,021 films had ONLY this tag on helix_sty and now have an empty helix_sty vector,
correctly treated as "no style tag" rather than as a false positive match. Verified: full 80-film
test_engine.py control suite still passes clean after the change; Whiplash → TÁR #1 / Black Swan #2
unaffected.

## Known Issues / Active Work

### Recommendation Quality (as of 2026-04-27)
- Whiplash: TÁR #1 ✓, Black Swan #3 ✓ (fixed 2026-04-27 — Smell Test helix escape hatch)
- The Ring: Ringu (Ring 1998) #2 ✓
- Nightcrawler: Shattered Glass #2 ✓ (verified 2026-08-18 via get_recommendations directly). Its actual
  vote count is 40,210 — well above the 20K obscure threshold — so the older "filtered by exclude_obscure"
  note here was simply wrong, not stale data. It surfaces fine at default settings; the toggle default
  was changed to off anyway (below) for other reasons.
- Social Network: Hackers still ranks too high — "hacking" keyword is shared; needs thematic deprioritization
- Oldboy (2013) match on "incest" keyword is a spoiler — noted, not yet addressed
- Send Help: matches survival horror too heavily; comedy-horror genre sub-split needed
- Pulp Fiction: Reservoir Dogs / Jackie Brown still absent from top 10

### UI
- Poster maintains aspect-ratio at all zoom levels (object-fit: cover, no fixed min/max-height)
- Name tags use CamelCase split + Mc/Mac/Di prefix rejoining for proper display
- "Exclude Lower Popularity Films" toggle, default OFF as of 2026-08-18 (was on; changed because
  it was suppressing deep-cut matches that are the product's actual differentiator — no vote-count
  number shown in the UI label itself; underlying threshold is 20,000 votes per recommender.py's
  exclude_obscure, set in app.py's session_state init, not the toggle label)
- Real UI pills are Balanced | Plot & Story | Genre | Style & Tone | Cast | Director | Writer — no
  separate "Mood" pill exists; "Style & Tone" (internal key "vibe") is the mood/atmosphere mode

### Future: limited series / miniseries support (not started, 2026-09-05)
Idea: let users opt into TV matches — El Camino -> Breaking Bad, Shot Caller ->
The Night Of. The engine is already agnostic about what a "title" is: overview,
plot, keywords, cast, crew and the helix taxonomy all apply to a series unchanged,
so all 17 scoring channels would work as-is.

Scope it to **tvMiniSeries first**, not all TV. Matching quality tracks narrative
cohesion: The Night Of (8 episodes, one story) has tight DNA, while a 200-episode
procedural has diffuse DNA and would mostly add noise.

What actually needs building:
- TMDB TV endpoints are a different shape: `/tv/{id}` not `/movie/{id}`, `name` not
  `title`, `first_air_date` not `release_date`, credits under `aggregate_credits`.
  The whole pipeline assumes the movie shape — this is the bulk of the work.
- `_parse_imdb_basics` filters `titleType != "movie"`; TV needs `tvSeries` /
  `tvMiniSeries`. One line, but it roughly doubles the index.
- Runtime is not comparable (a 62-hour series vs a 2-hour film), and a series
  `wiki_plot` is a season-by-season summary — much longer and structurally
  different, so wiki_semantic chunking likely needs its own handling.
- Add a `media_type` column, a UI toggle, and recommender filtering on it.

### Search box ordering (known limitation, 2026-09-05)
`_ordered_titles` IS sorted by vote_count DESC, but `st.selectbox` re-ranks matches
client-side by string-match quality, so typing "spider" surfaces Spider (2002,
42K votes) above Spider-Man: No Way Home (1.0M votes). No selectbox parameter
controls this. Fixing it means giving up the native widget:
`st.text_input` does NOT fire per keystroke (Enter/blur only), so the realistic
option is `streamlit-searchbox`, which trades a server round-trip per keystroke for
control over ranking. Native filtering is client-side and instant; any custom
ranking is inherently slower. Evaluate the latency before committing to it.

### Future Work
- MMR (Maximal Marginal Relevance) re-ranking to reduce result clustering
- Same-country bonus for matching
- Release-year proximity bonus
- "Mark as Seen" filtering
- Keyword spoiler detection / suppression
- LLM-generated thematic tags (e.g. "obsessive perfectionism", "toxic mentorship") to cover
  gaps that Wikipedia categories and TMDB keywords don't capture

## helix_dom / helix_sty Columns (added 2026-04-27) — SUPERSEDED, historical record only
This section describes an April 2026 milestone (2 of 8 helix columns added, tagger not yet run).
All 8 helix columns exist and are populated now (see "Current Status" above for live counts) —
do not use the "READY TO RUN but blocked" line below as current fact, and do not assume the
prompt-caching blocker is still open. Kept here for the allowed-tag-value lists, which are
still accurate.

Two new pipe-delimited tag columns added to movies.db:
- helix_dom: Domain/Milieu tags (0–2 per film). 21 allowed values: dom_creative_performance,
  dom_criminal_underworld, dom_criminal_justice, dom_penal_system, dom_military_combat,
  dom_corporate_finance, dom_political_arena, dom_academic_scientific, dom_domestic_suburban,
  dom_urban_civic, dom_high_society_aristocracy, dom_espionage_intelligence,
  dom_journalism_media, dom_sports_competition, dom_isolated_containment,
  dom_wilderness_frontier, dom_deep_space, dom_tech_corporate, dom_supernatural_occult,
  dom_afterlife_metaphysical, dom_civilization_collapse
- helix_sty: Style/Pacing tags (1–2 per film). 10 allowed values: style_classical_invisible,
  style_epic_operatic, style_hyper_kinetic, style_slow_burn, style_meditative_atmospheric,
  style_procedural_methodical, style_cold_clinical, style_surreal_expressionist,
  style_raw_verite, style_found_footage

Tagger script: **helix_tagger.py** — this is the merged one, use it.
- Writes ALL 8 helix columns in a SINGLE Claude call per film
  (helix_pro/dyn/thm/str/ton/spl/dom/sty)
- helix_domain_tagger.py is SUPERSEDED — it only writes helix_dom + helix_sty (2 of 8).
  Its later mtime is misleading (a CostTracker patch), it is not the current tagger.
  helix_mini_tagger.py is narrower still: it appends candidate tags to existing columns.
- Targets: is_valid=1 AND overview IS NOT NULL AND vote_count >= 1000
- Supports: --limit N, --sort-by-votes, --dry-run, --max-cost N
- Run: python helix_tagger.py --sort-by-votes --limit 200 --max-cost 1.00

### Documentaries are NOT tagged (default, enforced in helix_tagger.py)
The Helix taxonomy describes narrative fiction — protagonist archetype, character
dynamic, spoiler resolution. A documentary genuinely has none of those, so the model
correctly returns empty buckets. Those empties then satisfy the tagger's own
`empty_field_count >= 3` targeting condition, so the film re-enters the queue, gets
re-tagged, comes back sparse, and re-queues — a loop that costs money every run and
never converges.

Measured 2026-09-05: documentaries were **11.2x over-represented** in the untagged
queue (25% of it vs 2% of the library). Already-attempted ones sat at 0-4 of 8
dimensions: An Inconvenient Truth 1/8, Blackfish 2/8, Apollo 11 2/8, Senna 0/8.

`helix_tagger.py` now excludes `dna_genres LIKE '%Documentary%'` by default; use
`--include-documentaries` to override. This is consistent with the recommender,
which already excludes Documentary outright in the genre gate.

### helix_tagger.py aborts on unrecoverable API errors
Credit exhaustion, invalid key, and auth failures raise `FatalAPIError` and stop the
run immediately. They used to be caught by a blanket `except anthropic.APIError` and
retried with 10/30/90/270s backoff PER FILM — across a 900-film queue that is ~100
hours of sleeping on an error that cannot clear. Transient errors (429, 5xx,
timeouts) still retry as before.

### Helix tagging is deliberately NOT in weekly_refresh.py
It costs money per film. The user wants to run it separately and watch it, so paid
steps stay manual while free/deterministic ones are automated. Do not add any Claude-API
tagger to the weekly run — not even behind a default-off flag — unless asked.

### Prompt Caching Blocker (unresolved as of 2026-04-27)
cache_creation_input_tokens returns 0 on every call with claude-haiku-4-5.
Root cause: claude-haiku-4-5 requires a 4,096-token minimum for prompt caching.
Current system prompt is ~2,407 tokens — below the threshold.
Options:
  1. Pad system prompt to 4,096+ tokens by adding more annotated examples (preferred — also improves quality)
  2. Accept no caching and run at full price (~$90–100 total for 33K films)
CostTracker.record() already has defensive fallback reading both old flat field
(cache_creation_input_tokens) and new nested field (usage.cache_creation.ephemeral_5m_input_tokens).
helix_mini_tagger.py has the same fix applied.

## Test Protocol
Run headless: python test_engine.py
Single film:  python test_engine.py --film "Inception (2010)"
More results: python test_engine.py --top 20
Style & Tone mode: python test_engine.py --priority vibe
  (internal key is "vibe", not "mood" — "mood" isn't a PRIORITY_WEIGHTS key, so
  --priority mood silently falls back to balanced rather than erroring, which is
  its own trap: it looks like it worked)
Allow obscure: python test_engine.py --obscure

Control films and expected signals:
1. Whiplash (2014)              — TÁR #1 ✓, Black Swan #3 ✓ (Smell Test helix escape hatch, 2026-04-27)
2. The Ring (2002)              — Ringu #2 ✓, curse/videotape supernatural
3. Nightcrawler (2014)          — journalism/neo-noir; Shattered Glass #2 ✓ (not filtered — 40,210 votes)
4. La La Land (2016)            — musical/romance specificity
5. Oldboy (2003)                — Park Chan-wook body of work, revenge
6. The Social Network (2010)    — ambition/betrayal; Hackers still ranks too high
7. Mad Max: Fury Road (2015)    — post-apoc action, no unrelated bleed ✓
8. Parasite (2019)              — South Korea class thriller ✓
9. Hereditary (2018)            — horror family trauma; generic supernatural bleed present
10. Pulp Fiction (1994)         — nonlinear crime; Reservoir Dogs / Jackie Brown absent
11. Infernal Affairs (2002)     — The Departed #1 ✓ CONFIRMED
12. Inception (2010)            — Matrix #1-2 ✓; some generic spy bleed
13. Send Help (2026)            — Triangle of Sadness ✓; survival horror over-represented
14. Goodfellas (1990)           — The Irishman #1 ✓, organized crime EXCELLENT

## Data Notes
- Keywords stored as space-separated tokens in dna_keywords column
- Genres stored as space-separated tokens in dna_genres column
- category_tags stored as space-separated snake_case tokens in category_tags column (added 2026-04)
- wiki_categories stored as pipe-separated raw Wikipedia category strings in wiki_categories column
- Films imported via fetch_missing_by_imdb_id.py must match the original tmdb_data.csv column order
  (Cast before Genres). Prior bug caused 1,290 films to get runtime stored in dna_genres — fixed by fix_corrupted_genres.py
- tconst_x column = IMDb ID used for OMDb API poster/RT lookups
- Cache files to delete when rebuilding: semantic_embeddings_cache.npy, wiki_semantic_embeddings_cache.npy
