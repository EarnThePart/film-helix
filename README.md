# FilmHelix. A Content-Based Movie Recommendation Engine.

## Overview

FilmHelix is a film recommendation platform that addresses a problem I've often faced when searching for a movie to watch. Streaming services and movie tracking platforms such as JustWatch and ReelGood use algorithms to recommend movies and shows you are likely to enjoy based on your watch history, ratings (often utilizing binary like/dislike systems), and other users' ratings. This system is built on behavioral data that optimizes for continued engagement but not necessarily taste. The "since you liked X, you might like Y movie" convention of movie recommendation platforms is ubiquitous. FilmHelix differentiates itself with an algorithm that finds films which share maximum similarity with a user-supplied source film, using that film's "genes" as a guide, i.e. narrative aspects such as plot and story, themes, atmosphere, and genre.

FilmHelix operates on the same principles as Pandora's Music Genome Project. Instead of behavioral data ("users who listened to X also listened to Y"), it analyzes the intrinsic properties of the film itself to find films with shared DNA. A traditional movie recommendation engine might observe that the user loved *Nightcrawler* (2014) and recommend *Drive* (2011) and *Prisoners* (2012) for them to watch next. The engine's results are defensible, as these films are also acclaimed crime thrillers with a neo-noir atmosphere. However, FilmHelix goes deeper. The algorithm analyzes the narrative DNA of *Nightcrawler* and seeks to provide the film that shares the most genes, producing matches such as *Ace in the Hole* (1951) and *Shattered Glass* (2003), surfacing films that are also are character studies of manipulative journalists with questionable ethics.

The FilmHelix database houses approximately 31,000 films, filtered by IMDb popularity from a starting list of nearly 900,000 films. Its engine uses a 12-channel weighted similarity architecture combining TF-IDF keyword matching, sentence-transformer semantic embeddings, Wikipedia plot analysis, and a custom-built cinematic taxonomy. The taxonomy established a framework for mapping each film's narrative genome across eight dimensions: protagonist archetype, dramatic structure, mood/atmosphere, setting, theme(s), core dramatic dynamic, cinematic style, and narrative resolution.

A Content-Based Movie Recommendation Engine.

FilmHelix operates on the same principles as Pandora's Music Genome Project. Instead of behavioral data ("users who listened to X also listened to Y"), it analyzes the intrinsic properties of the film itself to find films with shared DNA.




---

## Tech Stack

| Layer | Tools |
|---|---|
| Core |        Python, Pandas, NumPy, SciPy |
| ML / NLP |    scikit-learn (TF-IDF, cosine similarity), sentence-transformers (all-MiniLM-L6-v2) |
| Database |    SQLite (691MB enriched dataset) |
| Interface |   Streamlit |
| Utilities |   wordninja (CamelCase tokenization) |
| Data Sources | TMDB (~1M film metadata), IMDb (ratings/vote counts), Wikipedia (plot summaries), DoesTheDogDie (content warnings), OMDb / TMDB APIs (posters, Rotten Tomatoes scores) |

---

## The Helix Tag System

The most distinctive part of FilmHelix is a custom cinematic taxonomy I built by running ~25,200+ "valid" films through Claude Haiku (Anthropic) and then manually auditing, correcting, and refining the output over several weeks. A film attains "valid" status by reaching minimum qualifications to be included in recommendations, such as length, content, visibility, and distribution.

Each film receives tags across seven dimensions:

| Column | What it captures | Example tags |
|----|----|---|
| `helix_dom` | Primary setting / milieu | `dom_criminal_justice`, `dom_deep_space`, `dom_wilderness_frontier` |
| `helix_sty` | Cinematographic style | `style_slow_burn`, `style_cold_clinical`, `style_raw_verite` |
| `helix_pro` | Protagonist archetype | `obsessed_artist`, `reluctant_hero`, `determined_outsider` |
| `helix_str` | Narrative structure | `quest_narrative`, `nested_narrative`, `nonlinear_timeline` |
| `helix_ton` | Tonal register | `bleak_and_oppressive`, `warm_and_nostalgic`, `relentlessly_tense` |
| `helix_dyn` | Core dramatic dynamic | `cat_and_mouse`, `toxic_mentorship`, `individual_vs_institution` |
| `helix_thm` | Central theme | `cost_of_ambition`, `grief_as_transformation`, `hubris_of_science` |

### IDF-Weighted Helix Scoring

A key architectural decision: helix tags are scored using IDF weighting rather than raw overlap counts. This means rare tags carry exponentially more signal than common ones.

For example: `obsessed_artist` appears in only 21 films. `reluctant_hero` appears in 10,648. Without IDF weighting, both tags contribute equally to a match score. As a result, *Interstellar* as the source film would match *Indiana Jones* above *Arrival* because they share the generic `reluctant_hero + quest_narrative` archetypes. With IDF weighting, sharing `obsessed_artist` between *Whiplash* and *Black Swan* is correctly worth ~500x more than sharing `reluctant_hero`.

The formula: `IDF = log(N / (1 + df))` computed across all 7 helix columns at load time. Vectors are L2-normalized, so cosine similarity remains well-defined.

---

## Scoring Architecture

FilmHelix uses 16 independent similarity channels. Scores are computed in parallel, then blended according to the user's chosen match focus area.

### Channels

| Channel | Method | Description |
|---|---|---|
| **Keywords**      | TF-IDF (unigram) | TMDB plot keywords, mood/atmosphere words stripped. Pure narrative DNA |
| **Mood**          | TF-IDF | Atmosphere and tone descriptors only (`tense`, `haunting`, `cerebral`). Feeds the "Vibe" match focus |
| **Wiki**          | TF-IDF (bigram) | Full Wikipedia plot summaries. Provides richer signal, especially for foreign and older films |
| **Overview**      | TF-IDF | Raw TMDB overview text. Lightweight backstop |
| **Semantic**                                          | Sentence-transformer embeddings | TMDB overview encoded with all-MiniLM-L6-v2. Captures meaning beyond keyword overlap |
| **Wiki Semantic**                                     | Sentence-transformer, chunked | Wikipedia plots chunked into 180-word segments, encoded, mean-pooled |
| **Category Tags**                                     | TF-IDF | Wikipedia-derived narrative/thematic category tags |
| **Cast / Director / Writer**                          | CountVectorizer | Exact overlap on crew and cast |
| **helix_dom / sty / pro / str / ton / dyn / thm**     | IDF-weighted cosine | Custom taxonomy channels (see above) |

### Keyword Architecture

TMDB keywords are split into two channels at load time:
- **Plot keywords**: narrative DNA only. Meta-production tags (`basedon*`, `sequel`, city names, content warnings) are burned entirely via a stopword list.
- **Mood keywords**: atmosphere descriptors (`tense`, `bleak`, `cerebral`) routed to a dedicated channel to prevent false plot matches.

Keywords are also normalized at runtime to collapse TMDB tagging inconsistencies: `court` / `trial` / `court_case` → `courtroom`; `journalist` / `reporter` / `newspaperman` → `journalism`; etc.

### Diversity Multipliers

To prevent a single incidental shared tag from dominating a match, diversity multipliers are applied before blending:
- **Keyword diversity**:        1 shared keyword = 0.50x; 2 = 0.65x; 3+ = 1.0x
- **Category tag diversity**:   1 = 0.20x; 2 = 0.45x; 3 = 0.75x; 4+ = 1.0x
- **Helix diversity**:          1 shared helix tag = 0.10x; 2 = 0.35x; 3 = 0.70x; 4+ = 1.0x

### Priority Modes

| Mode | What it emphasizes |
|---|---|
| **Balanced**                  | Broad narrative match across all channels |
| **Plot & Story**              | Heavy keyword + semantic + wiki weighting; suppresses cast/crew |
| **Vibe**                      | Tone, atmosphere, and cinematic style; helix_ton + helix_sty dominate |
| **Genre**                     | Continuous genre similarity multiplier instead of binary gate |
| **Cast / Director / Writer**  | Finds the body of work for a specific collaborator |

### Genre Gate

A hard filter applied before final ranking. Films below genre cosine similarity thresholds are zeroed out:
- Standard threshold: 0.20
- Strict genres (Comedy, Animation, Documentary, Romance, Musical): 0.35

Comedy and horror films require a stricter filter, as the mismatch in tone/genre is too dissonant to overwhelm incidental plot similarities. For instance, the aspect of class warfare in the slapstick comedy classic *Caddyshack* (1980) could result in the dark comedy/thriller *Parasite* (2019) emerging as a recommended similar film. Likewise, Animation uses a gate of 0.70 for non-animated films, ensuring that only animated films that share an extraordinary amount of similarities to a live-action film would be matched.

### Smell Test

A final safety check: if a candidate shares almost zero semantic overlap, almost zero category tag overlap, AND almost zero helix tag overlap with the source film, its score is forcibly diminished by 90% regardless of other dimensional similarities. This prevents incidental single-keyword matches from surfacing completely unrelated films (i.e. a film from the SpongeBob SquarePants franchise matching to the sci-fi thrillers *The Abyss* (1989) or *Sphere* (1998) as a result of sharing the "underwater" tag).

---

## Data Pipeline

```
TMDB dataset (~1M films)
    → etl.py                        # ingests raw CSV, builds movies.db
    → merge_layers.py               # merges IMDb ratings + DoesTheDogDie content warnings
    → fetch_missing_by_imdb_id.py   # finds films with 10K+ IMDb votes missing from DB
    → wiki_plot_fetch.py            # fetches Wikipedia plots into movies.db
    → fetch_posters_tmdb.py         # batch-fetches poster URLs from TMDB API
    → haiku_tagger.py               # tags ~24K films with helix taxonomy via Claude Haiku
    → weekly_refresh.py             # automated weekly pipeline (see below)
    → movies.db                     # enriched SQLite store
```

Valid films (~31,000) must meet the threshhold of 1,000+ IMDb votes. Similarity modeling runs on this subset; the full 864K dataset is retained for future use.

### Weekly Refresh Pipeline

`weekly_refresh.py` keeps the database current with new releases, updated IMDb votes and ratings for previous releases with a single command:

```bash
python weekly_refresh.py              # full run
python weekly_refresh.py --dry-run    # preview without writing
python weekly_refresh.py --skip-tmdb --skip-wiki --skip-posters --skip-cache  # IMDb only
python weekly_refresh,py --verify-plots, --fix-mismatches, --min-votes      #   data integrity checks, targeted refreshes
```

Phases:
1. **TMDB enrichment**:         fetches updated keywords/metadata for 2024-2026 films and any valid films missing keywords
2. **IMDb updates**:            downloads IMDb ratings TSV, updates vote counts where change exceeds 5% threshold
3. **Wikipedia plots**:         fetches missing plots for newly valid films
4. **Posters & RT scores**:     fetches TMDB poster URLs and OMDb Rotten Tomatoes scores for new valid films
5. **Cache rebuild**:           rebuilds `.npy` embedding files locally (requires `sentence-transformers` installed)

This pipeline can be resumed after interruptions, such that each phase checks if a film has already been updated and skips it if so.

---

## User-Facing Match Explainability Layer

Every result card surfaces the exact signals driving the match, organized into three rows:

- **Setting & World**:  shared `helix_dom` tags (milieu/setting)
- **Tone & Style**:     shared `helix_sty` + `helix_ton` tags (cinematic feel)
- **Story DNA**:        shared TMDB plot keywords + protagonist/structure/theme helix tags

Tags are formatted human-readable (`dom_criminal_justice` → `Criminal Justice`), deduplicated across buckets, and filtered to remove generic noise tags (`style_classical_invisible`, `dom_domestic_suburban`).

---

## Key Features

- **16-channel similarity scoring**         with user-selectable priority blending
- **IDF-weighted helix taxonomy**:          7 cinematic dimensions, ~25,200 films tagged
- **Adaptive genre gate**                   with stricter thresholds for genre-sensitive categories
- **Diversity multipliers**                 on keywords, category tags, and helix channels
- **Wikipedia plot integration**:           ~22,000 valid films have enriched plot summaries feeding TF-IDF and semantic channels (short wiki plots are too thin for inclusion)
- **Three-layer explainability**:           Setting, Tone & Style, Story DNA shown on every card
- **Content warning filters**:              11 grouped categories powered by DoesTheDogDie; hidden by default
- **Popularity filter**:                    exclude films under 25K IMDb votes
- **Advanced filters**:                     year range, min IMDb rating, min RT score, exclude sequels/remakes, exclude animated, exclude non-English
- **Automated weekly refresh pipeline**:    single-command database update

---

## Headless Testing

A CLI test harness runs the engine without Streamlit:

```bash
python test_engine.py                          # all 80+ control films
python test_engine.py --film "Whiplash (2014)" # single film
python test_engine.py --top 20                 # show top 20
python test_engine.py --priority vibe          # test vibe matching
python test_engine.py --obscure                # include films under 25K votes
python test_engine.py --debug                  # show per-channel score breakdown
```

---

## Setup / Run Locally

> **Note:** `movies.db` is not distributed with this repo, as it contains ~691MB of enriched film data. Contact me if you need the database for evaluation purposes.

1. Clone the repo:
```bash
git clone https://github.com/your-username/FilmHelix.git
cd FilmHelix
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the app:
```bash
streamlit run app.py
```

The `.npy` embedding caches are included in the repo. If they're missing or the database was updated, rebuild them:
```bash
python -c "import weekly_refresh; weekly_refresh.run_cache_rebuild(dry_run=False)"
```

---

## Known Limitations
gi
**Focus mode tradeoffs.**   A single algorithm cannot simultaneously optimize for crowd-pleasing blockbusters like *Project Hail Mary* as well as nuanced psychological character studies such as *Persona*. If the math perfectly connects *Whiplash* and *Black Swan*, it may also group generic hero's journey films together (such as *Interstellar* and *Star Wars*). FilmHelix solves this at the UI level: the "Vibe" and "Plot & Story" focus modes empower users to dictate which narrative dimensions matter most.

**Freshness lag**           Films with fewer than 25,000 IMDb votes are excluded to prevent low-quality data from polluting the results. Brand new releases will occasionally drop out of the candidate pool until the weekly ETL pipeline syncs enough votes to validate them.

**Sparse data matches**     Films with exceptionally thin metadata (few keywords, no Wikipedia plot) will correctly match to more fully-mapped films when used as a source, but may struggle to emerge as matches. This resolves naturally with the regular data updates.

**Metadata fragmentation**  Because TMDB relies on crowd-sourced tagging, inconsistencies are inevitable. FilmHelix employs a keyword normalization map (e.g. "ship"/"boat"/"yacht" combine to form the singular keyword "boat") to consolidate variants, but some minor fragmentation remains.

---

## Future Enhancements

- **Horror sub-genre gate:**        distinguish supernatural/body horror from psychological horror to reduce cross-contamination
- **MMR re-ranking:**               Maximal Marginal Relevance to reduce sequel/franchise clustering in results
- **Mark as Seen:**                 per-session exclusion of watched titles
- **Watchlist import:**             Letterboxd / JustWatch CSV
- **Where to Watch:**               JustWatch streaming availability integration
- **Feedback loop:**                Like/Dislike signals to refine weighting per session