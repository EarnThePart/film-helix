# FilmHelix: A Content-Based Movie Recommendation Engine.

**[Try it live: filmhelix.streamlit.app](https://filmhelix.streamlit.app)**

## Overview

FilmHelix is a film recommendation platform that addresses a problem I've often faced when searching for a movie to watch. Streaming services and movie tracking platforms such as JustWatch and ReelGood use algorithms to recommend movies and shows you are likely to enjoy based on your watch history, ratings (often utilizing binary like/dislike systems), and other users' ratings. This system is built on behavioral data that optimizes for continued engagement but not necessarily taste. The "since you liked X, you might like Y movie" convention of movie recommendation platforms is ubiquitous. FilmHelix differentiates itself with an algorithm that finds films which share maximum similarity with a user-supplied source film, using that film's "genes" as a guide, i.e. narrative aspects such as plot and story, themes, atmosphere, and genre.

FilmHelix operates on the same principles as Pandora's Music Genome Project. Instead of behavioral data ("users who listened to X also listened to Y"), it analyzes the intrinsic properties of the film itself to find films with shared DNA. A traditional movie recommendation engine might observe that the user loved *Nightcrawler* (2014) and recommend *Drive* (2011) and *Prisoners* (2012) for them to watch next. The engine's results are defensible, as these films are also acclaimed crime thrillers with a neo-noir atmosphere. However, FilmHelix goes deeper. The algorithm analyzes the narrative DNA of *Nightcrawler* and seeks to provide the film that shares the most genes, producing matches such as *Ace in the Hole* (1951) and *Shattered Glass* (2003), surfacing films that are also character studies of manipulative journalists with questionable ethics.

The FilmHelix database houses approximately 43,000 valid films, filtered by IMDb popularity from a starting list of 863,000 films. Its engine uses a 17-channel weighted similarity architecture combining TF-IDF keyword matching, sentence-transformer semantic embeddings, Wikipedia plot analysis, and a custom-built cinematic taxonomy. The taxonomy established a framework for mapping each film's narrative genome across eight dimensions: protagonist archetype, dramatic structure, mood/atmosphere, setting, theme(s), core dramatic dynamic, cinematic style, and narrative resolution (this last dimension is tagged in the data but not yet wired into live scoring — see below).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Core |        Python, Pandas, NumPy, SciPy |
| ML / NLP |    scikit-learn (TF-IDF, cosine similarity), sentence-transformers (all-MiniLM-L6-v2) |
| Database |    SQLite (661MB enriched dataset) |
| Interface |   Streamlit |
| Utilities |   wordninja (CamelCase tokenization) |
| Data Sources | TMDB (863K film metadata), IMDb (ratings/vote counts), Wikipedia (plot summaries), DoesTheDogDie (content warnings), OMDb / TMDB APIs (posters, Rotten Tomatoes scores) |

---

## The Helix Tag System

The most distinctive part of FilmHelix is a custom cinematic taxonomy: films are run through Claude Haiku (Anthropic) to LLM-tag them, then manually spot-audited and corrected across several re-tagging passes. 34,140 films carry a tag on at least one dimension; 18,231 are complete across all eight. A film attains "valid" status by reaching minimum qualifications to be included in recommendations, such as length, content, visibility, and distribution.

Each film receives tags across eight dimensions, seven of which currently feed live scoring (`helix_spl` / narrative resolution is tagged but not yet wired into the similarity engine):

| Column | What it captures | Example tags |
|----|----|---|
| `helix_dom` | Primary setting / milieu | `dom_criminal_justice`, `dom_deep_space`, `dom_wilderness_frontier` |
| `helix_sty` | Cinematographic style | `style_slow_burn`, `style_cold_clinical`, `style_raw_verite` |
| `helix_pro` | Protagonist archetype | `obsessed_artist`, `reluctant_hero`, `determined_outsider` |
| `helix_str` | Narrative structure | `quest_narrative`, `nested_narrative`, `nonlinear_timeline` |
| `helix_ton` | Tonal register | `bleak_and_oppressive`, `warm_and_nostalgic`, `relentlessly_tense` |
| `helix_dyn` | Core dramatic dynamic | `cat_and_mouse`, `toxic_mentorship`, `individual_vs_institution` |
| `helix_thm` | Central theme | `cost_of_ambition`, `grief_as_transformation`, `hubris_of_science` |
| `helix_spl` | Narrative resolution | `happy_ending`, `tragic_ending`, `bittersweet_ending` |

### IDF-Weighted Helix Scoring

A key architectural decision: helix tags are scored using IDF weighting rather than raw overlap counts. This means rare tags carry more signal than common ones — but on a logarithmic scale, which compresses rather than amplifies the raw frequency gap (see the worked example below).

For example: `obsessed_artist` appears in only 17 films. `reluctant_hero` appears in 10,191. Without IDF weighting, both tags contribute equally to a match score. As a result, *Interstellar* as the source film would match *Indiana Jones* above *Arrival* because they share the generic `reluctant_hero + quest_narrative` archetypes. With log-scaled IDF weighting, sharing `obsessed_artist` between *Whiplash* and *Black Swan* is correctly worth roughly 5x more than sharing `reluctant_hero` — a rare, specific signal is treated as meaningfully stronger evidence than a common, generic one, even though log-scaling deliberately keeps the raw 600:1 frequency gap from swinging the score by 600x.

The formula: `IDF = log(N / (1 + df))`, computed across the 7 helix columns currently wired into scoring (`helix_spl` is excluded — see above). Vectors are L2-normalized, so cosine similarity remains well-defined.

---

## Scoring Architecture

FilmHelix uses 17 independent similarity channels. Scores are computed in parallel, then blended according to the user's chosen match focus area.

### Channels

| Channel | Method | Description |
|---|---|---|
| **Keywords**      | TF-IDF (unigram) | TMDB plot keywords, mood/atmosphere words stripped. Pure narrative DNA |
| **Mood**          | TF-IDF | Atmosphere and tone descriptors only (`tense`, `haunting`, `cerebral`). Feeds the "Style & Tone" match focus |
| **Wiki**          | TF-IDF (bigram) | Full Wikipedia plot summaries. Provides richer signal, especially for foreign and older films |
| **Overview**      | TF-IDF | Raw TMDB overview text. Lightweight backstop |
| **Semantic**                                          | Sentence-transformer embeddings | TMDB overview encoded with all-MiniLM-L6-v2. Captures meaning beyond keyword overlap |
| **Wiki Semantic**                                     | Sentence-transformer, chunked | Wikipedia plots chunked into 180-word segments, encoded, mean-pooled |
| **Category Tags**                                     | TF-IDF | Wikipedia-derived narrative/thematic category tags |
| **Cast / Director / Writer**                          | CountVectorizer | Exact overlap on crew and cast |
| **helix_dom / sty / pro / str / ton / dyn / thm**     | IDF-weighted cosine | Custom taxonomy channels, 7 total (see above). `helix_spl` is tagged but not yet a live channel |

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
| **Style & Tone**              | Tone, atmosphere, and cinematic style; helix_ton + helix_sty dominate |
| **Genre**                     | Continuous genre similarity multiplier instead of binary gate |
| **Cast / Director / Writer**  | Finds the body of work for a specific collaborator |

### Genre Gate

Two layers, gating on different sides of the match:

**Base gate** — applies to every candidate uniformly, set by what genre the *source* film is:
- Standard: 0.20
- Source is Comedy, Animation, Documentary, or Romance ("strict" genres): 0.35
- Source is a true musical (Music genre + a literal "musical" keyword, to exclude jazz dramas/band comedies that only carry the broader Music genre tag): 0.45

**Cross-contamination floors** — an *additional*, stricter bar a candidate must separately clear if it carries one of these genres and the source doesn't, so a genre-typical candidate can't sneak in on generic plot similarity alone:
- Candidate is a documentary and source isn't: excluded entirely, no exception
- Candidate is Comedy and source's primary genre isn't Comedy: 0.60
- Candidate is Animation/Family and source isn't Animation/Family: 0.70 — e.g. the slapstick comedy *Caddyshack* (1980) sharing "class warfare" with *Parasite* (2019) isn't enough on its own to surface an animated match against a live-action source
- Candidate is a true musical and source isn't: 0.50

So Animation's two numbers aren't a contradiction: 0.35 is what applies when you search *for* an animated film (every candidate needs to clear it); 0.70 is the much higher bar an animated candidate must separately clear when the source isn't animated at all.

### Smell Test

A final safety check: if a candidate shares almost zero semantic overlap, almost zero category tag overlap, AND almost zero helix tag overlap with the source film, its score is forcibly diminished by 90% regardless of other dimensional similarities. This prevents incidental single-keyword matches from surfacing completely unrelated films (i.e. a film from the SpongeBob SquarePants franchise matching to the sci-fi thrillers *The Abyss* (1989) or *Sphere* (1998) as a result of sharing the "underwater" tag).

---

## Data Pipeline

```
TMDB dataset (863K films)
    → etl.py                        # ingests raw CSV, builds movies.db
    → merge_layers.py               # merges IMDb ratings + DoesTheDogDie content warnings
    → fetch_missing_by_imdb_id.py   # finds films with 10K+ IMDb votes missing from DB
    → wiki_plot_fetch.py            # fetches Wikipedia plots into movies.db
    → fetch_posters_tmdb.py         # batch-fetches poster URLs from TMDB API
    → haiku_tagger.py               # tags 34,140 films with helix taxonomy via Claude Haiku
    → weekly_refresh.py             # automated weekly pipeline (see below)
    → movies.db                     # enriched SQLite store
```

Valid films (~43,000) must meet the threshold of 1,000+ IMDb votes. Similarity modeling runs on this subset; the full 863K dataset is retained for future use.

### Weekly Refresh Pipeline

`weekly_refresh.py` keeps the database current with new releases, updated IMDb votes and ratings for previous releases with a single command:

```bash
python weekly_refresh.py              # full run
python weekly_refresh.py --dry-run    # preview without writing
python weekly_refresh.py --skip-tmdb --skip-wiki --skip-posters --skip-cache  # IMDb only
python weekly_refresh.py --verify-plots --fix-mismatches --min-votes       # data integrity checks, targeted refreshes
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

- **17-channel similarity scoring**         with user-selectable priority blending
- **IDF-weighted helix taxonomy**:          8 cinematic dimensions tagged (7 currently scored), 34,140 films tagged on at least one, 18,231 complete across all eight
- **Adaptive genre gate**                   with stricter thresholds for genre-sensitive categories
- **Diversity multipliers**                 on keywords, category tags, and helix channels
- **Wikipedia plot integration**:           98% coverage (42,562 of 43,357 valid films) with enriched plot summaries feeding TF-IDF and semantic channels
- **Three-layer explainability**:           Setting, Tone & Style, Story DNA shown on every card
- **Content warning filters**:              11 grouped categories powered by DoesTheDogDie; hidden by default
- **Popularity filter**:                    optionally exclude lesser-known films under 20K IMDb votes (on by default; valid films still require 1K+ votes)
- **Advanced filters**:                     year range, min IMDb rating, min RT score, exclude sequels/remakes, exclude animated, exclude non-English
- **Automated weekly refresh pipeline**:    single-command database update

---

## Headless Testing

A CLI test harness runs the engine without Streamlit:

```bash
python test_engine.py                          # all 80+ control films
python test_engine.py --film "Whiplash (2014)" # single film
python test_engine.py --top 20                 # show top 20
python test_engine.py --priority vibe          # test Style & Tone mode
python test_engine.py --obscure                # include films under 20K votes
python test_engine.py --debug                  # show per-channel score breakdown
```

---

## Setup / Run Locally

> **Note:** `movies.db` is not distributed with this repo, as it contains ~661MB of enriched film data. Contact me if you need the database for evaluation purposes.

1. Clone the repo:
```bash
git clone https://github.com/EarnThePart/film-helix.git
cd film-helix
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

**Focus mode tradeoffs.**   A single algorithm cannot simultaneously optimize for crowd-pleasing blockbusters like *Project Hail Mary* as well as nuanced psychological character studies such as *Persona*. If the math perfectly connects *Whiplash* and *Black Swan*, it may also group generic hero's journey films together (such as *Interstellar* and *Star Wars*). FilmHelix solves this at the UI level: the "Style & Tone" and "Plot & Story" focus modes empower users to dictate which narrative dimensions matter most.

**Freshness lag**           Films with fewer than 1,000 IMDb votes aren't yet considered "valid" and won't appear in results at all. Brand new releases will occasionally drop out of the candidate pool until the weekly ETL pipeline syncs enough votes to validate them.

**Sparse data matches**     Films with exceptionally thin metadata (few keywords, no Wikipedia plot) will correctly match to more fully-mapped films when used as a source, but may struggle to emerge as matches. This resolves naturally with the regular data updates.

**Metadata fragmentation**  Because TMDB relies on crowd-sourced tagging, inconsistencies are inevitable. FilmHelix employs a keyword normalization map (e.g. "ship"/"boat"/"yacht" combine to form the singular keyword "boat") to consolidate variants, but some minor fragmentation remains.

**LLM tagger bias toward modal categories**  Measuring my own tagger's output distribution surfaced a real skew: the modal tag on several helix dimensions covers a large share of tagged films, most notably `helix_pro` (protagonist archetype), where `reluctant_hero` alone accounts for 40.1% of the 25,496 films tagged on that dimension. `helix_str` (narrative structure) shows the same pattern — `character_study` covers 40.0%. `helix_sty` (cinematic style) was the most extreme: `style_classical_invisible` covered 75.7%. This is the classic LLM-tagging failure mode — when uncertain, the model defaults to the most generic available category. IDF weighting (above) is the general mitigation for `helix_pro` and `helix_str`, where the skew, while real, still leaves room for other values to carry signal. `helix_sty` crossed a different line: at 75.7% coverage, `style_classical_invisible` wasn't signaling "this film has a classical, invisible style" so much as "the tagger had nothing distinctive to say" — the UI's explainability layer had already independently flagged it as a generic noise tag to hide from display (see above), without anyone connecting that to the distribution problem. Fix applied: `style_classical_invisible` is now stripped before vectorization and treated as absent rather than as a real tag, in both the scoring layer and the shared-tag display, so films no longer match on sharing nothing.

---

## Future Enhancements

- **Horror sub-genre gate:**        distinguish supernatural/body horror from psychological horror to reduce cross-contamination
- **MMR re-ranking:**               Maximal Marginal Relevance to reduce sequel/franchise clustering in results
- **Mark as Seen:**                 per-session exclusion of watched titles
- **Watchlist import:**             Letterboxd / JustWatch CSV
- **Where to Watch:**               JustWatch streaming availability integration
- **Feedback loop:**                Like/Dislike signals to refine weighting per session


