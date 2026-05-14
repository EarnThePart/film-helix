import os
import re
import numpy as np
import pandas as pd
import sqlite3
import wordninja
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DB_PATH = 'movies.db'
EMBEDDINGS_CACHE         = 'semantic_embeddings_cache.npy'
WIKI_EMBEDDINGS_CACHE    = 'wiki_semantic_embeddings_cache.npy'
LOGLINE_EMBEDDINGS_CACHE = 'logline_embeddings_cache.npy'
TAGLINE_EMBEDDINGS_CACHE = 'tagline_embeddings_cache.npy'

META_KEYWORD_STOPWORDS = {
    'basedonshortfilm', 'basedonnovel', 'basedonbook', 'basedontruestory',
    'basedonplay', 'basedoncomicbook', 'basedoncomicseries', 'basedonscreenplay',
    'basedonnovelorbook', 'basedonnovella', 'basedonchildren', 'basedonwebseries',
    'basedonmanga', 'basedonvideogame', 'basedonrealpeople', 'basedonradioplay',
    'independentfilm', 'lowbudget', 'cultfilm', 'surrealism', 'silentfilm',
    'colorinfilm', 'directorscut', 'sequel', 'prequel',
    'reboot', 'remake', 'duringcreditssting', 'aftercreditssting',
    'basedonshort', 'newyorkcity', 'losangeles', 'london', 'paris', 'newyork',
    'boston', 'massachusetts', 'chicago', 'sanfrancisco', 'texas', 'california',
    'newjersey', 'washington', 'washington dc', 'hongkong', 'hong-kong', 'tokyo',
    'seoul', 'beijing', 'shanghai', 'mumbai', 'delhi', 'rome', 'berlin', 'sydney',
    'toronto', 'montreal', 'basedon', 'truestory', 'rosebud',
    'milkyway', 'universe', 'irish-american', 'irishamerican', 'italian-american', 
    'italianamerican', 'african-american', 'africanamerican', 'japanese-american', 'chinese-american',
    'mexican-american', 'jewish-american', 'korean-american',
    'childabuse', 'animalabuse', 'animalcruelty', 'animalkilling',
    'childmolestation', 'sexualabuse', 'domesticabuse', 'domesticviolence',
    'horrified', 'frightened', 'terrified', 'struggleforsurvival', 'morocco',
    # Newly audited TMDB format/meta tags
    'shortfilm', 'womandirector', 'lostfilm', 'experimental', 'pinkfilm',
    'stopmotion', 'documentaryshort', 'studentfilm', 'preservedfilm',
    'behindthescenes', 'basedonplayormusical', 'arthouse', 'essayfilm',
}

# Mood/atmosphere keywords — kept out of the main plot keyword channel so they
# don't create false matches (e.g. two films both tagged "tense" or "exciting").
# Instead they feed a dedicated 'mood' TF-IDF channel used by the Mood priority.
MOOD_KEYWORDS = {
    'atmospheric', 'tense', 'suspenseful', 'thrilling', 'exciting', 'excited',
    'awestruck', 'powerful', 'grand', 'epic', 'intimate', 'claustrophobic',
    'dreamlike', 'surreal', 'disturbing', 'unsettling', 'haunting', 'melancholic',
    'bleak', 'gritty', 'stylish', 'dark', 'darkhumor', 'darkcomedy',
    'heartwarming', 'uplifting', 'hopeful', 'emotional', 'poignant', 'moving',
    'bizarre', 'quirky', 'whimsical', 'lighthearted', 'funny', 'scary',
    'intense', 'visceral', 'brutal', 'shocking', 'mindblowing',
    'thoughtprovoking', 'cerebral', 'philosophical', 'meditative', 'slow-burn',
    'slowburn', 'fastpaced', 'actionpacked', 'charming', 'witty', 'satirical',
    'cynical', 'pessimistic', 'optimistic', 'nostalgic', 'romantic', 'erotic',
    'creepy', 'eerie', 'ominous', 'foreboding', 'dreadful', 'taut',
    # TMDB crowd-sourced tone/vibe descriptors
    'appreciative', 'bewildered', 'baffled', 'audacious', 'candid',
    'empathetic', 'frantic', 'disheartening', 'commanding', 'dignified',
    'blunt', 'biting', 'bold', 'bitter', 'ambivalent', 'distressing',
    'anxious', 'cautionary', 'didactic', 'dramatic', 'complex', 'critical',
    'direct', 'sincere', 'comforting', 'forceful', 'joyful', 'exhilarated',
    'introspective', 'provocative', 'serene', 'hilarious', 'cheerful', 'loving',
    'mysterious', 'sardonic', 'callous', 'macabre', 'grim', 'sinister',
    'clinical', 'antagonistic', 'inspiring', 'inspirational', 'steamy',
    'risque', 'affectation', 'curious', 'absurdist', 'absurd',
}

# Genres where a stricter genre gate applies — cross-genre contamination is
# more jarring for these (e.g. Horror shouldn't freely match Comedy/Crime).
STRICT_GENRES = {'comedy', 'animation', 'documentary', 'romance'}

# Blend weights per match priority
# wiki = separate TF-IDF channel on Wikipedia plot summaries (~70% film coverage)

PRIORITY_WEIGHTS = {
    #             keywords  semantic  wiki   wiki_sem  logline tagline mood   overview cast   director writer cattags h_pro  h_dyn  h_thm  h_str  h_ton  h_spl  h_dom  h_sty
    'balanced': dict(keywords=0.06, semantic=0.10, wiki=0.05, wiki_semantic=0.10, logline=0.04, tagline=0.02, mood=0.04, overview=0.02, cast=0.02, director=0.01, writer=0.00, cattags=0.07, helix_pro=0.11, helix_dyn=0.07, helix_thm=0.05, helix_str=0.07, helix_ton=0.07, helix_spl=0.00, helix_dom=0.08, helix_sty=0.03),
    'plot':     dict(keywords=0.08, semantic=0.08, wiki=0.08, wiki_semantic=0.12, logline=0.05, tagline=0.02, mood=0.02, overview=0.02, cast=0.00, director=0.00, writer=0.00, cattags=0.06, helix_pro=0.07, helix_dyn=0.07, helix_thm=0.07, helix_str=0.07, helix_ton=0.07, helix_spl=0.00, helix_dom=0.08, helix_sty=0.03),
    'cast':     dict(keywords=0.09, semantic=0.04, wiki=0.00, wiki_semantic=0.00, logline=0.04, tagline=0.02, mood=0.02, overview=0.02, cast=0.68, director=0.05, writer=0.00, cattags=0.01, helix_pro=0.01, helix_dyn=0.01, helix_thm=0.00, helix_str=0.00, helix_ton=0.00, helix_spl=0.00, helix_dom=0.01, helix_sty=0.00),
    'director': dict(keywords=0.09, semantic=0.04, wiki=0.00, wiki_semantic=0.00, logline=0.04, tagline=0.02, mood=0.02, overview=0.02, cast=0.05, director=0.68, writer=0.00, cattags=0.01, helix_pro=0.01, helix_dyn=0.01, helix_thm=0.00, helix_str=0.00, helix_ton=0.00, helix_spl=0.00, helix_dom=0.01, helix_sty=0.00),
    'writer':   dict(keywords=0.09, semantic=0.04, wiki=0.00, wiki_semantic=0.00, logline=0.04, tagline=0.02, mood=0.02, overview=0.02, cast=0.05, director=0.00, writer=0.68, cattags=0.01, helix_pro=0.01, helix_dyn=0.01, helix_thm=0.00, helix_str=0.00, helix_ton=0.00, helix_spl=0.00, helix_dom=0.01, helix_sty=0.00),
    'genre':    dict(keywords=0.12, semantic=0.08, wiki=0.04, wiki_semantic=0.04, logline=0.03, tagline=0.02, mood=0.09, overview=0.05, cast=0.08, director=0.04, writer=0.00, cattags=0.07, helix_pro=0.04, helix_dyn=0.04, helix_thm=0.04, helix_str=0.04, helix_ton=0.04, helix_spl=0.00, helix_dom=0.12, helix_sty=0.03),
    'narrative':dict(keywords=0.06, semantic=0.06, wiki=0.08, wiki_semantic=0.14, logline=0.05, tagline=0.02, mood=0.02, overview=0.02, cast=0.00, director=0.00, writer=0.00, cattags=0.04, helix_pro=0.08, helix_dyn=0.08, helix_thm=0.08, helix_str=0.08, helix_ton=0.08, helix_spl=0.00, helix_dom=0.08, helix_sty=0.03),
    'mood':     dict(keywords=0.05, semantic=0.08, wiki=0.03, wiki_semantic=0.05, logline=0.05, tagline=0.03, mood=0.35, overview=0.03, cast=0.00, director=0.00, writer=0.00, cattags=0.05, helix_pro=0.03, helix_dyn=0.03, helix_thm=0.03, helix_str=0.03, helix_ton=0.10, helix_spl=0.00, helix_dom=0.04, helix_sty=0.03),
}

class FilmHelixEngine:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.df = None
        self.titles_list = []
        self.vectorizers = {}
        self.matrices = {}
        self.semantic_model = None
        self.semantic_embeddings = None
        self.wiki_semantic_embeddings = None
        self.logline_embeddings = None
        self.tagline_embeddings = None

    def load_data(self):
        query = "SELECT * FROM movies WHERE overview IS NOT NULL AND is_valid = 1"
        self.df = pd.read_sql(query, self.conn)
        self.df = self.df.fillna("").reset_index(drop=True)

        self.df['vote_average'] = pd.to_numeric(self.df['vote_average'], errors='coerce').fillna(0)
        if 'rt_score' not in self.df.columns:
            self.df['rt_score'] = 0
        else:
            self.df['rt_score'] = pd.to_numeric(self.df['rt_score'], errors='coerce').fillna(0)

        self.df['year_int'] = pd.to_numeric(
            self.df['release_date'].astype(str).str[:4], errors='coerce'
        ).fillna(0)
        self.df['display_title'] = (
            self.df['title'] + " (" + self.df['year_int'].astype(int).astype(str) + ")"
        )
        self.df['search_title'] = self.df['display_title'].str.lower().str.strip()
        self.titles_list = sorted(self.df['display_title'].unique().tolist())

        # Semantic embeddings use TMDB overview only (256-token limit makes wiki too lossy).
        # Wiki plots get their own TF-IDF channel — no truncation issue for bag-of-words.
        self.df['narrative_text'] = self.df['overview']
        # Wiki plot: use fetched text when available (status ok/lead_section), else empty string
        self.df['vec_str_wiki'] = self.df.apply(
            lambda r: str(r['wiki_plot']) if str(r.get('wiki_plot_status', '')) in ('ok', 'lead_section') and str(r['wiki_plot']).strip() not in ('', 'nan') else '',
            axis=1
        )
        self.df['vec_str_overview']  = self.df['narrative_text']
        self.df['vec_str_keywords']  = self.df['dna_keywords'].apply(self._filter_plot_keywords)
        self.df['vec_str_mood']      = self.df['dna_keywords'].apply(self._filter_mood_keywords)
        # Logline proxy: first sentence of TMDB overview (~20 words, always available).
        self.df['vec_str_logline'] = self.df['overview'].apply(self._extract_logline)
        # Real TMDB tagline: fetched separately via fetch_taglines.py.
        # Empty string when not available — zero embedding contribution for those films.
        self.df['vec_str_tagline'] = self.df['tagline'].fillna('').astype(str).apply(
            lambda t: t.strip() if t.strip() not in ('', 'nan') else ''
        )
        self.df['vec_str_genre']     = self.df['dna_genres']
        # Wikipedia-derived narrative category tags (build_category_tags.py)
        self.df['vec_str_cattags']   = self.df['category_tags'].fillna('').astype(str).apply(
            lambda s: s.strip() if s.strip() not in ('', 'nan') else ''
        )
        # Haiku thematic DNA tags — pipe-separated, convert to space-separated for TF-IDF
        def _pipe_to_space(s):
            s = str(s).strip()
            return s.replace('|', ' ') if s not in ('', 'nan') else ''
        self.df['vec_str_helix_pro'] = self.df['helix_pro'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_dyn'] = self.df['helix_dyn'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_thm'] = self.df['helix_thm'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_str'] = self.df['helix_str'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_ton'] = self.df['helix_ton'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_spl'] = self.df['helix_spl'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_dom'] = self.df['helix_dom'].fillna('').apply(_pipe_to_space)
        self.df['vec_str_helix_sty'] = self.df['helix_sty'].fillna('').apply(_pipe_to_space)
        # Strip periods and hyphens so "JohnC.Reilly" / "TonyLeungChiu-Wai"
        # each stay as one token in CountVectorizer
        def _norm_names(s):
            return s.astype(str).str.replace('.', '', regex=False).str.replace('-', '', regex=False)
        self.df['vec_str_cast']      = _norm_names(self.df['dna_cast'])
        self.df['vec_str_director']  = _norm_names(self.df['dna_director'])
        self.df['vec_str_writer']    = _norm_names(self.df['dna_writer'])

        # Build token→display maps so shared cast/director/writer tags show
        # properly formatted names (e.g. "Park Chan-wook") instead of the
        # stripped/lowercased TF-IDF token ("parkchanwook").
        def _build_name_map(col):
            mapping = {}
            for val in self.df[col].dropna():
                for raw_token in str(val).split():
                    stripped = raw_token.replace('.', '').replace('-', '').lower()
                    mapping[stripped] = self._format_name_list(raw_token)
            return mapping
        self._cast_token_map     = _build_name_map('dna_cast')
        self._director_token_map = _build_name_map('dna_director')
        self._writer_token_map   = _build_name_map('dna_writer')

    def _filter_meta_keywords(self, keyword_str):
        if not keyword_str:
            return ""
        return " ".join(t for t in keyword_str.split() if t.lower() not in META_KEYWORD_STOPWORDS)

    @staticmethod
    def _norm_keyword(t):
        """Normalize a single keyword token: lowercase, strip hyphens.
        Ensures 'neo-noir' and 'neonoir' map to the same feature 'neonoir'
        rather than generating noise fragments 'neo', 'noir', 'neo noir'
        via ngram decomposition."""
        return t.lower().replace('-', '')

    def _filter_plot_keywords(self, keyword_str):
        """Plot-only keywords: strip both meta stopwords AND mood/atmosphere words.
        Hyphens are stripped so compound tags are consistent across films."""
        if not keyword_str:
            return ""
        return " ".join(
            self._norm_keyword(t) for t in keyword_str.split()
            if t.lower() not in META_KEYWORD_STOPWORDS
            and t.lower().replace('-', '') not in META_KEYWORD_STOPWORDS
            and t.lower() not in MOOD_KEYWORDS
        )

    def _filter_mood_keywords(self, keyword_str):
        """Mood-only keywords: keep only atmosphere/tone descriptors."""
        if not keyword_str:
            return ""
        return " ".join(
            self._norm_keyword(t) for t in keyword_str.split()
            if t.lower() in MOOD_KEYWORDS
        )

    @staticmethod
    def _extract_logline(overview):
        """Extract first sentence of overview as a proxy logline.
        Captures core conflict/protagonist in ~20 words — well within the
        256-token semantic model limit and avoids plot-detail noise from
        longer overviews."""
        if not overview or str(overview).strip() in ('', 'nan'):
            return ''
        text = str(overview).strip()
        for punct in ('. ', '! ', '? '):
            idx = text.find(punct)
            if idx > 20:  # skip very short fragments
                return text[:idx + 1].strip()
        return text[:200].strip()  # fallback: first 200 chars

    def train_model(self):
        self.vectorizers['overview'] = TfidfVectorizer(
            stop_words='english', min_df=2, max_df=0.85, dtype=np.float32
        )
        self.matrices['overview'] = self.vectorizers['overview'].fit_transform(
            self.df['vec_str_overview']
        )

        # ngram_range=(1,1): each dna_keywords token is already a complete multi-word
        # concept stored as a single joined string (e.g. 'serialkiller', 'nonlineartimeline').
        # Bigrams/trigrams across adjacent tags create false compound phrases from unrelated
        # sequential tags and cause stacking artifacts (jazz+musician, neo+noir+neonoir).
        self.vectorizers['keywords'] = TfidfVectorizer(
            stop_words='english', min_df=2, max_df=0.5,
            ngram_range=(1, 1), dtype=np.float32
        )
        self.matrices['keywords'] = self.vectorizers['keywords'].fit_transform(
            self.df['vec_str_keywords']
        )

        self.vectorizers['genre'] = CountVectorizer(min_df=1, dtype=np.float32)
        self.matrices['genre'] = self.vectorizers['genre'].fit_transform(
            self.df['vec_str_genre']
        )

        # Wiki plot TF-IDF — separate channel, full text, no truncation.
        # min_df=3 to filter noise; max_df=0.7 to drop plot-ubiquitous words.
        self.vectorizers['wiki'] = TfidfVectorizer(
            stop_words='english', min_df=3, max_df=0.7, dtype=np.float32,
            ngram_range=(1, 2),
        )
        self.matrices['wiki'] = self.vectorizers['wiki'].fit_transform(
            self.df['vec_str_wiki']
        )

        self.vectorizers['mood'] = TfidfVectorizer(
            min_df=2, max_df=0.9, dtype=np.float32
        )
        self.matrices['mood'] = self.vectorizers['mood'].fit_transform(
            self.df['vec_str_mood']
        )

        # Wikipedia category tags — human-curated narrative/thematic tokens.
        # min_df=2: a tag must appear in at least 2 films to be a feature.
        # max_df=0.4: drop tags so common they don't discriminate (e.g. generic drama).
        self.vectorizers['cattags'] = TfidfVectorizer(
            min_df=2, max_df=0.4, dtype=np.float32
        )
        self.matrices['cattags'] = self.vectorizers['cattags'].fit_transform(
            self.df['vec_str_cattags']
        )

        # Haiku thematic DNA channels — controlled vocabulary (~16-34 tags per bucket),
        # unigrams only (tags are already atomic concepts), min_df=2 so a tag must
        # appear in at least 2 films to be a feature, max_df=0.8 to drop near-universal tags.
        for helix_col in ('helix_pro', 'helix_dyn', 'helix_thm', 'helix_str', 'helix_ton', 'helix_spl', 'helix_dom', 'helix_sty'):
            self.vectorizers[helix_col] = TfidfVectorizer(
                min_df=2, max_df=0.8, dtype=np.float32
            )
            self.matrices[helix_col] = self.vectorizers[helix_col].fit_transform(
                self.df[f'vec_str_{helix_col}']
            )

        # lowercase=False preserves CamelCase in feature names so name tags
        # display correctly (e.g. "JodelleFerland" → "Jodelle Ferland" via
        # CamelCase split, not "jo delle fer land" via wordninja).
        self.vectorizers['cast'] = CountVectorizer(min_df=1, dtype=np.float32, lowercase=False)
        self.matrices['cast'] = self.vectorizers['cast'].fit_transform(
            self.df['vec_str_cast']
        )

        self.vectorizers['director'] = CountVectorizer(min_df=1, dtype=np.float32, lowercase=False)
        self.matrices['director'] = self.vectorizers['director'].fit_transform(
            self.df['vec_str_director']
        )

        self.vectorizers['writer'] = CountVectorizer(min_df=1, dtype=np.float32, lowercase=False)
        self.matrices['writer'] = self.vectorizers['writer'].fit_transform(
            self.df['vec_str_writer']
        )

        # Semantic layer — valid-only (~50K films), cached to disk after first run
        n = len(self.df)
        if os.path.exists(EMBEDDINGS_CACHE):
            cached = np.load(EMBEDDINGS_CACHE)
            if cached.shape[0] == n:
                self.semantic_embeddings = cached
            else:
                os.remove(EMBEDDINGS_CACHE)

        if self.semantic_embeddings is None:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_embeddings = self.semantic_model.encode(
                self.df['narrative_text'].tolist(),
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(EMBEDDINGS_CACHE, self.semantic_embeddings)

        # Wiki semantic embeddings — chunked mean-pooling to handle long plots
        n = len(self.df)
        if os.path.exists(WIKI_EMBEDDINGS_CACHE):
            cached = np.load(WIKI_EMBEDDINGS_CACHE)
            if cached.shape[0] == n:
                self.wiki_semantic_embeddings = cached
            else:
                os.remove(WIKI_EMBEDDINGS_CACHE)

        if self.wiki_semantic_embeddings is None:
            if self.semantic_model is None:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            CHUNK_WORDS = 180
            EMB_DIM = 384
            wiki_texts = self.df['vec_str_wiki'].tolist()
            # Build flat list of (film_idx, chunk_text) so we encode in one batch
            film_chunks = []   # list of (film_idx, chunk_text)
            no_wiki_idx = []   # indices with no wiki text
            for i, text in enumerate(wiki_texts):
                if not text or not text.strip():
                    no_wiki_idx.append(i)
                    continue
                words = text.split()
                for start in range(0, len(words), CHUNK_WORDS):
                    film_chunks.append((i, ' '.join(words[start:start + CHUNK_WORDS])))
            print(f"  Wiki semantic: encoding {len(film_chunks)} chunks for {n - len(no_wiki_idx)} films...")
            all_embs = np.zeros((len(film_chunks), EMB_DIM), dtype=np.float32)
            if film_chunks:
                texts_only = [c[1] for c in film_chunks]
                all_embs = self.semantic_model.encode(
                    texts_only,
                    batch_size=64,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            # Mean-pool chunks per film
            wiki_sem = np.zeros((n, EMB_DIM), dtype=np.float32)
            counts   = np.zeros(n, dtype=np.int32)
            for (film_idx, _), emb in zip(film_chunks, all_embs):
                wiki_sem[film_idx] += emb
                counts[film_idx] += 1
            has_wiki = counts > 0
            wiki_sem[has_wiki] /= counts[has_wiki, np.newaxis]
            norms = np.linalg.norm(wiki_sem, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            wiki_sem /= norms
            self.wiki_semantic_embeddings = wiki_sem
            np.save(WIKI_EMBEDDINGS_CACHE, self.wiki_semantic_embeddings)
            print(f"  Wiki semantic: {has_wiki.sum()} films with embeddings, {len(no_wiki_idx)} without.")

        # Logline semantic embeddings — first sentence of overview, fast to encode
        n = len(self.df)
        if os.path.exists(LOGLINE_EMBEDDINGS_CACHE):
            cached = np.load(LOGLINE_EMBEDDINGS_CACHE)
            if cached.shape[0] == n:
                self.logline_embeddings = cached
            else:
                os.remove(LOGLINE_EMBEDDINGS_CACHE)

        if self.logline_embeddings is None:
            if self.semantic_model is None:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  Logline semantic: encoding first-sentence overviews...")
            self.logline_embeddings = self.semantic_model.encode(
                self.df['vec_str_logline'].tolist(),
                batch_size=512,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(LOGLINE_EMBEDDINGS_CACHE, self.logline_embeddings)

        # Tagline embeddings — real TMDB taglines (fetched via fetch_taglines.py).
        # Films with no tagline get a zero vector (no contribution, no penalty).
        n = len(self.df)
        if os.path.exists(TAGLINE_EMBEDDINGS_CACHE):
            cached = np.load(TAGLINE_EMBEDDINGS_CACHE)
            if cached.shape[0] == n:
                self.tagline_embeddings = cached
            else:
                os.remove(TAGLINE_EMBEDDINGS_CACHE)

        if self.tagline_embeddings is None:
            if self.semantic_model is None:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            tagline_texts = self.df['vec_str_tagline'].tolist()
            has_tagline = [bool(t.strip()) for t in tagline_texts]
            coverage = sum(has_tagline)
            print(f"  Tagline semantic: {coverage:,}/{n:,} films have taglines ({coverage/n*100:.1f}%)")
            # Encode all — empty strings produce near-zero embeddings, then we zero them out
            raw_embs = self.semantic_model.encode(
                tagline_texts,
                batch_size=512,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Zero out embeddings for films with no tagline so they don't contribute
            tag_arr = raw_embs.astype(np.float32)
            for i, has in enumerate(has_tagline):
                if not has:
                    tag_arr[i] = 0.0
            self.tagline_embeddings = tag_arr
            np.save(TAGLINE_EMBEDDINGS_CACHE, self.tagline_embeddings)

    def _get_top_keyword_scores(self, idx_source, idx_target, top_n=5):
        """Return list of (token, overlap_tfidf_score) for the keywords channel.
        Used by test_engine --debug to show exactly which tokens drove a kw match."""
        vec = self.vectorizers['keywords']
        mat = self.matrices['keywords']
        intersection = mat[idx_source].multiply(mat[idx_target])
        if intersection.nnz == 0:
            return []
        dense = intersection.toarray().flatten()
        overlap_idx = np.where(dense > 0)[0]
        sorted_idx = overlap_idx[np.argsort(dense[overlap_idx])[::-1]]
        feature_names = vec.get_feature_names_out()
        return [(feature_names[i], round(float(dense[i]), 4)) for i in sorted_idx[:top_n]]

    def _get_top_overlapping_terms(self, idx_source, idx_target, feature_name, top_n=5):
        vec = self.vectorizers[feature_name]
        mat = self.matrices[feature_name]
        intersection = mat[idx_source].multiply(mat[idx_target])
        if intersection.nnz == 0:
            return ""
        dense = intersection.toarray().flatten()
        overlap_idx = np.where(dense > 0)[0]
        sorted_idx = overlap_idx[np.argsort(dense[overlap_idx])[::-1]]
        feature_names = vec.get_feature_names_out()
        name_map = {
            'cast':     self._cast_token_map,
            'director': self._director_token_map,
            'writer':   self._writer_token_map,
        }.get(feature_name)
        def _resolve(token):
            if name_map:
                key = token.replace('.', '').replace('-', '').lower()
                return name_map.get(key, self._format_name_list(token))
            return token
        return ", ".join(_resolve(feature_names[i]) for i in sorted_idx[:top_n])

    def get_recommendations(self, display_title, min_rating=0.0, min_rt=0,
                            year_range=(1900, 2030), exclude_foreign=False,
                            exclude_animated=False, exclude_obscure=False,
                            year_window=None, priority='balanced', exclude_sequels=False):
        title_clean = display_title.lower().strip()
        matches = self.df[self.df['search_title'] == title_clean]
        if matches.empty:
            return None
        idx = matches.index[0]

        s_overview  = cosine_similarity(self.matrices['overview'][idx],  self.matrices['overview']).flatten()
        s_keywords  = cosine_similarity(self.matrices['keywords'][idx],  self.matrices['keywords']).flatten()

        # Keyword diversity multiplier (Balanced/Plot/Narrative only):
        # Dampen results whose keyword match rests on only 1 or 2 shared tokens.
        # A single broad tag (neonoir, supernatural, hacker) is weak evidence;
        # 3+ shared tokens is genuine narrative overlap.
        _p = priority[0] if isinstance(priority, (list, tuple)) else priority
        s_keywords_raw = s_keywords.copy()  # preserve pre-multiplier value for floor check
        if _p not in ('cast', 'director', 'writer', 'genre'):
            src_kw_bool = (self.matrices['keywords'][idx] > 0)
            shared_kw_counts = np.array(
                src_kw_bool.dot((self.matrices['keywords'] > 0).T).todense()
            ).flatten()
            kw_diversity = np.where(shared_kw_counts >= 3, 1.0,
                           np.where(shared_kw_counts == 2, 0.65,
                           np.where(shared_kw_counts == 1, 0.50, 0.25)))
            s_keywords = s_keywords * kw_diversity

        s_wiki      = cosine_similarity(self.matrices['wiki'][idx],      self.matrices['wiki']).flatten()
        s_cattags   = cosine_similarity(self.matrices['cattags'][idx],   self.matrices['cattags']).flatten()

        # Category tag diversity multiplier: same logic as keyword diversity.
        # A single shared category tag (music, crime) is weak; 3+ specific shared
        # tags (about_music_and_musicians + about_educators + about_harassment) is real signal.
        if _p not in ('cast', 'director', 'writer', 'genre'):
            src_cat_bool = (self.matrices['cattags'][idx] > 0)
            shared_cat_counts = np.array(
                src_cat_bool.dot((self.matrices['cattags'] > 0).T).todense()
            ).flatten()
            cat_diversity = np.where(shared_cat_counts >= 4, 1.0,
                            np.where(shared_cat_counts == 3, 0.75,
                            np.where(shared_cat_counts == 2, 0.45,
                            np.where(shared_cat_counts == 1, 0.20, 0.0))))
            s_cattags = s_cattags * cat_diversity

        s_genre     = cosine_similarity(self.matrices['genre'][idx],     self.matrices['genre']).flatten()
        s_cast      = cosine_similarity(self.matrices['cast'][idx],      self.matrices['cast']).flatten()
        s_director  = cosine_similarity(self.matrices['director'][idx],  self.matrices['director']).flatten()
        s_writer    = cosine_similarity(self.matrices['writer'][idx],    self.matrices['writer']).flatten()
        s_helix_pro = cosine_similarity(self.matrices['helix_pro'][idx], self.matrices['helix_pro']).flatten()
        s_helix_dyn = cosine_similarity(self.matrices['helix_dyn'][idx], self.matrices['helix_dyn']).flatten()
        s_helix_thm = cosine_similarity(self.matrices['helix_thm'][idx], self.matrices['helix_thm']).flatten()
        s_helix_str = cosine_similarity(self.matrices['helix_str'][idx], self.matrices['helix_str']).flatten()
        s_helix_ton = cosine_similarity(self.matrices['helix_ton'][idx], self.matrices['helix_ton']).flatten()
        s_helix_spl = cosine_similarity(self.matrices['helix_spl'][idx], self.matrices['helix_spl']).flatten()
        s_helix_dom = cosine_similarity(self.matrices['helix_dom'][idx], self.matrices['helix_dom']).flatten()
        s_helix_sty = cosine_similarity(self.matrices['helix_sty'][idx], self.matrices['helix_sty']).flatten()

        # Helix diversity multiplier: dampen helix scores when shared tags are few.
        # Generic tags like reluctant_hero shared across hundreds of horror films
        # should not dominate over keyword/semantic channels.
        # Count shared tags across ALL helix columns combined.
        if _p not in ('cast', 'director', 'writer', 'genre'):
            helix_shared_counts = np.zeros(len(self.df), dtype=np.float32)
            for hc in ('helix_pro', 'helix_dyn', 'helix_thm', 'helix_str', 'helix_ton', 'helix_dom', 'helix_sty'):
                src_bool = (self.matrices[hc][idx] > 0)
                res_bool = (self.matrices[hc] > 0)
                helix_shared_counts += np.array(src_bool.dot(res_bool.T).todense()).flatten()

            helix_diversity = np.where(helix_shared_counts >= 4, 1.0,
                              np.where(helix_shared_counts == 3, 0.70,
                              np.where(helix_shared_counts == 2, 0.35,
                              np.where(helix_shared_counts == 1, 0.10, 0.0))))
            s_helix_pro = s_helix_pro * helix_diversity
            s_helix_dyn = s_helix_dyn * helix_diversity
            s_helix_thm = s_helix_thm * helix_diversity
            s_helix_str = s_helix_str * helix_diversity
            s_helix_ton = s_helix_ton * helix_diversity
            s_helix_dom = s_helix_dom * helix_diversity
            s_helix_sty = s_helix_sty * helix_diversity
        s_mood           = cosine_similarity(self.matrices['mood'][idx],      self.matrices['mood']).flatten()
        s_semantic       = (self.semantic_embeddings[idx] @ self.semantic_embeddings.T).astype(np.float32)
        s_wiki_semantic  = (self.wiki_semantic_embeddings[idx] @ self.wiki_semantic_embeddings.T).astype(np.float32)
        s_logline        = (self.logline_embeddings[idx] @ self.logline_embeddings.T).astype(np.float32)
        s_tagline        = (self.tagline_embeddings[idx] @ self.tagline_embeddings.T).astype(np.float32) if self.tagline_embeddings is not None else np.zeros(len(self.df), dtype=np.float32)

        # Adaptive genre gate
        source_genres = set(str(self.df.iloc[idx]['dna_genres']).lower().split())
        is_strict = bool(source_genres & STRICT_GENRES)
        gate_threshold = 0.35 if is_strict else 0.20

        # Priority can be a string or a list of 2 strings (dual-priority blend).
        # For dual: average the two weight dicts.
        if isinstance(priority, (list, tuple)) and len(priority) == 2:
            w1 = PRIORITY_WEIGHTS.get(priority[0], PRIORITY_WEIGHTS['balanced'])
            w2 = PRIORITY_WEIGHTS.get(priority[1], PRIORITY_WEIGHTS['balanced'])
            w = {k: (w1[k] + w2[k]) / 2 for k in w1}
            use_genre_boost = False
        else:
            p = priority[0] if isinstance(priority, (list, tuple)) else priority
            w = PRIORITY_WEIGHTS.get(p, PRIORITY_WEIGHTS['balanced'])
            use_genre_boost = (p == 'genre')

        base_scores = (
            s_keywords      * w['keywords'] +
            s_semantic      * w['semantic'] +
            s_wiki          * w.get('wiki', 0.0) +
            s_wiki_semantic * w.get('wiki_semantic', 0.0) +
            s_logline       * w.get('logline', 0.0) +
            s_tagline       * w.get('tagline', 0.0) +
            s_mood          * w.get('mood', 0.0) +
            s_cattags       * w.get('cattags', 0.0) +
            s_helix_pro     * w.get('helix_pro', 0.0) +
            s_helix_dyn     * w.get('helix_dyn', 0.0) +
            s_helix_thm     * w.get('helix_thm', 0.0) +
            s_helix_str     * w.get('helix_str', 0.0) +
            s_helix_ton     * w.get('helix_ton', 0.0) +
            s_helix_spl     * w.get('helix_spl', 0.0) +
            s_helix_dom     * w.get('helix_dom', 0.0) +
            s_helix_sty     * w.get('helix_sty', 0.0) +
            s_overview      * w['overview'] +
            s_cast          * w['cast'] +
            s_director      * w['director'] +
            s_writer        * w['writer']
        )

        # For single 'genre' priority: continuous genre multiplier rewards closer genre match.
        # For blended priorities, use the standard binary gate.
        if use_genre_boost:
            genre_factor = np.where(s_genre >= gate_threshold, s_genre, 0.0)
            final_scores = base_scores * genre_factor + s_genre * 0.20
        else:
            genre_multiplier = np.where(s_genre >= gate_threshold, 1.0, 0.0)
            final_scores = base_scores * genre_multiplier

        if 'documentary' not in source_genres:
            final_scores[self.df['dna_genres'].str.lower().str.contains('documentary', na=False)] = 0.0

        # Result-side strict genre gate:
        # Comedy results are always gated (0.60) — tone mismatch is too jarring regardless.
        # Other strict genre results (animation, documentary, musical, romance) are only
        # gated when the SOURCE is also a strict genre film. A non-strict source (e.g. Drama)
        # should not have Horror/Thriller results zeroed just because they contain a strict genre.
        genres_lower = self.df['dna_genres'].str.lower()
        result_has_comedy = genres_lower.str.contains('comedy', na=False)
        result_has_animation = genres_lower.str.contains('animation|family', na=False)
        source_primary_genre = str(self.df.iloc[idx].get('dna_genres', '') or '').split()[0].lower()
        if not source_genres.intersection({'comedy'}) or source_primary_genre != 'comedy':
            final_scores[result_has_comedy & (s_genre < 0.60)] = 0.0
        if not source_genres.intersection({'animation', 'family'}):
            final_scores[result_has_animation & (s_genre < 0.70)] = 0.0
        if is_strict:
            result_has_strict = genres_lower.apply(
                lambda g: bool(set(str(g).split()) & STRICT_GENRES)
            )
            final_scores[result_has_strict & ~result_has_comedy & (s_genre < 0.35)] = 0.0

        if min_rating > 0:
            final_scores[self.df['vote_average'] < min_rating] = 0.0
        if min_rt > 0:
            final_scores[self.df['rt_score'] < min_rt] = 0.0

        min_y, max_y = year_range
        final_scores[(self.df['year_int'] < min_y) | (self.df['year_int'] > max_y)] = 0.0

        if year_window is not None:
            sy = int(self.df.iloc[idx]['year_int'])
            if sy > 0:
                final_scores[
                    (self.df['year_int'] < sy - year_window) |
                    (self.df['year_int'] > sy + year_window)
                ] = 0.0

        if exclude_foreign:
            final_scores[~self.df['dna_lang'].str.lower().str.contains('en', na=False)] = 0.0

        # Same-language affinity boost: rewards same-language results by 20%
        # Naturally penalizes cross-language matches without hard exclusion,
        # allowing strong foreign matches (Oldboy, Parasite) to still surface.
        if not exclude_foreign:
            source_lang = str(self.df.iloc[idx].get('dna_lang', 'en') or 'en').lower()[:2]
            result_langs = self.df['dna_lang'].str.lower().str[:2].fillna('en')
            same_lang_mask = result_langs == source_lang
            final_scores[same_lang_mask] *= 1.20

        if exclude_animated:
            final_scores[self.df['dna_genres'].str.lower().str.contains('animation', na=False)] = 0.0

        if exclude_obscure:
            final_scores[pd.to_numeric(self.df['vote_count'], errors='coerce').fillna(0) < 25000] = 0.0

        if exclude_sequels:
            kw_lower = self.df['dna_keywords'].str.lower()
            # Only filter sequels/prequels — NOT remakes.
            # Remakes are the closest possible narrative match and should appear as results.
            seq_mask = kw_lower.str.contains('sequel', na=False) | kw_lower.str.contains('prequel', na=False)
            final_scores[seq_mask] = 0.0

        # Hard keyword floor: zero out results with near-zero keyword overlap.
        # Prevents single incidental keyword from carrying a match (e.g. Another Earth via 'solarsystem').
        # Uses raw (pre-diversity-multiplier) score so films with real but sparse keyword overlap
        # (e.g. TÁR vs Whiplash sharing "musician") aren't incorrectly excluded.
        # Only veto if BOTH keywords and semantic plot overlap are garbage
        final_scores = np.where(s_keywords_raw < 0.02, final_scores * 0.60, final_scores)

        # Remake bridge: results tagged as remakes with strong semantic similarity
        # are likely direct adaptations of the source film — boost score to surface them
        result_remake_mask = self.df['dna_keywords'].str.lower().str.contains(r'\bremake\b', na=False)
        final_scores[result_remake_mask & (s_semantic >= 0.40)] *= 1.5

        # --- HELIX GATING ---
        # If the source movie has Helix tags, heavily penalize candidates that share ZERO Helix tags.
        has_source_tags = (
            self.matrices['helix_pro'][idx].nnz > 0 or
            self.matrices['helix_dyn'][idx].nnz > 0 or
            self.matrices['helix_thm'][idx].nnz > 0 or
            self.matrices['helix_str'][idx].nnz > 0 or
            self.matrices['helix_ton'][idx].nnz > 0 or
            self.matrices['helix_dom'][idx].nnz > 0 or
            self.matrices['helix_sty'][idx].nnz > 0
        )

        # --- THE SMELL TEST (Genre & Plot Boundary) ---
        # Prevents tonal whiplash (e.g., Caddyshack on Parasite). 
        # If the literal plot (semantic) and the baseline genre (cattags) share almost zero overlap,
        # crush the score, regardless of how many abstract thematic tags they share.
        total_helix_sim = (s_helix_pro + s_helix_dyn + s_helix_thm + s_helix_str + s_helix_ton + s_helix_dom + s_helix_sty)
        final_scores[(s_semantic < 0.15) & (s_cattags < 0.10) & (total_helix_sim < 0.50)] *= 0.10

        if has_source_tags:
            final_scores[total_helix_sim == 0] *= 0.10

        final_scores[idx] = 0.0

        top_indices = np.argpartition(final_scores, -50)[-50:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
        top_indices = [i for i in top_indices if final_scores[i] > 0.0]

        results = []
        for i in top_indices:
            row = self.df.iloc[i]

            # Always compute all three overlap types independently — shown as
            # separate tag rows in the UI so users see the full match reason.
            overlap_k      = self._get_top_overlapping_terms(idx, i, 'keywords', 5)
            top_kw_scores  = self._get_top_keyword_scores(idx, i, top_n=5)
            overlap_cast   = self._get_top_overlapping_terms(idx, i, 'cast', 3)
            overlap_dir    = self._get_top_overlapping_terms(idx, i, 'director', 2)
            overlap_writer = self._get_top_overlapping_terms(idx, i, 'writer', 2)

            # Helix tag overlap: shared tags across all 5 scored helix columns
            shared_helix_tags = []
            for hcol in ('helix_pro', 'helix_dyn', 'helix_thm', 'helix_str', 'helix_ton'):
                src_tags = set(str(self.df.iloc[idx].get(hcol, '') or '').split('|'))
                res_tags = set(str(self.df.iloc[i].get(hcol, '') or '').split('|'))
                shared = src_tags & res_tags - {'', 'nan'}
                shared_helix_tags.extend(sorted(shared))
            shared_helix = ', '.join(shared_helix_tags)

            results.append({
                'title':            row['title'],
                'score':            f"{int(final_scores[i] * 100)}%",
                '_top_kw': top_kw_scores,
                '_ch': {
                    'kw':       round(float(s_keywords[i] * w['keywords']), 4),
                    'sem':      round(float(s_semantic[i] * w['semantic']), 4),
                    'wiki':     round(float(s_wiki[i] * w.get('wiki', 0)), 4),
                    'wiki_sem': round(float(s_wiki_semantic[i] * w.get('wiki_semantic', 0)), 4),
                    'logline':  round(float(s_logline[i] * w.get('logline', 0)), 4),
                    'tagline':  round(float(s_tagline[i] * w.get('tagline', 0)), 4),
                    'mood':     round(float(s_mood[i] * w.get('mood', 0)), 4),
                    'cattags':  round(float(s_cattags[i] * w.get('cattags', 0)), 4),
                    'raw_cat':  round(float(s_cattags[i]), 4),
                    'ov':       round(float(s_overview[i] * w['overview']), 4),
                    'cast':     round(float(s_cast[i] * w['cast']), 4),
                    'dir':      round(float(s_director[i] * w['director']), 4),
                    'raw_kw':   round(float(s_keywords[i]), 4),
                    'raw_sem':  round(float(s_semantic[i]), 4),
                    'raw_log':  round(float(s_logline[i]), 4),
                    'raw_tag':  round(float(s_tagline[i]), 4),
                },
                'year':             str(row['release_date'])[:4],
                'rating':           float(row.get('vote_average', 0)),
                'rt_score':         int(row.get('rt_score', 0) or 0),
                'poster':           str(row.get('poster_url', '')),
                'warnings':         str(row.get('warnings', '')),
                'shared_helix':     shared_helix,
                'shared_keywords':  overlap_k,
                'shared_cast':      overlap_cast,
                'shared_director':  overlap_dir,
                'shared_writer':    overlap_writer,
                'imdb_id':          str(row.get('tconst', '')),
                'overview':         str(row.get('overview', '')),
                'director':         self._format_name_list(str(row.get('dna_director', ''))),
                'writer':           self._format_name_list(str(row.get('dna_writer', ''))),
                'cast':             self._format_name_list(str(row.get('dna_cast', ''))),
                'runtime':          str(row.get('runtime', '')),
                'country':          self._format_country(str(row.get('dna_country', ''))),
                'lang':             str(row.get('dna_lang', '')),
                'genres':           str(row.get('dna_genres', '')),
                'vote_count':       float(row.get('vote_count', 0) or 0),
            })

        return {'source': self.df.iloc[idx]['display_title'], 'matches': results}

    def _format_name_list(self, raw):
        """CamelCase split: 'JakeGyllenhaal RizAhmed' → 'Jake Gyllenhaal, Riz Ahmed'
        Also strips trailing/leading punctuation (e.g. 'StacyMartin)' → 'Stacy Martin'),
        handles accented first chars (e.g. 'ChloëGrace' → 'Chloë Grace'), and
        rejoins split Irish/Scottish prefixes ('Ian Mc Shane' → 'Ian McShane').
        """
        if not raw or raw == 'nan':
            return ''
        formatted = []
        for t in raw.split():
            # Strip leading/trailing non-alphabetic chars (handles stray parentheses)
            t = re.sub(r'^[^a-zA-ZÀ-ÿ]+|[^a-zA-ZÀ-ÿ]+$', '', t)
            if not t:
                continue
            # Split on lowercase/accented-lowercase → uppercase boundaries
            spaced = re.sub(r'([a-zà-ÿ])([A-Z])', r'\1 \2', t)
            # Rejoin Irish/Scottish surname prefixes (Mc, Mac) split by regex
            parts = spaced.split()
            rejoined = []
            i = 0
            while i < len(parts):
                if parts[i] in ('Mc', 'Mac', 'Di') and i + 1 < len(parts):
                    rejoined.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    rejoined.append(parts[i])
                    i += 1
            formatted.append(' '.join(rejoined))
        return ', '.join(formatted)

    def _format_country(self, raw):
        """Use wordninja to split country tokens regardless of capitalization pattern.
        'UnitedStatesofAmerica' → 'United States Of America'
        'UnitedKingdom France'  → 'United Kingdom, France'
        """
        if not raw or raw == 'nan':
            return ''
        tokens = raw.split()
        formatted = [
            ' '.join(p.title() for p in wordninja.split(t.lower()))
            for t in tokens
        ]
        return ', '.join(formatted)
