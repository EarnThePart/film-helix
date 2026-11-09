"""
helix_tagger.py  (merged — single API call produces all 8 Helix columns)
-------------------------------------------------------------------------
Tags films with full Helix DNA using a single Claude Haiku call per film.
Writes: helix_pro, helix_dyn, helix_thm, helix_str, helix_ton, helix_spl,
        helix_dom, helix_sty, helix_low_confidence

Setup:
  export ANTHROPIC_API_KEY
  python helix_tagger.py

Options:
  --limit N             stop after N films (normal queue mode)
  --max-cost $X         halt if total cost exceeds $X (default: 20.00)
  --dry-run             print first 3 user messages without calling the API
  --ids-file PATH       retag only films whose TMDB IDs are in file (one per
                        line); bypasses checkpoint and NULL filter, overwrites
  --output-csv PATH     write all 8 tag columns + title/release_date/vote_count
                        to a CSV file as films are processed
  --sort-by-votes       process highest vote_count first (normal queue only)
  --progress-every N    print progress summary every N films (default: 50)
"""

import csv
import sqlite3
import json
import time
import sys
import os
import re
import unicodedata
import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import anthropic

#read ANTHROPIC_API_KEY (and the rest) from .env like every other script here.
#Without this the key has to be exported into the shell by hand, which is why a
#run could fail with "Set ANTHROPIC_API_KEY" despite .env holding it.
from dotenv import load_dotenv
load_dotenv()

DB_PATH         = "movies.db"
CHECKPOINT_PATH = "data/helix_processed_tmdb_ids.txt"
ERROR_LOG_PATH  = "data/helix_errors.log"
PROMPT_PATH     = "data/helix_system_prompt.txt"

MODEL           = "claude-haiku-4-5-20251001"
MAX_TOKENS      = 1024
SLEEP_BETWEEN   = 0.5
PROGRESS_EVERY  = 50

#Haiku 4.5 list prices, $/MTok (verified 2026-11-09). These were Haiku 3.5's rates
#(0.80/4.00/1.00/0.08), which made every cost readout and the --max-cost circuit
#breaker 20% too low.
PRICE_INPUT       = 1.00
PRICE_OUTPUT      = 5.00
PRICE_CACHE_WRITE = 1.25   #5-minute cache write = 1.25x input
PRICE_CACHE_READ  = 0.10   #cache read = 0.1x input

MAX_TAG_ATTEMPTS       = 2   # give up after this many genuinely-empty results
EXPECTED_COST_PER_FILM = 0.0045
VELOCITY_WINDOW        = 50
VELOCITY_CEILING       = 0.008

#Re-tagging policy (2026-11-09). A film that already has tags is NOT re-queued just
#because some fields are empty (that re-billed ~160 tagged films and let Haiku
#overwrite them with different judgment calls). It is re-queued only when the
#inputs it was tagged on were thin and have since been enriched.
STUB_PLOT_CHARS        = 500   #plot at tagging time shorter than this = "stub"
ENRICH_MIN_PLOT_GROWTH = 400   #...and it must have grown by at least this many chars
STUB_KEYWORDS          = 5     #keyword count at tagging time below this = "thin"
ENRICH_MIN_KW_GROWTH   = 5     #...and it must have gained at least this many tokens
ENRICH_WINDOW_DAYS     = 365   #enrichment re-tags only for films released this recently...
ENRICH_HIGH_VALUE_VOTES = 50000  #...or this popular (same bar as weekly_refresh HIGH_VALUE_VOTES)
AUDIT_CSV_PATH         = "data/helix_retag_audit.csv"
HELIX_FIELDS = ["helix_pro", "helix_dyn", "helix_thm", "helix_str",
                "helix_ton", "helix_spl", "helix_dom", "helix_sty"]
KW_COUNT_SQL = ("(CASE WHEN TRIM(COALESCE(dna_keywords,''))='' THEN 0 "
                "ELSE LENGTH(TRIM(dna_keywords)) - LENGTH(REPLACE(TRIM(dna_keywords),' ','')) + 1 END)")

VALID_TAGS = {
    #PROTAGONIST
    "obsessive_perfectionist", "ruthless_climber", "system_disruptor",
    "reluctant_hero", "haunted_by_past", "psychologically_vulnerable",
    "unreliable_narrator", "calculating_manipulator", "sympathetic_opportunist",
    "self_sabotaging_failure", "seduced_by_power", "determined_outsider",
    "introspective_idealist", "ensemble_no_single_lead", "reckless_hedonist",
    "reluctant_moral_hero",
    #DYNAMIC
    "toxic_mentorship", "psychological_manipulation", "ideological_rivals",
    "individual_vs_institution", "intellectual_partnership", "loyalty_and_betrayal",
    "class_collision", "cat_and_mouse", "marriage_as_warfare", "ensemble_mission",
    "friendship_under_strain", "community_as_predator", "protector_and_vulnerable",
    "strangers_becoming_intimate", "chaos_interrupts_order", "mutual_destruction",
    #THEME
    "cost_of_ambition", "corruption_of_power", "american_dream_corrupted",
    "media_as_moral_void", "challenging_orthodoxy", "loss_of_identity",
    "hubris_of_science", "man_vs_technology", "man_vs_environment",
    "grief_as_transformation", "reality_vs_constructed_reality",
    "institutional_corruption", "seduction_of_power", "randomness_of_fate",
    "redemption_vs_damnation", "class_inequality", "surveillance_state",
    "paranoia_and_conspiracy", "cult_and_indoctrination", "monster_as_mirror",
    "moral_awakening", "transience_of_connection", "fear_of_failure",
    "gender_as_trap", "transformed_by_journey", "belonging_and_family",
    "nihilism", "moral_ambiguity_as_theme", "satire_as_critique",
    "long_shot_odds",
    #STRUCTURE
    "escalating_tension", "slow_burn", "nonlinear_timeline", "ticking_clock",
    "nested_narrative", "dual_perspective", "mosaic_narrative", "character_study",
    "procedural", "episodic_journey", "genre_shifting", "real_time",
    "dialogue_as_plot", "documentary_realism", "unreliable_voiceover",
    "puzzle_box_narrative", "circular_narrative", "set_piece_driven",
    "meta_narrative", "quest_narrative", "contained_environment",
    #TONE
    "hyper_stylized", "bleak_and_oppressive", "darkly_comic", "meditative_and_slow",
    "relentlessly_tense", "dreamlike_and_surreal", "warm_and_nostalgic",
    "cold_and_clinical", "epic_and_grandiose", "playful_and_irreverent",
    "intimate_and_naturalistic", "dread_and_unease", "kinetic_and_propulsive",
    "melancholic_and_elegiac",
    #SPOILERS
    "pyrrhic_victory", "fall_from_grace", "dark_triumph", "tragedy_from_circumstance",
    "survival_at_a_cost", "trap_with_no_exit", "ambiguous_reality",
    "bittersweet_ending", "tentative_redemption", "liberation_through_destruction",
    "twist_recontextualizes_everything", "rise_and_fall", "happy_ending",
    "tragic_ending",
}

VALID_DOM_TAGS = {
    "dom_creative_performance", "dom_creative_solitude", "dom_criminal_underworld",
    "dom_criminal_justice", "dom_penal_system", "dom_military_combat",
    "dom_corporate_finance", "dom_political_arena", "dom_academic_scientific",
    "dom_domestic_suburban", "dom_urban_civic", "dom_high_society_aristocracy",
    "dom_espionage_intelligence", "dom_journalism_media", "dom_sports_competition",
    "dom_isolated_containment", "dom_wilderness_frontier", "dom_open_ocean",
    "dom_deep_space", "dom_tech_corporate", "dom_supernatural_occult",
    "dom_afterlife_metaphysical", "dom_civilization_collapse", "dom_dystopian_society",
}

VALID_STY_TAGS = {
    "style_classical_invisible", "style_epic_operatic", "style_hyper_kinetic",
    "style_hyper_stylized", "style_slow_burn", "style_meditative_atmospheric",
    "style_procedural_methodical", "style_cold_clinical", "style_surreal_expressionist",
    "style_raw_verite", "style_found_footage",
}


def load_system_prompt():
    p = Path(PROMPT_PATH)
    if not p.exists():
        print(f"\n[ERROR] System prompt not found at {PROMPT_PATH}")
        sys.exit(1)
    return p.read_text(encoding="utf-8").strip()


def ensure_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(movies)").fetchall()}
    for col, dtype in [
        ("helix_pro",            "TEXT"),
        ("helix_dyn",            "TEXT"),
        ("helix_thm",            "TEXT"),
        ("helix_str",            "TEXT"),
        ("helix_ton",            "TEXT"),
        ("helix_spl",            "TEXT"),
        ("helix_dom",            "TEXT"),
        ("helix_sty",            "TEXT"),
        ("helix_low_confidence", "INTEGER"),
        ("helix_attempts",       "INTEGER"),
        #Enrichment snapshot, recorded at tagging time (2026-11-09). Lets the queue
        #tell "this film's plot/keywords have since been filled in" from "nothing
        #changed" without re-tagging (and re-billing) everything.
        ("helix_wiki_chars",     "INTEGER"),   #LENGTH(wiki_plot) when last tagged
        ("helix_kw_count",       "INTEGER"),   ## of dna_keywords tokens when last tagged
        ("helix_tagged_at",      "TEXT"),      #ISO timestamp of last successful tagging
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {col} {dtype}")
    conn.commit()


def load_checkpoint():
    p = Path(CHECKPOINT_PATH)
    if not p.exists():
        return set()
    return set(line.strip() for line in p.read_text().splitlines() if line.strip())


def save_checkpoint(tmdb_id, checkpoint_set):
    checkpoint_set.add(str(tmdb_id))
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(f"{tmdb_id}\n")


def log_error(tmdb_id, title, reason):
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] id={tmdb_id} title={title!r} reason={reason}\n")


def log_hallucination(tmdb_id, title, bucket, tag):
    with open("data/helix_hallucinations.log", "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"id={tmdb_id} title={title!r} bucket={bucket} invalid_tag={tag!r}\n")


class FatalAPIError(Exception):
    """An API failure that retrying cannot fix — abort the whole run."""


#Substrings identifying account/request problems that will never resolve on retry.
#Credit exhaustion is the one that bit: the old code treated it as transient, so it
#backed off 10/30/90/270s per film and then repeated that for the NEXT film. Across
#900 queued films that is ~100 hours of sleeping on an error that cannot clear.
_FATAL_MARKERS = (
    "credit balance is too low",
    "invalid x-api-key",
    "authentication_error",
    "permission_error",
    "your account has been disabled",
)


def call_haiku(client, system_prompt, user_message, retry=0):
    try:
        return client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.RateLimitError:
        waits = [10, 30, 90, 270]
        wait = waits[min(retry, len(waits) - 1)]
        print(f"\n  [429] Rate limited — sleeping {wait}s...", flush=True)
        time.sleep(wait)
        return call_haiku(client, system_prompt, user_message, retry + 1) if retry < 4 else None
    except anthropic.APIError as e:
        msg = str(e).lower()
        #Account-level failures affect every film equally, so retrying this one —
        #or moving on to the next — is pure waste. Stop the run and say why.
        if any(marker in msg for marker in _FATAL_MARKERS):
            raise FatalAPIError(str(e)) from e
        waits = [10, 30, 90, 270]
        wait = waits[min(retry, len(waits) - 1)]
        print(f"\n  [API ERROR] {e} — sleeping {wait}s...", flush=True)
        time.sleep(wait)
        return call_haiku(client, system_prompt, user_message, retry + 1) if retry < 4 else None


def strip_prefix(tag_str):
    """'pro:obsessive_perfectionist' → 'obsessive_perfectionist'"""
    return re.sub(r"^[a-z]+:", "", tag_str).strip()


def parse_response(text, tmdb_id, title):
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    brace_depth = 0
    json_end = None
    for i, ch in enumerate(text):
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                json_end = i + 1
                break
    if json_end is not None:
        text = text[:json_end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log_error(tmdb_id, title, f"JSON parse error: {e} | raw: {text[:200]}")
        return None

    hallucination_count = 0

    def extract_thm(key, bucket_name):
        nonlocal hallucination_count
        raw = data.get(key, [])
        if not isinstance(raw, list):
            raw = []
        valid_tags = []
        for t in raw:
            if not isinstance(t, str):
                continue
            cleaned = strip_prefix(t.strip())
            if cleaned in VALID_TAGS:
                valid_tags.append(cleaned)
            else:
                log_hallucination(tmdb_id, title, bucket_name, t)
                hallucination_count += 1
        return "|".join(valid_tags) if valid_tags else ""

    def extract_dom(key, valid_set, bucket_name):
        nonlocal hallucination_count
        raw = data.get(key, [])
        if not isinstance(raw, list):
            raw = []
        valid_tags = []
        for t in raw:
            if not isinstance(t, str):
                continue
            cleaned = t.strip()
            if cleaned in valid_set:
                valid_tags.append(cleaned)
            else:
                log_hallucination(tmdb_id, title, bucket_name, t)
                hallucination_count += 1
        return "|".join(valid_tags) if valid_tags else ""

    dom_val = extract_dom("domain", VALID_DOM_TAGS, "domain")
    sty_val = extract_dom("style",  VALID_STY_TAGS, "style")

    if not sty_val:
        log_error(tmdb_id, title, f"No valid style tags in response: {text[:120]}")

    result = {
        "helix_pro": extract_thm("protagonist", "protagonist"),
        "helix_dyn": extract_thm("dynamic",     "dynamic"),
        "helix_thm": extract_thm("theme",       "theme"),
        "helix_str": extract_thm("structure",   "structure"),
        "helix_ton": extract_thm("tone",        "tone"),
        "helix_spl": extract_thm("spoilers",    "spoilers"),
        "helix_dom": dom_val,
        "helix_sty": sty_val,
        "helix_low_confidence": 1 if (data.get("low_confidence") or hallucination_count > 2) else 0,
    }
    return result


class CostTracker:
    def __init__(self, max_cost):
        self.max_cost     = max_cost
        self.total_cost   = 0.0
        self.cache_hits   = 0
        self.full_calls   = 0
        self.recent_costs = []

    def record(self, usage):
        input_tokens  = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_write   = getattr(usage, "cache_creation_input_tokens", 0)
        if cache_write == 0:
            cache_creation = getattr(usage, "cache_creation", None)
            if cache_creation:
                cache_write = getattr(cache_creation, "ephemeral_5m_input_tokens", 0)
        cache_read    = getattr(usage, "cache_read_input_tokens", 0)
        uncached_input = max(0, input_tokens - cache_write - cache_read)
        cost = (
            uncached_input * PRICE_INPUT       / 1_000_000 +
            cache_write    * PRICE_CACHE_WRITE / 1_000_000 +
            cache_read     * PRICE_CACHE_READ  / 1_000_000 +
            output_tokens  * PRICE_OUTPUT      / 1_000_000
        )
        self.total_cost += cost
        self.recent_costs.append(cost)
        if len(self.recent_costs) > VELOCITY_WINDOW:
            self.recent_costs.pop(0)
        if cache_read > 0:
            self.cache_hits += 1
        else:
            self.full_calls += 1
        return cost

    def check_circuit_breaker(self):
        if self.total_cost >= self.max_cost:
            return True
        if len(self.recent_costs) >= 5:
            avg = sum(self.recent_costs) / len(self.recent_costs)
            if avg > VELOCITY_CEILING:
                print(f"\n[VELOCITY ALERT] avg/film=${avg:.5f} > ceiling ${VELOCITY_CEILING:.4f} | total=${self.total_cost:.4f}")
                return True
        return False

    def status_line(self, done, total, elapsed_s):
        rate = done / elapsed_s if elapsed_s > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        eta = str(timedelta(seconds=int(remaining)))
        pct_cached = self.cache_hits / max(1, self.cache_hits + self.full_calls) * 100
        avg = sum(self.recent_costs) / len(self.recent_costs) if self.recent_costs else 0
        return (
            f"  {done}/{total} ({done/total*100:.1f}%) | "
            f"cost=${self.total_cost:.4f} | "
            f"avg/film=${avg:.4f} | "
            f"cache={pct_cached:.0f}% | "
            f"ETA {eta}"
        )


def _plot_is_trustworthy(row, year):
    """Reason this row's wiki_plot must not be sent to the model, or None if fine.

    FAIL-SAFE. Wikipedia lookups historically resolved to the wrong article — the
    old fetcher gave "(YYYY film)" formatting a bigger bonus than having the RIGHT
    year, so Arrival (2016) stored the plot of "The Arrival (1991 film)" and
    X2: X-Men United stored "Identity (2003 film)". Tagging on a wrong plot bakes
    another film's DNA into 8 helix columns and costs real money to undo, so a
    suspect plot is dropped and the film is tagged from its overview alone.
    Dropping a good plot is cheap; tagging on a wrong one is not.
    """
    plot = (row["wiki_plot"] or "").strip()
    if not plot:
        return None
    try:
        wiki_title = (row["wiki_title"] or "").strip()
    except (IndexError, KeyError):
        wiki_title = ""
    if not wiki_title:
        return None  # unverifiable; overview still anchors the prompt

    def _words(s):
        stop = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for",
                "is", "it", "be", "or", "by", "as", "up", "do"}
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        return {w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
                if w and w not in stop}

    if not (_words(row["title"]) & _words(wiki_title)):
        return f"plot came from unrelated article {wiki_title!r}"
    m = re.search(r"\((\d{4})\b", wiki_title)
    if m and str(year).isdigit() and abs(int(m.group(1)) - int(year)) > 1:
        return f"plot came from {wiki_title!r} but film is {year}"
    return None


def build_user_message(row, year):
    plot     = (row["wiki_plot"] or "").strip()
    overview = (row["overview"]  or "").strip()

    bad = _plot_is_trustworthy(row, year)
    if bad:
        print(f"    [PLOT GUARD] {row['title']} ({year}): {bad} — tagging from overview only")
        plot = ""

    words = plot.split()
    if len(words) > 1500:
        plot = " ".join(words[:1500]) + " [truncated]"
    parts = [f"Title: {row['title']} ({year})", f"TMDB ID: {row['id']}"]
    if overview:
        parts.append(f"Overview: {overview}")
    if plot:
        parts.append(f"Plot Summary: {plot}")
    return "\n".join(parts)


def load_ids_file(path):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] ids-file not found: {path}")
        sys.exit(1)
    ids = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(int(line))
            except ValueError:
                pass
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",          type=int,   default=None,
                        help="Max films to process")
    parser.add_argument("--max-cost",       type=float, default=20.00,
                        help="Circuit breaker cost limit in USD")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print first 3 messages, no API calls")
    parser.add_argument("--ids-file",       type=str,   default=None,
                        help="File of TMDB IDs to retag (one per line); bypasses checkpoint/NULL filter")
    parser.add_argument("--output-csv",     type=str,   default=None,
                        help="Write tag results to this CSV file as films are processed")
    parser.add_argument("--sort-by-votes",  action="store_true",
                        help="Process highest vote_count first (normal queue only)")
    parser.add_argument("--progress-every", type=int,   default=PROGRESS_EVERY,
                        help="Print progress summary every N films")
    parser.add_argument("--include-documentaries", action="store_true",
                        help="Also tag Documentary films. Off by default: the Helix taxonomy is built for "
                             "narrative fiction, so documentaries return mostly-empty buckets and then "
                             "re-enter the queue forever, costing money each run without converging.")
    parser.add_argument("--min-votes",      type=int,   default=2000,
                        help="Min IMDb vote_count for normal-queue targeting (default: 2000)")
    parser.add_argument("--skip-suspect-wiki", action="store_true",
                        help="Skip films whose stored wiki_title is a non-film page "
                             "(list/album/franchise/etc. = wrong plot). Checked with "
                             "weekly_refresh._non_film_article_type.")
    parser.add_argument("--retry-exhausted", action="store_true",
                        help=f"Also retry films that already returned empty {MAX_TAG_ATTEMPTS}+ times "
                             "(normally given up on — same non-convergence problem as documentaries, "
                             "just for anthology/sketch films with no single throughline protagonist)")
    parser.add_argument("--fill-partial",   action="store_true",
                        help="Also target films with 1-5 of 8 helix fields filled (3+ empty). Fill-only: "
                             "existing tags are kept, only blank fields are filled. Off by default.")
    parser.add_argument("--partial-only",   action="store_true",
                        help="Target ONLY partially tagged films (3-7 empty fields), fill-only. "
                             "Skips untagged and enrichment films. Implies the --fill-partial rules.")
    parser.add_argument("--validated-since", type=str, default=None,
                        help="Only films whose validated_at >= this ISO date (e.g. 2026-11-09) — "
                             "limits a run to newly-valid films.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    system_prompt = load_system_prompt()
    print(f"[INIT] System prompt loaded: {len(system_prompt):,} chars")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    ids_mode = args.ids_file is not None
    checkpoint = set()

    if ids_mode:
        target_ids = load_ids_file(args.ids_file)
        print(f"[INIT] ids-file mode: {len(target_ids):,} IDs loaded from {args.ids_file}")
        placeholders = ",".join("?" * len(target_ids))
        rows = conn.execute(f"""
            SELECT id, title, release_date, vote_count, wiki_plot, wiki_title, overview
            FROM movies
            WHERE rowid IN (
                SELECT MIN(rowid) FROM movies
                WHERE id IN ({placeholders})
                  AND is_valid = 1
                  AND overview IS NOT NULL AND overview != ''
                GROUP BY id
            )
            ORDER BY vote_count DESC
        """, list(target_ids)).fetchall()
        print(f"[INIT] Films matched in DB: {len(rows):,}")
    else:
        #DB column state (not the checkpoint file) is now the source of truth for
        #targeting. A film only qualifies for a (re)tagging pass if 3 or more of
        #the 8 helix_* fields are empty — a film missing just 1-2 fields was very
        #likely tagged successfully with Haiku simply lacking enough signal for
        #that one field (or a human deliberately left it blank), not a failed
        #pass, so re-running it would risk overwriting reviewed/curated tags for
        #no benefit. save_checkpoint() below still writes to a checkpoint file as
        #an audit trail only; it is never used to filter what gets targeted.
        order = "vote_count DESC" if args.sort_by_votes else "vote_count DESC"
        empty_field_count = """(
            (CASE WHEN helix_dom IS NULL OR helix_dom = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_sty IS NULL OR helix_sty = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_pro IS NULL OR helix_pro = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_dyn IS NULL OR helix_dyn = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_thm IS NULL OR helix_thm = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_str IS NULL OR helix_str = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_ton IS NULL OR helix_ton = '' THEN 1 ELSE 0 END) +
            (CASE WHEN helix_spl IS NULL OR helix_spl = '' THEN 1 ELSE 0 END)
        )"""
        #Documentaries are excluded by default. The Helix taxonomy describes
        #narrative fiction — protagonist archetype, character dynamic, spoiler
        #resolution — and a documentary genuinely has none of those, so the model
        #correctly returns empty buckets. Those empties then satisfy
        #`empty_field_count >= 3`, putting the film straight back in the queue to be
        #re-tagged, come back sparse, and re-queue again: a loop that costs money
        #every run and never converges. Measured 2026-09-05: documentaries were
        #11.2x over-represented in the queue (25% of it vs 2% of the library), and
        #already-attempted ones sat at 0-4 of 8 dimensions.
        #Use --include-documentaries to override.
        doc_clause = "" if args.include_documentaries else \
            "AND (dna_genres IS NULL OR dna_genres NOT LIKE '%Documentary%')"
        #Attempt cap. Without this, a film the model correctly and repeatedly
        #returns mostly-empty for (anthologies/sketch films: Monty Python's the
        #Meaning of Life, Movie 43, Paris Je T'aime — no single throughline
        #protagonist or domain, same root cause as the documentary problem above)
        #satisfies empty_field_count>=3 forever and gets re-tagged and re-billed
        #on every run with no cap. Verified 2026-09-07: 347 films were genuinely
        #processed and correctly returned empty, all still eligible for re-queue
        #before this existed. Use --retry-exhausted to override.
        attempts_clause = "" if args.retry_exhausted else \
            f"AND COALESCE(helix_attempts, 0) < {MAX_TAG_ATTEMPTS}"
        #Queue = (a) fresh: ALL 8 fields empty, within the attempt cap; OR
        #        (b) enrich: previously tagged on a thin plot/keyword set that has since
        #            been filled in, and the film is recent or high-vote (never an
        #            obscure old film); OR
        #        (c) --fill-partial only: 3-7 fields empty (fill-only, never overwrites).
        attempts_ok = "1=1" if args.retry_exhausted else \
            f"COALESCE(helix_attempts, 0) < {MAX_TAG_ATTEMPTS}"
        #grew_cond: the film's inputs were thin at the last attempt and have since been
        #filled in. The snapshot is recorded on EVERY attempt (success, empty, or
        #failed), so a film that came back empty is not re-billed until this is true.
        grew_cond = f"""(
                (COALESCE(helix_wiki_chars, 0) < {STUB_PLOT_CHARS}
                 AND LENGTH(COALESCE(wiki_plot, '')) >= COALESCE(helix_wiki_chars, 0) + {ENRICH_MIN_PLOT_GROWTH})
                OR
                (COALESCE(helix_kw_count, 0) < {STUB_KEYWORDS}
                 AND {KW_COUNT_SQL} >= COALESCE(helix_kw_count, 0) + {ENRICH_MIN_KW_GROWTH})
            )"""
        #Already-tagged films are only re-tagged if recent or high-vote (never obscure+old).
        enrich_cond = f"""(helix_tagged_at IS NOT NULL
            AND {grew_cond}
            AND (release_date >= date('now', '-{ENRICH_WINDOW_DAYS} days')
                 OR CAST(vote_count AS REAL) >= {ENRICH_HIGH_VALUE_VOTES}))"""
        #Never-successfully-tagged films: first try always; a retry only after the inputs
        #grew (no age gate — --min-votes already keeps obscure films out).
        fresh_ok = "1=1" if args.retry_exhausted else \
            f"(COALESCE(helix_attempts, 0) = 0 OR (helix_wiki_chars IS NOT NULL AND {grew_cond}))"
        partial_cond = f"({empty_field_count} BETWEEN 3 AND 7 AND {attempts_ok})"
        partial_clause = f"OR {partial_cond}" if args.fill_partial else ""
        #--partial-only: ONLY the partially-tagged films (fill-only), no fresh/enrich.
        queue_where = partial_cond if args.partial_only else f"""(
                    ({empty_field_count} = 8 AND {fresh_ok})
                    OR {enrich_cond}
                    {partial_clause}
              )"""
        since_clause = "AND validated_at >= ?" if args.validated_since else ""
        params = [args.min_votes] + ([args.validated_since] if args.validated_since else [])
        rows = conn.execute(f"""
            SELECT id, title, release_date, vote_count, wiki_plot, wiki_title, overview,
                   CASE WHEN {empty_field_count} = 8 THEN 'fresh'
                        WHEN {enrich_cond} THEN 'enrich'
                        ELSE 'partial' END AS queue_mode
            FROM movies
            WHERE is_valid = 1
              AND overview IS NOT NULL AND overview != ''
              AND {queue_where}
              {doc_clause}
              AND CAST(vote_count AS REAL) >= ?
              {since_clause}
            ORDER BY {order}
        """, params).fetchall()

    if args.skip_suspect_wiki and not ids_mode:
        from weekly_refresh import _non_film_article_type
        before = len(rows)
        rows = [r for r in rows if not _non_film_article_type(r["wiki_title"])]
        print(f"[INIT] --skip-suspect-wiki: skipped {before - len(rows):,} films with a non-film wiki page")

    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"[INIT] Films to tag: {total:,}")
    if not ids_mode:
        print("[INIT] By queue mode: " + ", ".join(
            f"{m}={n:,}" for m, n in Counter(r['queue_mode'] for r in rows).most_common()))
    print(f"[INIT] Max cost circuit breaker: ${args.max_cost:.2f}")
    print(f"[INIT] Estimated cost (with caching): ~${total * EXPECTED_COST_PER_FILM:.2f}")
    print()

    if args.dry_run:
        for row in rows[:3]:
            year = (row["release_date"] or "")[:4]
            msg  = build_user_message(row, year)
            print(f"── {row['title']} ({year}) ──")
            print(msg[:1000])
            print()
        print(f"[dry-run] {total:,} films in queue. No API calls made.")
        conn.close()
        return

    client = anthropic.Anthropic(api_key=api_key)
    tracker = CostTracker(max_cost=args.max_cost)
    start_time = time.time()
    errors = 0
    dom_counter = Counter()
    sty_counter = Counter()

    Path("data").mkdir(exist_ok=True)

    CSV_COLUMNS = [
        "tmdb_id", "title", "release_date", "vote_count",
        "helix_dom", "helix_sty",
        "helix_pro", "helix_dyn", "helix_thm",
        "helix_str", "helix_ton", "helix_spl",
    ]
    csv_file = None
    csv_writer = None
    if args.output_csv:
        csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()
        print(f"[INIT] CSV output: {args.output_csv}")

    for i, row in enumerate(rows):
        tmdb_id = row["id"]
        title   = row["title"]
        year    = (row["release_date"] or "")[:4]

        user_message = build_user_message(row, year)
        try:
            response = call_haiku(client, system_prompt, user_message)
        except FatalAPIError as e:
            #`i` is the loop index, so it is always bound here; the run's own
            #`processed` counter is computed further down and would NameError.
            print(f"\n{'='*64}", flush=True)
            print(f"  ABORTING — unrecoverable API error:\n    {e}", flush=True)
            print(f"  {i:,} of {len(rows):,} films attempted this run; all tags committed.", flush=True)
            print(f"  Fix the account issue and re-run; the queue resumes where it stopped.", flush=True)
            print(f"{'='*64}\n", flush=True)
            log_error(tmdb_id, title, f"FATAL: {e}")
            break

        if response is None:
            errors += 1
            log_error(tmdb_id, title, "API returned None after retries")
            if not ids_mode:
                save_checkpoint(tmdb_id, checkpoint)
            time.sleep(SLEEP_BETWEEN)
            continue

        if response.usage.output_tokens > 1500:
            log_error(tmdb_id, title, f"Anomalous output: {response.usage.output_tokens} tokens. Skipping.")
            conn.execute("UPDATE movies SET helix_attempts=COALESCE(helix_attempts,0)+1 WHERE id=?", (tmdb_id,))
            conn.commit()
            if not ids_mode:
                save_checkpoint(tmdb_id, checkpoint)
            time.sleep(SLEEP_BETWEEN)
            continue

        cost = tracker.record(response.usage)
        raw_text = response.content[0].text if response.content else ""
        parsed = parse_response(raw_text, tmdb_id, title)

        if parsed is None:
            errors += 1
            #Record the input snapshot even though nothing was tagged, so this film is
            #not retried until its plot/keywords actually grow.
            conn.execute(f"""UPDATE movies SET helix_attempts=COALESCE(helix_attempts,0)+1,
                helix_wiki_chars=LENGTH(COALESCE(wiki_plot,'')), helix_kw_count={KW_COUNT_SQL}
                WHERE id=?""", (tmdb_id,))
            conn.commit()
            if not ids_mode:
                save_checkpoint(tmdb_id, checkpoint)
            time.sleep(SLEEP_BETWEEN)
            continue

        #NEVER blank out an existing tag, and never silently replace one unless this is
        #an explicit enrichment/ids-file retag. Haiku is not deterministic: re-running
        #a tagged film can return different tags, or empty buckets, and on 2026-11-09
        #that wiped 30 previously-filled fields. A questionable tag beats no tag.
        mode = row["queue_mode"] if "queue_mode" in row.keys() else ("retag" if ids_mode else "fresh")
        overwrite = mode in ("enrich", "retag")
        cur_vals = conn.execute(
            f"SELECT {', '.join(HELIX_FIELDS)} FROM movies WHERE id=?", (tmdb_id,)).fetchone()
        merged = {}
        audit_rows = []
        for f in HELIX_FIELDS:
            old = ((cur_vals[f] if cur_vals else None) or "")
            new = (parsed[f] or "")
            if not new:
                val = old                          #empty result: keep what we had
            elif old and not overwrite:
                val = old                          #fill-only: keep existing tag
            else:
                val = new
            if old and new and old != new and overwrite:
                audit_rows.append([datetime.now().isoformat(timespec="seconds"), tmdb_id,
                                   title, mode, f, old, new])
            merged[f] = val if val else parsed[f]
        if audit_rows:
            new_file = not Path(AUDIT_CSV_PATH).exists()
            with open(AUDIT_CSV_PATH, "a", newline="", encoding="utf-8") as af:
                aw = csv.writer(af)
                if new_file:
                    aw.writerow(["ts", "tmdb_id", "title", "mode", "field", "old", "new"])
                aw.writerows(audit_rows)
        any_tag = any(merged[f] for f in HELIX_FIELDS)
        lowconf_sql = "helix_low_confidence=?," if mode != "partial" else ""
        lowconf_params = [parsed["helix_low_confidence"]] if mode != "partial" else []
        conn.execute(f"""
            UPDATE movies SET
                helix_pro=?, helix_dyn=?, helix_thm=?,
                helix_str=?, helix_ton=?, helix_spl=?,
                helix_dom=?, helix_sty=?,
                {lowconf_sql}
                helix_wiki_chars=LENGTH(COALESCE(wiki_plot, '')),
                helix_kw_count={KW_COUNT_SQL},
                helix_tagged_at=CASE WHEN ? THEN ? ELSE helix_tagged_at END,
                helix_attempts=COALESCE(helix_attempts, 0) + 1
            WHERE id=?
        """, [merged[f] for f in HELIX_FIELDS] + lowconf_params +
             [1 if any_tag else 0, datetime.now().isoformat(timespec="seconds"), tmdb_id])
        conn.commit()
        for f in HELIX_FIELDS:
            parsed[f] = merged[f]                 #so the printout/CSV show what was stored

        if not ids_mode:
            save_checkpoint(tmdb_id, checkpoint)

        dom_tags = [t for t in (parsed["helix_dom"] or "").split("|") if t]
        sty_tags = [t for t in (parsed["helix_sty"] or "").split("|") if t]
        dom_counter.update(dom_tags)
        sty_counter.update(sty_tags)

        if csv_writer:
            csv_writer.writerow({
                "tmdb_id":      tmdb_id,
                "title":        title,
                "release_date": row["release_date"] or "",
                "vote_count":   row["vote_count"] if "vote_count" in row.keys() else "",
                "helix_dom":    parsed["helix_dom"],
                "helix_sty":    parsed["helix_sty"],
                "helix_pro":    parsed["helix_pro"],
                "helix_dyn":    parsed["helix_dyn"],
                "helix_thm":    parsed["helix_thm"],
                "helix_str":    parsed["helix_str"],
                "helix_ton":    parsed["helix_ton"],
                "helix_spl":    parsed["helix_spl"],
            })
            csv_file.flush()

        print(
            f"  [{i+1:5d}] {title[:35]:<35} ({year})\n"
            f"           dom={parsed['helix_dom'] or '—'}\n"
            f"           sty={parsed['helix_sty'] or '—'}\n"
            f"           pro={parsed['helix_pro'] or '—'}\n"
            f"           dyn={parsed['helix_dyn'] or '—'}\n"
            f"           thm={parsed['helix_thm'] or '—'}\n"
            f"           str={parsed['helix_str'] or '—'}\n"
            f"           ton={parsed['helix_ton'] or '—'}\n"
            f"           spl={parsed['helix_spl'] or '—'}\n"
            f"           cost=${cost:.5f}",
            flush=True,
        )

        if tracker.check_circuit_breaker():
            avg = sum(tracker.recent_costs[-10:]) / max(1, len(tracker.recent_costs[-10:]))
            if tracker.total_cost >= args.max_cost:
                print(f"\n[CIRCUIT BREAKER] ${tracker.total_cost:.4f} reached limit ${args.max_cost:.2f}. Stopping.")
            else:
                print(f"\n[VELOCITY ALERT] avg/film=${avg:.5f}. Continue? [y/N] ", end="")
                if input().strip().lower() != "y":
                    break
                tracker.recent_costs.clear()

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - start_time
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {tracker.status_line(i+1, total, elapsed)} | errors={errors}", flush=True)

        time.sleep(SLEEP_BETWEEN)

    if csv_file:
        csv_file.close()
        print(f"[DONE] CSV written to {args.output_csv}")

    elapsed = time.time() - start_time
    processed = min(i + 1, total) if total > 0 else 0
    print(f"\n[DONE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Processed: {processed:,} / {total:,}")
    print(f"  Errors:    {errors:,}")
    print(f"  Total cost: ${tracker.total_cost:.4f}")
    print(f"  Cache hits: {tracker.cache_hits:,} / {tracker.cache_hits + tracker.full_calls:,}")
    print(f"  Elapsed:   {str(timedelta(seconds=int(elapsed)))}")
    if dom_counter:
        print(f"\nDomain tag frequency:")
        for tag, count in dom_counter.most_common(12):
            print(f"  {tag:<35s} {count:4d}")
    if sty_counter:
        print(f"\nStyle tag frequency:")
        for tag, count in sty_counter.most_common():
            print(f"  {tag:<35s} {count:4d}")
    conn.close()


if __name__ == "__main__":
    main()
