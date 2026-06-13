"""
helix_tagger.py
---------------
Tags films in movies.db with abstract thematic DNA using Claude Haiku.
Writes results to 7 columns: helix_pro, helix_dyn, helix_thm, helix_str,
helix_ton, helix_spl.

Setup:
  export ANTHROPIC_API_KEY
  python helix_tagger.py

Options:
  --limit N        stop after N films (default: all)
  --max-cost $X    halt run if total cost exceeds $X (default: 20.00)
  --dry-run        print first 3 user messages without calling the API
"""

import sqlite3
import json
import time
import sys
import os
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import anthropic

DB_PATH         = "movies.db"
CHECKPOINT_PATH = "data/helix_processed_tmdb_ids.txt"
ERROR_LOG_PATH  = "data/helix_errors.log"
PROMPT_PATH     = "data/helix_system_prompt.txt"

MODEL           = "claude-haiku-4-5-20251001"
MAX_TOKENS      = 400
SLEEP_BETWEEN   = 0.5
PROGRESS_EVERY  = 50
CIRCUIT_TRIP_X  = 8.0          #cost circuit breaker

PRICE_INPUT          = 0.80
PRICE_OUTPUT         = 4.00
PRICE_CACHE_WRITE    = 0.80
PRICE_CACHE_READ     = 0.08

EXPECTED_COST_PER_FILM = 0.0032
VELOCITY_WINDOW        = 50
VELOCITY_CEILING       = 0.005

VALID_BUCKETS = {"protagonist", "dynamic", "theme", "structure", "tone", "spoilers"}

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

def load_system_prompt():
    p = Path(PROMPT_PATH)
    if not p.exists():
        print(f"\n[ERROR] System prompt not found at {PROMPT_PATH}")
        print("  Save your system prompt text to that file, then re-run.")
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
        ("helix_low_confidence", "INTEGER"),
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
                f"id={tmdb_id} title={title!r} bucket={bucket} "
                f"invalid_tag={tag!r}\n")

def call_haiku(client, system_prompt, user_message, retry=0):
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )
        return response
    except anthropic.RateLimitError:
        waits = [10, 30, 90, 270]
        wait = waits[min(retry, len(waits) - 1)]
        print(f"\n  [429] Rate limited — sleeping {wait}s...", flush=True)
        time.sleep(wait)
        if retry < 4:
            return call_haiku(client, system_prompt, user_message, retry + 1)
        return None
    except anthropic.APIError as e:
        waits = [10, 30, 90, 270]
        wait = waits[min(retry, len(waits) - 1)]
        print(f"\n  [API ERROR] {e} — sleeping {wait}s...", flush=True)
        time.sleep(wait)
        if retry < 4:
            return call_haiku(client, system_prompt, user_message, retry + 1)
        return None

def strip_prefix(tag_str):
    """'pro:obsessive_perfectionist' → 'obsessive_perfectionist'"""
    return re.sub(r"^[a-z]+:", "", tag_str).strip()

def parse_response(text, tmdb_id, title):
    """
    Parse Haiku's JSON response. Returns dict of column values or None on failure.
    Strips bucket prefixes from tag strings before storing.
    """
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

    def extract(key, bucket_name):
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

    result = {
        "helix_pro": extract("protagonist", "protagonist"),
        "helix_dyn": extract("dynamic",     "dynamic"),
        "helix_thm": extract("theme",       "theme"),
        "helix_str": extract("structure",   "structure"),
        "helix_ton": extract("tone",        "tone"),
        "helix_spl": extract("spoilers",    "spoilers"),
        "helix_low_confidence": 1 if (data.get("low_confidence") or hallucination_count > 2) else 0,
    }
    return result

#cost tracking
class CostTracker:
    def __init__(self, max_cost):
        self.max_cost        = max_cost
        self.total_cost      = 0.0
        self.cache_hits      = 0
        self.full_calls      = 0
        self.recent_costs    = []

    def record(self, usage):
        """usage: anthropic Usage object"""
        input_tokens  = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_write   = getattr(usage, "cache_creation_input_tokens", 0)
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
        """Returns True (halt) if cost ceiling hit or velocity anomaly detected."""
        if self.total_cost >= self.max_cost:
            return True
        if len(self.recent_costs) < 5:
            return False
        avg_recent = sum(self.recent_costs) / len(self.recent_costs)
        if avg_recent > VELOCITY_CEILING:
            print(f"\n[VELOCITY ALERT] Rolling avg cost/film=${avg_recent:.5f} "
                  f"exceeds ceiling ${VELOCITY_CEILING:.4f} | total=${self.total_cost:.4f}")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",    type=int,   default=None,  help="Max films to process")
    parser.add_argument("--max-cost", type=float, default=20.00, help="Circuit breaker cost limit in USD")
    parser.add_argument("--dry-run",  action="store_true",       help="Print first 3 messages, no API calls")
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

    checkpoint = load_checkpoint()
    print(f"[INIT] Checkpoint: {len(checkpoint):,} films already processed")

    #tag films with only valid wiki plots
    rows = conn.execute("""
        SELECT id, title, release_date, wiki_plot, overview
        FROM movies
        WHERE is_valid = 1
          AND wiki_plot IS NOT NULL AND LENGTH(wiki_plot) >= 1000
          AND helix_pro IS NULL
        ORDER BY vote_count DESC
    """).fetchall()

    rows = [r for r in rows if str(r["id"]) not in checkpoint]
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"[INIT] Films to tag: {total:,}")
    print(f"[INIT] Max cost circuit breaker: ${args.max_cost:.2f}")
    print(f"[INIT] Estimated cost (with caching): ~${total * EXPECTED_COST_PER_FILM:.2f}")
    print()

    if args.dry_run:
        for row in rows[:3]:
            year = (row["release_date"] or "")[:4]
            msg = build_user_message(row, year)
            print(f"── {row['title']} ({year}) ──")
            print(msg[:800])
            print()
        return

    client = anthropic.Anthropic(api_key=api_key)
    tracker = CostTracker(max_cost=args.max_cost)
    start_time = time.time()
    errors = 0

    Path("data").mkdir(exist_ok=True)

    for i, row in enumerate(rows):
        tmdb_id = row["id"]
        title   = row["title"]
        year    = (row["release_date"] or "")[:4]

        user_message = build_user_message(row, year)
        response = call_haiku(client, system_prompt, user_message)

        if response is None:
            errors += 1
            log_error(tmdb_id, title, "API returned None after retries")
            save_checkpoint(tmdb_id, checkpoint)
            time.sleep(SLEEP_BETWEEN)
            continue

        if response.usage.output_tokens > 1000:
            log_error(tmdb_id, title, f"Anomalous output: {response.usage.output_tokens} tokens. Skipping.")
            save_checkpoint(tmdb_id, checkpoint)
            time.sleep(SLEEP_BETWEEN)
            continue

        tracker.record(response.usage)
        raw_text = response.content[0].text if response.content else ""
        parsed = parse_response(raw_text, tmdb_id, title)

        if parsed is None:
            errors += 1
            save_checkpoint(tmdb_id, checkpoint)
            time.sleep(SLEEP_BETWEEN)
            continue

        conn.execute("""
            UPDATE movies SET
                helix_pro=?, helix_dyn=?, helix_thm=?,
                helix_str=?, helix_ton=?, helix_spl=?,
                helix_low_confidence=?
            WHERE id=?
        """, (
            parsed["helix_pro"], parsed["helix_dyn"], parsed["helix_thm"],
            parsed["helix_str"], parsed["helix_ton"], parsed["helix_spl"],
            parsed["helix_low_confidence"], tmdb_id,
        ))
        conn.commit()
        save_checkpoint(tmdb_id, checkpoint)

        #cost circuit breaker
        if tracker.check_circuit_breaker():
            avg = sum(tracker.recent_costs[-10:]) / len(tracker.recent_costs[-10:])
            print(f"\n[CIRCUIT BREAKER] avg cost/film=${avg:.5f}, total=${tracker.total_cost:.4f}")
            if tracker.total_cost >= args.max_cost:
                print(f"  Total cost ${tracker.total_cost:.4f} reached limit ${args.max_cost:.2f}. Stopping.")
            else:
                print(f"  Per-film cost is {avg/EXPECTED_COST_PER_FILM:.1f}x expected. Something may be wrong.")
                ans = input("  Continue? [y/N] ").strip().lower()
                if ans != "y":
                    print("  Halting.")
                    break
                tracker.recent_costs.clear()  #reset

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {tracker.status_line(i+1, total, elapsed)} | errors={errors}", flush=True)

        time.sleep(SLEEP_BETWEEN)

    #final summary
    elapsed = time.time() - start_time
    processed = min(i+1, total) if total > 0 else 0
    print(f"\n[DONE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Processed: {processed:,} / {total:,}")
    print(f"  Errors:    {errors:,}")
    print(f"  Total cost: ${tracker.total_cost:.4f}")
    print(f"  Cache hits: {tracker.cache_hits:,} / {tracker.cache_hits + tracker.full_calls:,}")
    print(f"  Elapsed:   {str(timedelta(seconds=int(elapsed)))}")
    conn.close()


def build_user_message(row, year):
    plot = (row["wiki_plot"] or "").strip()
    overview = (row["overview"] or "").strip()
    #truncate wiki plot to 2K words
    words = plot.split()
    if len(words) > 2000:
        plot = " ".join(words[:2000]) + " [truncated]"
    return (
        f"Title: {row['title']} ({year})\n"
        f"TMDB ID: {row['id']}\n"
        f"Overview: {overview}\n\n"
        f"Plot Summary: {plot}"
    )


if __name__ == "__main__":
    main()