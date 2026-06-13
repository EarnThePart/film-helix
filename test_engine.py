"""
test_engine.py      Headless recommendation quality tester
--------------------------------------------------------
Loads the FilmHelixEngine directly (no Streamlit), runs a set of source
films, prints top N matches with scores and match reasons.

Usage:
    python test_engine.py                              # run all control films
    python test_engine.py --film "Inception (2010)"    # single film
    python test_engine.py --top 20                     # show top 20 per film
    python test_engine.py --priority plot              # override priority
    python test_engine.py --debug                      # show per-channel scores
    python test_engine.py --obscure                    # include < 50K vote films
"""

import argparse
from recommender import FilmHelixEngine

# #TOP CONTROL FILMS ONLY
# CONTROL_FILMS = [
#     ("Whiplash (2014)",                        "Tár, Black Swan — obsession/music, NOT generic musician films"),
#     ("Nightcrawler (2014)",                    "Shattered Glass, Zodiac — journalism ethics/neo-noir"),
#     ("The Ghost Writer (2010)",                "Political thriller, Polanski — conspiracy/isolation"),
#     ("Parasite (2019)",                        "South Korea class thriller — both Korean AND narrative match"),
#     ("F1 (2025)",                              "Formula 1 action — Kosinski, Apple, Brad Pitt"),
#     ("Casablanca (1943)",                      "Wartime romance — NOT generic WWII, NOT generic romance"),
#     ("The Ninth Gate (1999)",                  "Occult mystery — Polanski, rare books, devil"),
#     ("It Follows (2015)",                      "Horror — STD curse metaphor, relentless pursuit"),
#     ("Green Room (2016)",                      "Siege thriller — punk band, neo-Nazis, survival"),
#     ("Se7en (1995)",                           "Serial killer procedural — seven sins, bleak, Fincher"),
#     ("Enemy (2014)",                           "Psychological thriller — doppelganger, Villeneuve, spider"),
# ]

CONTROL_FILMS = [
    # ("Whiplash (2014)",                        "Tár, Black Swan — obsession/music, NOT generic musician films"),
    # ("The Ring (2002)",                        "Ringu, Ju-On — curse/videotape supernatural"),
    # ("Nightcrawler (2014)",                    "Shattered Glass, Zodiac — journalism ethics/neo-noir"),
    # ("La La Land (2016)",                      "Musical/romance — NOT generic romance (Notting Hill, Pretty Woman)"),
    # ("Oldboy (2003)",                          "Park Chan-wook body of work, revenge thriller"),
    # ("The Social Network (2010)",              "Ambition/betrayal — NOT hacking films (Hackers, The Net)"),
    # ("Mad Max: Fury Road (2015)",              "Post-apoc action — no unrelated bleed"),
    # ("Parasite (2019)",                        "South Korea class thriller — both Korean AND narrative match"),
    # ("Hereditary (2018)",                      "Horror family trauma — NOT generic supernatural (Thirteen Ghosts)"),
    # ("Pulp Fiction (1994)",                    "Reservoir Dogs, Jackie Brown — nonlinear crime"),
    # ("Infernal Affairs (2002)",                "The Departed #1 — undercover infiltration"),
    # ("Inception (2010)",                       "Matrix/Dark City — no kidnapping or generic heist films"),
    # ("Send Help (2026)",                       "Triangle of Sadness — survival island, comedy-horror tone"),
    # ("Goodfellas (1990)",                      "Casino, The Irishman — organized crime, Scorsese"),
    # ("The Girl with the Dragon Tattoo (2011)", "Investigation/hacker — Fincher, serial killer mystery"),
    # ("The Ghost Writer (2010)",                "Political thriller, Polanski — conspiracy/isolation"),
    # ("Jurassic Park (1993)",                   "Sci-fi adventure — dinosaurs, survival, not hacking films"),
    # ("There Will Be Blood (2007)",             "Ambition/obsession, period — character study"),
    # ("Enemy of the State (1998)",              "Surveillance thriller — government conspiracy, chase"),
    # ("The Player (1992)",                      "Hollywood satire, Altman — cynical insider drama"),
    # ("Crimson Tide (1995)",                    "Submarine/military thriller — claustrophobic command conflict"),
    # ("Midsommar (2019)",                       "Folk horror — cult, daylight dread, grief. NOT generic horror"),
    # ("Moneyball (2011)",                       "Sports analytics/outsider strategy — NOT generic baseball"),
    # # New Wave classics
    # ("American Graffiti (1973)",               "Coming-of-age/nostalgia — NOT generic teen comedy"),
    # ("Blade Runner (1982)",                    "Sci-fi noir — existential AI, NOT generic action sci-fi"),
    # ("The Godfather Part II (1974)",           "Organized crime epic — Corleone saga, parallel timelines"),
    # ("Alien (1979)",                           "Sci-fi horror — creature, claustrophobic survival"),
    # ("2001: A Space Odyssey (1968)",           "Philosophical sci-fi — HAL, transcendence, NOT generic space"),
    # ("Chinatown (1974)",                       "Neo-noir mystery — corruption, Polanski"),
    # ("Taxi Driver (1976)",                     "Alienation/vigilante — Scorsese character study"),
    # ("Network (1976)",                         "Media satire — corporate cynicism, prophetic"),
    # ("Apocalypse Now (1979)",                  "War/psychological descent — Heart of Darkness adaptation"),
    # ("Full Metal Jacket (1987)",               "Vietnam war — dehumanization, two-act structure"),
    # # Rom Coms
    # ("Crazy, Stupid, Love. (2011)",            "Modern romcom — ensemble, NOT generic love story"),
    # ("Hitch (2005)",                           "Romantic comedy — dating coach, genre-typical"),
    # ("10 Things I Hate About You (1999)",      "Shakespeare adaptation romcom — teen, Taming of the Shrew"),
    # # Classic rom coms
    # ("Roman Holiday (1953)",                   "Audrey Hepburn romance — European escapism, bittersweet"),
    # ("Sabrina (1954)",                         "Wilder romantic comedy — class/transformation"),
    # ("How to Marry a Millionaire (1953)",      "Screwball comedy — gold diggers, ensemble"),
    # # Class noir
    # ("Double Indemnity (1944)",                "Femme fatale noir — insurance fraud, Wilder"),
    # ("Laura (1944)",                           "Murder mystery noir — obsession, identity"),
    # ("The Maltese Falcon (1941)",              "Hard-boiled detective noir — Hammett, Bogart"),
    # ("Out of the Past (1947)",                 "Fatalistic noir — past catches up, double-cross"),
    # # Disaster films
    # ("The Towering Inferno (1974)",            "Disaster ensemble — skyscraper fire, ensemble survival"),
    # ("Earthquake (1974)",                      "Disaster spectacle — ensemble, practical effects era"),
    # ("The Poseidon Adventure (1972)",          "Disaster survival — capsized ship, ensemble"),
    # # 80s/90s action
    # ("Die Hard (1988)",                        "Confined action — single building, genre-defining"),
    # ("Lethal Weapon (1987)",                   "Buddy cop action — Riggs/Murtaugh dynamic"),
    # ("Total Recall (1990)",                    "Sci-fi action — identity/reality, Verhoeven"),
    # ("Speed (1994)",                           "Contained action thriller — bus bomb, ticking clock"),
    # ("The Rock (1996)",                        "Action blockbuster — Alcatraz, Cage/Connery"),
    # # Misc
    # ("Project Hail Mary (2026)",               "Sci-fi survival/problem-solving — isolation, first contact"),
    # ("Ex Machina (2015)",                      "AI thriller — contained, philosophical, Garland"),
    # ("The Lighthouse (2019)",                  "Psychological horror — isolation, two-hander, Eggers"),
    # ("The Exorcist (1973)",                    "Supernatural horror — possession, faith vs evil"),
    # ("Jaws (1975)",                            "Creature thriller — Spielberg, small-town siege"),
    # ("Fatal Attraction (1987)",               "Erotic thriller — obsession, domestic threat"),
    # ("Basic Instinct (1992)",                  "Erotic noir thriller — femme fatale, interrogation"),
    # ("American Psycho (2000)",                 "Satirical horror — unreliable narrator, Bateman"),
    # ("Starship Troopers (1997)",               "Sci-fi satire — fascism parody, Verhoeven"),
    # ("The Usual Suspects (1995)",              "Crime thriller — twist ending, unreliable narrator"),
    # ("Sweet Smell of Success (1957)",          "Cynical media/power noir — Lehman/Mackendrick"),
    # ("Casablanca (1943)",                      "Wartime romance — NOT generic WWII, NOT generic romance"),
    # ("The Asphalt Jungle (1950)",              "Heist noir — ensemble caper, Huston"),
    # ("Double Indemnity (1944)",                "Femme fatale noir — insurance fraud, Wilder"),
    # ("Laura (1944)",                           "Murder mystery noir — obsession, identity"),
    # ("The Maltese Falcon (1941)",              "Hard-boiled detective noir — Hammett, Bogart"),
    # ("Out of the Past (1947)",                 "Fatalistic noir — past catches up, double-cross"),
    # ("Touch of Evil (1958)",                   "Border noir — Welles, corruption, long take"),
    # ("Se7en (1995)",                           "Serial killer procedural — Fincher, bleak, seven sins"),
    # ("Adaptation. (2002)",                     "Meta-comedy — screenwriting, Kaufman, self-referential"),
    # ("Annie Hall (1977)",                      "Neurotic romantic comedy — Woody Allen, non-linear"),
    # ("Sideways (2004)",                        "Road trip comedy-drama — wine, male friendship, midlife"),
    # ("The Big Lebowski (1998)",                "Stoner noir comedy — Coens, NOT generic crime"),
    # ("The Grand Budapest Hotel (2014)",        "Wes Anderson — whimsy, caper, ensemble"),
    # ("Wedding Crashers (2005)",                "Raunch romcom — Vince Vaughn/Owen Wilson, crowd-pleaser"),
    # ("Crazy, Stupid, Love. (2011)",            "Modern romcom — ensemble, layered, NOT generic love story"),
    # ("Hitch (2005)",                           "Romantic comedy — dating coach, genre-typical"),
    # ("10 Things I Hate About You (1999)",      "Shakespeare adaptation romcom — teen, Taming of the Shrew"),
    # ("Home Alone (1990)",                      "Family comedy — booby traps, holiday, slapstick"),
    # ("Ghostbusters (1984)",                    "Comedy sci-fi — supernatural, ensemble, NYC"),
    # ("Back to the Future (1985)",              "Sci-fi comedy — time travel, NOT dark sci-fi"),
    # ("Michael Clayton (2007)",                 "Legal thriller — corporate conspiracy, Clooney"),
    # ("A Few Good Men (1992)",                  "Courtroom drama — military honor, Sorkin"),
    # ("Network (1976)",                         "Media satire — corporate cynicism, prophetic"),
    # ("Taxi Driver (1976)",                     "Alienation/vigilante — Scorsese character study"),
    # ("Hannah and Her Sisters (1986)",          "Ensemble family drama — Woody Allen, warmth"),
    # ("The Descendants (2011)",                 "Family drama — grief, Hawaii, Clooney"),
    # ("Force Majeure (2014)",                   "Family drama — cowardice, masculinity, Swedish"),
    # ("Babylon (2022)",                         "Hollywood excess — Chazelle, ensemble, maximalist"),
    # ("Requiem for a Dream (2000)",             "Addiction tragedy — Aronofsky, visceral, bleak"),
    # ("Joker (2019)",                           "Character study — incel rage, Scorsese homage, origin"),
    # ("Uncut Gems (2019)",                      "Anxiety thriller — gambling, compulsion, Sandler"),
    # ("American Psycho (2000)",                 "Satirical horror — unreliable narrator, Bateman"),
    # ("Fight Club (1999)",                      "Anti-consumerist thriller — twist, Fincher, Palahniuk"),
    # # Sci Fi
    # ("Blade Runner (1982)",                    "Sci-fi noir — existential AI, NOT generic action sci-fi"),
    # ("2001: A Space Odyssey (1968)",           "Philosophical sci-fi — HAL, transcendence, NOT generic space"),
    # ("Alien (1979)",                           "Sci-fi horror — creature, claustrophobic survival"),
    # ("Star Wars (1977)",                       "Space opera adventure — hero's journey, NOT dark sci-fi"),
    # ("Terminator 2: Judgment Day (1991)",      "Sci-fi action — time travel, Sarah Connor, NOT generic action"),
    # ("Total Recall (1990)",                    "Sci-fi action — identity/reality, Verhoeven"),
    # ("District 9 (2009)",                      "Sci-fi allegory — apartheid, found-footage hybrid"),
    # ("Cloverfield (2008)",                     "Found-footage monster — NYC, contained disaster"),
    # ("Ex Machina (2015)",                      "AI thriller — contained, philosophical, Garland"),
    # ("Starship Troopers (1997)",               "Sci-fi satire — fascism parody, Verhoeven"),
    # # Action/Thriller
    # ("Die Hard (1988)",                        "Confined action — single building, genre-defining"),
    # ("Lethal Weapon (1987)",                   "Buddy cop action — Riggs/Murtaugh dynamic"),
    # ("Speed (1994)",                           "Contained action thriller — bus bomb, ticking clock"),
    # ("The Rock (1996)",                        "Action blockbuster — Alcatraz, Cage/Connery"),
    # ("Iron Man (2008)",                        "Superhero origin — MCU, Favreau, self-aware"),
    # ("The Dark Knight (2008)",                 "Superhero noir — Joker, Nolan, moral philosophy"),
    # ("Gladiator (2000)",                       "Historical epic — revenge, arena, Scott"),
    # ("Black Hawk Down (2001)",                 "Military action — Somalia, ensemble, Ridley Scott"),
    # ("Sicario (2015)",                         "Border drug war thriller — moral ambiguity, Villeneuve"),
    # ("The Fugitive (1993)",                    "Chase thriller — wrongly accused, Harrison Ford"),
    # ("Cape Fear (1991)",                       "Psychological thriller — DeNiro stalker, Scorsese"),
    # ("Fatal Attraction (1987)",                "Erotic thriller — obsession, domestic threat"),
    # ("Basic Instinct (1992)",                  "Erotic noir thriller — femme fatale, interrogation"),
    # ("The Blair Witch Project (1999)",         "Found-footage horror — woods, ambiguous, minimalist"),
    # ("Drive (2011)",                           "Neo-noir action — Refn, silent protagonist, style"),
    # ("1917 (2019)",                            "War thriller — single-shot gimmick, WWI, Mendes"),
    # ("Apocalypse Now (1979)",                  "War/psychological descent — Heart of Darkness"),
    # ("Full Metal Jacket (1987)",               "Vietnam — dehumanization, two-act structure, Kubrick"),
    # ("The Great Escape (1963)",                "WWII POW ensemble — heist, camaraderie"),
    # ("Chinatown (1974)",                       "Neo-noir mystery — corruption, Polanski, water"),
    # ("Black Book (2006)",                      "WWII Dutch resistance — Verhoeven, moral complexity"),
    # ("Once Upon a Time in the West (1968)",    "Spaghetti Western — Leone, operatic revenge"),
    # ("The Good, the Bad and the Ugly (1966)",  "Spaghetti Western — Leone, Civil War, iconic"),
    # ("Shoah (1985)",                           "Holocaust documentary — 9-hour testimony, Lanzmann"),
    # ("The Right Stuff (1983)",                 "Space race drama — test pilots, American mythology"),
    # ("First Man (2018)",                       "Biopic — Apollo 11, Chazelle, grief/isolation"),
    # ("Gran Turismo (2023)",                    "Sports underdog — gamer to racer, based on true story"),
    # ("F1 (2025)",                              "Formula 1 action — Kosinski, Apple, Brad Pitt"),
    # # Horror
    # ("The Exorcist (1973)",                    "Supernatural horror — possession, faith vs evil"),
    # ("The Lighthouse (2019)",                  "Psychological horror — isolation, two-hander, Eggers"),
    # ("The Ninth Gate (1999)",                  "Occult mystery — Polanski, rare books, devil"),
    # # International
    # ("In the Mood for Love (2000)",            "Hong Kong romance — Wong Kar-wai, restraint, longing"),
    # ("Crouching Tiger, Hidden Dragon (2000)",  "Wuxia epic — Ang Lee, female protagonists, poetic"),
    # ("Princess Mononoke (1997)",               "Anime epic — Miyazaki, environmentalism, war"),
    # ("Police Story (1985)",                    "Hong Kong action — Jackie Chan, practical stunts"),
    # # Disaster
    # ("Twister (1996)",                         "Disaster action — tornado chasers, summer blockbuster"),
    # ("The Poseidon Adventure (1972)",          "Disaster survival — capsized ship, ensemble"),
    # ("The Towering Inferno (1974)",            "Disaster ensemble — skyscraper fire, star-studded"),
    # ("Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)",
    #                                            "Cold War satire — Kubrick, nuclear absurdism"),
    # ("Rear Window (1954)",                     "Hitchcock voyeurism — confined, suspense, voyeur"),
    # ("The Usual Suspects (1995)",              "Crime thriller — twist ending, unreliable narrator"),
    # ("American Graffiti (1973)",               "Coming-of-age/nostalgia — NOT generic teen comedy"),
    # ("The Godfather Part II (1974)",           "Organized crime epic — Corleone saga, parallel timelines"),
    # ("Roman Holiday (1953)",                   "Audrey Hepburn romance — European escapism, bittersweet"),
    # ("Sabrina (1954)",                         "Wilder romantic comedy — class/transformation"),
    # ("How to Marry a Millionaire (1953)",      "Screwball comedy — gold diggers, ensemble"),
    # ("Project Hail Mary (2026)",               "Sci-fi survival/problem-solving — isolation, first contact"),
    # ("Jaws (1975)",                            "Creature thriller — Spielberg, small-town siege"),
    # ("Star Wars (1977)",                       "Space opera — rebellion, hero's journey, franchise origin"),
    # ("The Empire Strikes Back (1980)",         "Space opera sequel — darker, Vader reveal, AT-AT"),
    # ("Return of the Jedi (1983)",              "Space opera conclusion — redemption, Ewoks, Death Star"),
    # ("Dune (2021)",                            "Sci-fi epic — messiah, desert planet, political intrigue"),
    # ("F1 (2025)",                              "Racing drama — redemption, mentor/rookie, championship"),
    # ("A House of Dynamite (2025)",             "Political thriller — nuclear crisis, Bigelow, procedural"),
    # ("Twelve Monkeys (1995)",                  "Time travel noir — apocalypse, unreliable narrator, Gilliam"),
    # ("Oppenheimer (2023)",                     "Biographical epic — atomic bomb, moral weight, Nolan"),
    # ("Everything Everywhere All at Once (2022)", "Multiverse comedy-drama — immigrant family, chaos, heart"),
    # ("Birdman or (The Unexpected Virtue of Ignorance) (2014)", "Hollywood satire — washed-up actor, Broadway, one-take"),
    # ("The Witch (2016)",                       "Folk horror — Puritan family, isolation, supernatural"),
    # ("It Follows (2015)",                      "Horror — STD curse metaphor, relentless pursuit"),
    # ("Green Room (2016)",                      "Siege thriller — punk band, neo-Nazis, survival"),
    # ("Se7en (1995)",                           "Serial killer procedural — seven sins, bleak, Fincher"),
    # ("Enemy (2014)",                           "Psychological thriller — doppelganger, Villeneuve, spider"),
    # ("The Adventures of Tintin (2011)",        "Animated adventure — Spielberg, globe-trotting mystery"),
    # ("I Heart Huckabees (2004)",               "Existential comedy — detectives, interconnectedness"),
    ("When Harry Met Sally... (1989)",            "Rom-com — friendship to love, Nora Ephron, NOT generic romance"),
    # ("Notting Hill (1999)",                    "Rom-com — celebrity/ordinary man, Hugh Grant, British charm"),
    # ("Crazy, Stupid, Love. (2011)",            "Modern romcom — ensemble, layered, NOT generic love story"),
    # ("The Conversation (1974)",                "Paranoia thriller — surveillance, Coppola, character study"),
    # ("Dog Day Afternoon (1975)",               "Crime drama — bank robbery gone wrong, Pacino, NYC"),
    # ("All the President's Men (1976)",         "Political journalism — Watergate, procedural, Pakula"),
    # ("Ronin (1998)",                           "Action thriller — mercenaries, car chases, ambiguous loyalties"),
    # ("Collateral (2004)",                      "Neo-noir thriller — hitman/cabbie, one night, Mann"),
    # ("Man on Fire (2004)",                     "Revenge thriller — bodyguard, kidnapping, Mexico City"),
    # ("Manchester by the Sea (2016)",           "Slow grief drama — Lonergan, New England, NOT redemption arc"),
    # ("Ordinary People (1980)",                 "Family trauma drama — guilt, therapy, Redford"),
    # ("Mulholland Drive (2001)",                "Surrealist neo-noir — Lynch, identity, Hollywood nightmare"),
    # ("Synecdoche, New York (2008)",            "Experimental — Kaufman, mortality, meta-theatrical"),
    # ("L.A. Confidential (1997)",               "Neo-noir — 1950s LAPD corruption, ensemble, Curtis Hanson"),
    ("Tenet (2020)",                           "Sci-fi action — time inversion, spy thriller, Nolan"),
    # ("The Shining (1980)",                     "Horror — isolation, Kubrick, supernatural/psychological"),
    # ("The Panic in Needle Park (1971)",        "Addiction drama — heroin, NYC, Pacino, raw naturalism"),
    # ("The Princess Bride (1987)",              "Fantasy adventure — fairy tale, comedy, self-aware"),
    # ("Labyrinth (1986)",                       "Fantasy — Bowie, Henson, coming-of-age dream world"),
    # ("Being John Malkovich (1999)",            "Surrealist comedy — identity, celebrity, Kaufman/Jonze"),
]

#deduplicate, preserving order
seen = set()
PROBES = []
for film, hint in CONTROL_FILMS:
    if film not in seen:
        seen.add(film)
        PROBES.append((film, hint))


def run_probe(engine, source_title, hint, top_n=10, priority="balanced",
              exclude_obscure=False, debug=False):
    result = engine.get_recommendations(
        display_title=source_title,
        priority=priority,
        exclude_obscure=exclude_obscure,
    )
    if result is None:
        print(f"\nSOURCE: {source_title}  [NOT FOUND IN ENGINE]")
        return

    films = result.get("matches", [])[:top_n]
    print(f"\n── SOURCE: {source_title} ──")
    print(f"   expect: {hint}")
    if not films:
        print("  (no matches — check genre gate, exclude_obscure, keyword floor)")
        return

    for i, m in enumerate(films, 1):
        title    = m.get("title", "?")
        year     = m.get("year", "?")
        helix    = m.get("shared_helix", "")
        kw       = m.get("shared_keywords", "")
        director = m.get("shared_director", "")
        cast     = m.get("shared_cast", "")
        bubbles  = ", ".join(filter(None, [helix, kw, director, cast]))
        bubble_str = f"  [{bubbles[:120]}]" if bubbles else ""
        print(f"  {i:2}. {title} ({year}){bubble_str}")

        if debug:
            ch = m.get("_ch", {})
            if ch:
                parts = (
                    f"kw={ch.get('kw',0):.3f} "
                    f"sem={ch.get('sem',0):.3f} "
                    f"log={ch.get('logline',0):.3f} "
                    f"tag={ch.get('tagline',0):.3f} "
                    f"wiki={ch.get('wiki',0):.3f} "
                    f"wiki_sem={ch.get('wiki_sem',0):.3f} "
                    f"mood={ch.get('mood',0):.3f} "
                    f"ov={ch.get('ov',0):.3f} "
                    f"cast={ch.get('cast',0):.3f} "
                    f"dir={ch.get('dir',0):.3f} "
                    f"cattags={ch.get('cattags',0):.3f} | "
                    f"raw_kw={ch.get('raw_kw',0):.3f} "
                    f"raw_sem={ch.get('raw_sem',0):.3f} "
                    f"raw_cat={ch.get('raw_cat',0):.3f} "
                    f"raw_log={ch.get('raw_log',0):.3f} "
                    f"raw_tag={ch.get('raw_tag',0):.3f}"
                )
                print(f"       {parts}")
            top_kw = m.get("_top_kw", [])
            if top_kw:
                kw_str = "  ".join(f"{tok}({score:.3f})" for tok, score in top_kw)
                print(f"       kw tokens: {kw_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--film",     default=None,  help="Single film to test")
    parser.add_argument("--top",      type=int, default=10)
    parser.add_argument("--priority", default="balanced")
    parser.add_argument("--obscure",  action="store_true", help="Include < 50K vote films")
    parser.add_argument("--debug",    action="store_true", help="Show per-channel score breakdown")
    args = parser.parse_args()

    print("Loading FilmHelixEngine...")
    engine = FilmHelixEngine()
    engine.load_data()
    engine.train_model()
    print("Engine ready.\n")

    exclude_obscure = not args.obscure

    probes = [(args.film, "manual test")] if args.film else PROBES

    for source, hint in probes:
        run_probe(engine, source, hint,
                  top_n=args.top,
                  priority=args.priority,
                  exclude_obscure=exclude_obscure,
                  debug=args.debug)


if __name__ == "__main__":
    main()
