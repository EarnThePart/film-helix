import os
import re
import math
import numpy as np
import pandas as pd
import sqlite3
import wordninja
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = '/tmp/movies.db'
EMBEDDINGS_CACHE         = 'semantic_embeddings_cache.npy'
WIKI_EMBEDDINGS_CACHE    = 'wiki_semantic_embeddings_cache.npy'

META_KEYWORD_STOPWORDS = {
    'basedonshortfilm', 'basedonnovel', 'basedonbook', 'basedontruestory',
    'basedonplay', 'basedoncomicbook', 'basedoncomicseries', 'basedonscreenplay',
    'basedonnovelorbook', 'basedonnovella', 'basedonchildren', 'basedonwebseries',
    'basedonmanga', 'basedonvideogame', 'basedonrealpeople', 'basedonradioplay',
    'lowbudget', 'cultfilm', 'surrealism', 'cartoon',
    'colorinfilm', 'directorscut', 'sequel', 'prequel',
    'reboot', 'remake', 'duringcreditssting', 'aftercreditssting',
    'duringcreditsstinger', 'aftercreditsstinger', 'postcreditsstinger', 'postcreditssting',
    'gatekeeper', 'slime', 'receptionist', 'worldtradecenter', 'enchant', 'amused',
    'bmovie', 'relationship', 'friends', 'family', 'death', 'evil', 'hero',
    'escape', 'fight', 'gun', 'money', 'travel', 'father', 'mother', 'sister',
    'brother', 'daughter', 'fear', 'admiring', 'adoring', 'casual', 'arrogant', 'dramatic',
    'blackpeople', 'van', 'fword',
    'hilarious', 'intense', 'playful', 'cautionary', 'awestruck', 'antagonistic',
    'bold', 'anxious', 'whimsical', 'tense', 'hopeful', 'cheerful', 'angry',
    'audacious', 'romantic', 'inspirational', 'shocking', 'nostalgic',
    'affectation', 'comforting', 'complex', 'ambiguous', 'nerd', 'urbansetting',
    'basedonshort', 'basedon', 'truestory', 'rosebud',
    'milkyway', 'universe', 'irish-american', 'irishamerican', 'italian-american', 
    'italianamerican', 'african-american', 'africanamerican', 'japanese-american', 'chinese-american',
    'mexican-american', 'jewish-american', 'korean-american',
    'childabuse', 'animalabuse', 'animalcruelty', 'animalkilling',
    'childmolestation', 'sexualabuse', 'domesticabuse', 'domesticviolence',
    'struggleforsurvival',
    #new TMDB format tags
    'shortfilm', 'womandirector', 'lostfilm', 'experimental', 'pinkfilm',
    'stopmotion', 'documentaryshort', 'studentfilm', 'preservedfilm',
    'behindthescenes', 'basedonplayormusical', 'arthouse', 'essayfilm',
    'hologram', 'humanity', 'praise', 'aggressive', 'symbolism', 'vindictive',
    'selfishness', 'awkwardness', 'indifferent',
    'immaturity', 'lifestyle', 'productplacement', 'pseudo-documentary',
    'pseudodocumentary', 'historicaldocumentary',
    'cigarsmoking', 'cigar smoking', 'cigarette', 'smoking',
    'houseboat',
    'hatter', 'mad hatter', 'scarecrow',
    'umbertoeco',
}

#dedicated vibe/atmosphere TF-IDF channel used by the "Vibe" match focus
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
    #TMDB crowd-sourced tone/vibe descriptors
    'appreciative', 'bewildered', 'audacious', 'candid',
    'empathetic', 'frantic', 'disheartening', 'commanding', 'dignified',
    'blunt', 'biting', 'bold', 'bitter', 'ambivalent', 'distressing',
    'anxious', 'cautionary', 'didactic', 'dramatic', 'complex', 'critical',
    'direct', 'sincere', 'comforting', 'forceful', 'joyful', 'exhilarated',
    'introspective', 'provocative', 'serene', 'hilarious', 'cheerful', 'loving',
    'sardonic', 'callous', 'grim',
    'clinical', 'antagonistic', 'inspiring', 'inspirational', 'steamy',
    'risque', 'affectation', 'curious', 'absurdist', 'absurd',
}

#define genres for cross-contamination blocking unless overwhelming shared DNA
STRICT_GENRES = {'comedy', 'animation', 'documentary', 'romance', 'musical'}

HELIX_COLUMNS = ('helix_pro', 'helix_dyn', 'helix_thm', 'helix_str', 'helix_ton', 'helix_dom', 'helix_sty')

#location tags for settings bucket (excluded from TF-IDF)
GEO_DISPLAY_TOKENS = {
    'newyorkcity', 'losangeles', 'london', 'paris', 'newyork',
    'boston', 'massachusetts', 'chicago', 'sanfrancisco', 'texas', 'california',
    'newjersey', 'washington', 'washingtondc', 'hongkong', 'hong-kong', 'tokyo',
    'seoul', 'beijing', 'shanghai', 'mumbai', 'delhi', 'rome', 'berlin', 'sydney',
    'toronto', 'montreal', 'morocco', 'ireland', 'france', 'germany', 'italy',
    'japan', 'korea', 'southkorea', 'china', 'india', 'russia', 'mexico', 'spain',
    'brazil', 'egypt', 'australia', 'canada', 'argentina', 'colombia', 'peru',
    'chile', 'venezuela', 'nigeria', 'southafrica', 'kenya', 'ethiopia',
    'vietnam', 'thailand', 'indonesia', 'pakistan', 'iran', 'iraq', 'turkey',
    'greece', 'portugal', 'sweden', 'norway', 'denmark', 'finland', 'poland',
    'ukraine', 'romania', 'hungary', 'czechoslovakia', 'yugoslavia', 'austria',
    'switzerland', 'belgium', 'netherlands', 'scotland', 'england', 'wales',
    'cuba', 'sicily', 'moscow', 'istanbul', 'cairo', 'lagos', 'nairobi',
    'bangkok', 'singapore', 'amsterdam', 'vienna', 'prague', 'warsaw', 'budapest',
    'dublin', 'lisbon', 'barcelona', 'madrid', 'milan', 'naples',
    'miami', 'lasvegas', 'neworleans', 'philadelphia', 'detroit', 'seattle',
    'atlanta', 'dallas', 'houston', 'hawaii', 'alaska', 'florida',
}

KEYWORD_NORMALIZATIONS = {
    #genre descriptors that collapse into their subject
    'disastermovie': 'disaster', 'naturaldisaster': 'disaster',
    #medical/psychological shorthand
    'post-traumaticstressdisorder(ptsd)': 'ptsd', 'posttraumaticstressdisorder': 'ptsd',
    #parole/release variants
    'parole': 'parolee',
    #"set in X" tokens — collapse into the place name
    'setinafrica': 'africa',
    #UK regional normalizations
    'northireland': 'northernireland',
    'lapland': 'england', 'northumberland': 'england', 'lancashire': 'england',
    'surrey': 'england', 'suffolk': 'england', 'norfolk': 'england',
    'essex': 'england', 'kent': 'england',
    #tone/genre normalization
    'darkcomedy': 'darklycomedic',
    'galaxy': 'deepspace',
    #subject normalization
    'unemployed': 'unemployment', 'unemploymentbenefits': 'unemployment',
    #psychiatric institution variants (bare 'asylum' excluded — also used in refugee context)
    'insaneasylum': 'psychiatrichospital', 'mentalasylum': 'psychiatrichospital',
    'lunaticasylum': 'psychiatrichospital', 'psychiatricward': 'psychiatrichospital',
    'psychiatricclinic': 'psychiatrichospital',
    #relationship role variants
    'badmother-in-law': 'mother-in-law', 'interferingmother-in-law': 'mother-in-law',
    'mother-in-lawdaughter-in-lawrelationship': 'mother-in-law',
    #legal
    'court': 'courtroom', 'trial': 'courtroom', 'court_case': 'courtroom',
    #media
    'journalist': 'journalismandmedia', 'reporter': 'journalismandmedia',
    'newspaperman': 'journalismandmedia', 'journalism': 'journalismandmedia',
    #crime/violence
    'murderer': 'murder', 'murders': 'murder', 'killing': 'murder',
    'serial': 'serialkiller', 'psychopathic': 'psychopath', 'theft': 'robbery',
    'gangsters': 'gangster', 'killers': 'killer', 'criminals': 'criminal',
    #relationship dynamics
    'adultery': 'infidelity', 'extramaritalaffair': 'infidelity',
    'bestfriend': 'friendship', 'bestfriends': 'friendship',
    'malefriendship': 'friendship', 'femalefriendship': 'friendship',
    'romantic': 'romance', 'brothers': 'brother', 'sisters': 'sister',
    #horror related
    'spider': 'spiders', 'spiderbite': 'spiders', 'spiderqueen': 'spiders',
    'spidergeneral': 'spiders', 'mutantspider': 'spiders', 'poisonousspider': 'spiders',
    'giantspider': 'spiders',
    'bug': 'bugs', 'insect': 'bugs', 'insects': 'bugs', 'giantinsect': 'bugs',
    'monsters': 'monster', 'zombies': 'zombie', 'vampires': 'vampire',
    'ghosts': 'ghost', 'demons': 'demon', 'aliens': 'alien', 'creatures': 'creature',
    #misc
    'memoryloss': 'amnesia',
    'spacecraft': 'space',
    'scienceteacher': 'teacher',
    'perfection': 'perfectionism',
    'oslo': 'norway',
    'fatherandsonrelationship': 'fathersonrelationship',
    'cliché': 'cliche',
    'epidemic': 'outbreak',
    'pandemic': 'outbreak',
    'contagion': 'outbreak',
    'roulette': 'gambling',
    'rarebook': 'books',
    'bibliophilia': 'books',
    'bookstore': 'books',
    'novelist': 'writer',
    'author': 'writer',
    #based-on variants
    'basedonyoungadultnovel': 'basedonbook', 'basedonshortstory': 'basedonbook',
    'basedonmemoirorautobiography': 'basedonbook', 'basedonmemoirortobiography': 'basedonbook',
    'badonchildrensbook': 'basedonbook', 'basedonchildrensbook': 'basedonbook',
    'basedonnovelorbook': 'basedonbook',
    #comic/manga
    'basedoncomic': 'basedoncomic', 'basedonmanga': 'basedoncomic',
    #surreal
    'absurd': 'surreal', 'surrealism': 'surreal',
    #director
    'womandirector': 'femaledirector',
    #time
    'time-manipulation': 'timemanipulation',
    #world-saving
    'worlddestruction': 'savingtheworld', 'worldending': 'savingtheworld',
    'savetheplanet': 'savingtheworld', 'heroicmission': 'savingtheworld',
    'armageddon': 'savingtheworld',
    #alien contact
    'firstcontact': 'aliencontact', 'alienlanguage': 'aliencontact',
    #hostile aliens
    'alienattack': 'hostilealien', 'evilalien': 'hostilealien', 'humanvsalien': 'hostilealien',
    #alien creatures
    'alienlife-form': 'alien', 'aliencreature': 'alien', 'alienrace': 'alien',
    'wwi': 'worldwari', 'worldwar1': 'worldwari',
    'wwii': 'worldwarii', 'worldwar2': 'worldwarii',
    'imprisonment': 'prison', 'nazism': 'nazi', 'nazioccupation': 'nazi',
    'dreams': 'dream', 'nightmares': 'nightmare', 'flashbacks': 'flashback',
    'doctors': 'doctor', 'soldiers': 'soldier', 'scientists': 'scientist',
    'detectives': 'detective', 'assassins': 'assassin', 'hostages': 'hostage',
    'sicilianmafia': 'mafia',
    'periodfilm': 'periodpiece',
    'horrified': 'terror', 'frightened': 'terror', 'terrified': 'terror',
    '1500s': '16th_century', '1510s': '16th_century', '1520s': '16th_century',
    '1530s': '16th_century', '1540s': '16th_century', '1550s': '16th_century',
    '1560s': '16th_century', '1570s': '16th_century', '1580s': '16th_century',
    '1590s': '16th_century', '16thcentury': '16th_century',
    '1600s': '17th_century', '1610s': '17th_century', '1620s': '17th_century',
    '1630s': '17th_century', '1640s': '17th_century', '1650s': '17th_century',
    '1660s': '17th_century', '1670s': '17th_century', '1680s': '17th_century',
    '1690s': '17th_century', '17thcentury': '17th_century',
    '1700s': '18th_century', '1710s': '18th_century', '1720s': '18th_century',
    '1730s': '18th_century', '1740s': '18th_century', '1750s': '18th_century',
    '1760s': '18th_century', '1770s': '18th_century', '1780s': '18th_century',
    '1790s': '18th_century', '18thcentury': '18th_century',
    '1800s': '19th_century', '1810s': '19th_century', '1820s': '19th_century',
    '1830s': '19th_century', '1840s': '19th_century', '1850s': '19th_century',
    '1860s': '19th_century', '1870s': '19th_century', '1880s': '19th_century',
    '1890s': '19th_century', '19thcentury': '19th_century',
    'slavetrade': 'slavery', 'enslavement': 'slavery',
    'amazinggracehymn': 'hymn', 'gospelmusic': 'hymn',
    'givingbirth': 'birth', 'childbirth': 'birth', 'naturalbirth': 'birth',
    'homebirth': 'birth', 'ceasareanbirth': 'birth', 'complicatedbirth': 'birth',
    'birthoftwins': 'birth',
    'xmas': 'christmas', 'xmaseve': 'christmaseve',
    'nationalsecurityagency(nsa)': 'nsa', 'nsaagent': 'nsa',
    'unitedstatesofamerica(usa)': 'usa',
    'aids': 'hiv', 'hivaids': 'hiv', 'hiv/aids': 'hiv', 'aidsepidemic': 'hiv',
    'policeman': 'policeofficer', 'cop': 'policeofficer',
    'femalepoliceofficer': 'policewoman',
    'corruptcop': 'crookedcop', 'crookedcops': 'crookedcop',
    'corruptedcops': 'crookedcop', 'crookedsheriff': 'crookedcop',
    'corruptsheriff': 'crookedcop',
    #comma-joined city,country TMDB tokens — normalize to whichever part is known
    'losangeles,california': 'losangeles', 'london,england': 'london',
    'paris,france': 'paris', 'chicago,illinois': 'chicago',
    'seattle,washington': 'seattle', 'tokyo,japan': 'tokyo',
    'miami,florida': 'miami', 'atlanta,georgia': 'atlanta',
    'boston,massachusetts': 'boston', 'amsterdam,netherlands': 'amsterdam',
    'seoul,southkorea': 'seoul', 'berlin,germany': 'berlin',
    'rome,italy': 'rome', 'prague,czechrepublic': 'prague',
    'montreal,canada': 'montreal', 'moscow,russia': 'moscow',
    'philadelphia,pennsylvania': 'philadelphia', 'manhattan,newyorkcity': 'newyork',
    'brooklyn,newyorkcity': 'newyork', 'newark,newjersey': 'newjersey',
    'venice,italy': 'italy', 'oslo,norway': 'norway',
    'mexicocity,mexico': 'mexico', 'ontario,canada': 'canada',
    'zurich,switzerland': 'switzerland', 'beirut,lebanon': 'lebanon',
    'cologne,germany': 'germany', 'saopaulo,brazil': 'brazil',
    'stockholm,sweden': 'sweden', 'harbin,china': 'china',
    'granada,spain': 'spain', 'akita,japan': 'japan',
    'nashville,tennessee': 'usa', 'richmond,virginia': 'usa',
    'oregon,usa': 'usa', 'pennsylvania,usa': 'usa', 'wyoming,usa': 'usa',
    #compound basedon forms
    'basedonmagazine,newspaperorarticle': 'basedonarticle',
    'basedonmyths,legendsorfolklore': 'mythology',
    'basedonnovelorbook': 'basedonbook', 'basedonshortstory': 'basedonbook',
    'android': 'robot',
    'homicideinvestigation': 'murderinvestigation',
    'coveredinvestigation': 'investigation',
    '1stcentury': 'antiquity', '2ndcentury': 'antiquity', '3rdcentury': 'antiquity',
    '4thcentury': 'antiquity', 'ancientrome': 'antiquity', 'ancientgreece': 'antiquity',
    'ancientegypt': 'antiquity', 'ancientworld': 'antiquity', 'biblical': 'antiquity',
    'biblicaltimes': 'antiquity', 'antiquity': 'antiquity',
    '5thcentury': 'medieval', '6thcentury': 'medieval', '7thcentury': 'medieval',
    '8thcentury': 'medieval', '9thcentury': 'medieval', '10thcentury': 'medieval',
    '11thcentury': 'medieval', '12thcentury': 'medieval', '13thcentury': 'medieval',
    '14thcentury': 'medieval', '15thcentury': 'medieval', 'middleages': 'medieval',
    'medieval': 'medieval',
    #law enforcement / investigators
    'fbiagent': 'fbi', 'fbiofficer': 'fbi',
    'privatedetective': 'detective', 'policedetective': 'detective',
    'homicidedetective': 'detective', 'amateurdetective': 'detective',
    'privateinvestigator': 'detective',
    'undercoveragent': 'secretagent', 'governmentagent': 'secretagent',
    'dirtycop': 'crookedcop',
    #killers / hired violence
    'contractkiller': 'hitman', 'hiredkiller': 'hitman',
    'femaleassassin': 'assassin',
    'psychokiller': 'psychopath',
    #mafia / organised crime
    'mafiaboss': 'mafia', 'chinesemafia': 'mafia',
    'bratva(russianmafia)': 'mafia',
    'broker': 'stockbroker',
    'plannedcoup': 'coupdetat', 'militarycoup': 'coupdetat',
    'insurrection': 'uprising',
    'sedition': 'treason',
    'gangwarfare': 'gangwar',
    #drug normalizations — collapse consumption variants into 'drugs'; crime-side stays separate
    'drug': 'drugs', 'drugabuse': 'drugs', 'substanceabuse': 'drugs',
    'illegaldrugs': 'drugs', 'drugusage': 'drugs', 'drugscene': 'drugs',
    'drugaddict': 'drugaddiction',
    #surfing
    'surf': 'surfing', 'surfer': 'surfing', 'surfers': 'surfing',
    'surfboard': 'surfing', 'surfingcontest': 'surfing', 'femalesurfer': 'surfing',
    #space travel
    'spacetravel': 'deepspace', 'outerspace': 'deepspace', 'interstellarspace': 'deepspace',
    'spaceflights': 'deepspace',
    'spacecraft': 'spaceship',
    #wealth
    'millionaire': 'wealth', 'billionaire': 'wealth',
    #entertainment industry
    'hollywood': 'moviebusiness', 'moviestudio': 'moviebusiness',
    'movie_studio': 'moviebusiness', 'filmstudio': 'moviebusiness',
    'showbusiness': 'moviebusiness', 'oldhollywood': 'moviebusiness',
    'blaxploitationcinema': 'blaxploitation',
    'filmdirector': 'director',
    'musicbusiness': 'musicindustry',
    'rich': 'wealth', 'wealthy': 'wealth',
    'poor': 'poverty',
    'terroristgroup': 'terrorism', 'terroristplot': 'terrorism',
    'terroristattack': 'terrorism', 'terroristbombing': 'terrorism',
    'terroristthreat': 'terrorism', 'terrorcell': 'terrorism',
    'terroristbase': 'terrorism', 'ploterroristgroup': 'terrorism',
    'etaterroristgroup': 'terrorism', 'hamasterroristgroup': 'terrorism',
    'waronterror': 'terrorism', 'terrorist': 'terrorism',
    '9/11': 'post911', 'post9/11': 'post911',
    's.w.a.t.': 'swat',
    'maya': 'mayacivilization', 'mayatemple': 'mayacivilization',
    'mayancalendar': 'mayacivilization', 'mayans': 'mayacivilization',
    'mexica(aztec)': 'aztec',
    'blackcomedy': 'darkcomedy',
    'unclenephewrelationship': 'familial_relationship',
    'uncleniecerelationship': 'familial_relationship',
    'auntnephewrelationship': 'familial_relationship',
    'auntniecerelationship': 'familial_relationship',
    'cousin': 'familial_relationship', 'cousins': 'familial_relationship',
    'cousinrelationship': 'familial_relationship',
    'cousincousinrelationship': 'familial_relationship',
    'cousinmarriage': 'familial_relationship',
    'cousinsinlove': 'familial_relationship',
    'brotherinlaw': 'familial_relationship', 'sisterinlaw': 'familial_relationship',
    'halfbrother': 'brother', 'halfsister': 'sister',
    'grandfather': 'grandparent', 'grandmother': 'grandparent',
    'grandparents': 'grandparent',
    'grandson': 'grandchild', 'granddaughter': 'grandchild',
    'grandchildren': 'grandchild',
    'grandfathergranddaughterrelationship': 'grandparentgrandchildrelationship',
    'grandfathergrandsonrelationship': 'grandparentgrandchildrelationship',
    'grandmothergranddaughterrelationship': 'grandparentgrandchildrelationship',
    'grandmothergrandsonrelationship': 'grandparentgrandchildrelationship',
    'whodunit': 'murdermystery',
    'mysterymurder': 'murdermystery',
    'cozymystery': 'murdermystery',
    'cuckoldedhusband': 'cuckold',
    'cuckholdedhusband': 'cuckold',
    'cuckolded': 'cuckold',
    'murderspree': 'killingspree',
    'ss': 'nazi',
    'exhilarated': 'exhilarating',
    'indianausa': 'indiana',
    'petrol': 'oil',
    'unsociability': 'antisocial',
    'centralandsouthamerica': 'southamerica',
}


def normalize_keyword_token(tn):
    normed = KEYWORD_NORMALIZATIONS.get(tn)
    if normed is None:
        tn_stripped = re.sub(r'\(.*\)$', '', tn)
        normed = KEYWORD_NORMALIZATIONS.get(tn_stripped, tn_stripped)
    return normed


#weight balancing w. wiki as separate TF-IDF channel for Wikipedia plot summaries where availalbe

PRIORITY_WEIGHTS = {
    'balanced': dict(keywords=0.10, semantic=0.22, wiki=0.05, wiki_semantic=0.16, logline=0.00, tagline=0.00, mood=0.04, overview=0.02, cast=0.02, director=0.01, writer=0.00, cattags=0.07, helix_pro=0.05, helix_dyn=0.05, helix_thm=0.05, helix_str=0.04, helix_ton=0.05, helix_spl=0.00, helix_dom=0.05, helix_sty=0.03),
    'plot': dict(keywords=0.08, semantic=0.15, wiki=0.08, wiki_semantic=0.12, logline=0.00, tagline=0.00, mood=0.02, overview=0.02, cast=0.00, director=0.00, writer=0.00, cattags=0.06, helix_pro=0.07, helix_dyn=0.07, helix_thm=0.07, helix_str=0.07, helix_ton=0.07, helix_spl=0.00, helix_dom=0.08, helix_sty=0.03),
    'vibe': dict(keywords=0.05, semantic=0.16, wiki=0.03, wiki_semantic=0.05, logline=0.00, tagline=0.00, mood=0.35, overview=0.03, cast=0.00, director=0.00, writer=0.00, cattags=0.05, helix_pro=0.03, helix_dyn=0.03, helix_thm=0.03, helix_str=0.03, helix_ton=0.10, helix_spl=0.00, helix_dom=0.04, helix_sty=0.03),
    'genre': dict(keywords=0.12, semantic=0.13, wiki=0.04, wiki_semantic=0.04, logline=0.00, tagline=0.00, mood=0.09, overview=0.05, cast=0.08, director=0.04, writer=0.00, cattags=0.07, helix_pro=0.04, helix_dyn=0.04, helix_thm=0.04, helix_str=0.04, helix_ton=0.04, helix_spl=0.00, helix_dom=0.12, helix_sty=0.03),
    'cast': dict(keywords=0.09, semantic=0.10, wiki=0.00, wiki_semantic=0.00, logline=0.00, tagline=0.00, mood=0.02, overview=0.02, cast=0.68, director=0.05, writer=0.00, cattags=0.01, helix_pro=0.01, helix_dyn=0.01, helix_thm=0.00, helix_str=0.00, helix_ton=0.00, helix_spl=0.00, helix_dom=0.01, helix_sty=0.00),
    'director': dict(keywords=0.09, semantic=0.10, wiki=0.00, wiki_semantic=0.00, logline=0.00, tagline=0.00, mood=0.02, overview=0.02, cast=0.05, director=0.68, writer=0.00, cattags=0.01, helix_pro=0.01, helix_dyn=0.01, helix_thm=0.00, helix_str=0.00, helix_ton=0.00, helix_spl=0.00, helix_dom=0.01, helix_sty=0.00),
    'writer': dict(keywords=0.09, semantic=0.10, wiki=0.00, wiki_semantic=0.00, logline=0.00, tagline=0.00, mood=0.02, overview=0.02, cast=0.05, director=0.00, writer=0.68, cattags=0.01, helix_pro=0.01, helix_dyn=0.01, helix_thm=0.00, helix_str=0.00, helix_ton=0.00, helix_spl=0.00, helix_dom=0.01, helix_sty=0.00),
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

        #semantic embeddings use TMDB overview only (due to 256-token limit)
        #wiki plots get dedicated TF-IDF channel
        self.df['narrative_text'] = self.df['overview']
        self.df['vec_str_wiki'] = self.df.apply(
            lambda r: str(r['wiki_plot']) if str(r.get('wiki_plot_status', '')) in ('ok', 'lead_section') and str(r['wiki_plot']).strip() not in ('', 'nan') else '',
            axis=1
        )
        self.df['vec_str_overview']  = self.df['narrative_text']
        self.df['vec_str_keywords']  = self.df['dna_keywords'].apply(self._filter_plot_keywords)
        self.df['vec_str_mood']      = self.df['dna_keywords'].apply(self._filter_mood_keywords)
        self.df['vec_str_logline'] = self.df['overview'].apply(self._extract_logline)
        self.df['vec_str_tagline'] = self.df['tagline'].fillna('').astype(str).apply(
            lambda t: t.strip() if t.strip() not in ('', 'nan') else ''
        )
        self.df['vec_str_genre']     = self.df['dna_genres']
        self.df['vec_str_cattags']   = self.df['category_tags'].fillna('').astype(str).apply(
            lambda s: s.strip() if s.strip() not in ('', 'nan') else ''
        )
        #thematic DNA tags
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
        #strip periods and hyphens ()"JohnC.Reilly" / "TonyLeungChiu-Wai")
        def _norm_names(s):
            return s.astype(str).str.replace('.', '', regex=False).str.replace('-', '', regex=False)
        self.df['vec_str_cast']      = _norm_names(self.df['dna_cast'])
        self.df['vec_str_director']  = _norm_names(self.df['dna_director'])
        self.df['vec_str_writer']    = _norm_names(self.df['dna_writer'])

        #token --> display maps so shared cast/director/writer tags are properly formatted ("Park Chan-wook", not instead not TDF token "parkchanwook")
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

        #helix IDF: log(N / (1 + df)) per unique tag across all 7 scored helix columns
        #document frequency = # of films where tag appears in ANY helix column
        N = len(self.df)
        df_counts = {}
        for col in HELIX_COLUMNS:
            seen_per_film = self.df[f'vec_str_{col}'].apply(
                lambda s: set(s.split()) - {''}
            )
            for tagset in seen_per_film:
                for tag in tagset:
                    df_counts[tag] = df_counts.get(tag, 0) + 1
        self.helix_idf = {tag: math.log(N / (1 + dfc)) for tag, dfc in df_counts.items()}
        self._helix_vocab = {tag: i for i, tag in enumerate(sorted(self.helix_idf.keys()))}

    def _filter_meta_keywords(self, keyword_str):
        if not keyword_str:
            return ""
        return " ".join(t for t in keyword_str.split() if t.lower() not in META_KEYWORD_STOPWORDS)

    @staticmethod
    def _norm_keyword(t):
        """Normalize a single keyword token: lowercase, strip hyphens to ensure "neo-noir"/"neonoir" are treated the same"""
        return t.lower().replace('-', '')

    def _filter_plot_keywords(self, keyword_str):
        """Plot keywords: strip meta stopwords, normalize variants. Mood tokens are included here
        so they score in the keyword channel too — they get extra weight only when Vibe is selected."""
        if not keyword_str:
            return ""
        tokens = []
        for t in keyword_str.split():
            tl = t.lower()
            tn = tl.replace('-', '')
            if tl in META_KEYWORD_STOPWORDS or tn in META_KEYWORD_STOPWORDS:
                continue
            if tl in GEO_DISPLAY_TOKENS or tn in GEO_DISPLAY_TOKENS:
                continue
            tokens.append(normalize_keyword_token(tn))
        return " ".join(tokens)

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
        """Extract first sentence of overview as a proxy logline. Captures core conflict/protagonist in ~20 words"""
        if not overview or str(overview).strip() in ('', 'nan'):
            return ''
        text = str(overview).strip()
        for punct in ('. ', '! ', '? '):
            idx = text.find(punct)
            if idx > 20:
                return text[:idx + 1].strip()
        return text[:200].strip()

    def train_model(self):
        self.vectorizers['overview'] = TfidfVectorizer(
            stop_words='english', min_df=2, max_df=0.85, dtype=np.float32
        )
        self.matrices['overview'] = self.vectorizers['overview'].fit_transform(
            self.df['vec_str_overview']
        )

        #ngram_range=(1,1): each dna_keywords token stored as a single joined string (e.g. "serialkiller", "nonlineartimeline")
        #eliminate false compound phrases from unrelated sequential tags (e.g. jazz+musician, neo+noir+neonoir)
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

        #wiki plot is separate TF-IDF channel
        #min_df=3 to filter noise
        #max_df=0.7 to drop ultra-common plot keywords
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

        #wikipedia category tags
        #min_df=2: tag must appear in at least 2 films to be a feature
        #max_df=0.4: drop common tags to avoid generic matches
        self.vectorizers['cattags'] = TfidfVectorizer(
            min_df=2, max_df=0.4, dtype=np.float32
        )
        self.matrices['cattags'] = self.vectorizers['cattags'].fit_transform(
            self.df['vec_str_cattags']
        )

        #helix channels: manual IDF-weighted vectors (formula: log(N / (1 + df))).
        #shared vocab across all 7 scored helix columns so rare archetypes carry more weight than common ones
        vocab = self._helix_vocab
        V = len(vocab)
        N = len(self.df)
        for helix_col in HELIX_COLUMNS:
            rows, cols, vals = [], [], []
            for i, s in enumerate(self.df[f'vec_str_{helix_col}']):
                tags = set(s.split()) - {''}
                for tag in tags:
                    j = vocab.get(tag)
                    if j is None:
                        continue
                    rows.append(i); cols.append(j); vals.append(self.helix_idf[tag])
            mat = csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.float32)
            sq = mat.multiply(mat).sum(axis=1)
            norms = np.sqrt(np.asarray(sq).flatten())
            norms[norms == 0] = 1.0
            inv = 1.0 / norms
            d = mat.tocoo()
            mat = csr_matrix(
                (d.data * inv[d.row], (d.row, d.col)),
                shape=(N, V), dtype=np.float32,
            )
            self.matrices[helix_col] = mat
        #helix_spl is no longer scored (weight=0 everywhere), only referenced for zero-vector dot products elsewhere
        self.matrices['helix_spl'] = csr_matrix((N, max(V, 1)), dtype=np.float32)

        #lowercase=False preserves CamelCase in feature names so name tags display correctly
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

        #semantic layer for the valid 50K films, cached after first run
        n = len(self.df)
        if os.path.exists(EMBEDDINGS_CACHE):
            cached = np.load(EMBEDDINGS_CACHE)
            if cached.shape[0] == n:
                self.semantic_embeddings = cached
            else:
                os.remove(EMBEDDINGS_CACHE)

        if self.semantic_embeddings is None:
            raise RuntimeError(f"Semantic embeddings cache missing or wrong size. Re-run with sentence-transformers installed to rebuild {EMBEDDINGS_CACHE}.")

        #wiki semantic embeddings (chunked to handle long plots)
        n = len(self.df)
        if os.path.exists(WIKI_EMBEDDINGS_CACHE):
            cached = np.load(WIKI_EMBEDDINGS_CACHE)
            if cached.shape[0] == n:
                self.wiki_semantic_embeddings = cached
            else:
                os.remove(WIKI_EMBEDDINGS_CACHE)

        if self.wiki_semantic_embeddings is None:
            raise RuntimeError(f"Wiki semantic embeddings cache missing or wrong size. Re-run with sentence-transformers installed to rebuild {WIKI_EMBEDDINGS_CACHE}.")

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
        _MONONYMS = {
            'cher', 'madonna', 'brandy', 'adele', 'prince', 'seal', 'bjork',
            'aaliyah', 'rihanna', 'beyonce', 'eminem', 'drake', 'zendaya',
            'sting', 'bono', 'moby', 'beck', 'pink', 'sia', 'kesha',
        }
        terms = []
        for i in sorted_idx:
            token = feature_names[i]
            resolved = _resolve(token)
            #name corrections for cast (mononyms)
            if feature_name == 'cast' and ' ' not in resolved and resolved.lower() not in _MONONYMS:
                continue
            terms.append(resolved)
            if len(terms) >= top_n:
                break
        return ", ".join(terms)

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

        #keyword diversity multiplier, dampens results whose keyword match rests on only 1 or 2 shared tokens
        #single broad tag (neonoir, supernatural, hacker) is weak evidence, 3+ shared tokens is strong
        _p = priority[0] if isinstance(priority, (list, tuple)) else priority
        s_keywords_raw = s_keywords.copy()  #preserves pre-multiplier value for floor check
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

        #category tag diversity multiplier (same logic as keyword diversity)
        #single shared tag = weak, 3+ shared = strong
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

        #dampen helix scores when there are few shared tags
        #count shared tags across ALL helix columns combined
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
        s_logline        = np.zeros(len(self.df), dtype=np.float32)
        s_tagline        = np.zeros(len(self.df), dtype=np.float32)

        #ensure adaptive genre gate
        source_genres = set(str(self.df.iloc[idx]['dna_genres']).lower().split())
        is_strict = bool(source_genres & STRICT_GENRES)
        gate_threshold = 0.35 if is_strict else 0.20

        #dual-priority blend (string, list of two strings)
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

        #continuous genre multiplier rewards closer genre match
        if use_genre_boost:
            genre_factor = np.where(s_genre >= gate_threshold, s_genre, 0.0)
            final_scores = base_scores * genre_factor + s_genre * 0.20
        else:
            genre_multiplier = np.where(s_genre >= gate_threshold, 1.0, 0.0)
            final_scores = base_scores * genre_multiplier

        if 'documentary' not in source_genres:
            final_scores[self.df['dna_genres'].str.lower().str.contains('documentary', na=False)] = 0.0

        #strict genre gate: comedy gated to 0.6, animation/doc/musical/romance gated when source also a strict genre film
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

        #same-language affinity boost: rewards same-language results by 10%, but allows strong foreign matches (Oldboy and Parasite)
        if not exclude_foreign:
            source_lang = str(self.df.iloc[idx].get('dna_lang', 'en') or 'en').lower()[:2]
            result_langs = self.df['dna_lang'].str.lower().str[:2].fillna('en')
            same_lang_mask = result_langs == source_lang
            final_scores[same_lang_mask] *= 1.10

        if exclude_animated:
            final_scores[self.df['dna_genres'].str.lower().str.contains('animation', na=False)] = 0.0

        if exclude_obscure:
            vote_counts = pd.to_numeric(self.df['vote_count'], errors='coerce').fillna(0)
            obscure_mask = vote_counts < 20000
            # exempt films sharing the source title (originals, remakes, reboots)
            src_title = self.df.iloc[idx]['title'].strip().lower()
            title_match = self.df['title'].str.strip().str.lower() == src_title
            final_scores[obscure_mask & ~title_match] = 0.0

        if exclude_sequels:
            kw_lower = self.df['dna_keywords'].str.lower()
            seq_mask = (
                kw_lower.str.contains(r'\bsequel\b', na=False) |
                kw_lower.str.contains(r'\bprequel\b', na=False) |
                kw_lower.str.contains(r'\bremake\b', na=False) |
                kw_lower.str.contains(r'\breboot\b', na=False)
            )
            final_scores[seq_mask] = 0.0

        #hard keyword floor: zero out results with near-zero keyword overlap, prevents matches from single incidental keywrod
        #only veto if BOTH keywords and semantic plot overlap are garbage
        final_scores = np.where(s_keywords_raw < 0.02, final_scores * 0.60, final_scores)

        #Helix tag gate: penalize matches if source has helix tags but potential matches don't
        has_source_tags = (
            self.matrices['helix_pro'][idx].nnz > 0 or
            self.matrices['helix_dyn'][idx].nnz > 0 or
            self.matrices['helix_thm'][idx].nnz > 0 or
            self.matrices['helix_str'][idx].nnz > 0 or
            self.matrices['helix_ton'][idx].nnz > 0 or
            self.matrices['helix_dom'][idx].nnz > 0 or
            self.matrices['helix_sty'][idx].nnz > 0
        )

        #final "smell test" for genre/plot boundaries to prevent cross-genre matches (Caddyshack and Parasite)
        #heavily penalize score if plot and base genre have ~0 overlap
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

            #compute all three overlap types independently
            #separate tag rows in UI so users see full match explanation
            overlap_k      = self._get_top_overlapping_terms(idx, i, 'keywords', 5)
            top_kw_scores  = self._get_top_keyword_scores(idx, i, top_n=5)
            overlap_cast   = self._get_top_overlapping_terms(idx, i, 'cast', 3)
            overlap_dir    = self._get_top_overlapping_terms(idx, i, 'director', 2)
            overlap_writer = self._get_top_overlapping_terms(idx, i, 'writer', 2)

            #shared helix tag overlap across all scored helix columns
            shared_helix_tags = []
            for hcol in ('helix_dom', 'helix_sty', 'helix_pro', 'helix_str', 'helix_ton', 'helix_dyn', 'helix_thm'):
                src_tags = set(str(self.df.iloc[idx].get(hcol, '') or '').split('|'))
                res_tags = set(str(self.df.iloc[i].get(hcol, '') or '').split('|'))
                shared = src_tags & res_tags - {'', 'nan'}
                shared_helix_tags.extend(sorted(shared))
            shared_helix = ', '.join(shared_helix_tags)
            total_helix_raw_i = float(
                s_helix_pro[i] + s_helix_dyn[i] + s_helix_thm[i] +
                s_helix_str[i] + s_helix_ton[i] + s_helix_dom[i] + s_helix_sty[i]
            )

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
                'poster':           str(row.get('poster', '')),
                'warnings':         str(row.get('warnings', '')),
                'shared_helix':     shared_helix,
                'total_helix_raw':  round(total_helix_raw_i, 4),
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
        """Format a name list for display, handling DB storage formats for CamelCase space-separated and 
        already-split comma/pipe-separated"""
        if not raw or raw == 'nan':
            return ''
        sep = ',' if ',' in raw else ('|' if '|' in raw else None)
        if sep:
            names = [n.strip() for n in raw.split(sep) if n.strip()]
            return ', '.join(names)
        #CamelCase
        formatted = []
        for t in raw.split():
            t = re.sub(r'^[^a-zA-ZÀ-ÿ]+|[^a-zA-ZÀ-ÿ]+$', '', t)
            if not t:
                continue
            spaced = re.sub(r'([a-zà-ÿ])(von|van|der|den|del)([A-Z])', r'\1 \2 \3', t)
            spaced = re.sub(r'([a-zà-ÿ])([A-Z])', r'\1 \2', spaced)
            spaced = re.sub(r'([A-Za-z]\.)([A-Za-z])', r'\1 \2', spaced)
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
        #plain space-separated words (e.g. "Alex Garland") form one name, not a comma list
        if len(raw.split()) > 1 and not any(re.search(r'[a-z][A-Z]', t) for t in raw.split()):
            return ' '.join(formatted)
        return ', '.join(formatted)

    def _format_country(self, raw):
        """Use wordninja to split country tokens regardless of capitalization pattern."""
        if not raw or raw == 'nan':
            return ''
        tokens = raw.split()
        formatted = [
            ' '.join(p.title() for p in wordninja.split(t.lower()))
            for t in tokens
        ]
        return ', '.join(formatted)
