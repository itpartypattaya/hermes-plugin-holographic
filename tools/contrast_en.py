# -*- coding: utf-8 -*-
"""Contrast set EN for tools/benchmark_rerank.py (RFC #9, gates 2 and 3).

Byte-compatible in structure with tools/contrast_ru.py (FACTS / PAIRS /
CHITCHAT / self_check) — same shapes, so every loader that reads contrast_ru
reads this file unchanged.

Design (see cross-encoder-rerank/references/validation-datasets.md):
  - every PAIRS entry is a minimal PAIR: two queries differing in ONE factor,
    both targets live in the store, the metric is PAIRED (both-or-nothing);
  - one shared store for all contrasts (~64 facts) — a distractor of one
    contrast is filler for another;
  - CHITCHAT — 45 messages that must recall NOTHING (side 2 of contrast C5
    for gate_reads_score; side 1 — the "hard" C2 queries);
  - the expected behaviour of the base (off) and the traps live in the `note`
    field — those are design ESTIMATES and a mandatory check: run the off mode
    and make sure the base really fails the pairs (self_check prints the plan).

Types: c1_entity — name swap (Peter/Lena, Denis/Dina, Sasha/Sonya); c2
(+subtype number) — paraphrase bridge, the target shares no content words
with the hard query while the trap does (2 of 5 carry a digit in the fact and
the number spelled out in the query); c3 — one-subject predicate (documented
weakness of retrieval.py:173-176), 6 subjects loaded with 6+ facts plus one
deliberately weak control pair (Sasha/Sonya, 2 facts each).

Synthetic data, English half of the reranker benchmark.
"""

FACTS = {
    # --- c1e-1: birthdays (one "dates" type so the type stays loaded) ---
    "peter_bd": ("Peter's birthday is on March 12.", ["Peter"]),
    "lena_bd": ("Lena's birthday is on June 3.", ["Lena"]),
    "olga_wedding": ("Olga's wedding was on August 18, 2023.", ["Olga"]),
    "app_deadline": ("The application deadline is September 25.", []),
    "barsik_shot": ("Barsik the cat has a vaccination booked for November 14.", ["Barsik"]),

    # --- c1e-2: work (the company name is shared by both targets) ---
    "denis_work": ("Denis works for Norilsk Nickel in Moscow.", ["Denis"]),
    "dina_work": ("Dina works for Norilsk Nickel in Yekaterinburg.", ["Dina"]),
    "sergey_nornickel": ("Sergey quit Norilsk Nickel back in 2022.", ["Sergey"]),
    "nornickel_office": ("The Norilsk Nickel office on Pyatnitskaya is closed for repairs.", []),

    # --- c1e-3: pool (Sasha/Sonya — close names, same content) ---
    "sasha_pool": ("Sasha goes to the pool on Tuesdays.", ["Sasha"]),
    "sonya_pool": ("Sonya goes to the pool on Thursdays.", ["Sonya"]),
    "masha_pool": ("Masha signed up for the pool in January.", ["Masha"]),
    "pool_hours": ("The pool on Sportivnaya is open until ten at night.", []),
    "sasha_chess": ("Sasha plays chess on Saturdays.", ["Sasha"]),
    "sonya_yoga": ("Sonya does yoga in the mornings.", ["Sonya"]),

    # --- c2-numeric: digits in the fact, the number spelled out in the query ---
    "denis_car": ("Denis bought a car in 2024.", ["Denis"]),
    "denis_tires": ("Denis changed his winter tires in October.", ["Denis"]),
    "denis_sochi": ("Denis spent the year 2023 in Sochi.", ["Denis"]),
    "denis_passport": ("Denis's passport expires in 2027.", ["Denis"]),
    "denis_license": ("Denis's driver's license expires in 2025.", ["Denis"]),
    "denis_rights2019": ("Denis got his license back in 2019.", ["Denis"]),
    "dina_flat": ("Dina bought her flat in 2021.", ["Dina"]),

    # --- c3 subject andrey: exactly 6 facts ---
    "andrey_vacation": ("Andrey's vacation runs from May 7 to May 20.", ["Andrey"]),
    "andrey_flex": ("Andrey works a flexible schedule.", ["Andrey"]),
    "andrey_meet": ("Andrey's Friday meetings start at 11:00.", ["Andrey"]),
    "andrey_dacha": ("Andrey spends his weekends at the summer house.", ["Andrey"]),
    "andrey_dentist": ("Andrey has a dentist appointment on Saturday.", ["Andrey"]),
    "andrey_cat": ("Andrey feeds the cat twice a day.", ["Andrey"]),

    # --- c3 subject marina: 7 facts ---
    "marina_night": ("Marina walks through the park late at night.", ["Marina"]),
    "marina_cams": ("Marina is often seen on the cameras outside the metro.", ["Marina"]),
    "marina_gagarin": ("Marina lives on Gagarin Street.", ["Marina"]),
    "marina_chinese": ("Marina is learning Chinese in the evenings.", ["Marina"]),
    "marina_office": ("Marina works at the office on weekdays.", ["Marina"]),
    "marina_diet": ("Marina doesn't eat meat.", ["Marina"]),
    "marina_car": ("Marina drives a silver sedan.", ["Marina"]),

    # --- c3 subject sergey: 7 facts (nornickel above + these) ---
    "sergey_pattaya": ("Sergey flew to Pattaya instead of Bangkok.", ["Sergey"]),
    "sergey_fear": ("Sergey was scared of flying to Thailand.", ["Sergey"]),
    "sergey_dog": ("Sergey walks the dog twice a day.", ["Sergey"]),
    "sergey_guitar": ("Sergey is learning guitar in the evenings.", ["Sergey"]),
    "sergey_market": ("Sergey goes to the market on Saturdays.", ["Sergey"]),
    "sergey_flat": ("Sergey rents a flat near the metro.", ["Sergey"]),

    # --- c2 paraphrase bridge: connectivity (no subject, shared traps) ---
    "net_off": ("The internet in the flat was cut off for non-payment.", []),
    "light_evening": ("The power in the flat goes off on schedule in the evening.", []),
    "provider": ("The provider promised to bring the speed back within a week.", []),

    # --- masha: filler crowd for top-5 ---
    "masha_dress": ("Masha bought a dress for 4,000 rubles.", ["Masha"]),
    "masha_return": ("Masha took the dress back to the store.", ["Masha"]),
    "masha_savings": ("Masha puts money aside every month.", ["Masha"]),

    # --- c3 subject olga: 7 facts (wedding above + these) ---
    "olga_teach": ("Olga teaches math.", ["Olga"]),
    "olga_run": ("Olga runs in the morning.", ["Olga"]),
    "olga_ielts": ("Olga scored 7.5 on IELTS.", ["Olga"]),
    "olga_kids": ("Olga takes the kids to their clubs after school.", ["Olga"]),
    "olga_ski": ("Olga goes skiing once a year.", ["Olga"]),
    "olga_car": ("Olga drives a small hatchback.", ["Olga"]),

    # --- c3 subject dina: 7 facts (work / flat above + these) ---
    "dina_cat": ("Dina got a cat in March.", ["Dina"]),
    "dina_course": ("Dina is taking an accounting course this year.", ["Dina"]),
    "dina_suv": ("Dina drives a big SUV.", ["Dina"]),
    "dina_trip": ("Dina is planning a trip to Georgia in the fall.", ["Dina"]),
    "dina_serials": ("Dina binges TV shows on Fridays.", ["Dina"]),

    # --- common fillers: crowd for top-5 and chit-chat distractors ---
    "filler_train": ("The train to St. Petersburg leaves at 07:30.", []),
    "filler_coffee": ("Coffee at the office got a third more expensive.", []),
    "filler_berries": ("Berry season at the dacha starts in July.", []),
    "filler_sneakers": ("The new sneakers drop in September.", []),
    "filler_market": ("The Saturday market runs until two in the afternoon.", []),
    "filler_winter_tires": ("Winter tires go on after the first frost.", []),
}

# 15 pairs = 30 queries. Paired metric: both queries must hit their target.
PAIRS = [
    # ---------- C1: swap the differing token (the name) ----------
    {"id": "c1e-1", "type": "c1_entity", "queries": [
        ("When is Peter's birthday?", "peter_bd"),
        ("When is Lena's birthday?", "lena_bd")],
     "note": "The names separate the targets. names_fit ON: the base passes "
             "both (the wrong birthday is cut by the name); OFF: risk of "
             "taking someone else's birthday — the rerank gain shows only on "
             "OFF. Separate date facts (wedding, deadline, vaccination) load "
             "the type so a date alone is not enough."},
    {"id": "c1e-2", "type": "c1_entity", "queries": [
        ("Where does Denis work?", "denis_work"),
        ("Where does Dina work?", "dina_work")],
     "note": "Both targets contain Norilsk Nickel — the shared content cannot "
             "discriminate, only the name does. Traps: sergey_nornickel and "
             "nornickel_office share the company with BOTH sides, so BM25 "
             "ranks them on par with the targets."},
    {"id": "c1e-3", "type": "c1_entity", "queries": [
        ("When does Sasha go to the pool?", "sasha_pool"),
        ("When does Sonya go to the pool?", "sonya_pool")],
     "note": "Sasha/Sonya — close names, same verb and the same content "
             "(pool); the facts differ only in the day of the week. Other "
             "pool facts (masha_pool, pool_hours) share the query word pool."},

    # ---------- C2: paraphrase bridge (2 numeric + 3 plain) ----------
    {"id": "c2n-1", "type": "c2", "subtype": "number", "queries": [
        ("When did Denis buy a car?", "denis_car"),
        ("What did Denis do in the twenty-fourth year?", "denis_car")],
     "note": "B carries no digits: 2024 is spelled out as 'the twenty-fourth "
             "year', so BM25 can match on 'Denis' alone. Trap denis_sochi "
             "('the year 2023') shares the word 'year' and the name with the "
             "query; denis_tires shares only the name and still competes. "
             "A is the control (exact 'buy a car')."},
    {"id": "c2n-2", "type": "c2", "subtype": "number", "queries": [
        ("In what year did Dina buy her flat?", "dina_flat"),
        ("In the twenty-first year, what did Dina purchase?", "dina_flat")],
     "note": "B spells 2021 out as 'the twenty-first year' and says 'purchase' "
             "instead of 'bought' — the target shares only the name with the "
             "query. Trap dina_course ('this year') shares 'year' and the "
             "name, so the base can land on it. A is the control (exact "
             "'buy her flat')."},
    {"id": "c2-1", "type": "c2", "queries": [
        ("Where is Marina usually seen after dark?", "marina_night"),
        ("What time is Marina out on the street?", "marina_night")],
     "note": "A: trap marina_cams shares the word 'seen' with the query while "
             "the target shares only the name ('after dark' vs 'late at "
             "night'). B: trap marina_gagarin shares 'street'; the target is "
             "the park, matched by 'time' -> 'night' only."},
    {"id": "c2-2", "type": "c2", "queries": [
        ("Where in Thailand did Sergey actually go?", "sergey_pattaya"),
        ("What did Sergey choose instead of Bangkok?", "sergey_pattaya")],
     "note": "A: 'Thailand' never appears in the target ('flew to Pattaya "
             "instead of Bangkok') — the bridge word is only in the trap "
             "sergey_fear. B is the control (exact 'Bangkok')."},
    {"id": "c2-3", "type": "c2", "queries": [
        ("Why did the Wi-Fi go out in the evening?", "net_off"),
        ("What happened to the internet in the flat?", "net_off")],
     "note": "A: 'Wi-Fi' vs 'internet' — no shared root, so the target scores "
             "on nothing; trap light_evening shares 'evening' and even 'cut "
             "off'. B is the control ('internet', 'flat')."},

    # ---------- C3: one-subject predicate ----------
    {"id": "c3-1", "type": "c3", "queries": [
        ("What does Marina do late at night?", "marina_night"),
        ("What does Marina do at the office?", "marina_office")],
     "note": "7 facts about Marina, all hanging on her name equally "
             "(documented weakness of retrieval.py:173-176): top-5 out of 7 "
             "cuts one of the two targets and the base loses the pair."},
    {"id": "c3-2", "type": "c3", "queries": [
        ("In what year does Denis's passport expire?", "denis_passport"),
        ("When does Denis's driving license run out?", "denis_license")],
     "note": "7 facts about Denis, both queries are temporal — only the "
             "predicate separates them (passport / license). Traps: "
             "denis_rights2019 also answers 'when did he get his license', "
             "denis_car and denis_sochi also carry a year."},
    {"id": "c3-3", "type": "c3", "queries": [
        ("What does Andrey do on weekends?", "andrey_dacha"),
        ("What time does Andrey's Friday meeting start?", "andrey_meet")],
     "note": "6 facts about Andrey; 'weekends' lands on andrey_dacha exactly, "
             "but every other Andrey fact crowds top-5 on the name alone. "
             "Target B is the time (11:00); andrey_dentist sits on Saturday "
             "instead of Friday."},
    {"id": "c3-4", "type": "c3", "queries": [
        ("What does Sasha do on Saturdays?", "sasha_chess"),
        ("What does Sonya do in the mornings?", "sonya_yoga")],
     "note": "WEAKER than the rest of c3 (2 facts per subject) — held "
             "together only by the Sasha/Sonya name similarity. Kept on "
             "purpose as the easy control pair, the one pair exempt from the "
             "6-facts-per-subject rule."},
    {"id": "c3-5", "type": "c3", "queries": [
        ("What does Sergey do in the evenings?", "sergey_guitar"),
        ("What animal does Sergey keep?", "sergey_dog")],
     "note": "7 facts about Sergey; B says 'animal', a word that never "
             "appears in the target 'dog' (bridge), with sergey_fear as a "
             "name-only trap. A: 'evenings' sits only in sergey_guitar, but "
             "the other six Sergey facts fight for top-5."},
    {"id": "c3-6", "type": "c3", "queries": [
        ("What kind of car does Dina drive?", "dina_suv"),
        ("What is Dina studying?", "dina_course")],
     "note": "7 facts about Dina; two different predicates (car / studies). "
             "Traps dina_work, dina_trip and dina_serials share her name and "
             "crowd top-5; dina_flat is claimed by pair c2n-2."},
    {"id": "c3-7", "type": "c3", "queries": [
        ("What does Olga do in the morning?", "olga_run"),
        ("Which exam did Olga take?", "olga_ielts")],
     "note": "7 facts about Olga; A: 'morning' points at olga_run, yet every "
             "Olga fact ranks equally on the name. B: 'exam' has no shared "
             "root with 'IELTS' (bridge) — only olga_kids/olga_ski-level "
             "name traps compete."},
]

# 45 messages (15 per category), all must return an EMPTY recall.
# Rule of three: 0 noise at n=45 -> noise < 7% per language.
CHITCHAT = [
    # greetings with an entity (shared words exist -> coverage admits,
    # gate_reads_score must still refuse on score)
    ("Hi, Marina! How's it going?", "greeting_entity"),
    ("Morning, Denis — did you sleep well?", "greeting_entity"),
    ("Olga, hello, how are you?", "greeting_entity"),
    ("Sergey, are you home yet?", "greeting_entity"),
    ("Good evening, Andrey!", "greeting_entity"),
    ("Sasha, how's life?", "greeting_entity"),
    ("Dina, can I ask you something?", "greeting_entity"),
    ("Lena, tell me later.", "greeting_entity"),
    ("Sonya, is that you?", "greeting_entity"),
    ("Hi there, Peter!", "greeting_entity"),
    ("Masha, got a minute?", "greeting_entity"),
    ("Hey Marina, you around?", "greeting_entity"),
    ("Denis, what's new?", "greeting_entity"),
    ("Happy Monday, Andrey!", "greeting_entity"),
    ("Olga, long time no see!", "greeting_entity"),
    # small talk sharing words with the facts (coverage trap, false recall)
    ("It's always noisy in the park at night.", "topic_shared_words"),
    ("The pool is freezing in winter.", "topic_shared_words"),
    ("Norilsk Nickel is hiring again.", "topic_shared_words"),
    ("Everyone says 2024 flew by.", "topic_shared_words"),
    ("It always rains in May.", "topic_shared_words"),
    ("There are always queues at the office on weekdays.", "topic_shared_words"),
    ("Traffic on Pyatnitskaya is a nightmare today.", "topic_shared_words"),
    ("The cat next door yowls at night.", "topic_shared_words"),
    ("Is it worth renewing your license early?", "topic_shared_words"),
    ("They ask for your passport everywhere now.", "topic_shared_words"),
    ("Yoga in the morning is all the rage.", "topic_shared_words"),
    ("Everyone travels somewhere on weekends.", "topic_shared_words"),
    ("Summer in the office is so stuffy.", "topic_shared_words"),
    ("Winter tires are such a hassle.", "topic_shared_words"),
    ("Coffee at the office got crazy expensive.", "topic_shared_words"),
    # irrelevant generic tail
    ("How's the weather this week?", "vague"),
    ("What should I watch tonight?", "vague"),
    ("Who said that?", "vague"),
    ("Tell me a joke.", "vague"),
    ("What day is it today?", "vague"),
    ("Any idea what time it is?", "vague"),
    ("Help me pick a movie.", "vague"),
    ("What is photosynthesis?", "vague"),
    ("Translate this sentence for me.", "vague"),
    ("How do I minimize a window?", "vague"),
    ("Remind me how recursion works.", "vague"),
    ("What's the humidity right now?", "vague"),
    ("How do I set up a fan?", "vague"),
    ("Find me some music for the mood.", "vague"),
    ("Which file format is better?", "vague"),
]


def self_check():
    """Prints the plan and catches data errors. Run: python contrast_en.py"""
    errors = []
    seen = {}
    for p in PAIRS:
        if len(p["queries"]) != 2:
            errors.append(f'{p["id"]}: expected exactly 2 queries')
        for q, key in p["queries"]:
            if key not in FACTS:
                errors.append(f'{p["id"]}: no fact {key!r}')
            if q in seen:
                errors.append(f'{p["id"]}: query duplicates {seen[q]!r}: {q!r}')
            seen[q] = p["id"]
    chit_seen = {}
    for q, cat in CHITCHAT:
        if q in seen:
            errors.append(f"chit-chat duplicates {seen[q]!r}: {q!r}")
        if q in chit_seen:
            errors.append(f"chit-chat duplicates itself: {q!r}")
        chit_seen[q] = True
    subj = {}
    for key, (text, ents) in FACTS.items():
        for e in ents:
            subj.setdefault(e, []).append(key)

    # --- composition: 3 c1_entity + 5 c2 (2 numeric) + 7 c3, 15/15/15 chit ---
    by_type = {}
    for p in PAIRS:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    n_num = sum(1 for p in PAIRS if p.get("subtype") == "number")
    by_cat = {}
    for q, cat in CHITCHAT:
        by_cat[cat] = by_cat.get(cat, 0) + 1

    n_pairs = len(PAIRS)
    n_q = sum(len(p["queries"]) for p in PAIRS)
    print(f"facts in store: {len(FACTS)}")
    print(f"pairs: {n_pairs} (queries: {n_q}), chit-chat: {len(CHITCHAT)}")
    print("by type:", by_type, f"(subtype=number: {n_num})")
    print("chit-chat by category:", by_cat)
    print("subject -> number of facts (c3 needs >=6, except the weak control pair):")
    for e, keys in sorted(subj.items(), key=lambda kv: -len(kv[1])):
        print(f"   {e}: {len(keys)}")
    if n_pairs != 15:
        errors.append(f"expected 15 pairs, got {n_pairs}")
    if n_q != 30:
        errors.append(f"expected 30 queries, got {n_q}")
    if len(CHITCHAT) != 45:
        errors.append(f"expected 45 chit-chat, got {len(CHITCHAT)}")
    if by_type.get("c1_entity") != 3:
        errors.append(f"expected 3 c1_entity pairs, got {by_type.get('c1_entity', 0)}")
    if by_type.get("c2") != 5:
        errors.append(f"expected 5 c2 pairs, got {by_type.get('c2', 0)}")
    if n_num != 2:
        errors.append(f"expected 2 c2 pairs with subtype='number', got {n_num}")
    if by_type.get("c3") != 7:
        errors.append(f"expected 7 c3 pairs, got {by_type.get('c3', 0)}")
    for cat, want in (("greeting_entity", 15), ("topic_shared_words", 15), ("vague", 15)):
        if by_cat.get(cat, 0) != want:
            errors.append(f"expected {want} chit-chat {cat}, got {by_cat.get(cat, 0)}")

    # --- c3 subject load: >=6 facts each, one pair is the weak control ---
    weak_control = {"c3-4"}  # Sasha/Sonya, 2 facts each — deliberately easy
    for p in PAIRS:
        if p["type"] != "c3" or p["id"] in weak_control:
            continue
        for q, key in p["queries"]:
            for e in FACTS[key][1]:
                if len(subj[e]) < 6:
                    errors.append(
                        f'{p["id"]}: subject {e!r} has only {len(subj[e])} facts (need >=6)')

    print("DATA ERRORS:" if errors else "No data errors.")
    for e in errors:
        print("  -", e)
    return not errors


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)