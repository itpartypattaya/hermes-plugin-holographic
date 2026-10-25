# -*- coding: utf-8 -*-
"""Contrast set RU for tools/benchmark_rerank.py (RFC #9, gates 2 and 3).

Design (see cross-encoder-rerank/references/validation-datasets.md):
  - every PAIRS entry is a minimal PAIR: two queries differing by a single
    factor; both targets live in the store; the metric is PAIRED (both-or-nothing);
  - the store is shared by all contrasts (~70 facts); the distractors of one
    contrast are fillers of the other;
  - CHITCHAT — 15 queries that MUST recall nothing (side 2
    of contrast C5 for gate_reads_score; side 1 — the C2 "hard" queries);
  - the expected behaviour of the base (off) and the traps are recorded in the `note` field —
    these are design ASSESSMENTS, a mandatory check: run the off mode and make sure
    the base actually fails the pairs (the self-check below prints the plan).

Types: c1_entity — a name swap; c2 (+subtype number) — a paraphrase bridge with no
shared roots; c3 — a predicate over a single subject (the documented weakness
retrieval.py:173-176).

Synthetic data, written for PR itpartypattaya/hermes-plugin-holographic#10.
"""

FACTS = {
    # --- c1e-1: birthdays (one "date" type so the type gets loaded) ---
    "peter_bd": ("День рождения Петра — 12 марта.", ["Петр"]),
    "lena_bd": ("День рождения Лены — 3 июня.", ["Лена"]),
    "olga_wedding": ("Свадьба Ольги — 18 августа 2023 года.", ["Ольга"]),
    "app_deadline": ("Крайний срок подачи заявки — 25 сентября.", []),
    "barsik_shot": ("Прививка кота Барсика — 14 ноября.", ["Барсик"]),

    # --- c1e-2: work (the shared word Nornickel in both targets) ---
    "denis_work": ("Денис работает в Норникеле в Москве.", ["Денис"]),
    "dina_work": ("Дина работает в Норникеле в Екатеринбурге.", ["Дина"]),
    "sergey_nornickel": ("Сергей уволился из Норникеля в 2022 году.", ["Сергей"]),
    "nornickel_office": ("Офис Норникеля на Пятницкой закрыт на ремонт.", []),

    # --- c1e-3: pool (Sasha/Sonya: verb-form agreement) ---
    "sasha_pool": ("Саша ходит в бассейн по вторникам.", ["Саша"]),
    "sonya_pool": ("Соня ходит в бассейн по четвергам.", ["Соня"]),
    "masha_pool": ("Маша записалась в бассейн на январь.", ["Маша"]),
    "pool_hours": ("Бассейн на Спортивной открыт до десяти вечера.", []),
    "sasha_chess": ("Саша играет в шахматы по субботам.", ["Саша"]),
    "sonya_yoga": ("Соня занимается йогой по утрам.", ["Соня"]),

    # --- c2-numeric: digits in the fact, spelled out in the query ---
    "denis_car": ("Денис купил машину в 2024 году.", ["Денис"]),
    "denis_tires": ("Денис менял зимнюю резину в октябре.", ["Денис"]),
    "denis_sochi": ("Денис отдыхал в Сочи летом.", ["Денис"]),
    "denis_passport": ("Паспорт Дениса истекает в 2027 году.", ["Денис"]),
    "denis_license": ("Водительские права Дениса истекают в 2025 году.", ["Денис"]),
    "denis_rights2019": ("Денис получил права в 2019 году.", ["Денис"]),
    "dina_flat": ("Дина купила квартиру в 2021 году.", ["Дина"]),
    "andrey_vacation": ("Отпуск Андрея — с 7 по 20 мая.", ["Андрей"]),
    "andrey_flex": ("Андрей работает по гибкому графику.", ["Андрей"]),
    "andrey_meet": ("Пятничные встречи Андрея в 11:00.", ["Андрей"]),
    "andrey_dacha": ("Андрей отдыхает на даче по выходным.", ["Андрей"]),
    "andrey_dentist": ("Андрей записан к стоматологу на субботу.", ["Андрей"]),
    "andrey_cat": ("Андрей кормит кота дважды в день.", ["Андрей"]),

    # --- c2: paraphrase bridge (the target shares no query words; the trap does) ---
    "marina_night": ("Марина гуляет по ночам в парке.", ["Марина"]),
    "marina_cams": ("Марина видна по камерам у метро.", ["Марина"]),
    "marina_gagarin": ("Марина живёт на улице Гагарина.", ["Марина"]),
    "marina_chinese": ("Марина изучает китайский по вечерам.", ["Марина"]),
    "marina_office": ("Марина работает в офисе по будням.", ["Марина"]),
    "marina_diet": ("Марина не ест мясо.", ["Марина"]),
    "marina_car": ("Марина водит серебристый седан.", ["Марина"]),
    "sergey_pattaya": ("Сергей поехал в Паттайю вместо Бангкока.", ["Сергей"]),
    "sergey_fear": ("Сергей боялся лететь в Таиланд.", ["Сергей"]),
    "sergey_dog": ("Сергей выгуливает собаку дважды в день.", ["Сергей"]),
    "sergey_guitar": ("Сергей учится играть на гитаре по вечерам.", ["Сергей"]),
    "net_off": ("Интернет в квартире отключили из-за неуплаты.", []),
    "light_evening": ("Вечером в квартире отключают свет по графику.", []),
    "provider": ("Провайдер пообещал вернуть скорость через неделю.", []),
    "masha_dress": ("Маша купила платье за 4 000 рублей.", ["Маша"]),
    "masha_return": ("Маша вернула покупку в магазин.", ["Маша"]),
    "masha_savings": ("Маша экономит на продуктах каждый месяц.", ["Маша"]),

    # --- c3: predicate over a single subject (the subject must carry
    #     at least 6 facts, else top-5 does not cut and the base never fails) ---
    "olga_teach": ("Ольга преподаёт математику.", ["Ольга"]),
    "olga_run": ("Ольга бегает по утрам.", ["Ольга"]),
    "olga_ielts": ("Ольга сдала IELTS на 7.5.", ["Ольга"]),
    "olga_kids": ("Ольга водит детей в секцию после школы.", ["Ольга"]),
    "olga_ski": ("Ольга катается на лыжах раз в год.", ["Ольга"]),
    "olga_car": ("Ольга водит маленький хэтчбек.", ["Ольга"]),
    "dina_cat": ("Дина завела кота в марте.", ["Дина"]),
    "dina_course": ("Дина учится на курсах бухгалтеров.", ["Дина"]),
    "dina_suv": ("Дина водит большой внедорожник.", ["Дина"]),
    "dina_trip": ("Дина планирует поездку в Грузию осенью.", ["Дина"]),
    "dina_serials": ("Дина смотрит сериалы по пятницам.", ["Дина"]),
    "sergey_market": ("Сергей ходит на рынок по субботам.", ["Сергей"]),
    "sergey_flat": ("Сергей снимает квартиру рядом с метро.", ["Сергей"]),

    # --- shared fillers: crowd for top-5 and chit-chat distractors ---
    "filler_train": ("Поезд в Питер отправляется в 07:30.", []),
    "filler_coffee": ("Кофе в офисе подорожал на треть.", []),
    "filler_berries": ("Сезон дачных ягод начинается в июле.", []),
    "filler_sneakers": ("Новые кроссовки выходят в сентябре.", []),
    "filler_market": ("Субботний рынок работает до двух дня.", []),
    "filler_winter_tires": ("Зимнюю резину ставят после первых заморозков.", []),
}

# 15 pairs = 30 queries. Paired metric: both queries must hit the target.
PAIRS = [
    # ---------- C1: swap of the discriminating token ----------
    {"id": "c1e-1", "type": "c1_entity", "queries": [
        ("Когда день рождения Петра?", "peter_bd"),
        ("Когда день рождения Лены?", "lena_bd")],
     "note": "Имена различают цели. names_fit ON: база проходит обе (ошибочная "
             "цель отсекается именем); OFF: риск взять чужое birthday — выигрыш "
             "rerank виден только на OFF. Отдельные факты-даты нагружают тип."},
    {"id": "c1e-2", "type": "c1_entity", "queries": [
        ("Где работает Денис?", "denis_work"),
        ("Где работает Дина?", "dina_work")],
     "note": "Обе цели содержат Норникель — общий контент не различает; "
             "различие только в имени. Ловушка: Sergey_nornickel делится "
             "Норникелем и с той, и с другой."},
    {"id": "c1e-3", "type": "c1_entity", "queries": [
        ("Когда Саша ходит в бассейн?", "sasha_pool"),
        ("Когда Соня ходит в бассейн?", "sonya_pool")],
     "note": "Саша/Соня — близкие имена + согласование (ходил/ходила). "
             "Факты отличаются только днём недели; прочие факты бассейна "
             "делят слова запроса (бассейн)."},

    # ---------- C2: paraphrase bridge (2 numeric + 3 plain) ----------
    {"id": "c2n-1", "type": "c2", "subtype": "number", "queries": [
        ("Когда Денис купил машину?", "denis_car"),
        ("Что Денис сделал в двадцать четвёртом году?", "denis_car")],
     "note": "Запрос B не содержит 2024 и «год»-цифр: ловушки denis_tires / "
             "denis_sochi делят «Денис» и участвуют в BM25 наравне с целью. "
             "A — контроль (точное «купил машину»)."},
    {"id": "c2n-2", "type": "c2", "subtype": "number", "queries": [
        ("Когда начинается отдых Андрея?", "andrey_vacation"),
        ("Когда у Андрея отпуск?", "andrey_vacation")],
     "note": "B — контроль (точное «отпуск»). A: «отдых» vs «отпуск» — разные "
             "корни; ловушка andrey_dacha («отдыхает») лексически ближе цели."},
    {"id": "c2-1", "type": "c2", "queries": [
        ("Где обычно видят Марину после полуночи?", "marina_night"),
        ("Во сколько Марина бывает на улице?", "marina_night")],
     "note": "A: ловушка marina_cams делится «видят/видна»; цель делится только "
             "именем. B: ловушка marina_gagarin делится «улице»; цель — «парк»."},
    {"id": "c2-2", "type": "c2", "queries": [
        ("Где оказался Сергей вместо столицы Таиланда?", "sergey_pattaya"),
        ("Что Сергей выбрал вместо Бангкока?", "sergey_pattaya")],
     "note": "A: «столица Таиланда» ≠ Бангкок по словам; ловушка sergey_fear "
             "делится «Таиланд». B — контроль (точный Бангкок)."},
    {"id": "c2-3", "type": "c2", "queries": [
        ("Почему пропал вай-фай вечером?", "net_off"),
        ("Что случилось с интернетом в квартире?", "net_off")],
     "note": "A: «вай-фай» ≠ «интернет» по корню; ловушка light_evening делится "
             "«вечером/квартире». B — контроль («интернет», «квартире»)."},

    # ---------- C3: predicate over a single subject ----------
    {"id": "c3-1", "type": "c3", "queries": [
        ("Что Марина делает по ночам?", "marina_night"),
        ("Что Марина делает в офисе?", "marina_office")],
     "note": "7 фактов про Марину: все висят на «Марина» одинаково (слабость "
             "retrieval.py:173-176), top-5 из 7 режет — база может потерять цель."},
    {"id": "c3-2", "type": "c3", "queries": [
        ("В каком году истекает паспорт Дениса?", "denis_passport"),
        ("Когда Денис покупал машину?", "denis_car")],
     "note": "7 фактов про Дениса, два «временных» запроса; различает только "
             "предикат (паспорт/машину). Ловушки: denis_license, denis_rights2019 "
             "тоже отвечают «в каком году»."},
    {"id": "c3-3", "type": "c3", "queries": [
        ("Что Андрей делает по выходным?", "andrey_dacha"),
        ("Во сколько у Андрея встреча в пятницу?", "andrey_meet")],
     "note": "6 фактов про Андрея; «выходным» делится с andrey_dacha (точно), "
             "но и остальные Андреи-факты в top-5 наравне; цель B — время 11:00."},
    {"id": "c3-4", "type": "c3", "queries": [
        ("Чем Саша занимается по субботам?", "sasha_chess"),
        ("Чем Соня занимается по утрам?", "sonya_yoga")],
     "note": "Слабее прочих c3 (у каждого субъекта 2 факта) — держится на "
             "похожести Саша/Соня. Оставлен как контроль лёгкости."},
    {"id": "c3-5", "type": "c3", "queries": [
        ("Что Сергей делает по вечерам?", "sergey_guitar"),
        ("Какое животное содержит Сергей?", "sergey_dog")],
     "note": "5 фактов про Сергея; «вечерам» есть и у sergey_pattaya-контекста? "
             "нет — различает предикат; B использует «животное» ≠ «собака» "
             "по корню (мост) с ловушкой sergey_fear («Сергей»)."},
    {"id": "c3-6", "type": "c3", "queries": [
        ("Что Дина сделала в двадцать первом году?", "dina_flat"),
        ("На каком курсе учится Дина?", "dina_course")],
     "note": "5 фактов про Дину; A — числовая парафраза (2021 прописью), "
             "ловушки dina_work / dina_suv делят «Дина»; B — предикат."},
    {"id": "c3-7", "type": "c3", "queries": [
        ("Что Ольга делает по утрам?", "olga_run"),
        ("Какой экзамен сдала Ольга?", "olga_ielts")],
     "note": "5 фактов про Ольгу; A: «утрам» делится только с целью, но прочие "
             "Ольги-факты в top-5 наравне; B: «экзамен» ≠ IELTS по корню."},
]

# 45 queries (15 per category), must return EMPTY recall.
# Rule of three: 0 noise at n=45 → noise < 7% per language.
CHITCHAT = [
    # greetings with an entity (shared words with facts exist → coverage admits,
    # gate_reads_score must refuse on score)
    ("Привет, Марина! Как дела?", "greeting_entity"),
    ("Марина, доброе утро!", "greeting_entity"),
    ("Денис, что нового?", "greeting_entity"),
    ("Ольга, привет, как ты?", "greeting_entity"),
    ("Сергей, ты уже дома?", "greeting_entity"),
    ("Марина, ты тут?", "greeting_entity"),
    ("Денис, привет!", "greeting_entity"),
    ("Добрый вечер, Андрей!", "greeting_entity"),
    ("Саша, как жизнь?", "greeting_entity"),
    ("Дина, дай совет.", "greeting_entity"),
    ("Лена, рассказывай потом.", "greeting_entity"),
    ("Соня, это ты?", "greeting_entity"),
    ("Привет, Петруся!", "greeting_entity"),
    ("Ольга, здравствуй!", "greeting_entity"),
    ("Андрей, ты где?", "greeting_entity"),
    # chatter sharing words with facts (coverage trap)
    ("По ночам в городе шумно.", "topic_shared_words"),
    ("В бассейн зимой идти холодно.", "topic_shared_words"),
    ("В Норникеле сейчас много вакансий.", "topic_shared_words"),
    ("Двадцать четвёртый год вообще странный.", "topic_shared_words"),
    ("В мае бывает дождливо.", "topic_shared_words"),
    ("По будням в офисе всегда очереди.", "topic_shared_words"),
    ("На Пятницкой сегодня пробки.", "topic_shared_words"),
    ("Коты по ночам шумят.", "topic_shared_words"),
    ("Стоит ли менять права заранее?", "topic_shared_words"),
    ("Паспорта сейчас везде требуют.", "topic_shared_words"),
    ("Зимой в бассейн реже попадаю.", "topic_shared_words"),
    ("Йога по утрам — модная тема.", "topic_shared_words"),
    ("Дачный сезон скоро откроется.", "topic_shared_words"),
    ("По выходным все куда-то едут.", "topic_shared_words"),
    ("В офисе летом душно.", "topic_shared_words"),
    # irrelevant shared tail
    ("Как тебе погода на этой неделе?", "vague"),
    ("Что посмотреть вечером?", "vague"),
    ("Кто это сказал?", "vague"),
    ("Расскажи анекдот.", "vague"),
    ("Какой сегодня день?", "vague"),
    ("Сколько будет завтра?", "vague"),
    ("Помоги выбрать фильм.", "vague"),
    ("Что такое фотосинтез?", "vague"),
    ("Переведи фразу на английский.", "vague"),
    ("Как свернуть окно в винде?", "vague"),
    ("Напомни, как работает рекурсия.", "vague"),
    ("Какая сейчас влажность?", "vague"),
    ("Как настроить вентилятор?", "vague"),
    ("Найди музыку под настроение.", "vague"),
    ("Какой формат файла лучше?", "vague"),
]


def self_check():
    """Prints the plan and catches data errors. Run: python contrast_ru.py"""
    errors = []
    seen = {}
    for p in PAIRS:
        if len(p["queries"]) != 2:
            errors.append(f'{p["id"]}: exactly 2 queries are required')
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

    n_pairs = len(PAIRS)
    n_q = sum(len(p["queries"]) for p in PAIRS)
    by_type = {}
    for p in PAIRS:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    print(f"facts in the store: {len(FACTS)}")
    print(f"pairs: {n_pairs} (queries: {n_q}), chit-chat: {len(CHITCHAT)}")
    print("by type:", by_type)
    print("subjects with their fact counts (c3 needs >=5, better >=6):")
    for e, keys in sorted(subj.items(), key=lambda kv: -len(kv[1])):
        print(f"   {e}: {len(keys)}")
    if n_pairs != 15:
        errors.append(f"expected 15 pairs, got {n_pairs}")
    if n_q != 30:
        errors.append(f"expected 30 queries, got {n_q}")
    if len(CHITCHAT) != 45:
        errors.append(f"expected 45 chit-chat, got {len(CHITCHAT)}")
    print("ERRORS:" if errors else "No data errors.")
    for e in errors:
        print("  -", e)
    return not errors


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)