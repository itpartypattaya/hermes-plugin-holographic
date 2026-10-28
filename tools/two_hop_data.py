# -*- coding: utf-8 -*-
"""Dataset of the two-hop links experiment (plans/2026-10-28-two-hop.md).

Synthetic and invented: no real person, company or household. Three store files under `scope: chat`:
  family  — two fictional households (English and Russian), ~300 facts
  work-a  — a fictional company (English and Russian), ~2000 facts, with a hub entity
  work-b  — another chat of the same company: the far end of the cross-scope chains

SCENARIO facts are labelled (key -> store, text, entities); FILLER facts are generated from templates that reuse the
questions' relation words ("responsible for", "отвечает за", "runs on", "holds", "проводит") so a lone relation word
is no give-away. OPS are applied after every add, in order (change / end / forget), through fact_store.

QUESTIONS: (qid, type, store, lang, text, gold keys, forbidden keys). Gold = every key must be in the block;
forbidden = must never be in it. Held-out = index % 5 in {1, 3}.
"""
from __future__ import annotations

import itertools
import random

STORES = ("family", "work-a", "work-b")

# key -> (store, text, entities)
SCENARIO = {
    # ---------------- 2-hop, work, English ----------------
    "alpha_on": ("work-a", "Project Alpha runs on the server vega-01.", ["Alpha", "vega-01"]),
    "vega_resp": ("work-a", "Karen Holt is responsible for vega-01.", ["Karen Holt", "vega-01"]),
    "billing_db": ("work-a", "The Billing service stores its data in the database pg-ledger.", ["Billing", "pg-ledger"]),
    "ledger_backup": ("work-a", "pg-ledger is backed up every night at 02:00.", ["pg-ledger"]),
    "harbor_lead": ("work-a", "Tom Reyes leads the Harbor team.", ["Tom Reyes", "Harbor"]),
    "harbor_standup": ("work-a", "The Harbor team holds its stand-up at 10:30 on weekdays.", ["Harbor"]),
    "lynx_courier": ("work-a", "The mobile app Lynx sends push notifications through the service Courier.",
                     ["Lynx", "Courier"]),
    "courier_contract": ("work-a", "Courier's vendor contract renews on March 1.", ["Courier"]),
    # ---------------- 2-hop, work, Russian ----------------
    "borey_on": ("work-a", "Проект Борей работает на сервере кедр-2.", ["Борей", "кедр-2"]),
    "kedr_resp": ("work-a", "За сервер кедр-2 отвечает Инна Соколова.", ["Инна Соколова", "кедр-2"]),
    "kashtan_db": ("work-a", "Сервис Каштан хранит данные в базе orion-db.", ["Каштан", "orion-db"]),
    "orion_backup": ("work-a", "Резервная копия orion-db делается каждое воскресенье.", ["orion-db"]),
    "priboy_lead": ("work-a", "Команду Прибой возглавляет Олег Ветров.", ["Прибой", "Олег Ветров"]),
    "priboy_meeting": ("work-a", "Планёрка команды Прибой проходит по средам в 11:00.", ["Прибой"]),
    "soyka_golub": ("work-a", "Приложение Сойка отправляет письма через сервис Голубь.", ["Сойка", "Голубь"]),
    "golub_contract": ("work-a", "Договор с поставщиком сервиса Голубь продлевается 1 апреля.", ["Голубь"]),
    # ---------------- 2-hop, family, English ----------------
    "mia_piano": ("family", "Mia's piano teacher is Clara Voss.", ["Mia", "Clara Voss"]),
    "voss_lessons": ("family", "Clara Voss gives lessons at the Linden music school on Thursdays.", ["Clara Voss"]),
    "leo_doctor": ("family", "Leo's pediatrician is Dr. Ahmed Noor.", ["Leo", "Ahmed Noor"]),
    "noor_days": ("family", "Dr. Ahmed Noor sees patients at the Riverside clinic on Mondays and Wednesdays.",
                  ["Ahmed Noor"]),
    "pepper_groomer": ("family", "Our dog Pepper goes to the groomer Tidy Paws.", ["Pepper", "Tidy Paws"]),
    "tidy_closed": ("family", "Tidy Paws is closed on Sundays.", ["Tidy Paws"]),
    # ---------------- 2-hop, family, Russian ----------------
    "varya_coach": ("family", "Тренер Вари по плаванию — Глеб Орлов.", ["Варя", "Глеб Орлов"]),
    "orlov_pool": ("family", "Глеб Орлов ведёт тренировки в бассейне «Волна».", ["Глеб Орлов"]),
    "timur_doctor": ("family", "Педиатр Тимура — Нина Белова.", ["Тимур", "Нина Белова"]),
    "belova_days": ("family", "Нина Белова принимает по вторникам в поликлинике на Садовой.", ["Нина Белова"]),
    "pushok_food": ("family", "Кот Пушок ест корм марки Мурлыка.", ["Пушок", "Мурлыка"]),
    "murlyka_shop": ("family", "Корм Мурлыка продаётся в зоомагазине у рынка.", ["Мурлыка"]),
    # ---------------- 1-hop controls (their own facts; the 2-hop facts above serve too) ----------------
    "mia_dentist": ("family", "Mia sees the dentist Dr. Hale on November 12.", ["Mia", "Dr. Hale"]),
    "hale_office": ("family", "Dr. Hale's office needs an appointment two weeks ahead.", ["Dr. Hale"]),
    "leo_allergy": ("family", "Leo is allergic to peanuts.", ["Leo"]),
    "timur_school": ("family", "Тимур учится в школе №12 у учительницы Анны Ивановны Гром.", ["Тимур", "Анна Гром"]),
    "grom_meeting": ("family", "Анна Гром проводит родительское собрание в последний четверг месяца.", ["Анна Гром"]),
    # ---------------- namesakes: a one-word name two people share ----------------
    "mia_swim": ("family", "Mia's swimming coach is Anna.", ["Mia", "Anna"]),
    "anna_practice": ("family", "Anna holds practice at the Delfin pool on Saturdays.", ["Anna"]),
    "anna_neighbour": ("family", "Anna from next door holds the spare key to our flat.", ["Anna"]),
    "timur_tutor": ("family", "Репетитор Тимура по математике — Ольга.", ["Тимур", "Ольга"]),
    "olga_lessons": ("family", "Ольга проводит занятия по субботам в 10:00.", ["Ольга"]),
    "olga_neighbour": ("family", "Ольга с третьего этажа проводит собрания жильцов.", ["Ольга"]),
    "quartz_designer": ("work-a", "The Quartz project's designer is Sam.", ["Quartz", "Sam"]),
    "sam_mockups": ("work-a", "Sam reviews mockups every Friday.", ["Sam"]),
    "sam_accounting": ("work-a", "Sam from accounting reviews expense reports on Mondays.", ["Sam"]),
    "granit_analyst": ("work-a", "Аналитик проекта Гранит — Денис.", ["Гранит", "Денис"]),
    "denis_report": ("work-a", "Денис готовит отчёт по метрикам каждый понедельник.", ["Денис"]),
    "denis_accounting": ("work-a", "Денис из бухгалтерии готовит акты сверки в конце месяца.", ["Денис"]),
    # ---------------- changed / ended link ----------------
    "nimbus_on": ("work-a", "Project Nimbus runs on the server ash-03.", ["Nimbus", "ash-03"]),
    "ash_resp": ("work-a", "Rita Gomez is responsible for ash-03.", ["Rita Gomez", "ash-03"]),
    "elm_resp": ("work-a", "Paul Diaz is responsible for elm-07.", ["Paul Diaz", "elm-07"]),
    "lastochka_on": ("work-a", "Сервис Ласточка работает на сервере дуб-5.", ["Ласточка", "дуб-5"]),
    "dub_resp": ("work-a", "За сервер дуб-5 отвечает Марат Ким.", ["Марат Ким", "дуб-5"]),
    "leo_guitar": ("family", "Leo's guitar teacher is Ben Cole.", ["Leo", "Ben Cole"]),
    "cole_lessons": ("family", "Ben Cole holds lessons at the Oak Street studio.", ["Ben Cole"]),
    "reid_lessons": ("family", "Omar Reid holds lessons at the Harbour arts centre.", ["Omar Reid"]),
    "varya_art": ("family", "Варя ходит в художественную студию Палитра.", ["Варя", "Палитра"]),
    "palitra_hours": ("family", "Палитра открыта до 20:00.", ["Палитра"]),
    # ---------------- forgotten fact on the path ----------------
    "delta_on": ("work-a", "Project Delta runs on the server fir-09.", ["Delta", "fir-09"]),
    "fir_resp": ("work-a", "Nadia Brook is responsible for fir-09.", ["Nadia Brook", "fir-09"]),
    "varya_dentist": ("family", "Стоматолог Вари — Елена Жук.", ["Варя", "Елена Жук"]),
    "zhuk_days": ("family", "Елена Жук ведёт приём в клинике Улыбка по пятницам.", ["Елена Жук"]),
    "mia_friend": ("family", "Mia's best friend is Zoe Grant.", ["Mia", "Zoe Grant"]),
    "zoe_party": ("family", "Zoe Grant's birthday party is on December 3.", ["Zoe Grant"]),
    # ---------------- cross-scope: the far end lives in another chat's file ----------------
    "orbit_on": ("work-a", "Project Orbit runs on the server pine-04.", ["Orbit", "pine-04"]),
    "pine_resp": ("work-b", "Lena Park is responsible for pine-04.", ["Lena Park", "pine-04"]),
    "klever_db": ("work-a", "Сервис Клевер хранит данные в базе vega-db.", ["Клевер", "vega-db"]),
    "vegadb_backup": ("work-b", "Резервная копия vega-db делается ежедневно в 03:00.", ["vega-db"]),
    "comet_relay": ("work-a", "The Comet app sends emails through the service Relay.", ["Comet", "Relay"]),
    "relay_contract": ("work-b", "Relay's vendor contract renews on June 15.", ["Relay"]),
    # ---------------- no answer / inadmissible inference ----------------
    "kim_alpha": ("work-a", "Kim Walsh works on project Alpha.", ["Kim Walsh", "Alpha"]),
    "kim_pays": ("work-a", "Kim Walsh pays the team's lunch bills on Fridays.", ["Kim Walsh"]),
    "vetrov_borey": ("work-a", "Олег Ветров работает над проектом Борей.", ["Олег Ветров", "Борей"]),
    "vetrov_pays": ("work-a", "Олег Ветров оплачивает обеды команды по пятницам.", ["Олег Ветров"]),
}

# Applied in order after every fact is stored: (action, key, new text, entities)
OPS = [
    ("change", "nimbus_on", "Project Nimbus runs on the server elm-07.", ["Nimbus", "elm-07"]),
    ("end", "lastochka_on", None, None),
    ("change", "leo_guitar", "Leo's guitar teacher is Omar Reid.", ["Leo", "Omar Reid"]),
    ("end", "varya_art", None, None),
    ("forget", "delta_on", None, None),
    ("forget", "varya_dentist", None, None),
    ("forget", "mia_friend", None, None),
]

# (qid, type, store, lang, question, gold, forbidden) — types interleaved so the held-out subset gets each type
QUESTIONS = [
    ("q01", "two_hop", "work-a", "en", "Who is responsible for the server that project Alpha runs on?",
     ["alpha_on", "vega_resp"], []),
    ("q02", "two_hop", "work-a", "ru", "Кто отвечает за сервер, на котором работает Борей?",
     ["borey_on", "kedr_resp"], []),
    ("q03", "one_hop", "work-a", "en", "Who is responsible for vega-01?", ["vega_resp"], []),
    ("q04", "namesake", "family", "en", "Where does Mia's swimming coach hold practice?",
     ["mia_swim", "anna_practice"], []),
    ("q05", "changed", "work-a", "en", "Who is responsible for the server project Nimbus runs on?",
     ["nimbus_on", "elm_resp"], ["ash_resp"]),
    ("q06", "two_hop", "work-a", "en", "When is the database of the Billing service backed up?",
     ["billing_db", "ledger_backup"], []),
    ("q07", "forgotten", "work-a", "en", "Who is responsible for the server project Delta runs on?",
     [], ["delta_on", "fir_resp"]),
    ("q08", "cross_scope", "work-a", "en", "Who is responsible for the server project Orbit runs on?",
     ["orbit_on"], ["pine_resp"]),
    ("q09", "no_answer", "work-a", "en", "Who pays for the server of project Alpha?", [], []),
    ("q10", "two_hop", "work-a", "ru", "Когда делается резервная копия базы сервиса Каштан?",
     ["kashtan_db", "orion_backup"], []),
    ("q11", "one_hop", "work-a", "ru", "Кто отвечает за сервер кедр-2?", ["kedr_resp"], []),
    ("q12", "two_hop", "work-a", "en", "What time is the stand-up of the team Tom Reyes leads?",
     ["harbor_lead", "harbor_standup"], []),
    ("q13", "namesake", "family", "ru", "Когда проводит занятия репетитор Тимура по математике?",
     ["timur_tutor", "olga_lessons"], []),
    ("q14", "changed", "work-a", "ru", "Кто отвечает за сервер, на котором работает Ласточка?",
     [], ["lastochka_on", "dub_resp"]),
    ("q15", "two_hop", "work-a", "ru", "Когда планёрка у команды, которую возглавляет Олег Ветров?",
     ["priboy_lead", "priboy_meeting"], []),
    ("q16", "forgotten", "family", "ru", "Когда ведёт приём стоматолог Вари?", [], ["varya_dentist", "zhuk_days"]),
    ("q17", "one_hop", "family", "en", "When is Mia's next dentist appointment?", ["mia_dentist"], []),
    ("q18", "cross_scope", "work-a", "ru", "Когда делается резервная копия базы сервиса Клевер?",
     ["klever_db"], ["vegadb_backup"]),
    ("q19", "two_hop", "work-a", "en", "When does the vendor contract renew for the service Lynx sends notifications "
     "through?", ["lynx_courier", "courier_contract"], []),
    ("q20", "no_answer", "work-a", "ru", "Кто оплачивает сервер проекта Борей?", [], []),
    ("q21", "two_hop", "work-a", "ru", "Когда продлевается договор сервиса, через который Сойка отправляет письма?",
     ["soyka_golub", "golub_contract"], []),
    ("q22", "one_hop", "work-a", "ru", "Когда планёрка у команды Прибой?", ["priboy_meeting"], []),
    ("q23", "namesake", "work-a", "en", "When does the Quartz designer review mockups?",
     ["quartz_designer", "sam_mockups"], []),
    ("q24", "changed", "family", "en", "Where are the lessons of Leo's guitar teacher held?",
     ["leo_guitar", "reid_lessons"], ["cole_lessons"]),
    ("q25", "two_hop", "family", "en", "Where does Mia's piano teacher give lessons?", ["mia_piano", "voss_lessons"], []),
    ("q26", "forgotten", "family", "en", "When is the birthday party of Mia's best friend?",
     [], ["mia_friend"]),
    ("q27", "one_hop", "family", "en", "Is Leo allergic to anything?", ["leo_allergy"], []),
    ("q28", "cross_scope", "work-a", "en", "When does the vendor contract renew for the service Comet sends emails "
     "through?", ["comet_relay"], ["relay_contract"]),
    ("q29", "two_hop", "family", "en", "On which days does Leo's pediatrician see patients?",
     ["leo_doctor", "noor_days"], []),
    ("q30", "no_answer", "family", "en", "What is the name of the dog of Leo's pediatrician?", [], []),
    ("q31", "two_hop", "family", "en", "When is the place where Pepper gets groomed closed?",
     ["pepper_groomer", "tidy_closed"], []),
    ("q32", "one_hop", "family", "ru", "Кто педиатр Тимура?", ["timur_doctor"], []),
    ("q33", "namesake", "work-a", "ru", "Когда аналитик проекта Гранит готовит отчёт?",
     ["granit_analyst", "denis_report"], []),
    ("q34", "changed", "family", "ru", "До скольки открыто место, куда ходит Варя?", [], ["varya_art", "palitra_hours"]),
    ("q35", "two_hop", "family", "ru", "В каком бассейне ведёт тренировки тренер Вари по плаванию?",
     ["varya_coach", "orlov_pool"], []),
    ("q36", "one_hop", "family", "ru", "Где учится Тимур?", ["timur_school"], []),
    ("q37", "two_hop", "family", "ru", "По каким дням принимает педиатр Тимура?", ["timur_doctor", "belova_days"], []),
    ("q38", "no_answer", "family", "ru", "Какая машина у тренера Вари по плаванию?", [], []),
    ("q39", "two_hop", "family", "ru", "Где продаётся корм, который ест Пушок?", ["pushok_food", "murlyka_shop"], []),
    ("q40", "one_hop", "work-a", "en", "Which team does Tom Reyes lead?", ["harbor_lead"], []),
]


def held_out(index: int) -> bool:
    return index % 5 in (1, 3)


# ---------------------------------------------------------------------------------------------------------------
# Filler. Deterministic (seed 20261028). Names are invented and distinct from the scenario's.

EN_FIRST = ["Adam", "Bella", "Carl", "Dana", "Eli", "Fiona", "Gus", "Hana", "Ian", "Jade", "Kurt", "Lola", "Milo",
            "Nora", "Owen", "Pia", "Quinn", "Rosa", "Seth", "Tara", "Uma", "Vic", "Wes", "Xena", "Yuri", "Zara"]
EN_LAST = ["Abbott", "Baird", "Crane", "Dunn", "Ellis", "Frost", "Gale", "Hart", "Irwin", "Joyce", "Knox", "Lane",
           "Mead", "Nash", "Oakes", "Pratt", "Quill", "Rowe", "Shaw", "Thorne"]
RU_MALE = ["Аркадий", "Вадим", "Давид", "Захар", "Илья", "Лев", "Никита", "Пётр", "Савва", "Фёдор", "Эрик", "Ярослав"]
RU_FEMALE = ["Белла", "Галина", "Ева", "Жанна", "Кира", "Мила", "Оксана", "Рената", "Таисия", "Ульяна", "Юлия"]
RU_LAST = ["Абрамов", "Быков", "Воронин", "Гусев", "Дьяков", "Ершов", "Жуков", "Зайцев", "Ильин", "Карпов", "Лаптев",
           "Мухин", "Носов", "Осипов", "Панов", "Рябов", "Сорокин", "Тихонов", "Уваров", "Фролов"]
EN_PROJECTS = ["Apex", "Bolt", "Cobalt", "Drift", "Ember", "Flux", "Glint", "Haven", "Iris", "Jolt", "Kite", "Lumen",
               "Mosaic", "Nova", "Onyx", "Pulse", "Quest", "Rally", "Sable", "Tide", "Umbra", "Vector", "Wave",
               "Zephyr", "Beacon", "Canyon", "Dune", "Falcon", "Garnet", "Helix", "Indigo", "Jasper", "Kestrel",
               "Lagoon", "Meadow", "Nectar", "Opal", "Prism", "Raven", "Summit"]
RU_PROJECTS = ["Азимут", "Бриз", "Вектор", "Гейзер", "Дельфин", "Енисей", "Жемчуг", "Зенит", "Иней", "Лира", "Маяк",
               "Нептун", "Обсидиан", "Пламя", "Радуга", "Сапфир", "Тайга", "Ураган", "Феникс", "Хронос", "Циклон",
               "Шторм", "Эфир", "Юпитер", "Янтарь", "Аврора", "Байкал", "Волга", "Гроза", "Дюна", "Ельник", "Заря",
               "Изумруд", "Корвет", "Ладога", "Метель", "Нарвал", "Орион", "Полюс", "Рубин"]
EN_TREES = ["birch", "cedar", "maple", "larch", "rowan", "hazel", "alder", "yew", "spruce", "willow"]
RU_TREES = ["липа", "ольха", "осина", "рябина", "сосна", "ель", "ива", "клён", "бук", "граб"]
STARS = ["sirius", "lyra", "deneb", "altair", "rigel", "mira", "castor", "pollux", "spica", "antares", "capella",
         "procyon", "regulus", "bellatrix", "mizar"]
EN_TEAMS = ["Anchor", "Bridge", "Compass", "Dock", "Ferry", "Galley", "Helm", "Keel", "Lantern", "Mast", "Oar",
            "Pier", "Quay", "Rudder", "Sail", "Tiller", "Wharf", "Buoy", "Cove", "Reef"]
RU_TEAMS = ["Якорь", "Мост", "Компас", "Причал", "Паром", "Штурвал", "Киль", "Фонарь", "Мачта", "Весло", "Пирс",
            "Руль", "Парус", "Бакен", "Риф", "Залив", "Фарватер", "Лоцман", "Шкипер", "Бриг"]
EN_SERVICES = ["Ledger", "Mailer", "Pager", "Router", "Scheduler", "Indexer", "Notifier", "Gateway", "Tracker",
               "Uploader", "Resizer", "Search", "Auth", "Metrics", "Exporter", "Importer", "Signer", "Payroll",
               "Invoicer", "Archiver"]
RU_SERVICES = ["Почтальон", "Счётчик", "Курьер-2", "Диспетчер", "Сторож", "Летопись", "Маршрут", "Глашатай",
               "Сборщик", "Хранитель", "Переводчик", "Ключник", "Казначей", "Кассир", "Табель", "Печать", "Сигнал",
               "Архивариус", "Навигатор", "Сканер"]
EN_APPS = ["Finch", "Wren", "Heron", "Robin", "Swift", "Kiwi", "Crane", "Egret", "Plover", "Tern"]
RU_APPS = ["Синица", "Чиж", "Снегирь", "Щегол", "Дрозд", "Иволга", "Зяблик", "Клёст", "Свиристель", "Кулик"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
RU_WEEKDAYS_DAT = ["понедельникам", "вторникам", "средам", "четвергам", "пятницам"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
RU_MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
                 "ноября", "декабря"]
RU_MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре", "октябре",
                  "ноябре", "декабре"]


def _ru_people(rng, n):
    out = []
    for first, last in itertools.product(RU_MALE + RU_FEMALE, RU_LAST):
        out.append((f"{first} {last + 'а' if first in RU_FEMALE else last}", first in RU_FEMALE))
    rng.shuffle(out)
    return out[:n]


def _work_filler(rng) -> list[tuple[str, list[str]]]:
    facts = []
    en_people = [f"{a} {b}" for a, b in itertools.product(EN_FIRST, EN_LAST)]
    rng.shuffle(en_people)
    en_people = en_people[:170]
    ru_people = _ru_people(rng, 170)
    en_servers = [f"{t}-{n}" for t in EN_TREES for n in range(10, 16)]
    ru_servers = [f"{t}-{n}" for t in RU_TREES for n in range(10, 16)]
    hub = "Orbital Labs"
    t = lambda: f"{rng.randint(8, 18):02d}:{rng.choice(['00', '15', '30', '45'])}"  # noqa: E731
    # English
    for i, p in enumerate(EN_PROJECTS):
        facts.append((f"Project {p} runs on the server {en_servers[i]}.", [p, en_servers[i]]))
        facts.append((f"Project {p} release {rng.randint(1, 9)}.{rng.randint(0, 20)} is planned for "
                      f"{rng.choice(MONTHS)} {rng.randint(1, 28)}.", [p]))
    for s in en_servers:
        facts.append((f"{rng.choice(en_people)} is responsible for {s}.", [None, s]))
        facts.append((f"{s} has {rng.choice([16, 32, 64, 128])} GB of RAM and runs Debian.", [s]))
        facts.append((f"{s} restarts automatically every {rng.choice(WEEKDAYS)} night.", [s]))
    for p in en_people:
        facts.append((f"{p} works on project {rng.choice(EN_PROJECTS)}.", [p, None]))
        facts.append((f"{p} joined {hub} in {rng.randint(2015, 2026)}.", [p, hub]))
    for i, team in enumerate(EN_TEAMS):
        facts.append((f"The {team} team holds its stand-up at {t()} on weekdays.", [team]))
        facts.append((f"{en_people[i]} leads the {team} team.", [en_people[i], team]))
    for i, svc in enumerate(EN_SERVICES):
        db = f"{STARS[i % len(STARS)]}-{i}"
        facts.append((f"The {svc} service stores its data in the database {db}.", [svc, db]))
        facts.append((f"{db} is backed up every {rng.choice(WEEKDAYS)} at {t()}.", [db]))
        facts.append((f"The {svc} service's vendor contract renews on {rng.choice(MONTHS)} {rng.randint(1, 28)}.",
                      [svc]))
        for _ in range(3):
            facts.append((f"{rng.choice(en_people)} is on call for the {svc} service in {rng.choice(MONTHS)}.",
                          [None, svc]))
    for app in EN_APPS:
        svc = rng.choice(EN_SERVICES)
        facts.append((f"The {app} app sends notifications through the {svc} service.", [app, svc]))
    for _ in range(80):
        p, proj = rng.choice(en_people), rng.choice(EN_PROJECTS)
        facts.append((f"{p} reviews pull requests for project {proj} on {rng.choice(WEEKDAYS)}s.", [p, proj]))
    for _ in range(20):
        facts.append((f"{rng.choice(en_people)} pays for the team's {rng.choice(['Figma', 'Notion', 'Miro', 'Zoom'])} "
                      "subscription.", [None]))
    for _ in range(40):
        a, b = rng.sample(en_people, 2)
        facts.append((f"{a} holds a weekly one-on-one with {b} on {rng.choice(WEEKDAYS)}s.", [a, b]))
    # Russian
    for i, p in enumerate(RU_PROJECTS):
        facts.append((f"Проект {p} работает на сервере {ru_servers[i]}.", [p, ru_servers[i]]))
        facts.append((f"Релиз {rng.randint(1, 9)}.{rng.randint(0, 20)} проекта {p} запланирован на "
                      f"{rng.randint(1, 28)} {rng.choice(RU_MONTHS_GEN)}.", [p]))
    for s in ru_servers:
        facts.append((f"За сервер {s} отвечает {rng.choice(ru_people)[0]}.", [s, None]))
        facts.append((f"Сервер {s}: {rng.choice([16, 32, 64, 128])} ГБ памяти, Debian.", [s]))
        facts.append((f"Сервер {s} перезагружается автоматически по {rng.choice(RU_WEEKDAYS_DAT)}.", [s]))
    for p, _female in ru_people:
        facts.append((f"{p} работает над проектом {rng.choice(RU_PROJECTS)}.", [p, None]))
        facts.append((f"{p} работает в компании {hub} с {rng.randint(2015, 2026)} года.", [p, hub]))
    for i, team in enumerate(RU_TEAMS):
        facts.append((f"Планёрка команды {team} проходит по будням в {t()}.", [team]))
        facts.append((f"Команду {team} возглавляет {ru_people[i][0]}.", [team, ru_people[i][0]]))
    for i, svc in enumerate(RU_SERVICES):
        db = f"{STARS[(i + 7) % len(STARS)]}-db{i}"
        facts.append((f"Сервис {svc} хранит данные в базе {db}.", [svc, db]))
        facts.append((f"Резервная копия {db} делается по {rng.choice(RU_WEEKDAYS_DAT)} в {t()}.", [db]))
        facts.append((f"Договор с поставщиком сервиса {svc} продлевается {rng.randint(1, 28)} "
                      f"{rng.choice(RU_MONTHS_GEN)}.", [svc]))
        for _ in range(3):
            facts.append((f"{rng.choice(ru_people)[0]} дежурит по сервису {svc} в {rng.choice(RU_MONTHS_PREP)}.",
                          [None, svc]))
    for app in RU_APPS:
        svc = rng.choice(RU_SERVICES)
        facts.append((f"Приложение {app} отправляет уведомления через сервис {svc}.", [app, svc]))
    for _ in range(80):
        (p, _f), proj = rng.choice(ru_people), rng.choice(RU_PROJECTS)
        facts.append((f"{p} проверяет пул-реквесты проекта {proj} по {rng.choice(RU_WEEKDAYS_DAT)}.", [p, proj]))
    for _ in range(20):
        facts.append((f"{rng.choice(ru_people)[0]} оплачивает подписку команды на "
                      f"{rng.choice(['Figma', 'Notion', 'Miro', 'Zoom'])}.", [None]))
    for _ in range(40):
        (a, _fa), (b, _fb) = rng.sample(ru_people, 2)
        facts.append((f"{a} проводит встречу один на один с {b} по {rng.choice(RU_WEEKDAYS_DAT)}.", [a]))
    return _resolve(facts)


def _resolve(facts):
    """[None] in an entity list = the person or project named in the text: the template fills it from the text's
    capitalised words; here it simply means "declare every name the text holds" — the runner declares the listed ones
    and leaves the rest to the plugin's own capitalised-name linking, as an agent that names only some would."""
    return [(text, [e for e in ents if e]) for text, ents in facts]


RU_GEN = {"Варя": "Вари", "Тимур": "Тимура", "Савелий": "Савелия", "Алиса": "Алисы"}


def _family_filler(rng) -> list[tuple[str, list[str]]]:
    facts = []
    en_kids = ["Mia", "Leo", "Noah", "Ella"]
    ru_kids = ["Варя", "Тимур", "Савелий", "Алиса"]
    activities = ["chess", "football", "drawing", "robotics", "ballet", "judo", "choir"]
    ru_activities = ["шахматы", "футбол", "рисование", "робототехника", "балет", "дзюдо", "хор"]
    foods = ["pasta", "pancakes", "rice with chicken", "tomato soup", "dumplings", "fish sticks"]
    ru_foods = ["макароны", "блины", "гречка с котлетой", "борщ", "пельмени", "сырники"]
    en_adults = [f"{a} {b}" for a, b in itertools.product(EN_FIRST[:13], EN_LAST[10:])]
    rng.shuffle(en_adults)
    ru_adults = _ru_people(rng, 60)
    for kid in en_kids:
        for act in rng.sample(activities, 4):
            facts.append((f"{kid} has {act} on {rng.choice(WEEKDAYS)}s at {rng.randint(15, 18)}:00.", [kid]))
        facts.append((f"{kid}'s favourite food is {rng.choice(foods)}.", [kid]))
        facts.append((f"{kid} goes to bed at {rng.choice(['20:30', '21:00', '21:30'])} on school nights.", [kid]))
        for subj in rng.sample(["maths", "English", "art", "music", "science"], 3):
            teacher = en_adults.pop()
            facts.append((f"{kid}'s {subj} teacher is {teacher}.", [kid, teacher]))
            facts.append((f"{teacher} holds a parents' meeting on the first {rng.choice(WEEKDAYS)} of the month.",
                          [teacher]))
    for kid in ru_kids:
        for act in rng.sample(ru_activities, 4):
            facts.append((f"У {RU_GEN[kid]} {act} по {rng.choice(RU_WEEKDAYS_DAT)} в "
                          f"{rng.randint(15, 18)}:00.", [kid]))
        facts.append((f"Любимая еда: {kid} — {rng.choice(ru_foods)}.", [kid]))
        for subj in rng.sample(["математике", "английскому", "рисованию", "музыке", "чтению"], 3):
            teacher, _female = ru_adults.pop()
            facts.append((f"Учитель {RU_GEN[kid]} по {subj} — {teacher}.", [kid, teacher]))
            facts.append((f"{teacher} проводит консультации по {rng.choice(RU_WEEKDAYS_DAT)}.", [teacher]))
    for _ in range(25):
        doc = en_adults.pop()
        facts.append((f"Dr. {doc} sees patients at the {rng.choice(['Hillside', 'Park', 'Central', 'Bay'])} clinic on "
                      f"{rng.choice(WEEKDAYS)}s.", [doc]))
        facts.append((f"Dr. {doc} is a {rng.choice(['dermatologist', 'dentist', 'ophthalmologist', 'ENT doctor'])}.",
                      [doc]))
    for _ in range(25):
        doc, _female = ru_adults.pop()
        facts.append((f"{doc} принимает в поликлинике на {rng.choice(['Лесной', 'Речной', 'Полевой'])} по "
                      f"{rng.choice(RU_WEEKDAYS_DAT)}.", [doc]))
        facts.append((f"{doc} — {rng.choice(['дерматолог', 'окулист', 'лор', 'ортопед'])}.", [doc]))
    places = ["The pharmacy on Elm Road", "The bakery", "The library", "The post office", "The bike shop"]
    for place in places:
        facts.append((f"{place} is closed on {rng.choice(WEEKDAYS)}s.", []))
        facts.append((f"{place} is open until {rng.randint(17, 22)}:00.", []))
    ru_places = ["Аптека на углу", "Пекарня", "Библиотека", "Почта", "Веломастерская"]
    for place in ru_places:
        facts.append((f"{place} открыта до {rng.randint(17, 22)}:00.", []))
        facts.append((f"{place} не работает по {rng.choice(RU_WEEKDAYS_DAT)}.", []))
    for i in range(12):
        coach = en_adults.pop()
        facts.append((f"{coach} holds practice at the {rng.choice(['North', 'East', 'Lake'])} sports hall on "
                      f"{rng.choice(WEEKDAYS)}s.", [coach]))
        coach_ru, _f = ru_adults.pop()
        facts.append((f"{coach_ru} ведёт тренировки в спорткомплексе {rng.choice(['Север', 'Восток', 'Озеро'])}.",
                      [coach_ru]))
    for i in range(30):
        facts.append((f"The family budget for {rng.choice(['groceries', 'transport', 'clubs', 'gifts'])} is "
                      f"{rng.randint(100, 900)} per month (note {i + 1}).", []))
        facts.append((f"Бюджет семьи на {rng.choice(['продукты', 'транспорт', 'кружки', 'подарки'])} — "
                      f"{rng.randint(5, 60)} тысяч в месяц (запись {i + 1}).", []))
    return facts


def filler() -> dict[str, list[tuple[str, list[str]]]]:
    rng = random.Random(20261028)
    return {"family": _family_filler(rng), "work-a": _work_filler(rng), "work-b": [
        ("Lena Park is on call for the Orbit project in July.", ["Lena Park", "Orbit"]),
        ("pine-04 has 64 GB of RAM.", ["pine-04"]),
        ("Сервис Клевер обновляется по четвергам.", ["Клевер"]),
    ]}
