"""Memory map (experimental): a read-only snapshot of ONE store as JSON, for a person to read (docs/memory-map.md:
what it shows, what it does not, its risks and limits; the format may change while it is experimental).

The data layer of the map: a dashboard tab or a standalone page renders it; every privacy rule is applied here, so
the renderers cannot differ on them. The store is opened read-only (``mode=ro``, ``query_only``): nothing is written,
migrated or created, no model and no network are used. What the provider itself would not show is not shown:

- one store per export (one scope), read in one read transaction; of the night's files (dreaming) the report and the
  approval queue only for the profile store, a merge question only when it names this very store;
- a forgotten fact is its number and the date, nothing else (its text, keywords and names are gone from the store);
- a corrected version has no text (it never held); a text the provider's screen keeps out of answers (a credential,
  an instruction to the assistant, hidden markup) is not shown, current or past: the fact is listed under HIDDEN
  with its number and date.

    python -m <package>.memory_map --store PATH [--hermes-home DIR] [--tz ZONE] [--config YAML] [--json]
    python <plugin dir>/memory_map.py --store PATH ...     (no Hermes core needed)

``export_map(store_path, hermes_home=None, tz=None, now=None)`` returns the same dict for a caller in Python."""

from __future__ import annotations

if __name__ == "__main__" and not __package__:  # run as a file: load the plugin's modules without its __init__
    import importlib
    import sys
    import types
    from pathlib import Path as _Path

    _here = str(_Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if _Path(p or ".").resolve() != _Path(_here)]  # its modules are not top-level
    _pkg = types.ModuleType("_holographic_map")
    _pkg.__path__ = [_here]
    sys.modules["_holographic_map"] = _pkg
    sys.exit(importlib.import_module("_holographic_map.memory_map").main())

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import threading
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path, PurePosixPath, PureWindowsPath

from .entity_names import entity_key, fold
from .history import error_kind, source_of
from .retrieval import FactRetriever
from .screen import _recall_problem, _secret_in
from .store import MemoryStore
from .text import NEGATORS, analyze, name_forms, name_keys

SCHEMA_VERSION = 1
# Words for trust (design, decided 29.10.2026), on the number rounded to hundredths (0.30000000000000004 is stored)
TRUST_NORMAL, TRUST_STRONG = 0.45, 0.65
STALE_DAYS = 60  # review()'s "never recalled" and "unconfirmed inference"
WEEK_DAYS = 7
_BIG = 100_000  # "all of them" for the provider's limits
_DREAM_FILES = {"proposals": ("cache", "dream-proposals.json"), "report": ("cache", "dream.json"),
                "queue": ("pending", "memory")}
_PROPOSAL_ID = re.compile(r"D-\d{1,9}")
_CODE = re.compile(r"[a-z_]{1,40}")
_STAT_KEY = re.compile(r"[a-z][a-z0-9_]{0,59}")
# What dream writes as a MEMORY conflict's numbers (its _numbers: a number, a date or a time as written, of any length
# — an id or an account number too —, a month or a day of the week said in words), and the longest negated phrase it
# writes (a word or two after the negation)
_NIGHT_NUMBER = re.compile(r"\d(?:[\d.,:/-]*\d)?|month:\d{2}|weekday:[1-7]")
_NEGATION_MAX = 120
_QUEUE_FILE = re.compile(r"[A-Za-z0-9_.-]{1,80}\.json")
_HIT_LISTS = ("fact_numbers", "memory_numbers", "fact_negations", "memory_negations")
_FEED_ORDER = {"night": 0, "del": 1, "mrg": 2, "end": 3, "chg": 4, "conf": 5, "new": 6}
_REVIEW_REASONS = (("low trust", "low_trust"), ("never recalled", "never_recalled"),
                   ("an inference nobody confirmed", "unconfirmed_inference"), ("near-duplicate of #", "near_duplicate"))


class _ReadOnlyStore(MemoryStore):
    """The provider's store over a read-only connection of its own: the provider's read code (history, origins,
    entities, review, contradict) runs unchanged, and any write it might attempt fails instead of happening. Not in
    the shared connection registry, never migrated.

    One read transaction from the first read to close(): every query of the export, the provider's included (its
    _tx() joins an open transaction and never commits it), sees the store at one moment, so a forget or a change
    another process makes meanwhile is either all in the map or not at all (Codex review, 30.10.2026). In WAL mode
    writers go on meanwhile; in rollback-journal mode (DELETE, what the core keeps on a SQLite it deems
    WAL-reset-vulnerable) a writer's COMMIT waits until close(), up to its busy timeout (10 s in the provider)."""

    def __init__(self, db_path, tz: tzinfo | None = None) -> None:  # not MemoryStore's: that one opens for writing
        path = Path(db_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no store at {path}")
        self.db_path, self.tz = path, tz
        self.default_trust, self.hrr_dim, self._hrr_available, self._key = 0.5, 1024, False, None
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False,
                               isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.create_function("hfold", 1, fold, deterministic=True)
            conn.execute("PRAGMA query_only = ON")
            conn.execute("BEGIN")  # deferred: the snapshot is taken at the first read, the list of tables below
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        except BaseException:
            conn.close()  # closing ends the transaction too
            raise
        self.tables = tables
        self._entry = {"conn": conn, "lock": threading.RLock(), "refs": 1, "ready": True, "migrated": False,
                       "generation": 0}
        self._conn, self._lock = conn, self._entry["lock"]

    def _meta(self, key: str) -> str | None:
        return super()._meta(key) if "meta" in self.tables else None  # before 0.4.0 there was no meta table

    def close(self) -> None:
        """End the read transaction (nothing was written: a rollback) and close; always closes, even when the
        rollback fails."""
        if self._entry is not None:
            self._entry = None
            try:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
            finally:
                self._conn.close()

    def in_snapshot(self) -> bool:
        """Is the export's read transaction still open (nothing in between committed or ended it)?"""
        return self._entry is not None and self._conn.in_transaction


# -- helpers --------------------------------------------------------------------------------------------------------

def _zone(tz) -> tuple[tzinfo, str]:
    """(tzinfo, its name): None = UTC; "+07:00" / "-03:30" a fixed offset; else an IANA name (zoneinfo)."""
    if tz is None or tz == "" or tz == "UTC":
        return timezone.utc, "UTC"
    if isinstance(tz, tzinfo):
        return tz, str(getattr(tz, "key", None) or tz)
    text = str(tz).strip()
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", text)
    if m:
        delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
        return timezone(-delta if m.group(1) == "-" else delta), text
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        return ZoneInfo(text), text
    except ZoneInfoNotFoundError:  # a KeyError: the caller's ValueError did not catch it, a traceback (H17)
        raise ValueError(f"unknown time zone {text!r}: give an IANA name such as Asia/Bangkok, or +HH:MM") from None


def _screen(text) -> str | None:
    """Why the provider would keep *text* out of an answer or away from a network call, as a code, or None: the
    strictest of its screens (recall's and the sendable one)."""
    value = str(text or "")
    problem = _recall_problem(value)
    if problem:
        return "instruction" if "instruction" in problem else "credential" if "credential" in problem else "markup"
    return "credential" if _secret_in(value) else None


def _shown(text):
    """*text* if it may be shown, else None."""
    return None if text is None or _screen(text) else text


def _hit_lists(hit: dict) -> tuple[dict, str | None]:
    """(the numbers and negations of a MEMORY conflict, the first field of another shape or None). Each is what dream
    writes there — numbers: a number, a date or a time as text (or a JSON number), `month:MM`, `weekday:N`;
    negations: a short phrase — and passes the screen; anything else (a structure, a sentence among the numbers, a
    negation the screen refuses) makes the field wrong, never text in the JSON."""
    out = {}
    for key in _HIT_LISTS:
        value = hit.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            return {}, key
        if key.endswith("_numbers"):
            items = [str(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in value]
            ok = all(isinstance(x, str) and _NIGHT_NUMBER.fullmatch(x) for x in items)
        else:
            items = value
            ok = all(isinstance(x, str) and 0 < len(x) <= _NEGATION_MAX for x in items)
        if not ok or any(_screen(x) for x in items):
            return {}, key
        out[key] = list(items)
    return out, None


def _round(value):
    return None if value is None else round(float(value), 2)


def _trust_level(trust) -> str | None:
    if trust is None:
        return None
    t = round(float(trust), 2)
    return "strong" if t >= TRUST_STRONG else "normal" if t >= TRUST_NORMAL else "weak"


def _day(store: _ReadOnlyStore, timestamp) -> str | None:
    """The local day of a UTC timestamp SQLite wrote, or None."""
    return store._local_day(timestamp) if timestamp else None


def _iso_day(value, zone: tzinfo) -> str | None:
    """The day of an ISO 8601 moment in *zone* (a date alone, or a moment without an offset, as written); None when
    it is none, or no calendar holds it in *zone* (past the year 9999 once moved there)."""
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(str(value)[:10]).isoformat()
        except ValueError:
            return None
    try:
        return (moment.astimezone(zone) if moment.tzinfo else moment).date().isoformat()
    except OverflowError:
        return None


def _moment(value, zone: tzinfo) -> datetime | None:
    """A text that is an ISO 8601 moment and nothing else, in *zone* when it has an offset; else None."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
        return moment.astimezone(zone) if moment.tzinfo else moment
    except (ValueError, OverflowError):
        return None


def _finite(value) -> bool:
    """A JSON number that is a number: not a bool, not NaN or Infinity (json.loads takes those; strict JSON not)."""
    return isinstance(value, int) and not isinstance(value, bool) or isinstance(value, float) and math.isfinite(value)


def _same_file(a, b) -> bool:
    """Do two paths name one file: the same file when both exist (links, ".." and, on Windows, letter case
    included), else the same resolved path. A relative path is read from the current folder — so only the caller's
    own arguments may be one; a path read from a night file is checked to be absolute first (_record_store)."""
    try:
        a, b = Path(a).expanduser(), Path(b).expanduser()
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
        return a.resolve() == b.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _record_store(value, mine: Path) -> str:
    """Whose a night record is by its `store`: none (absent or empty: the dream's profile store), this (an absolute
    path naming this very file), other (another absolute path, or one absolute only on another system), relative
    (not absolute: never resolved, or the current folder would decide whose it is — in the report it is a chat
    store's name, in a merge question it proves nothing), bad (not text)."""
    if value is None or value == "":
        return "none"
    if not isinstance(value, str) or "\0" in value:
        return "bad"
    if Path(value).is_absolute():
        return "this" if _same_file(mine, value) else "other"
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return "other"
    return "relative"


def _load_json(path: Path):
    """(state, data): state ok | absent | unreadable."""
    if not path.exists():
        return "absent", None
    try:
        return "ok", json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unreadable", None


# -- who is a person (docs/memory-map.md, "Entity types") -------------------------------------------------------------
# The store records no types: a name is a name. A person is told only by what the store itself holds — the name is
# someone who spoke in a chat (a recorded author), or a stored fact puts a word for a relative or a role right beside
# it ("дочери Варе", "my daughter Varya", "Varya is my daughter"). A topic only by its form: a person's name has no
# digit, no dot or slash inside a word, no lower-case start before a capital (iPhone) and is not one word in capitals
# (USDT). Neither, or both: null. Capitals prove nothing (a month, a brand and a pet are capitalized too).
_PERSON_WORDS_TEXT = """
    дочь дочери дочерью дочка дочки дочке дочку дочкой сын сына сыну сыном сыне сынок сынка сынку
    жена жены жене жену женой муж мужа мужу мужем муже мама мамы маме маму мамой мать матери матерью
    папа папы папе папу папой отец отца отцу отцом брат брата брату братом брате сестра сестры сестре сестру сестрой
    бабушка бабушки бабушке бабушку бабушкой дедушка дедушки дедушке дедушку дедушкой дед деда деду дедом
    внук внука внуку внуком внучка внучки внучке внучку внучкой тётя тёти тёте тётю тётей тетя тети тете тетю тетей
    дядя дяди дяде дядю дядей племянник племянника племяннику племянница племянницы племяннице
    друг друга другу другом подруга подруги подруге подругу подругой коллега коллеги коллеге коллегу коллегой
    няня няни няне няню няней учитель учителя учителю учителем учительница учительницы учительнице учительницу
    врач врача врачу врачом
    daughter son wife husband mother mom mum father dad brother sister grandmother grandma grandfather grandpa
    granddaughter grandson aunt uncle niece nephew cousin friend girlfriend boyfriend colleague nanny teacher doctor
    mr mrs ms miss dr"""
# Between a name and the word after it only these ("Varya is my daughter", «Варя — наша дочь»): not "and" / «и», so
# «Сосновка и мама» says nothing of Sosnovka.
_LINK_WORDS_TEXT = "is was my our his her their моя мой моё мое мои наша наш наше наши его её ее их это"
_RE_INNER_MARK = re.compile(r"\w[._:/]\w")
_person_words: frozenset | None = None
_link_words: frozenset | None = None


def _lexicon() -> tuple[frozenset, frozenset]:
    """The two word lists, folded the way analyze folds a fact's words."""
    global _person_words, _link_words
    if _person_words is None:
        _person_words = frozenset(w.text for w in analyze(_PERSON_WORDS_TEXT).words)
        _link_words = frozenset(w.text for w in analyze(_LINK_WORDS_TEXT).words)
    return _person_words, _link_words


def _name_spans(words, name: str) -> list[tuple[int, int]]:
    """(start, end) of every mention of *name* among a fact's *words*, in any case form (text.name_keys: «Варе» and
    «Вари» mention «Варя»; a Latin name matches as written, or with 's)."""
    keys = name_keys(name)
    if not keys:
        return []
    spans = []
    for i in range(len(words) - len(keys) + 1):
        if all(words[i + j].text in forms if exact else not forms.isdisjoint(name_forms(words[i + j].text))
               for j, (forms, exact) in enumerate(keys)):
            spans.append((i, i + len(keys)))
    return spans


def _beside_person_word(words, start: int, end: int) -> bool:
    person, link = _lexicon()
    if start > 0 and words[start - 1].text in person:
        return True
    j = end
    while j < len(words) and j - end < 2 and words[j].text in link:
        j += 1
    return end < j < len(words) and words[j].text in person


def _topic_form(names) -> str | None:
    """Why a name cannot be a person's by its form, or None."""
    for name in names:
        if any(c.isdigit() for c in name):
            return "digit"
        if _RE_INNER_MARK.search(name):
            return "identifier"
        for word in name.split():
            letters = [c for c in word if c.isalpha() and c.lower() != c.upper()]
            if letters and letters[0].islower() and any(c.isupper() for c in letters[1:]):
                return "mixed_case"
        cased = [c for c in name if c.isalpha() and c.lower() != c.upper()]
        if len(name.split()) == 1 and len(cased) >= 2 and all(c.isupper() for c in cased):
            return "acronym"
    return None


def _entity_type(names, speakers: set, texts) -> tuple[str | None, str | None, int | None]:
    """(person | topic | None, the rule, the fact that showed it). *names*: the name and its aliases; *speakers*: the
    entity keys of the authors the store recorded; *texts*: (fact id, text) of the visible facts linked to it."""
    by, fact = None, None
    if any(entity_key(n) in speakers for n in names):
        by = "speaker"
    else:
        for fid, text in texts:
            words = analyze(text).words
            if any(_beside_person_word(words, a, b) for n in names for a, b in _name_spans(words, n)):
                by, fact = "relative_or_role", fid
                break
    topic = _topic_form(names)
    if by and topic:
        return None, "both", None  # a person's sign and a topic's: not decided
    if by:
        return "person", by, fact
    return ("topic", topic, None) if topic else (None, None, None)


# -- what search puts first (docs/memory-map.md, "Contradictions") ----------------------------------------------------
_RANKING_DEFAULTS = {"min_trust_threshold": 0.3, "temporal_decay_half_life": 0}
_TRUE = {"1", "true", "yes", "on"}


def _load_yaml(text: str):
    """config.yaml as Hermes reads it: its own loader (hermes_yaml, ruamel; Hermes does not install PyYAML), else
    PyYAML outside Hermes. Neither present raises ImportError: the settings are then unknown."""
    try:
        from hermes_yaml import safe_load
    except ImportError:
        from yaml import safe_load
    return safe_load(text)


def _ranking(config) -> dict:
    """The settings search ranks by: the trust floor, the decay half-life, and whether the reranker (which ignores
    trust) or search by meaning is on. *config*: the plugin's section as a dict, a config.yaml path, or None (the
    plugin's defaults). ``source``: defaults | config | absent (no such file: the provider runs on the defaults too) |
    unreadable (nothing about the order is said then)."""
    source, section = "defaults", {}
    if isinstance(config, dict):
        source, section = "config", config
    elif config is not None:
        path = Path(config).expanduser()
        if not path.is_file():
            source = "absent"
        else:
            try:
                data = _load_yaml(path.read_text(encoding="utf-8")) or {}
                plugins = data.get("plugins") if isinstance(data, dict) else None
                section = (plugins or {}).get("hermes-memory-store") if isinstance(plugins, dict) else None
                source, section = "config", section if isinstance(section, dict) else {}
            except Exception:  # noqa: BLE001 - no yaml, or a file it cannot parse: the order is not said
                return {"source": "unreadable", "min_trust": None, "half_life": None, "rerank": None, "meaning": None}
    try:
        min_trust = float(section.get("min_trust_threshold", _RANKING_DEFAULTS["min_trust_threshold"]))
        half_life = int(section.get("temporal_decay_half_life", _RANKING_DEFAULTS["temporal_decay_half_life"]))
    except (TypeError, ValueError, OverflowError):  # the provider turns memory off for these: nothing to rank
        return {"source": "unreadable", "min_trust": None, "half_life": None, "rerank": None, "meaning": None}
    flag = lambda key: section.get(key) is True or str(section.get(key, "")).strip().lower() in _TRUE  # noqa: E731
    return {"source": source, "min_trust": min_trust, "half_life": half_life, "rerank": flag("rerank"),
            "meaning": flag("embeddings")}


def _difference(text_a: str, text_b: str, entity_words: set) -> dict:
    """What tells two claims apart, taken from their own words: the words of each the other lacks (by their keys,
    so «вторникам» and «вторник» are one word; function words and the shared names left out) and whether one is a
    negation of the other (contradict's own reading)."""
    def words(text):
        return [w for w in analyze(text).words if not w.stop and w.text not in NEGATORS and w.text not in entity_words]

    wa, wb = words(text_a), words(text_b)
    only = lambda xs, ys: list(dict.fromkeys(  # noqa: E731
        w.text for w in xs if all(w.keys.isdisjoint(y.keys) for y in ys)))[:8]
    neg_a = FactRetriever._claim(text_a, entity_words)[1]
    neg_b = FactRetriever._claim(text_b, entity_words)[1]
    return {"a": only(wa, wb), "b": only(wb, wa), "negation": neg_a != neg_b}


# -- the snapshot ---------------------------------------------------------------------------------------------------

class _Snapshot:
    def __init__(self, store: _ReadOnlyStore, made: datetime, zone_name: str, hermes_home, ranking: dict):
        self.s, self.made, self.zone_name, self.home, self.ranking = store, made, zone_name, hermes_home, ranking
        self.today = made.date().isoformat()
        self.week_from = (made.date() - timedelta(days=WEEK_DAYS)).isoformat()
        self.history = "fact_versions" in store.tables and store.history_ready()
        self.has = store.tables.__contains__

    def rows(self, sql, params=()):
        return [dict(r) for r in self.s.read(sql, params)]

    # -- facts --
    def build(self) -> dict:
        s = self.s
        current = {r["fact_id"]: r for r in self.rows(
            "SELECT fact_id, content, category, trust_score, retrieval_count, helpful_count, created_at, updated_at "
            "FROM facts ORDER BY fact_id")}
        versions: dict[int, list[dict]] = {}
        if self.history:
            for r in self.rows("SELECT * FROM fact_versions ORDER BY fact_id, seq"):
                versions.setdefault(r["fact_id"], []).append(r)
        merges = {r["from_fact_id"]: r for r in self.rows(
            "SELECT from_fact_id, into_fact_id, merged_at FROM fact_merges ORDER BY from_fact_id")} \
            if self.has("fact_merges") else {}
        forgotten = self.rows("SELECT fact_id, forgotten_at FROM forgotten ORDER BY fact_id") \
            if self.has("forgotten") and self.history else []
        gone = {r["fact_id"] for r in forgotten}
        # a fact that ended for good: no current version and not merged since (history_counts' rule)
        ended = {fid: [v for v in vs if v["status"] == "ended"][-1] for fid, vs in versions.items()
                 if fid not in current and fid not in gone and any(v["status"] == "ended" for v in vs)
                 and not any(v["status"] in ("current", "merged") for v in vs)}
        origins = s.origins(list(current))
        confirmations = self.rows("SELECT fact_id, confirmed_at, user_id FROM fact_confirmations "
                                  "ORDER BY confirmed_at, fact_id") if self.has("fact_confirmations") else []
        names = self._user_names(versions)

        hidden, facts = [], {}
        for fid, row in current.items():
            cur = next((v for v in versions.get(fid, []) if v["status"] == "current"), None)
            why = _screen(row["content"])
            if why:
                hidden.append({"id": fid, "d": cur["recorded_on"] if cur else _day(s, row["created_at"]),
                               "why": why, "status": "current"})
                continue
            facts[fid] = self._fact(fid, row["content"], row["category"], row["trust_score"], "current", cur,
                                    versions.get(fid, []), (origins.get(fid) or {}).get("origin"), row)
        for fid, last in sorted(ended.items()):
            why = _screen(last["content"])
            if why:
                hidden.append({"id": fid, "d": last["recorded_on"], "why": why, "status": "ended"})
                continue
            facts[fid] = self._fact(fid, last["content"], last["category"], last["trust_score"], "ended", last,
                                    versions[fid], last.get("src_origin"), None)
        hidden.sort(key=lambda h: h["id"])

        for c in confirmations:
            fact = facts.get(c["fact_id"])
            if fact is not None:
                fact["conf"].append({"d": _day(s, c["confirmed_at"]), "by": _shown(names.get(c["user_id"]))})
                if fact["origin"] not in ("said", "confirmed") and fact["status"] == "ended":
                    fact["origin"] = "confirmed"  # as origins() counts it for a current fact

        merged = []
        for from_id, m in merges.items():
            copy = [v for v in versions.get(from_id, []) if v["status"] == "merged"]
            if not copy:
                continue  # its versions are gone: the next writer drops the row
            text = None if _screen(copy[-1]["content"]) else copy[-1]["content"]
            entry = {"from": from_id, "into": m["into_fact_id"], "d": _day(s, m["merged_at"]), "prop": None,
                     "text": text}
            merged.append(entry)
            if m["into_fact_id"] in facts:
                facts[m["into_fact_id"]]["mergedFrom"].append({"id": from_id, "text": text, "d": entry["d"],
                                                               "prop": None})

        speakers = {entity_key(name) for vs in versions.values() for v in vs
                    if (name := _shown(v.get("src_user_name"))) and entity_key(name)}
        ents, links = self._entities(facts, speakers)
        for fid, fact in facts.items():
            fact["ents"] = links.get(fid, [])
        self._weak(facts)
        conflicts = self._conflicts(facts, links, current, ents)
        dreaming, pending, nights = self._dreaming(facts, gone, versions, merged)
        pending = [{"kind": "conflict", "id": c["id"], "a": c["a"], "b": c["b"], "status": "open"}
                   for c in conflicts] + pending

        tombs = sorted(({"id": r["fact_id"], "d": _day(s, r["forgotten_at"])} for r in forgotten),
                       key=lambda t: (t["d"] or "", t["id"]), reverse=True)
        feed = self._feed(facts, versions, confirmations, names, forgotten, merged, nights)
        visible = list(facts.values())
        counts = {
            "current": sum(f["status"] == "current" for f in visible),
            "ended": sum(f["status"] == "ended" for f in visible),
            "merged": len(merged), "forgotten": len(forgotten),
            "corrected": sum(v["status"] == "corrected" for fid in facts for v in versions.get(fid, [])),
            "hidden": len(hidden), "entities": len(ents),
            "provisional": sum(e["standing"] == "provisional" for e in ents),
            "legacy": sum(f["status"] == "current" and f["via"] == "legacy" and f["origin"] == "unknown" for f in visible),
        }
        snap = {
            "made": self.made.isoformat(timespec="seconds"), "tz": self.zone_name, "store": s.db_path.name,
            "schema": {"store": s._meta("schema_version"), "history": s._meta("history_schema"),
                       "entity_rules": s._meta("entity_rules"), "experience": s._meta("experience_schema")},
            "counts": counts, "week_from": self.week_from,
            "excluded": [{"what": "hidden_by_screen", "count": len(hidden)}] if hidden else [],
            "ranking": self.ranking,
            "experience": {"included": False}, "dreaming": dreaming,
        }
        return {"schema_version": SCHEMA_VERSION, "SNAP": snap, "ENT": ents, "FACTS": sorted(visible, key=lambda f: f["id"]),
                "CONFLICTS": conflicts, "PENDING": pending, "TOMBS": tombs, "MERGES": merged, "HIDDEN": hidden,
                "FEED": feed, "ATTEMPTS": []}

    def _user_names(self, versions) -> dict:
        """user_id -> the name the store recorded with it (the newest), to say who confirmed a fact."""
        out = {}
        for vs in versions.values():
            for v in vs:
                seen = out.get(v.get("src_user_id"))  # versions come by fact, not by time: keep the newest (H17)
                if v.get("src_user_id") and v.get("src_user_name") and (seen is None or v["recorded_at"] >= seen[0]):
                    out[v["src_user_id"]] = (v["recorded_at"], v["src_user_name"])
        return {k: name for k, (_, name) in out.items()}

    def _fact(self, fid, text, category, trust, status, head, rows, origin, row) -> dict:
        s = self.s
        src = source_of(head) if head else None
        first = rows[0] if rows else None
        return {
            "id": fid, "text": text, "ents": [], "cat": _shown(category),
            "origin": origin if origin in ("said", "confirmed", "inferred") else "unknown",
            "trust": _round(trust), "trust_level": _trust_level(trust), "status": status,
            "from": head["valid_from"] if head else None,
            "to": head["valid_to"] if head and status == "ended" else None,
            "to_exact": bool(head["valid_to_exact"]) if head and status == "ended" else None,
            "rec": head["recorded_on"] if head else _day(s, row["created_at"]),
            "first": first["recorded_on"] if first else _day(s, row["created_at"]),
            "via": (head.get("src_via") or "legacy") if head else "legacy",
            "chat": _shown((src or {}).get("chat_type")), "by": _shown((src or {}).get("user_name")),
            "conf": [], "recalls": row["retrieval_count"] if row else None,
            "helpful": row["helpful_count"] if row else None, "last": None,
            "versions": [self._version(v) for v in rows if v["status"] != "merged"],
            "mergedFrom": [], "weak": [],
        }

    def _version(self, v) -> dict:
        out = {"seq": v["seq"], "status": v["status"], "text": None, "from": v["valid_from"], "to": v["valid_to"],
               "to_exact": bool(v["valid_to_exact"]), "rec": v["recorded_on"], "closed": _day(self.s, v["closed_at"]),
               "origin": v.get("src_origin") or None, "via": v.get("src_via") or None}
        if v["status"] != "corrected":  # a corrected version never held: never shown as the past
            if _screen(v["content"]):
                out["hidden"] = True
            else:
                out["text"] = v["content"]
        return out

    # -- entities --
    def _entities(self, facts, speakers) -> tuple[list[dict], dict[int, list[int]]]:
        s = self.s
        try:
            rows = s.list_entities(limit=_BIG)
        except sqlite3.OperationalError:  # a file older than entity_aliases (0.4.0): names only
            rows = [{"entity_id": r["entity_id"], "name": r["name"], "declared": r["entity_type"] == "declared",
                     "aliases": [], "text_aliases": [], "status": "established"}
                    for r in self.rows("SELECT entity_id, name, entity_type FROM entities")]
        since = {r["entity_id"]: r["created_at"] for r in self.rows("SELECT entity_id, created_at FROM entities")}
        stale = {r["entity_id"] for r in s.stale_provisional_entities(days=STALE_DAYS, limit=_BIG)} \
            if "entity_aliases" in s.tables else set()
        by_entity: dict[int, list[int]] = {}
        for r in self.rows("SELECT fact_id, entity_id FROM fact_entities ORDER BY entity_id, fact_id"):
            if r["fact_id"] in facts:
                by_entity.setdefault(r["entity_id"], []).append(r["fact_id"])
        out, links = [], {}
        for r in sorted(rows, key=lambda e: e["entity_id"]):
            eid = r["entity_id"]
            # a name only facts the screen hides gave, or a name the screen would hide itself, is not shown
            if not by_entity.get(eid) or _screen(r["name"]):
                continue
            aliases = [a for a in r.get("aliases", []) if _shown(a) is not None]
            text_aliases = [a for a in r.get("text_aliases", []) if _shown(a) is not None]
            kind, rule, shown_by = _entity_type([r["name"], *aliases, *text_aliases], speakers,
                                                [(fid, facts[fid]["text"]) for fid in by_entity[eid]])
            out.append({"id": eid, "name": r["name"], "type": kind, "type_by": rule, "type_fact": shown_by,
                        "declared": bool(r.get("declared")), "aliases": aliases, "text_aliases": text_aliases,
                        "standing": r.get("status") or "established", "stale": eid in stale,
                        "since": _day(s, since.get(eid)), "facts": len(by_entity[eid])})
            for fid in by_entity[eid]:
                links.setdefault(fid, []).append(eid)
        return out, links

    # -- weak basis, conflicts --
    def _weak(self, facts) -> None:
        """review()'s reasons for the current facts it lists, plus "no source" for a fact whose origin is unknown —
        except the records written before origins existed (via legacy): not weak for that (decided 29.10.2026)."""
        retriever = FactRetriever(self.s, hrr_weight=0.0)
        for item in retriever.review(limit=max(20, len(facts) + 1), stale_days=STALE_DAYS):
            fact = facts.get(item["fact_id"])
            if fact is None or fact["status"] != "current":
                continue
            code = next((c for prefix, c in _REVIEW_REASONS if item["reason"].startswith(prefix)), None)
            if code is None:
                continue
            fact["weak"].append(code)
            if code == "near_duplicate":
                of = int(item["reason"].rsplit("#", 1)[1])
                if of in facts:
                    fact["dup_of"] = of
        for fact in facts.values():
            if fact["status"] == "current" and fact["origin"] == "unknown" and fact["via"] != "legacy":
                fact["weak"].append("no_source")

    def _conflicts(self, facts, links, current, ents) -> list[dict]:
        retriever = FactRetriever(self.s, hrr_weight=0.0)
        names = {e["id"]: e["name"] for e in ents}
        pairs = []
        for c in retriever.contradict(limit=_BIG):
            a, b = sorted((c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]))
            if a in facts and b in facts:
                shared = sorted(set(links.get(a, [])) & set(links.get(b, [])))
                # the names of either fact are not what differs (contradict leaves them out of the claims too)
                entity_words = {w for e in set(links.get(a, [])) | set(links.get(b, [])) if e in names
                                for w in analyze(names[e]).bag}
                pairs.append({"a": a, "b": b, "score": c["contradiction_score"], "shared": shared,
                              "since": max(facts[a]["rec"] or "", facts[b]["rec"] or "") or None,
                              "differ": _difference(facts[a]["text"], facts[b]["text"], entity_words),
                              "relies": self._relies(current[a], current[b])})
        pairs.sort(key=lambda p: (-p["score"], p["a"], p["b"]))
        return [{"id": f"C{i}", **p} for i, p in enumerate(pairs, 1)]

    def _relies(self, row_a, row_b) -> dict:
        """Which of two current facts search puts first, by the part of its score no question changes: a fact below
        the trust floor is never found; otherwise trust x decay (FactRetriever._rerank) decides between two facts
        the question fits equally. ``on`` is null when that says nothing: a tie, both below the floor, the reranker
        on (it orders by text alone), or settings that could not be read."""
        r = self.ranking
        out = {"on": None, "why": None, "sides": {}}
        if r["source"] == "unreadable":
            out["why"] = "settings_unknown"
            return out
        weights = {}
        # search's own decay (FactRetriever.decays: from when the current text was recorded, 0.23.0), at the
        # snapshot's moment
        decays = FactRetriever(self.s, temporal_decay_half_life=r["half_life"], hrr_weight=0.0).decays(
            [row_a, row_b], now=self.made)
        for row, decay in zip((row_a, row_b), decays):
            trust = float(row["trust_score"])
            found = trust >= r["min_trust"]
            weights[row["fact_id"]] = (found, trust * decay)
            out["sides"][str(row["fact_id"])] = {"trust": _round(trust), "found": found,
                                                 "decay": round(decay, 3) if r["half_life"] else None}
        (ia, (fa, wa)), (ib, (fb, wb)) = weights.items()
        if r["rerank"]:
            out["why"] = "rerank"
        elif not fa and not fb:
            out["why"] = "neither_found"
        elif fa != fb:
            out["on"], out["why"] = (ia if fa else ib), "floor"
        elif math.isclose(wa, wb, rel_tol=1e-9, abs_tol=1e-12):
            out["why"] = "tie"
        else:
            out["on"], out["why"] = (ia if wa > wb else ib), "weight"
        return out

    # -- dreaming --
    def _dreaming(self, facts, gone, versions, merged):
        """(SNAP.dreaming, pending items, night events). Only when the caller names the Hermes home. Whose each piece
        is, is checked piece by piece, never inferred from another (Codex review, 30.10.2026): the report (MEMORY.md
        entries, its conflicts, the night's stats) and the approval queue are the profile store's — <home>/memory_store.db
        — and never reach a map of another store; a merge question reaches a map only when it names this very store
        by an absolute path (one without a store is the profile's; a relative one proves nothing: second review,
        30.10.2026); a report item of any section that names another store is that store's, its text and fact number
        too. A file that is missing, unreadable or of another shape is a state, never an error; a record of another
        shape (a text the screen refuses or a date no calendar holds included) is left out and listed in ``invalid``
        with the field that is wrong."""
        if self.home is None:
            return {"read": False, "reason": "not_requested", "files": {}, "invalid": []}, [], []
        home = Path(self.home).expanduser()
        profile = _same_file(self.s.db_path, home / "memory_store.db")
        files, data, invalid = {}, {}, []
        for key, parts in _DREAM_FILES.items():
            if key != "proposals" and not profile:
                files[key] = "profile_only"  # not even opened: nothing in it is this store's
                continue
            path = home.joinpath(*parts)
            if key == "queue":
                files[key] = "absent" if not path.exists() else "ok" if path.is_dir() else "unreadable"
                continue
            files[key], data[key] = _load_json(path)
            if files[key] == "ok" and not isinstance(data[key], dict):
                files[key], data[key] = "unexpected", None
        proposals = (data.get("proposals") or {}).get("proposals")
        if files["proposals"] == "ok" and not isinstance(proposals, dict):
            files["proposals"], proposals = "unexpected", None
        records = self._own_proposals(proposals or {}, profile, invalid)
        if not profile and not records:
            return {"read": False, "reason": "other_store", "files": files, "invalid": invalid}, [], []
        current_version = {fid: v["version_id"] for fid, vs in versions.items() for v in vs if v["status"] == "current"}
        pending = self._proposals(records, facts, gone, current_version, merged)
        nights = []
        if profile and files["report"] == "ok":
            report = data["report"]
            # the night's moment: an ISO moment and nothing else (it is shown as written), in a calendar's range
            at = report.get("generated_at")
            moment = _moment(at, self.made.tzinfo) if at is not None else None
            if at is not None and moment is None:
                invalid.append({"file": "report", "id": "generated_at", "why": "generated_at"})
            day = moment.date().isoformat() if moment else None
            pending += self._report_items(report, facts, day, invalid)
            stats = report.get("stats") if isinstance(report.get("stats"), dict) else {}
            if any(not (isinstance(k, str) and _STAT_KEY.fullmatch(k)) for k in stats):
                invalid.append({"file": "report", "id": "stats", "why": "key"})  # the key itself is not repeated
            if day:
                alerts = report.get("alerts") if isinstance(report.get("alerts"), list) else []
                nights.append({"d": day, "k": "night", "at": at,
                               "alerts": sorted(str(a.get("kind")) for a in alerts
                                                if isinstance(a, dict) and _CODE.fullmatch(str(a.get("kind")))),
                               "stats": {k: v for k, v in sorted(stats.items())
                                         if isinstance(k, str) and _STAT_KEY.fullmatch(k) and _finite(v)}})
        if profile and files["queue"] == "ok":
            queue = self._queue(home.joinpath(*_DREAM_FILES["queue"]), invalid)
            if queue:
                pending.append(queue)
        return {"read": True, "reason": None, "files": files, "invalid": invalid}, pending, nights

    def _own_proposals(self, proposals: dict, profile: bool, invalid: list) -> list[tuple[str, dict, dict, dict]]:
        """(id, record, keep, copy) of the merge questions about this store, by number; the ones of another shape go
        to *invalid* (an id the map cannot read is not repeated: null)."""
        def bad(pid, why):
            invalid.append({"file": "proposals", "id": pid, "why": why})

        ids = [k for k in proposals if isinstance(k, str) and _PROPOSAL_ID.fullmatch(k)]
        for _ in range(len(proposals) - len(ids)):
            bad(None, "id")
        out = []
        for pid in sorted(ids, key=lambda k: int(k[2:])):
            rec = proposals[pid]
            if not isinstance(rec, dict):
                bad(pid, "record")
                continue
            whose = _record_store(rec.get("store"), self.s.db_path)
            if whose in ("bad", "relative"):
                bad(pid, "store" if whose == "bad" else "store_relative")
                continue
            if whose == "other" or (whose == "none" and not profile):
                continue  # about another store; or none named: the dream's profile store, whatever numbers match
            keep, copy = rec.get("keep"), rec.get("copy")
            if not isinstance(keep, dict) or not isinstance(copy, dict):
                bad(pid, "keep" if not isinstance(keep, dict) else "copy")
                continue
            if not all(isinstance(i, int) and not isinstance(i, bool) for i in (keep.get("fact_id"), copy.get("fact_id"))):
                bad(pid, "fact_id")
                continue
            out.append((pid, rec, keep, copy))
        return out

    def _proposals(self, records, facts, gone, current_version, merged) -> list[dict]:
        out = []
        for pid, rec, keep, copy in records:
            ids = (keep["fact_id"], copy["fact_id"])
            if set(ids) & gone:
                continue  # a question about a forgotten fact is gone with it
            if rec.get("status") == "done":
                for m in merged:
                    if (m["into"], m["from"]) == ids and m["prop"] is None:
                        m["prop"] = pid
                        if m["into"] in facts:
                            for f in facts[m["into"]]["mergedFrom"]:
                                if f["id"] == m["from"]:
                                    f["prop"] = pid
                continue
            if rec.get("status") != "open" or not all(i in facts for i in ids):
                continue  # declined / stale questions, or one about a fact the screen hides
            extra = copy.get("extra_words") if isinstance(copy.get("extra_words"), list) else []
            out.append({"kind": "merge", "id": pid, "keep": ids[0], "copy": ids[1], "covered": bool(copy.get("covered")),
                        "extra": [w for w in extra if isinstance(w, str) and _shown(w) is not None],
                        "created": _iso_day(rec.get("created"), self.made.tzinfo),
                        "asked": _iso_day(rec.get("last_published"), self.made.tzinfo),
                        "status": "open",
                        "current": all(current_version.get(i) == side.get("version_id")
                                       for i, side in zip(ids, (keep, copy)))})
        return out

    def _forgotten_text(self, text: str) -> bool:
        """Does *text* say what a forgotten fact said (the provider's tombstone check)?"""
        return bool(self.history and self.s._meta("tombstone_key") and self.s.forgotten_match(text))

    def _report_items(self, report, facts, day, invalid) -> list[dict]:
        """The report's questions for the profile store (the caller checked whose the report is); records of another
        shape go to *invalid*, by section and index. Every item of every section is checked for whose it is (an item
        that names another store — dream names a chat's own database by its key — is that store's, its text and its
        fact number too), and every text of it that reaches the JSON passes the screen."""
        def bad(where, why):
            invalid.append({"file": "report", "id": where, "why": why})

        def section(name):
            items = report.get(name)
            if items is not None and not isinstance(items, list):
                bad(name, "list")
                return []
            return items or []

        def ours(item, where) -> bool:
            whose = _record_store(item.get("store"), self.s.db_path)
            if whose == "bad":
                bad(where, "store")
            return whose in ("none", "this")

        out = []
        n = 0
        for i, item in enumerate(section("conflicts")):
            where = f"conflicts[{i}]"
            if not isinstance(item, dict):
                bad(where, "record")
                continue
            if not ours(item, where):
                continue  # a chat's own fact: its number is that store's, not this one's
            fid, hits = item.get("fact_id"), item.get("conflicts")
            if not isinstance(fid, int) or isinstance(fid, bool):
                bad(where, "fact_id")
                continue
            if fid not in facts:
                continue  # a fact forgotten (or hidden) since the night: nothing of it, its numbers included
            if not isinstance(hits, list):
                bad(where, "conflicts")
                continue
            for j, hit in enumerate(hits):
                entry = hit.get("entry") if isinstance(hit, dict) else None
                if not isinstance(entry, str):
                    bad(f"{where}.conflicts[{j}]", "record" if not isinstance(hit, dict) else "entry")
                    continue
                if hit.get("replace_unsafe") or _screen(entry) or self._forgotten_text(entry):
                    continue
                lists, wrong = _hit_lists(hit)
                if wrong:
                    bad(f"{where}.conflicts[{j}]", wrong)
                    continue
                n += 1
                out.append({"kind": "memory_conflict", "id": f"M-{n}", "fact": fid, "entry": entry, **lists,
                            "asked": day, "status": "open"})
        for i, item in enumerate(section("md_decays")):
            where = f"md_decays[{i}]"
            if not isinstance(item, dict) or not isinstance(item.get("entry"), str):
                bad(where, "record" if not isinstance(item, dict) else "entry")
                continue
            if not ours(item, where):
                continue  # MEMORY.md is the profile's; an item that names another store is that store's
            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                bad(where, "index")  # dream's index of the entry in MEMORY.md, a number: the map shows it as R-<n>
                continue
            if _screen(item["entry"]) or self._forgotten_text(item["entry"]):
                continue
            days = item.get("last_mention_days")
            out.append({"kind": "relevant", "id": f"R-{index}", "text": item["entry"],
                        "last_mention_days": days if _finite(days) else None, "asked": day, "status": "open"})
        return out

    def _queue(self, base: Path, invalid: list) -> dict | None:
        """Writes to long-term memory waiting for approval: how many and since when — never their text (they are
        not memory yet). A record's date no calendar holds is that record's: it is counted, its date is not, and it
        goes to *invalid* by its file name (when the name is a plain one; else null)."""
        records = writes = unreadable = 0
        oldest = None
        for path in sorted(base.glob("*.json")):
            state, data = _load_json(path)
            if state != "ok" or not isinstance(data, dict):
                unreadable += 1
                continue
            records += 1
            payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
            ops = payload.get("operations")
            writes += len([o for o in ops if isinstance(o, dict)]) if isinstance(ops, list) else 1
            created = data.get("created_at")
            if isinstance(created, (int, float)) and not isinstance(created, bool):
                try:
                    moment = datetime.fromtimestamp(created, self.made.tzinfo)
                except (OverflowError, OSError, ValueError):  # 1e100, NaN, before the platform's epoch range
                    name = path.name if _QUEUE_FILE.fullmatch(path.name) and not _screen(path.name) else None
                    invalid.append({"file": "queue", "id": name, "why": "created_at"})
                    continue
                oldest = moment if oldest is None else min(oldest, moment)
        if not records and not unreadable:
            return None
        return {"kind": "queue", "id": "Q", "records": records, "writes": writes, "unreadable": unreadable,
                "oldest": oldest.date().isoformat() if oldest is not None else None, "status": "open"}

    # -- the week --
    def _feed(self, facts, versions, confirmations, names, forgotten, merged, nights) -> list[dict]:
        s, start, end = self.s, self.week_from, self.today
        inside = lambda d: d is not None and start <= d <= end  # noqa: E731
        out = []
        for fid, fact in facts.items():
            rows = versions.get(fid, [])
            if rows and inside(rows[0]["recorded_on"]):
                out.append({"d": rows[0]["recorded_on"], "k": "new", "f": fid})
            elif not rows and inside(fact["rec"]):
                out.append({"d": fact["rec"], "k": "new", "f": fid})
            by_id = {v["version_id"]: v for v in rows}
            for v in rows[1:]:
                if v["status"] in ("current", "changed", "ended") and inside(v["recorded_on"]):
                    before = by_id.get(v["supersedes"])
                    out.append({"d": v["recorded_on"], "k": "chg", "f": fid,
                                "how": "corrected" if before is not None and before["status"] == "corrected" else "changed"})
            if fact["status"] == "ended" and rows:
                closed = _day(s, rows[-1]["closed_at"]) if rows[-1]["status"] == "ended" else None
                if inside(closed):
                    out.append({"d": closed, "k": "end", "f": fid})
        for c in confirmations:
            d = _day(s, c["confirmed_at"])
            if c["fact_id"] in facts and inside(d):
                out.append({"d": d, "k": "conf", "f": c["fact_id"], "by": _shown(names.get(c["user_id"]))})
        for r in forgotten:
            d = _day(s, r["forgotten_at"])
            if inside(d):
                out.append({"d": d, "k": "del", "id": r["fact_id"]})
        for m in merged:
            if inside(m["d"]):
                out.append({"d": m["d"], "k": "mrg", "from": m["from"], "into": m["into"]})
        out += [n for n in nights if inside(n["d"])]
        key = lambda e: (e["d"], -_FEED_ORDER[e["k"]], -(e.get("f") or e.get("id") or e.get("from") or 0))  # noqa: E731
        return sorted(out, key=key, reverse=True)


# -- the size limit (docs/memory-map.md, "Size") -------------------------------------------------------------------
MAX_FACTS = 1000
MAX_ENTITIES = 300
_FACT_REFS = ("a", "b", "keep", "copy", "fact", "f", "into")


def _refs(items) -> list[int]:
    """Fact ids the records name, in their order, each once."""
    out: list[int] = []
    for item in items:
        if isinstance(item, dict):
            for key in _FACT_REFS:
                v = item.get(key)
                if isinstance(v, int) and not isinstance(v, bool) and v not in out:
                    out.append(v)
    return out


def _limit(snapshot: dict, max_facts: int | None, max_entities: int | None) -> dict:
    """Cut a snapshot to at most *max_facts* facts and *max_entities* names (None: no limit) and say so in
    SNAP.limits. Facts a person is asked about or that changed this week come first (contradictions, then the
    nightly review's questions, then the week's feed, newest first), then current facts and then closed ones, each
    newest first by when the current text was recorded. Names with the most facts stay. SNAP.counts and a name's
    `facts` stay the whole store's; a kept fact and a contradiction list only the kept names. Records that name a cut
    fact (a merge, a question, a feed line) keep their own fields."""
    facts, ents = snapshot["FACTS"], snapshot["ENT"]
    limits = {"facts": {"max": max_facts, "total": len(facts), "shown": len(facts)},
              "entities": {"max": max_entities, "total": len(ents), "shown": len(ents)}}
    if max_facts is not None and len(facts) > max_facts:
        by_id = {f["id"]: f for f in facts}
        feed = sorted((x for x in snapshot["FEED"] if isinstance(x, dict)), key=lambda x: str(x.get("d") or ""),
                      reverse=True)
        first = [i for i in _refs(snapshot["CONFLICTS"]) + _refs(snapshot["PENDING"]) + _refs(feed) if i in by_id]
        rest = sorted(facts, key=lambda f: (str(f.get("rec") or ""), f["id"]), reverse=True)
        rest = sorted(rest, key=lambda f: f["status"] != "current")  # stable: current first, each newest first
        keep = set(list(dict.fromkeys(first + [f["id"] for f in rest]))[:max_facts])
        snapshot["FACTS"] = [f for f in facts if f["id"] in keep]
        limits["facts"]["shown"] = len(snapshot["FACTS"])
    if max_entities is not None and len(ents) > max_entities:
        ids = {e["id"] for e in sorted(ents, key=lambda e: (-(e.get("facts") or 0), e["id"]))[:max_entities]}
        snapshot["ENT"] = [e for e in ents if e["id"] in ids]
        for f in snapshot["FACTS"]:
            f["ents"] = [e for e in f["ents"] if e in ids]
        for c in snapshot["CONFLICTS"]:
            c["shared"] = [e for e in c.get("shared") or [] if e in ids]
        limits["entities"]["shown"] = len(snapshot["ENT"])
    snapshot["SNAP"]["limits"] = limits
    return snapshot


def export_map(store_path, *, hermes_home=None, tz=None, now: datetime | None = None, config=None,
               max_facts: int | None = MAX_FACTS, max_entities: int | None = MAX_ENTITIES) -> dict:
    """The memory map of the store at *store_path* as a dict (schema_version, SNAP, ENT, FACTS, CONFLICTS, PENDING,
    TOMBS, MERGES, HIDDEN, FEED, ATTEMPTS). *hermes_home*: where the dreaming files are (none read without it).
    *tz*: the zone of the snapshot's dates (IANA name or "+07:00"; UTC by default). *now*: the snapshot's moment.
    *config*: the plugin's settings search ranks by (its config.yaml path, or the plugins.hermes-memory-store section
    as a dict); None = the plugin's defaults. *max_facts* / *max_entities*: the size limit (None: none; see
    _limit). Read-only; raises FileNotFoundError when there is no store file."""
    for name, value in (("max_facts", max_facts), ("max_entities", max_entities)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{name} must be a positive whole number or None")
    zone, zone_name = _zone(tz)
    made = (now or datetime.now(timezone.utc))
    made = (made if made.tzinfo else made.replace(tzinfo=timezone.utc)).astimezone(zone)
    store = _ReadOnlyStore(store_path, tz=zone)
    try:
        snapshot = _Snapshot(store, made, zone_name, hermes_home, _ranking(config)).build()
        if not store.in_snapshot():  # something ended the read transaction midway: the reads are not one moment
            raise RuntimeError("memory_map: the read transaction ended before the export did")
        return _limit(snapshot, max_facts, max_entities)
    finally:
        store.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only memory map of one holographic store, as JSON.")
    parser.add_argument("--store", required=True, help="the store file (memory_store.db or a scope's .db)")
    parser.add_argument("--hermes-home", help="read the dreaming files under this directory (only for its store)")
    parser.add_argument("--tz", help="zone of the dates: an IANA name or +HH:MM (default UTC)")
    parser.add_argument("--now", help="the snapshot's moment, ISO 8601 (default: now)")
    parser.add_argument("--config", help="Hermes config.yaml: the search settings that order contradictions "
                                         "(default: the plugin's defaults)")
    parser.add_argument("--max-facts", type=int, default=MAX_FACTS,
                        help=f"at most this many facts in the map (default {MAX_FACTS}; 0: no limit)")
    parser.add_argument("--max-entities", type=int, default=MAX_ENTITIES,
                        help=f"at most this many names in the map (default {MAX_ENTITIES}; 0: no limit)")
    parser.add_argument("--json", action="store_true", help="print JSON (the only output format so far)")
    args = parser.parse_args(argv)
    try:
        now = datetime.fromisoformat(args.now) if args.now else None
        data = export_map(args.store, hermes_home=args.hermes_home, tz=args.tz, now=now, config=args.config,
                          max_facts=args.max_facts or None, max_entities=args.max_entities or None)
    except FileNotFoundError as exc:
        print(f"memory_map: {exc}", file=sys.stderr)
        return 2
    except sqlite3.DatabaseError as exc:  # the kind only: SQLite may quote a stored value in its message
        print(f"memory_map: the store could not be read ({error_kind(exc)})", file=sys.stderr)
        return 1
    except ValueError as exc:  # --now or --tz
        print(f"memory_map: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
