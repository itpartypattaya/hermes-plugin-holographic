"""Fact retrieval for the memory store: FTS5 keyword candidates reranked with Jaccard and HRR similarity, weighted by
trust (ported from KIK memory_agent.py); entity recall straight from the entity index."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

from . import holographic as hrr
from . import text as _text
from .embeddings import Meaning, VectorIndex
from .history import CLOSED, error_kind, version_state
from .text import (CJK_NEGATORS, CLAIM_WORDS, CONTRASTIVE, NEGATORS, UNSEGMENTED, analyze, content_chars,
                   content_keys, content_stems, gram_coverage, is_chinese, is_greeting, is_sea, name_forms, name_lemmas,
                   query_grams, query_terms, question_parts, run_grams, soft_jaccard, soft_overlap, words,
                   zh_char_text)

logger = logging.getLogger(__name__)

_FACT_COLUMNS = "fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at"
_F_COLUMNS = ", ".join(f"f.{c.strip()}" for c in _FACT_COLUMNS.split(","))
_ROLE_ENTITY, _ROLE_CONTENT = hrr.ROLE_ENTITY, hrr.ROLE_CONTENT
# FTS5 operator characters. Inside a token they become spaces of a quoted phrase: "pve-01" -> "pve 01", the same two
# tokens unicode61 indexed, adjacent (deleting them gave "pve01", which matches nothing).
_FTS_OPERATORS = str.maketrans({c: " " for c in '"()*^:-+'})
_RE_UNSEGMENTED = UNSEGMENTED
# The commonest Indonesian function words: a store with none of them holds no Indonesian, and Latin words of a query
# of undetermined language need no word scan (Indonesian prefixes are what hides a word from FTS5).
_INDONESIAN_PROBE = '"yang" OR "dengan" OR "untuk" OR "tidak" OR "adalah" OR "sudah" OR "belum"'
# ... and of German, whose compounds hide their last word from a prefix query ("Katze" in "Familienkatze").
_GERMAN_PROBE = '"und" OR "nicht" OR "ist" OR "sind" OR "der" OR "das" OR "eine" OR "mit"'
# ... Dutch (plurals that change the word: "raam" / "ramen"), Turkish (consonant changes: "kitap" / "kitabı") and
# Polish (ł, and words typed without their marks), with words no other supported language writes as a word of its own.
_DUTCH_PROBE = '"het" OR "een" OR "niet" OR "zijn" OR "wordt" OR "naar"'
_TURKISH_PROBE = '"bir" OR "için" OR "değil" OR "gibi" OR "olarak" OR "çok"'
_POLISH_PROBE = '"się" OR "jest" OR "oraz" OR "który" OR "przez" OR "jak"'
_SCAN_PROBE = " OR ".join((_INDONESIAN_PROBE, _GERMAN_PROBE, _DUTCH_PROBE, _TURKISH_PROBE, _POLISH_PROBE))
# A question told to be in one of these languages scans whatever the store holds: the probe words above are a sample,
# and a store of short facts may have none of them ("Kitabı masada bıraktım." asked as "Kitap nerede?")
_SCAN_LANGUAGES = frozenset({"id", "de", "tr", "pl", "nl"})


# Search by meaning: what a message names (_inner_names).
_RE_NAME_WORD = re.compile(r"[^\W\d_]+")
_RE_SENTENCE_END = re.compile(r"[.!?…。！？]+(?:\s+|$)")


def _inner_names(query: str) -> list[str]:
    """Capitalized words inside a sentence ("When is Peter's birthday?" -> Peter): what a message names. Not a
    sentence's first word, and nothing in German, where every noun is capitalized."""
    if not query or analyze(query).german:
        return []
    return _inner_capitals(query)


def _inner_capitals(query: str) -> list[str]:
    """Capitalized words inside a sentence, German nouns among them: what may be a name."""
    names = []
    for sentence in _RE_SENTENCE_END.split(query):
        for i, m in enumerate(_RE_NAME_WORD.finditer(sentence)):
            word = m.group(0)
            if i and len(word) > 1 and word[0].isupper() and word[1:].islower():
                names.append(word)
    return names


def _capitalized(text: str, inner: bool) -> str:
    """*text* with every word that is not written with a capital (or, when *inner*, is the first of its sentence) cut
    out as a line break: what store.mentioned_entity_ids may read as a name, and nothing a lower-case word says («чай»
    is no mention of a «Чай»)."""
    ends = [m.end() for m in _RE_SENTENCE_END.finditer(text or "")]
    out, last, e, first = [], 0, 0, True
    for m in _RE_NAME_WORD.finditer(text or ""):
        while e < len(ends) and ends[e] <= m.start():
            e, first = e + 1, True
        word = m.group(0)
        capital = len(word) > 1 and word[0].isupper() and word[1:].islower() and not (inner and first)
        out += [text[last:m.start()], word if capital else "\n"]
        last, first = m.end(), False
    return "".join(out) + (text or "")[last:]


def _name_forms(analysis) -> frozenset:
    """Keys of a fact's words read as names (text.name_forms): the short stems a plain text drops, by which a case
    form of a known name is found in it («Яны» holds «ян»). Not its function words: «и» is no form of «Ия»."""
    return frozenset().union(*(name_forms(w.text) for w in analysis.words if not w.stop and not w.identifier))


def _fact_text(fact: dict) -> str:
    """What a fact is searched by: its text and its tags, where the agent's keywords are kept (README, keywords)."""
    return f'{fact.get("content", "")} {fact.get("tags") or ""}'


_WARNED: set[str] = set()  # failures said once each (a damaged index fails on every turn)
# Each line is a fixed place and an error's kind, so the set stays small by itself; the bound only stops a fault
# from growing it for ever. 64 was reached by the provider's actions alone and hid the next new failure (Codex
# review of 0.17.8).
_WARNED_MAX = 4096


def _warn_once(what: str, exc: Exception) -> None:
    text = f"{what} ({error_kind(exc)})"
    if text not in _WARNED and len(_WARNED) < _WARNED_MAX:
        _WARNED.add(text)
        logger.warning("holographic: %s", text)


def _fact_row(row) -> dict:
    """A fact row as a dict with our own columns only: a store migrated from another embedding deployment can carry
    e.g. a ``facts.embedding`` BLOB, and ``SELECT *`` / ``f.*`` would hand it to the tool result, where JSON
    serialization fails on every search (reported on a live store, 0.14.1) — and a foreign text column would leak
    whatever it holds. ``hrr_vector`` stays: scoring reads it and pops it before anything is returned."""
    return {k: v for k, v in dict(row).items() if k in _ROW_KEYS}


def _scan_language(query: str) -> bool:
    """Is *query* in a language whose other word forms the FTS index cannot reach (told by its function words)?"""
    if not query:
        return False
    analysis = analyze(query)
    return analysis.determined and not _SCAN_LANGUAGES.isdisjoint(analysis.langs)


class _Corpus:
    """What a search runs over: the current facts, or (``as_of``) the versions that may have held on a date. *source*
    is a table or a subquery with the columns of ``facts``, *key* its row id, *fts* the FTS5 index over it."""

    __slots__ = ("source", "params", "fts", "key", "cache")

    def __init__(self, source: str, params: tuple, fts: str, key: str, cache: dict):
        self.source, self.params, self.fts, self.key, self.cache = source, params, fts, key, cache


# Versions that may have held on a date (as_of): corrected ones never did. Trust: the fact's while the version is
# current, else the snapshot taken when it was closed.
_AS_OF_TEMPLATE = (
    "(SELECT v.version_id, v.fact_id, v.seq, v.content, v.category, v.tags, v.status, v.valid_from, v.valid_to, "
    "v.valid_to_exact, v.recorded_on, v.recorded_at AS created_at, COALESCE(v.closed_at, v.recorded_at) AS updated_at, "
    "COALESCE(c.trust_score, v.trust_score, ?) AS trust_score, 0 AS retrieval_count, NULL AS hrr_vector{merged} "
    "FROM fact_versions v LEFT JOIN facts c ON c.fact_id = v.fact_id AND v.status = 'current' {join}"
    "WHERE v.status != 'corrected' AND (v.valid_from IS NULL OR v.valid_from <= ?) AND (v.valid_to IS NULL OR v.valid_to > ?)"
    "{gone})")
# A merged version (0.19.0) carries its merge row: the day it was merged clips it (HistoryMixin.clip_merged). A file
# no writer of 0.19.0 has opened has no fact_merges table: a reader of it takes the source without the join.
_AS_OF_MERGED = {"merged": ", m.merged_at, m.into_fact_id AS merged_into",
                 "join": "LEFT JOIN fact_merges m ON m.from_fact_id = v.fact_id AND v.status = 'merged' "}
_AS_OF_SOURCE = _AS_OF_TEMPLATE.format(**_AS_OF_MERGED, gone="")
_AS_OF_SOURCE_PLAIN = _AS_OF_TEMPLATE.format(merged="", join="", gone="")
# Every column a candidate read may return: the facts table's, the as_of source's extras and the FTS rank. Anything
# else in a row (a column left by another program) is not ours and never reaches a result (_fact_row).
_AS_OF_EXTRA = ("version_id", "seq", "status", "valid_from", "valid_to", "valid_to_exact", "recorded_on", "merged_at",
                "merged_into")
_ROW_KEYS = frozenset([c.strip() for c in _FACT_COLUMNS.split(",")] + ["hrr_vector", "fts_rank_raw", *_AS_OF_EXTRA])


def _shift(sim: float) -> float:
    """Cosine similarity [-1, 1] -> [0, 1]."""
    return (sim + 1.0) / 2.0


def _related_floor(dim: int) -> float:
    """Residual similarity a fact must reach to count as structurally tied to an entity in `related`. Measured at
    dim=1024 on 400 facts with one to four entities: tied facts >= 0.35, the best of 17k untied pairs 0.088; 0.2 sits
    in the gap with a wide margin. Small dimensions get a proportionally higher floor (8 sd of chance similarity)."""
    return min(0.5, max(0.2, 8.0 * math.sqrt(0.5 / max(dim, 1))))


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(self, store: MemoryStore, temporal_decay_half_life: int = 0,  # days, 0 = disabled
                 fts_weight: float = 0.4, jaccard_weight: float = 0.3, hrr_weight: float = 0.3, hrr_dim: int = 1024):
        self.store, self.half_life, self.hrr_dim = store, temporal_decay_half_life, hrr_dim
        if hrr_weight > 0 and not hrr.has_numpy():  # redistribute weights without numpy
            fts_weight, jaccard_weight, hrr_weight = 0.6, 0.4, 0.0
        self.fts_weight, self.jaccard_weight, self.hrr_weight = fts_weight, jaccard_weight, hrr_weight
        # row id -> (text, Analysis), for word-level scans: one cache for the current facts, one for versions
        self._current = _Corpus("facts", (), "facts_fts", "fact_id", {})
        self._version_cache: dict[int, tuple[str, object]] = {}
        self._vectors: VectorIndex | None = None  # search by meaning, built on first use
        # prefetch_clause_coverage (an experiment, off by default): relevant() gives a fact it left out a second look
        # against each of a message's two questions (_clause_rescue)
        self.clause_coverage = False

    def _as_of(self, day: str) -> _Corpus:
        if self.store._merges_ready():
            # Merged versions that no longer hold on the day leave before ranking (they would take the pool).
            gone = self.store.merged_gone_by(day)
            source = _AS_OF_TEMPLATE.format(**_AS_OF_MERGED, gone=f" AND v.version_id NOT IN "
                                            f"({','.join(str(int(i)) for i in gone)})" if gone else "")
        else:
            source = _AS_OF_SOURCE_PLAIN
        return _Corpus(source, (self.store.default_trust, day, day), "fact_versions_fts", "version_id",
                       self._version_cache)

    def _atom(self, word: str):
        return hrr.encode_atom(word, self.hrr_dim)

    def _phases(self, blob: bytes):
        """Decode a stored vector; None for a blob of another dimension or a damaged one (logged, row skipped)."""
        try:
            return hrr.bytes_to_phases(blob, dim=self.hrr_dim)
        except (ValueError, TypeError) as exc:
            logger.warning("holographic: skipping an unreadable HRR vector (%s)", exc)
            return None

    def search(self, query: str, category: str | None = None, min_trust: float = 0.3, limit: int = 10,
               corpus: _Corpus | None = None, roots: bool = True, meaning: Meaning | None = None) -> list[dict]:
        """FTS5 candidates (limit*3) → Jaccard + HRR rerank → trust weighting → optional temporal decay
        0.5^(age_days / half_life). Returns fact dicts with 'score', sorted desc. *corpus*: the current facts by
        default; search_as_of passes the versions of a date. *roots*=False matches words exactly, as 0.8.0 did (the
        candidates of write_check: a word of the same root must not crowd a conflict out). *meaning* (search by
        meaning on: prefetch and fact_store search only) adds closeness in meaning to the rerank; the candidates are
        still the words' (by_meaning offers the rest)."""
        corpus = corpus or self._current
        key = corpus.key
        terms = query_terms(query or "")
        candidates = {f[key]: f for f in self._fts_candidates(query, category, min_trust, limit * 3, corpus)
                      if self._contains_a_term(terms, f, exact=not roots)}
        extra = " OR ".join(e for e in (t.fts_extra() for t in terms if t.fts_ok) if e) if roots else ""
        if extra:  # words of the same root: only what the words' own forms missed, ranked below them
            for fact in self._fts_query(extra, category, min_trust, limit * 3, corpus):
                if fact[key] not in candidates and self._contains_a_term(terms, fact):
                    fact["fts_rank"] *= _text._DERIV_WEIGHT
                    candidates[fact[key]] = fact
        for fact in self._ngram_candidates(query, category, min_trust, limit * 3, corpus):
            if fact[key] not in candidates or candidates[fact[key]].get("fts_rank", 0) < fact["fts_rank"]:
                candidates[fact[key]] = fact
        for fact in self._scan_candidates(terms, category, min_trust, limit * 3, corpus, exact=not roots, query=query):
            candidates.setdefault(fact[key], fact)  # fills in what FTS cannot see; FTS ranks what it found
        # BM25 does not see roots: a fact holding "Марина" and "по ночам" ranks with every other fact about Marina for
        # "ночная подработка Марины", and "The walker" matches the prefix "walk"* as fully as a fact with "walk". When
        # any candidate meets any term only through a root (its own or the fact's), all of them are ranked by the
        # weighted share of the query they hold, which counts the root at _DERIV_WEIGHT (as as_of always is).
        if corpus is not self._current:
            self._corpus_rank(query, terms, list(candidates.values()), category, min_trust, corpus)
        elif roots and any(0.0 < t.match(analyze(_fact_text(f))) < 1.0
                           for f in candidates.values() for t in terms):
            self._coverage_rank(query, terms, list(candidates.values()))
        return self._rerank(query, list(candidates.values()), limit, meaning if corpus is self._current else None)

    def _corpus_rank(self, query: str, terms, candidates: list[dict], category, min_trust, corpus: _Corpus) -> None:
        """FTS5's BM25 counts every row of its index; the versions index holds corrected versions and every other
        date too, so for as_of its rank would weigh words by versions that are not in the corpus. Here the weights
        come from the corpus itself: the IDF-weighted share of the query's terms a version contains, or (an
        unsegmented script) the share of the query's n-grams, whichever is higher."""
        if not candidates:
            return
        texts = self._texts(category, min_trust, corpus)
        df = {t: sum(1 for _, _, a in texts if t.found_in(a)) for t in terms}
        weight = {t: math.log(1.0 + len(texts) / df[t]) for t in terms if df[t]}
        # A word no version of the date contains still says what the query is about (as in `relevant`): it counts
        # against coverage, weighted like the commonest present word.
        total = sum(weight.values()) + sum(1 for t in terms if not df[t]) * (min(weight.values()) if weight else 0.0)
        grams, zh = query_grams(query), is_chinese(query or "")
        for fact in candidates:
            analysis = analyze(_fact_text(fact))
            share = sum(w * t.match(analysis) for t, w in weight.items()) / total if total else 0.0
            fact["fts_rank"] = max(share, gram_coverage(grams, analysis, zh) if grams else 0.0)

    def _coverage_rank(self, query: str, terms, candidates: list[dict]) -> None:
        """The same weighted share for the current facts, cheaply: document frequencies from the FTS index (one COUNT
        per term, as the recall gate takes them), the share computed on the candidates only, not on every fact."""
        total_facts = max(self.store.count_facts(), 1)
        df = {t: self._doc_freq(t) for t in terms}
        weight = {t: math.log(1.0 + total_facts / df[t]) for t in terms if df[t]}
        if not weight:
            return
        total = sum(weight.values()) + sum(1 for t in terms if not df[t]) * min(weight.values())
        grams, zh = query_grams(query), is_chinese(query or "")
        for fact in candidates:
            analysis = analyze(_fact_text(fact))
            share = sum(w * t.match(analysis) for t, w in weight.items()) / total
            fact["fts_rank"] = max(share, gram_coverage(grams, analysis, zh) if grams else 0.0)

    def search_as_of(self, query: str, day: str, category: str | None = None, min_trust: float = 0.3,
                     limit: int = 10) -> dict:
        """What held on *day*: the search runs over the versions that may have held then (candidates, IDF, the
        floor and the limit are all computed over them, so limit=1 is the best version for that date). Versions
        known to hold go to 'results', those whose start or end is not known to the day to 'undetermined'."""
        self.store.require_history()
        held, maybe = [], {}
        # A wide pool: FTS5 picks candidates by its global rank, which knows nothing of the date; the corpus rank
        # (_corpus_rank) then orders them by the versions of that date only.
        for found in self.search(query, category=category, min_trust=min_trust, limit=max(limit * 10, 100),
                                 corpus=self._as_of(day)):
            v = self.store.clip_merged(found)  # a merged version (0.19.0) holds until the day of the merge
            if v is None:
                continue
            state, why = version_state(v, day)
            entry = {"fact_id": v["fact_id"], "version": v["seq"], "content": v["content"], "category": v["category"],
                     "valid_from": v["valid_from"] or "unknown",
                     "valid_to": v["valid_to"] if not v["valid_to"] or v["valid_to_exact"] else f"before {v['valid_to']}",
                     "status": v["status"], "score": round(v["score"], 4)}
            if v["status"] == "merged":
                entry["merged_into"] = v["merged_into"]
            if state == "holds" and all(h["fact_id"] != v["fact_id"] for h in held):
                held.append(entry)
            elif state == "undetermined":
                maybe.setdefault(v["fact_id"], {**entry, "reason": why})
        known = {h["fact_id"] for h in held}
        undetermined = [e for fact_id, e in maybe.items() if fact_id not in known]
        return {"as_of": day, "results": held[:limit], "undetermined": undetermined[:limit]}

    @staticmethod
    def _contains_a_term(terms, fact: dict, exact: bool = False) -> bool:
        """A stem prefix ("contra"*) also matches other words: keep a candidate only when it really contains a query
        term (same word, same stem, the same identifier with word boundaries, or, unless *exact*, a word of the same
        root)."""
        if not terms:
            return True
        analysis = analyze(_fact_text(fact))
        return any(t._exact(analysis) if exact else t.found_in(analysis) for t in terms)

    def _texts(self, category: str | None = None, min_trust: float | None = None,
               corpus: _Corpus | None = None) -> list[tuple[int, float, object]]:
        """(row id, trust, Analysis) of every fact (or version) passing the filters. The rows are read on every call
        (the table is the truth; another process may have written); a row is analysed again only when its text
        changed, so the first scan after start analyses every fact and later ones only what changed."""
        corpus = corpus or self._current
        where = [c for c, on in (("f.trust_score >= ?", min_trust is not None), ("f.category = ?", category)) if on]
        params = [p for p, on in ((min_trust, min_trust is not None), (category, category)) if on]
        rows = self.store.read(f"SELECT f.{corpus.key} AS rid, f.content, f.tags, f.trust_score FROM {corpus.source} f"
                               + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY rid",
                               [*corpus.params, *params])
        cache, out = corpus.cache, []
        for row in rows:
            text = f'{row["content"]} {row["tags"] or ""}'
            hit = cache.get(row["rid"])
            if hit is None or hit[0] != text:
                hit = cache[row["rid"]] = (text, analyze(text))
            out.append((row["rid"], row["trust_score"], hit[1]))
        if corpus is self._current and not where and len(cache) > len(out) + 1024:  # deleted facts: only an
            kept = {fact_id: cache[fact_id] for fact_id, _, _ in out}  # unfiltered read of the facts sees every one
            cache.clear()
            cache.update(kept)
        elif corpus is not self._current and len(cache) > 20_000:
            cache.clear()
        return out

    def _scan_terms(self, terms, corpus: _Corpus | None = None, query: str = "") -> list:
        """The terms matched word by word: Arabic and Indic always; Latin ones (Indonesian, German, Dutch, Turkish,
        Polish, or a query of undetermined language) only when the store holds facts in one of those languages, so an
        English store never pays for a scan."""
        fts = (corpus or self._current).fts
        scan = [t for t in terms if t.scan]
        if any(t.fts_ok for t in scan) and not _scan_language(query):
            try:
                needed = bool(self.store.read(f"SELECT 1 FROM {fts} WHERE {fts} MATCH ? LIMIT 1", (_SCAN_PROBE,)))
            except Exception:  # noqa: BLE001 - a damaged index: the scan does not need it
                needed = True
            if not needed:
                scan = [t for t in scan if not t.fts_ok]
        return scan

    def _rows(self, ids: list[int], category: str | None = None, min_trust: float | None = None,
              corpus: _Corpus | None = None) -> list[dict]:
        """Full rows of *ids*, the filters applied again: the fact may have changed since its text was read."""
        if not ids:
            return []
        corpus = corpus or self._current
        sql = (f"SELECT * FROM {corpus.source} f WHERE f.{corpus.key} IN ({','.join('?' * len(ids))})"
               + (" AND f.trust_score >= ?" if min_trust is not None else "") + (" AND f.category = ?" if category else ""))
        params = [*corpus.params, *ids, *([min_trust] if min_trust is not None else []), *([category] if category else [])]
        return [_fact_row(r) for r in self.store.read(sql, params)]

    def _scan_candidates(self, terms, category: str | None, min_trust: float, limit: int,
                         corpus: _Corpus | None = None, exact: bool = False, query: str = "") -> list[dict]:
        """Facts containing the query's scan terms (Arabic, Indic, Indonesian: words whose other forms the FTS5
        index cannot find), matched word by word against the text of every fact and ranked by the IDF-weighted share
        of those terms they contain ('fts_rank'). A scan, like the n-gram path: milliseconds at the size this store
        is meant for."""
        corpus = corpus or self._current
        scan = self._scan_terms(terms, corpus, query)
        if not scan:
            return []
        texts = self._texts(category, min_trust, corpus)
        strength = (lambda t, a: float(t._exact(a))) if exact else (lambda t, a: t.match(a))
        found = [(rid, trust, {t: s for t in scan if (s := strength(t, a))}) for rid, trust, a in texts]
        df = {t: sum(t in hit for _, _, hit in found) for t in scan}
        weight = {t: math.log(1.0 + len(texts) / df[t]) for t in scan if df[t]}
        total = sum(weight.values()) or 1.0
        # ties: the more trusted fact, then the newer one
        ranked = sorted(((sum(weight[t] * s for t, s in hit.items()) / total, trust, rid) for rid, trust, hit in found
                         if hit), reverse=True)[:limit]
        rank = {rid: score for score, _, rid in ranked}
        rows = [f for f in self._rows([rid for _, _, rid in ranked], category, min_trust, corpus)
                if self._contains_a_term(scan, f, exact=exact)]
        for fact in rows:
            fact["fts_rank"] = rank[fact[corpus.key]]
        return rows

    def _ngram_candidates(self, query: str, category: str | None, min_trust: float, limit: int,
                          corpus: _Corpus | None = None) -> list[dict]:
        """Facts sharing character n-grams with the query's unsegmented runs, ranked by the share of n-grams they
        contain ('fts_rank'). A substring scan: no extra index, so nothing to migrate or keep in sync; at the size
        this store is meant for (hundreds to thousands of facts) it takes milliseconds."""
        corpus = corpus or self._current
        grams = query_grams(query)
        if not grams:
            return []
        # Matched against each fact's folded text (content and tags: the agent's keywords are searched like the text),
        # so Traditional and Simplified characters meet; the analysis is cached per fact, as for the word scan. A
        # Chinese query's pairs are matched as they stand in Chinese facts (gram_coverage, inlined: it runs per fact).
        zh, total = is_chinese(query or ""), sum(weight for _, weight in grams)
        plain = [(g.replace(_text.SEAM, ""), w) for g, w in grams]
        texts = self._texts(category, min_trust, corpus)
        held = ((sum(w for g, w in grams if g in a.seamed) if zh and a.seamed else
                 sum(w for g, w in plain if g in a.text), trust, rid) for rid, trust, a in texts)
        ranked = sorted(((share / total, trust, rid) for share, trust, rid in held), reverse=True)
        rank = {rid: score for score, _, rid in ranked[:limit] if score > 0}
        rows = self._rows(list(rank), category, min_trust, corpus)
        for fact in rows:
            fact["fts_rank"] = rank[fact[corpus.key]]
        return sorted(rows, key=lambda f: f["fts_rank"], reverse=True)

    def _rerank(self, query: str, candidates: list[dict], limit: int, meaning: Meaning | None = None) -> list[dict]:
        query_tokens = content_stems(query)
        # Query vector is loop-invariant; encode lazily on the first candidate that carries an HRR vector
        # so stores whose hrr_vector was never backfilled don't pay for it.
        query_vec = None  # bind(encode_text(query), ROLE_CONTENT): the shape a fact's content is stored in
        decays = self.decays(candidates) if self.half_life > 0 else None
        for i, fact in enumerate(candidates):
            jaccard = self._jaccard_similarity(query_tokens, content_stems(_fact_text(fact)))
            hrr_sim = 0.5  # neutral
            fact_vec = self._phases(fact["hrr_vector"]) if self.hrr_weight > 0 and fact.get("hrr_vector") else None
            if fact_vec is not None:
                if query_vec is None:
                    query_vec = hrr.bind(hrr.encode_text(query, self.hrr_dim), self._atom(_ROLE_CONTENT))
                hrr_sim = _shift(hrr.similarity(query_vec, fact_vec))
            relevance = self.fts_weight * fact.get("fts_rank", 0.0) + self.jaccard_weight * jaccard + self.hrr_weight * hrr_sim
            if meaning is not None:  # the words keep (1 - weight) of the say; off, the score is 0.13.0's to the bit
                cos = meaning.cos(fact["fact_id"]) if meaning.fits(fact["fact_id"], fact["content"]) else meaning.neutral
                relevance = (1.0 - meaning.weight) * relevance + meaning.weight * cos
            fact["score"] = relevance * fact["trust_score"]
            if decays is not None:
                fact["score"] *= decays[i]
        results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
        return results

    # -- search by meaning (embeddings.py; optional, online) -----------------------------------------------------

    def vector_index(self, tag: str) -> VectorIndex:
        """This store's vectors of model *tag* (one index, kept between queries)."""
        if self._vectors is None or self._vectors.tag != tag:
            self._vectors = VectorIndex(self.store, tag)
        return self._vectors

    def meaning(self, vector, tag: str, weight: float) -> Meaning | None:
        """Closeness of every current fact with a vector of *tag* to a message whose vector is *vector*; None when no
        fact has one of that size yet."""
        ids, matrix, hashes = self.vector_index(tag).matrix_with_hashes(int(vector.shape[0]))
        return Meaning(ids, matrix @ vector, weight, hashes) if ids else None

    def names_fit(self, query: str, fact: dict) -> bool:
        """Does a fact offered on meaning alone mention whoever the message names: every capitalized word inside a
        sentence, in any word form (or among the fact's keywords), and every known entity the message mentions? In
        meaning "When is Peter's birthday?" is as close to "Lena's birthday is on June 3" as a paraphrase is to its
        fact (calibration 2026-10-14); only the name tells them apart."""
        analysis, capitals = analyze(_fact_text(fact)), None
        for name in _inner_names(query):
            terms = query_terms(name)
            if not terms or any(t.found_in(analysis) for t in terms):
                continue
            # «Ева» and «Еву» share only «ев», a stem under _MIN_STEM that plain text keeps out of a word's keys, so the
            # fact read as being about someone else (2026-10-30); «Лев» and «Льва» share no stem at all. A name also
            # meets a word of the fact written with a capital that may be a case form of the same first form
            # (text.name_lemmas), and never a lower-case word: «Мая» is no «в мае». A shared short stem alone was
            # not enough: it joined «Аня» and «Анна», «Ян» and «Яной» (0.25.1).
            if capitals is None:
                capitals = frozenset().union(*(name_lemmas(w.text) for w in analyze(
                    _capitalized(fact.get("content", ""), inner=False)).words if not w.stop and not w.identifier))
            if frozenset().union(*(name_lemmas(t.word) for t in terms if not t.identifier)).isdisjoint(capitals):
                return False
        known = self.store.mentioned_entity_ids(query)
        if known:
            linked = {r[0] for r in self.store.read("SELECT entity_id FROM fact_entities WHERE fact_id = ?",
                                                    (fact["fact_id"],))}
            if not known <= linked:
                return False
        return True

    def by_meaning(self, meaning: Meaning, *, exclude=(), category: str | None = None, min_trust: float = 0.3,
                   limit: int = 1, margin: float, min_score: float | None = None) -> list[dict]:
        """The current facts closest in meaning that stand out from the rest of the store for this message
        (Meaning.standout), closest first, filters applied: 'similarity' is the cosine, 'score' similarity x trust."""
        # The filters first, so facts of another category or below the trust floor never use up the places.
        eligible = {r[0] for r in self.store.read(
            "SELECT fact_id FROM facts WHERE trust_score >= ?" + (" AND category = ?" if category else ""),
            [min_trust] + ([category] if category else []))}
        picked = []
        for fact_id, _ in meaning.ranked():  # standout falls with similarity: the first miss ends the list
            if not meaning.standout(fact_id, margin, min_score):
                break
            if fact_id not in exclude and fact_id in eligible:
                picked.append(fact_id)
                if len(picked) >= limit * 3 + 10:  # room for a row changed meanwhile
                    break
        rows = {f["fact_id"]: f for f in self._rows(picked, category, min_trust)}
        out = []
        for fact_id in picked:
            fact = rows.get(fact_id)
            if fact is not None and meaning.fits(fact_id, fact["content"]):  # changed meanwhile: not its score
                fact.pop("hrr_vector", None)
                fact["similarity"] = round(meaning.sims[fact_id], 4)
                fact["score"] = meaning.sims[fact_id] * fact["trust_score"]
                out.append(fact)
                if len(out) >= limit:
                    break
        return out

    def beyond_words(self, meaning: Meaning, query: str, words: list[dict], *, exclude=(), category: str | None = None,
                     min_trust: float = 0.3, k: int, beat: float, z_min: float, min_score: float | None = None) -> list[dict]:
        """fact_store search: of the *k* facts closest in meaning, those closer than the best fact the words found (by
        *beat*), standing out a little (z >= *z_min*) and naming whoever the query names. The words' results can fill
        the limit while the fact asked for is the store's closest in meaning — then by_meaning is never asked (issue
        #3: semantic rank 1 of 78, twice). Not before a turn: there a fact arrives unasked and by_meaning's bar stays."""
        if not words:
            return []
        have = {f["fact_id"] for f in words} | set(exclude)
        bar = max(meaning.cos(f["fact_id"]) if meaning.fits(f["fact_id"], f["content"]) else meaning.neutral
                  for f in words) + beat
        top = [fact_id for fact_id, s in meaning.ranked()[:k]
               if fact_id not in have and s > bar and meaning.z(fact_id) >= z_min and (min_score is None or s >= min_score)]
        rows = {f["fact_id"]: f for f in self._rows(top, category, min_trust)} if top else {}
        out = []
        for fact_id in top:
            fact = rows.get(fact_id)
            if fact is None or not meaning.fits(fact_id, fact["content"]) or not self.names_fit(query, fact):
                continue
            fact.pop("hrr_vector", None)
            fact["similarity"] = round(meaning.sims[fact_id], 4)
            fact["score"] = meaning.sims[fact_id] * fact["trust_score"]
            out.append(fact)
        return out

    def _linked_facts(self, entity_groups: list[list[int]], category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Facts linked to at least one entity of EVERY group (each group = the ids one name resolves to)."""
        joins = " INTERSECT ".join(
            f"SELECT fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(g))})" for g in entity_groups)
        params = [i for g in entity_groups for i in g] + [min_trust] + ([category] if category else []) + [limit]
        sql = (f"SELECT {_F_COLUMNS} FROM facts f WHERE f.fact_id IN ({joins}) AND f.trust_score >= ? "
               + ("AND f.category = ? " if category else "") + "ORDER BY f.trust_score DESC, f.updated_at DESC, f.fact_id DESC LIMIT ?")
        results = [dict(r) for r in self.store.read(sql, params)]
        for fact in results:
            fact["score"] = fact["trust_score"]
        return results

    def entity_facts(self, entity_id: int, *, min_trust: float, limit: int) -> list[dict]:
        """The facts linked to one entity, most trusted first (recall of the names a message mentions)."""
        return self._linked_facts([[entity_id]], None, min_trust, limit)

    def drop_cached(self, fact_id: int) -> None:
        """Forget what this retriever holds of a fact: its analysed text, every version's, and its vector (forget)."""
        self._current.cache.pop(fact_id, None)
        self._version_cache.clear()
        if self._vectors is not None:
            self._vectors.drop(fact_id)

    def probe(self, entity: str, category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Every fact linked to the entity (exact, by the entity index; name or alias, any letter case), most trusted
        first. A name never linked as an entity (e.g. a one-word name) falls back to keyword search."""
        ids = self.store.entity_ids(entity)
        if not ids:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        return self._linked_facts([ids], category, min_trust, limit)

    def reason(self, entities: list[str], category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Facts linked to ALL the given entities. When some name is not a known entity, keyword search that still
        requires every name (FTS5 AND of the names as phrases), never facts about just one of them."""
        names = [e for e in (entities or []) if isinstance(e, str) and e.strip()]
        if not names:
            return []
        groups = [self.store.entity_ids(name) for name in names]
        if all(groups):
            return self._linked_facts(groups, category, min_trust, limit)
        unknown = [name for name, ids in zip(names, groups) if not ids]
        known = [ids for ids in groups if ids]
        spaced = [n for n in unknown if not UNSEGMENTED.search(n)]
        glued = [n.strip().lower() for n in unknown if UNSEGMENTED.search(n)]
        required = []
        for name in spaced:  # every word of the name, in any word form ("allergy" finds "allergic")
            terms = query_terms(name)
            if not terms:
                return []
            required += terms
        if required:  # every constraint first, then the 500 most trusted (newest on ties)
            allowed = {f["fact_id"] for f in self._linked_facts(known, category, min_trust, 10_000)} if known else None
            hits = sorted(((trust, fact_id) for fact_id, trust, a in self._texts(category, min_trust)
                           if (allowed is None or fact_id in allowed) and all(t.found_in(a) for t in required)),
                          reverse=True)[:500]
            lexical = self._rows([fact_id for _, fact_id in hits], category, min_trust)
        elif known:
            lexical = self._linked_facts(known, category, min_trust, 500)
        else:
            lexical = self._ngram_candidates(" ".join(glued), category, min_trust, 500)
        lexical = [f for f in lexical if all(g in (f.get("content") or "").lower() for g in glued)]
        if known:  # known names by identity (aliases included), unknown names by their words
            allowed = {f["fact_id"] for f in self._linked_facts(known, category, min_trust, 10_000)}
            lexical = [f for f in lexical if f["fact_id"] in allowed]
        return self._rerank(" ".join(names), lexical, limit)

    def related(self, entity: str, category: str | None = None, limit: int = 10, min_trust: float = 0.3) -> list[dict]:
        """Facts structurally connected to an entity (the entity's atom unbinds to a role), above a noise floor.
        Keyword search without numpy, vectors, or any fact over the floor."""
        if not hrr.has_numpy():
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        entity_vec = self._atom(entity.strip().lower())  # bare atom, not role-bound: ANY structural match
        roles = (self._atom(_ROLE_ENTITY), self._atom(_ROLE_CONTENT))  # loop-invariant: encode once
        floor = _related_floor(self.hrr_dim)
        scored = []
        for row in self._vector_rows(category, min_trust):
            fact = dict(row)
            fact_vec = self._phases(fact.pop("hrr_vector"))
            if fact_vec is None:
                continue
            sim = max(hrr.similarity(hrr.unbind(fact_vec, entity_vec), role) for role in roles)
            if sim >= floor:
                fact["score"] = _shift(sim) * fact["trust_score"]
                scored.append(fact)
        if not scored:
            return self.search(entity, category=category, min_trust=min_trust, limit=limit)
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]

    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10,
                   min_trust: float = 0.0) -> list[dict]:
        """Candidate conflicts: pairs about the same subject (shared linked entities) making a similar but not
        identical claim ("Alice lives in Paris" / "Alice lives in London", a changed number, a flipped preference).
        Score = entity overlap x word overlap of the two claims with the entity names left out. Identical claims,
        a claim that only adds detail to the other ("likes apples" / "likes apples and bananas") and different
        topics about the same subject do not qualify. ``content_similarity`` is that word overlap. Word sets, so
        word order is ignored. Works without numpy."""
        sql = ("SELECT fact_id, content, category, tags, trust_score, created_at, updated_at FROM facts "
               "WHERE trust_score >= ?" + (" AND category = ?" if category else "")
               + " ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 500")  # bound: the most recently changed facts
        rows = self.store.read(sql, [min_trust] + ([category] if category else []))
        if len(rows) < 2:
            return []
        ids = [r["fact_id"] for r in rows]
        names_by_fact: dict[int, set[str]] = {}
        for r in self.store.read("SELECT fe.fact_id, e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                                 f"WHERE fe.fact_id IN ({','.join('?' * len(ids))})", ids):
            names_by_fact.setdefault(r["fact_id"], set()).add(r["name"].lower())
        facts = []  # (public dict, lower-cased entity names, claim)
        for row in rows:
            names = names_by_fact.get(row["fact_id"])
            if names:
                entity_words = {w for name in names for w in words(name)}
                facts.append((dict(row), names, self._claim(row["content"], entity_words)))
        # Only pairs that share an entity can qualify: group by entity instead of comparing every pair.
        by_entity: dict[str, list[int]] = {}
        for index, (_, names, _) in enumerate(facts):
            for name in names:
                by_entity.setdefault(name, []).append(index)
        pairs = sorted({(a, b) for group in by_entity.values() for i, a in enumerate(group) for b in group[i + 1:]})
        contradictions = []
        for i, j in pairs:
            (f1, ents1, claim1), (f2, ents2, claim2) = facts[i], facts[j]
            entity_overlap = len(ents1 & ents2) / len(ents1 | ents2)
            if entity_overlap < 0.3:
                continue  # different subjects
            claim_sim = self._conflicting(claim1, claim2)
            contradiction_score = entity_overlap * claim_sim
            if contradiction_score >= threshold:
                contradictions.append({
                    "fact_a": f1, "fact_b": f2,
                    "entity_overlap": round(entity_overlap, 3),
                    "content_similarity": round(claim_sim, 3),
                    "contradiction_score": round(contradiction_score, 3),
                    "shared_entities": sorted(ents1 & ents2),
                })
        return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)[:limit]

    def _vector_rows(self, category: str | None, min_trust: float = 0.0) -> list:
        """All facts that carry an HRR vector and pass the trust filter, optionally within a category."""
        where = "WHERE hrr_vector IS NOT NULL AND trust_score >= ?" + (" AND category = ?" if category else "")
        return self.store.read(f"SELECT {_FACT_COLUMNS}, hrr_vector FROM facts {where}",
                               [min_trust] + ([category] if category else []))

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int,
                        corpus: _Corpus | None = None) -> list[dict]:
        """FTS5 candidates for a natural-language query; [] when nothing in it is searchable."""
        expression = self._sanitize_fts_query(query)
        return self._fts_query(expression, category, min_trust, limit, corpus) if expression else []

    def _fts_query(self, expression: str, category: str | None, min_trust: float, limit: int,
                   corpus: _Corpus | None = None) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
        corpus = corpus or self._current
        fts = corpus.fts
        category_clause = "AND f.category = ? " if category else ""
        params = [*corpus.params, expression] + ([category] if category else []) + [min_trust, limit]
        sql = (f"SELECT f.*, {fts}.rank as fts_rank_raw FROM {fts} JOIN {corpus.source} f ON f.{corpus.key} = {fts}.rowid "
               f"WHERE {fts} MATCH ? {category_clause}AND f.trust_score >= ? ORDER BY {fts}.rank LIMIT ?")
        try:
            results = [_fact_row(row) for row in self.store.read(sql, params)]
        except Exception as exc:  # noqa: BLE001 - a damaged index (expressions are built to be valid)
            # Logged, not raised: recall must not take the turn down; a silent [] hides a broken index.
            logger.warning("holographic: FTS5 search failed (%s)", error_kind(exc))
            return []
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero)
        max_rank = max([abs(f["fts_rank_raw"]) for f in results] + [1e-6])
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Natural-language query -> FTS5-safe OR expression. FTS5 AND-joins a multi-word MATCH by default, which
        tanks recall on prose: informative terms only (English and Russian stopwords dropped), each as its exact
        word OR its stem as a prefix ("маши" -> ("маши" OR "маш"*)), FTS5 operator characters inside a token
        turned into a phrase ("pve-01" -> "pve 01"). Nothing left -> "" (no search, never a raw MATCH)."""
        return " OR ".join(term.fts(_FTS_OPERATORS) for term in query_terms(query or "") if term.fts_ok)

    # -- prefetch relevance --------------------------------------------------------------------------------------

    def _doc_freq(self, term) -> int:
        """Facts that may hold the term: its own forms, and the words of its root when it has any (the population the
        coverage counts it in: "ночные" is weighed by every fact with a night word, not only by the one that says
        "ночные")."""
        extra = term.fts_extra()
        expression = term.fts(_FTS_OPERATORS) if not extra else f"({term.fts(_FTS_OPERATORS)} OR {extra})"
        try:
            return int(self.store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", (expression,))[0][0])
        except Exception as exc:  # noqa: BLE001 - a damaged index: the term weighs nothing, and that is said (as _fts_query does)
            _warn_once("FTS5 count failed; recall weighs that word as absent", exc)
            return 0

    def _named_terms(self, terms) -> dict:
        """term -> the keys it shares with a name the store knows (an entity's name or alias) and not with its own
        forms. «Яне», «Яны» and «Яна» share the stem «ян», which plain Russian text keeps out of a word's keys
        (_MIN_STEM: "ян"* would find «январь» in the index) but name_keys reads in a name, so a fact that says «у Яны»
        did not meet a question about «Яне» though «Маши» meets «Маша» (0.18.3). Only by the keys of names the store
        holds: «ель» and «ела» share «ел» too, and stay apart. A Latin name matches as written and gives no key."""
        known = self.store.known_name_forms()
        out = {}
        if known:
            for t in terms:
                if not t.identifier and len(t.tokens) == 1 and (probe := (name_forms(t.word) - t.keys) & known):
                    out[t] = probe
        return out

    def relevant(self, query: str, candidates: list[dict], min_coverage: float = 0.3,
                 rescued: set | None = None) -> list[dict]:
        """Keep the candidates that answer the query, not merely share a word with it.

        Coverage = IDF-weighted share of the query's informative terms the fact contains (exact word or same stem).
        Terms no fact contains are left out of the total (they cannot be matched by anything), and a matched
        identifier (pve-01, v0.21.6) is decisive on its own. A query with no informative term ("ok, thanks",
        "да, это так") recalls nothing, and a fact sharing one word of a longer message, a word other facts hold too
        and no name inside the sentence ("What day is today?"), is left out. Order is preserved. *rescued*, when given,
        receives the ids of the facts let in only by prefetch_clause_coverage (_clause_rescue): they come after the
        others here, and the final selection gives them only the places the others leave free."""
        if is_greeting(query):  # "good morning", "доброе утро", "早上好": a courtesy, not a question about mornings
            return []
        grams, zh = query_grams(query or ""), is_chinese(query or "")
        gram_kept = []
        if grams and candidates:
            # Unsegmented script: weighted coverage of the query's n-grams, or (stricter, since one character says
            # less than a word) IDF-weighted coverage of its meaningful characters: in Chinese a one-character
            # word (猫, cat) or a reordered phrase (名字 / 名叫) shares characters, not bigrams.
            char_weights = self._char_weights(content_chars(query))
            pairs = [(g, w) for g, w in grams if len(g.replace(_text.SEAM, "")) > 1]  # not the one-character words
            char_total = sum(char_weights.values())
            for f in candidates:
                analysis = analyze(_fact_text(f))  # lower-cased and folded
                chars = zh_char_text(grams, analysis) if zh else analysis.text  # not the halves of a split pair
                char_cov = sum(w for c, w in char_weights.items() if c in chars) / char_total if char_total else 0.0
                # a Chinese one-character word (猫) never recalls alone: a pair of the question is there too, or
                # (characters one by one) two of its characters ("猫几岁了" / "合同里有猫腻" share 猫 only)
                paired = not zh or not pairs or gram_coverage(pairs, analysis, zh, 1.0) > 0
                two = not zh or len(char_weights) < 2 or sum(1 for c in char_weights if c in chars) >= 2
                if (gram_coverage(grams, analysis, zh) >= min_coverage and paired) or \
                        (char_cov >= max(0.5, min_coverage) and two):
                    gram_kept.append(f)
        terms = [t for t in query_terms(_RE_UNSEGMENTED.sub(" ", query or "")) if t.word]
        identifiers = [t for t in terms if t.identifier]
        if identifiers:  # "hr-vps" inside Chinese text: the character overlap must not stand in for the host
            gram_kept = [f for f in gram_kept
                         if any(t.found_in(analyze(_fact_text(f))) for t in identifiers)]
        if not terms or not candidates:
            return gram_kept
        total_facts = max(self.store.count_facts(), 1)
        scan = self._scan_terms(terms, query=query)
        named = self._named_terms(terms)
        texts = self._texts() if scan or named else []
        scan_df = {t: sum(1 for _, _, a in texts if t.found_in(a)) for t in scan}
        # A form of a name the store knows is held by the facts that hold any form of it («Яне» by «у Яны балет»),
        # which the FTS index cannot count (it finds "яне"*, not "яны"); counted over the texts, as a scan term is.
        scan_df.update({t: sum(1 for _, _, a in texts if t.found_in(a) or not probe.isdisjoint(_name_forms(a)))
                        for t, probe in named.items()})
        weights, unknown, freq = {}, [], {}
        for term in terms:
            df = scan_df[term] if term in scan_df else self._doc_freq(term)
            if df:
                weights[term], freq[term] = math.log(1.0 + total_facts / df), df
            else:
                unknown.append(term)
        if not weights:
            return gram_kept
        # A word no fact contains still says what the message is about: it counts against coverage, weighted like the
        # most common known word, so "do submarines deploy on Jupiter?" is not "about" a fact that mentions deploy.
        total = sum(weights.values()) + len(unknown) * min(weights.values())
        # keys of the words that may be names: compared by keys, as "José" is "jose" and "Peter's" "peter" to a term
        name_keys = {k for name in _inner_capitals(query or "") for w in analyze(name).words for k in w.keys}
        held: dict = {}  # term -> facts that hold the word itself (the FTS count also takes "carpet" for "car")

        def by_name(term, analysis) -> bool:
            """A form of a name the store knows, in a fact holding another form of it: «Яне» in «у Яны балет»."""
            return term in named and not named[term].isdisjoint(_name_forms(analysis))

        def held_by_others(term) -> bool:
            if term not in held:
                held[term] = sum(1 for _, _, a in self._texts() if term.match(a) == 1.0 or by_name(term, a))
            return held[term] > 1

        kept, left_out = [], []
        for fact in candidates:
            analysis = analyze(_fact_text(fact))
            if identifiers and not any(t.found_in(analysis) for t in identifiers):
                continue  # a message naming pve-01 or v0.21.6 is about that, not about pve-010
            strength = {t: t.match(analysis) for t in weights}  # 1.0 the word itself, _DERIV_WEIGHT its root
            for t in named:  # the name's other case form is the word itself: «Яна» and «Яны» are «Яне»
                if t in strength and strength[t] < 1.0 and by_name(t, analysis):
                    strength[t] = 1.0
            # A root alone never recalls: suffix rules cannot tell a derivation from a look-alike ("войной" / "вой",
            # "полный" / "пол"), so a fact needs a word of the message itself, and its roots only add to that.
            exact = any(s == 1.0 for s in strength.values())
            # One shared word picks a fact out only when no other fact holds it: "What day is today?" shares "day"
            # with everything that happens every day and is about none of it. A message of that word alone ("Lena?")
            # is about it, and so is one that may name someone inside the sentence ("What is Dina studying?" of
            # "Dina takes a pottery course"; in German any capitalized word; in a script without capitals, such as
            # Hebrew or Hindi, any word); a known entity's facts the message mentions come by the entity (prefetch).
            matched = [t for t, s in strength.items() if s]
            lone = len(terms) > 1 and len(matched) == 1 and freq[matched[0]] > 1 \
                and any(c.lower() != c.upper() for c in matched[0].word) and matched[0].keys.isdisjoint(name_keys) \
                and held_by_others(matched[0])
            if any(t.identifier and s for t, s in strength.items()) or fact in gram_kept or (
                    exact and not lone and sum(weights[t] * s for t, s in strength.items()) / total >= min_coverage):
                kept.append(fact)
            elif self.clause_coverage and exact and not grams:
                left_out.append((fact, strength))
        if left_out:
            back = self._clause_rescue(query, terms, weights, set(unknown), named, name_keys, left_out, min_coverage)
            kept += back
            if rescued is not None:
                rescued.update(f["fact_id"] for f in back)
        return kept

    def _clause_rescue(self, query: str, terms, weights: dict, unknown: set, named: dict, name_keys: set,
                       left_out: list, min_coverage: float) -> list:
        """Facts the gate left out that answer one of a message's two questions (prefetch_clause_coverage, off by
        default): each question's own terms, read in the whole message (text.question_parts), weighed as the gate
        weighs them. A fact is let in when it holds at least two of that question's own words exactly and covers
        min_coverage of the question, and only when whose answer it is can be told:
        - a question with a word no fact holds lets nothing in: it may be a name the store does not know («какой чай
          любит клара?»);
        - a question about a person is bound to exactly one entity the store knows, by a name written with a capital
          inside the sentence (or, for «она» / «он», to the one person the other question names); a word that is also a
          form of a known name but is written in lower case («чай» when a «Чай» is known, «злата») leaves it unbound,
          and so does a name the store does not know or two people: nothing is let in;
        - the fact must then name that person itself, with a capital: an ordinary word or a shared stem («чай» for
          «Чай») never says who it is about.
        In the order of the candidates, after every fact the gate kept."""
        parts = question_parts(query)
        if parts is None or not weights:
            return []
        known = self.store.known_name_forms()

        def name_like(t) -> bool:  # a word that may name someone (as the gate reads names)
            return t in named or not t.keys.isdisjoint(name_keys) or bool(
                known and not (t.keys | name_forms(t.word)).isdisjoint(known))

        def bound(own, said) -> tuple[bool, set]:
            """(can tell, the entities the question is about: empty = about nobody in particular)."""
            names = {t for t in own if name_like(t)}
            if not names:
                return True, set()
            if any(t.keys.isdisjoint(name_keys) for t in names):
                return False, set()  # a known name's form in lower case, or at the start: a word or a name
            people = self.store.mentioned_entity_ids(_capitalized(said, inner=True))
            return len(people) == 1, people

        by_key = {t.stem: t for t in terms}
        questions = []  # (own terms, «она»?, whose it is can be told, nothing unknown in it, the people, the names)
        for keys, pronoun, said in parts:
            own = {by_key[k] for k in keys if k in by_key}
            told, people = bound(own, said)
            questions.append((own, pronoun, told, not (own & unknown), people, {t for t in own if name_like(t)}))
        floor = min(weights.values())
        rescued, about = [], {}
        for fact, strength in left_out:
            for i, (own, pronoun, told, all_known, people, names) in enumerate(questions):
                if not told or not all_known:
                    continue
                said = set(own)
                if not people and pronoun:
                    _, _, other_told, _, people, other_names = questions[1 - i]
                    if not other_told or len(people) != 1:
                        continue  # «она» of nobody, or of one of two: whose answer it is cannot be told
                    said |= other_names
                    names = other_names
                strength = dict(strength)
                if people:
                    if fact["fact_id"] not in about:
                        about[fact["fact_id"]] = self.store.mentioned_entity_ids(
                            _capitalized(_fact_text(fact), inner=False))
                    if not people <= about[fact["fact_id"]]:
                        continue
                    strength.update({t: 1.0 for t in names})  # the fact names that person, in whatever case form
                if sum(1 for t in own if strength.get(t, 0.0) == 1.0) < 2:
                    continue
                share = sum(weights[t] * strength.get(t, 0.0) for t in said if t in weights) / \
                    sum(weights.get(t, floor) for t in said)
                if share >= min_coverage:
                    rescued.append(fact)
                    break
        return rescued

    def _char_weights(self, chars: list[str]) -> dict[str, float]:
        """IDF weight of each character over the store; a character no fact contains weighs like the most common
        one that is present (it still says what the message is about)."""
        if not chars:
            return {}
        texts = [a.text for _, _, a in self._texts()]  # folded: 醫 counts where 医 is written
        total = max(len(texts), 1)
        weights, missing = {}, []
        for c in chars[:16]:
            df = sum(1 for t in texts if c in t)
            if df:
                weights[c] = math.log(1.0 + total / df)
            else:
                missing.append(c)
        if weights:
            floor = min(weights.values())
            weights.update({c: floor for c in missing})
        return weights

    # -- write-time checks and housekeeping ----------------------------------------------------------------------

    @staticmethod
    def _claim(text: str, entity_words: set[str]) -> tuple[tuple, bool]:
        """(the claim without the entity names, negators and function words, negated?). Words compare by their keys
        (word and stems); unsegmented text (Chinese, Thai) by its characters, which is what a short claim shares with
        a contradicting one."""
        claim, negated, contrast = [], False, False
        for word in analyze(text).words:
            w = word.text
            if w in entity_words:
                continue
            if UNSEGMENTED.search(w):
                for name in entity_words:
                    if UNSEGMENTED.search(name):
                        w = w.replace(name, " ")
                if is_sea(w):  # Thai letters mean nothing alone: trigrams, and "ไม่" (not)
                    negated = negated or ("ไม่" in w and not contrast)
                    claim += [frozenset({g}) for run in UNSEGMENTED.findall(w) for g, weight in run_grams(run) if weight]
                    continue
                for ch in w:
                    if ch in CJK_NEGATORS:
                        negated = negated or not contrast
                    elif ch.isalnum():
                        claim.append(frozenset({ch}))
            elif w in CONTRASTIVE:  # "likes cats but not dogs": the negation is another clause's
                contrast = True
            elif w in NEGATORS:
                negated = negated or not contrast
            elif not word.stop or w in CLAIM_WORDS:  # "with" / "without" sugar is the claim
                claim.append(word.keys - word.forms if word.forms else word.keys)  # a case-form group: recall only
        return tuple(dict.fromkeys(claim)), negated

    @classmethod
    def _conflicting(cls, a: tuple[frozenset, bool], b: tuple[frozenset, bool]) -> float:
        """Claim overlap when two claims about one subject look like a conflict, else 0: the same statement with
        the opposite polarity ("likes apples" / "does not like apples": at least 60 % of the shorter claim is in the
        other, so "eats meat every day" / "does not eat meat" counts and "likes cats" / "does not like cooking" does
        not), or a similar but different statement where neither only adds detail to the other ("lives in Paris" /
        "lives in London"). Lexical candidates for a person to judge; word order (who paid whom) is not modelled."""
        (claim_a, neg_a), (claim_b, neg_b) = a, b
        a_in_b, b_in_a = soft_overlap(list(claim_a), list(claim_b))
        if neg_a != neg_b:
            shorter = min(len(claim_a), len(claim_b))
            contained = min(a_in_b, b_in_a) / shorter if shorter else 0.0
            return contained if contained >= 0.6 else 0.0
        overlap = soft_jaccard(list(claim_a), list(claim_b))
        if a_in_b == len(claim_a) or b_in_a == len(claim_b):  # identical, or one only adds detail
            return 0.0
        return overlap

    def write_check(self, fact_id: int, limit: int = 3) -> dict:
        """Facts that look like a near-duplicate of fact *fact_id* or a possible conflict with it (advisory: the
        agent decides; nothing is changed). Candidates: facts sharing an entity, plus the keyword neighbours."""
        rows = self.store.read("SELECT fact_id, content FROM facts WHERE fact_id = ?", (fact_id,))
        if not rows:
            return {"similar": [], "possible_conflicts": []}
        content = rows[0]["content"]
        own_entities = {r[0] for r in self.store.read("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fact_id,))}
        candidate_ids = {r[0] for r in self.store.read(
            "SELECT DISTINCT fact_id FROM fact_entities WHERE entity_id IN ({}) AND fact_id != ? ORDER BY fact_id DESC LIMIT 50".format(
                ",".join("?" * len(own_entities)) or "NULL"), [*own_entities, fact_id])}
        candidate_ids |= {f["fact_id"] for f in self.search(content, min_trust=0.0, limit=10, roots=False)} - {fact_id}
        if not candidate_ids:
            return {"similar": [], "possible_conflicts": [],
                    "past_versions": self._past_versions(fact_id, content, content_keys(content), limit)}
        entity_names = {r["entity_id"]: r["name"] for r in self.store.read(
            "SELECT entity_id, name FROM entities WHERE entity_id IN (SELECT entity_id FROM fact_entities WHERE fact_id IN ({}))".format(
                ",".join("?" * (len(candidate_ids) + 1))), [fact_id, *sorted(candidate_ids)])}
        links: dict[int, set[int]] = {}
        for r in self.store.read("SELECT fact_id, entity_id FROM fact_entities WHERE fact_id IN ({})".format(
                ",".join("?" * len(candidate_ids))), sorted(candidate_ids)):
            links.setdefault(r["fact_id"], set()).add(r["entity_id"])
        own_words = {w for e in own_entities for w in words(entity_names.get(e, ""))}
        own_keys, own_claim = content_keys(content), self._claim(content, own_words)
        similar, conflicts = [], []
        for other in self.store.read("SELECT fact_id, content FROM facts WHERE fact_id IN ({})".format(
                ",".join("?" * len(candidate_ids))), sorted(candidate_ids)):
            other_entities = links.get(other["fact_id"], set())
            overlap = soft_jaccard(own_keys, content_keys(other["content"]))
            entry = {"fact_id": other["fact_id"], "content": other["content"], "word_overlap": round(overlap, 2)}
            shared = own_entities & other_entities
            other_words = {w for e in other_entities for w in words(entity_names.get(e, ""))}
            other_claim = self._claim(other["content"], other_words)
            score = self._conflicting(own_claim, other_claim)
            if (shared and score >= 0.3) or (not own_entities and not other_entities and score >= 0.5):
                conflicts.append(entry)  # before duplicates: "budget is 500" / "budget is 600" is an update
            elif overlap >= 0.7 and own_claim[1] == other_claim[1]:
                similar.append(entry)
        by_overlap = lambda xs: sorted(xs, key=lambda e: e["word_overlap"], reverse=True)[:limit]  # noqa: E731
        return {"similar": by_overlap(similar), "possible_conflicts": by_overlap(conflicts),
                "past_versions": self._past_versions(fact_id, content, own_keys, limit)}

    def near_duplicate(self, content: str) -> int | None:
        """A current fact that says nearly what *content* says — write_check's "similar" (word overlap of 0.7 or
        more, the same polarity), checked before anything is stored — or None. A possible conflict is not a duplicate:
        "the budget is 500" / "the budget is 600" is a change."""
        own_keys, own_claim = content_keys(content), self._claim(content, set())
        for other in self.search(content, min_trust=0.0, limit=10, roots=False):
            other_claim = self._claim(other["content"], set())
            if soft_jaccard(own_keys, content_keys(other["content"])) >= 0.7 and own_claim[1] == other_claim[1] \
                    and self._conflicting(own_claim, other_claim) < 0.3:
                return other["fact_id"]
        return None

    def _past_versions(self, fact_id: int, content: str, own_keys, limit: int) -> list[dict]:
        """Closed versions (changed / ended, never corrected) of other facts that say nearly the same: "a fact that
        came back" is a new version of that fact (change), not a fact of its own. A merged version (0.19.0) says
        which current fact it is part of now (`merged_into`): that fact is the one to change or confirm."""
        expression = self._sanitize_fts_query(content)
        if not expression or not self.store.history_ready():
            return []
        statuses = (*CLOSED, "merged") if self.store._merges_ready() else CLOSED
        marks = ",".join("?" * len(statuses))
        try:
            rows = self.store.read(
                "SELECT v.fact_id, v.seq, v.content, v.status, v.valid_to, v.valid_to_exact FROM fact_versions_fts "
                "JOIN fact_versions v ON v.version_id = fact_versions_fts.rowid WHERE fact_versions_fts MATCH ? "
                f"AND v.status IN ({marks}) AND v.fact_id != ? ORDER BY fact_versions_fts.rank LIMIT 20",
                (expression, *statuses, fact_id))
        except Exception as exc:  # noqa: BLE001 - a damaged index: the hint is advisory
            logger.warning("holographic: history search failed (%s)", error_kind(exc))
            return []
        out: dict[int, dict] = {}
        for r in rows:
            overlap = soft_jaccard(own_keys, content_keys(r["content"]))
            if overlap < 0.7 or r["fact_id"] in out:
                continue
            entry = {"fact_id": r["fact_id"], "content": r["content"], "status": r["status"],
                     "word_overlap": round(overlap, 2)}
            if r["status"] == "merged":
                target = self.store.merge_target(r["fact_id"])
                if target is None or int(target["into_fact_id"]) == fact_id or not self.store.read(
                        "SELECT 1 FROM facts WHERE fact_id = ?", (target["into_fact_id"],)):
                    continue  # merged into this very fact, or into one that is gone: nothing to point at
                entry["merged_into"] = int(target["into_fact_id"])
            else:
                entry["until"] = r["valid_to"] if r["valid_to_exact"] else f"before {r['valid_to']}"
            out[r["fact_id"]] = entry
        return sorted(out.values(), key=lambda e: e["word_overlap"], reverse=True)[:limit]

    def review(self, limit: int = 20, stale_days: int = 60) -> list[dict]:
        """Clean-up candidates with a reason each: low trust; never recalled and older than *stale_days*; a
        near-duplicate of a more trusted fact. Advisory: nothing is removed."""
        out, seen = [], set()
        def add(fact, reason):
            if fact["fact_id"] not in seen:
                seen.add(fact["fact_id"])
                out.append({"fact_id": fact["fact_id"], "content": fact["content"], "trust_score": fact["trust_score"],
                            "retrieval_count": fact["retrieval_count"], "reason": reason})
        for fact in self.store.read(f"SELECT {_FACT_COLUMNS} FROM facts WHERE trust_score < 0.3 ORDER BY trust_score LIMIT ?", (limit,)):
            add(dict(fact), "low trust (rated unhelpful)")
        for fact in self.store.read(
                f"SELECT {_FACT_COLUMNS} FROM facts WHERE retrieval_count = 0 AND julianday('now') - julianday(created_at) > ? "
                "ORDER BY created_at LIMIT ?", (stale_days, limit)):
            add(dict(fact), f"never recalled in {stale_days}+ days")
        # the agent's own conclusions that nobody confirmed in that time (0.15.0): confirm, correct or forget
        # (the age of the version that made it an inference, not of the fact; selected in SQL, at most *limit*)
        stale = self.store.stale_inferences(stale_days, limit)
        rows = {r["fact_id"]: dict(r) for r in self._rows(stale, None, 0.0)} if stale else {}
        for fact_id in stale:
            if fact_id in rows:
                add(rows[fact_id], f"an inference nobody confirmed in {stale_days}+ days")
        if len(out) >= limit:
            return out[:limit]
        # near-duplicates among the most recent 200 facts (bounded pair scan), the more trusted one kept
        facts = sorted((dict(r) for r in self.store.read(f"SELECT {_FACT_COLUMNS} FROM facts ORDER BY fact_id DESC LIMIT 200")),
                       key=lambda f: f["trust_score"], reverse=True)
        keyed = [(f, content_stems(f["content"])) for f in facts]
        for i, (better, a) in enumerate(keyed):
            for worse, b in keyed[i + 1:]:
                if self._jaccard_similarity(a, b) >= 0.8:
                    add(worse, f"near-duplicate of #{better['fact_id']}")
        return out[:limit]

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

    def decays(self, facts: list[dict], now: datetime | None = None) -> list[float]:
        """The temporal decay search multiplies each fact's score by (1.0 each with decay off), at *now* (the present
        by default; the memory map passes its snapshot's moment). *facts*: rows of `facts` (fact_id, created_at,
        updated_at) or of a search as_of (with version_id).

        Decay counts from when the version was recorded, not from updated_at: feedback stamped that until 0.22.2,
        so an old fact marked unhelpful came out "younger" (F1 research, 2026-10-29). A version of as_of carries
        its own recorded_at as created_at; a store without history falls back to the last change."""
        if self.half_life <= 0:
            return [1.0] * len(facts)
        recorded = self.store.recorded_times([f["fact_id"] for f in facts if "version_id" not in f])
        return [self._temporal_decay(f.get("created_at") if "version_id" in f else (
            recorded.get(f["fact_id"]) or f.get("updated_at") or f.get("created_at")), now) for f in facts]

    def _temporal_decay(self, timestamp_str: str | None, now: datetime | None = None) -> float:
        """0.5^(age_days / half_life) at *now* (the present by default); 1.0 if disabled, missing, unparseable, or in
        the future."""
        if not self.half_life or not timestamp_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if isinstance(timestamp_str, str) else timestamp_str
            age_days = ((now or datetime.now(timezone.utc)) -(ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))).total_seconds() / 86400
            return 1.0 if age_days < 0 else math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
