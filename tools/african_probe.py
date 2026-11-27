"""Language probe: how the matcher reads FLORES-200 and MasakhaNER text in every language it supports (not shipped;
data not in the repo). Named for the African languages it was written for (0.33.0).

Data (CC BY-SA 4.0 / CC BY 4.0, downloaded by hand, read only):
    FLORES-200: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz  -> <flores>/{dev,devtest}/<code>.<split>
    MasakhaNER: https://github.com/masakhane-io/masakhane-ner  MasakhaNER2.0/data/<lang>/test.txt, data/amh/test.txt

Run (the venv has snowballstemmer and numpy):
    .venv/Scripts/python -I -X utf8 tools/african_probe.py --flores <dir> --ner <dir> [--split devtest] [--out f.json]

Measures, per language:
  detect   — Latin text: share of sentences whose language is determined and is the right one alone (exact), the
             right one among tied ones (tied), and old-language sentences that get a new language among theirs (leak).
             Cyrillic text (Russian, Ukrainian, Belarusian): the same of Analysis.cyrillic; "undetermined" is a
             Ukrainian or Belarusian sentence read as Russian (what nothing telling it apart gives), "leak" a Russian
             sentence that gets Ukrainian or Belarusian among its languages. Code without Analysis.cyrillic reads all
             Cyrillic as Russian.
  dropped  — share of tokens dropped as function words.
  ner      — MasakhaNER: share of name tokens (B-/I- tags) dropped as function words (should stay near 0).
  hooks    — Hausa: a question that types ɓ ɗ ƙ ƴ as b d k y (its hooked words plus up to two neighbours) finds its
             sentence among all of them: recall@1 / @5 (FactRetriever.search, words only, no embeddings).
  span     — four words from the middle of a sentence find it among all sentences of its language: recall@1 / @5.
             Chinese, Japanese and Thai have no spaces between words: there the span is 12 characters (Thai 16) from
             the middle, and the "common words" below are the commonest character pairs.
  gate     — automatic recall (relevant): hit rate for questions about held facts, false-recall rate for questions
             about facts the store does not hold (--only-gate runs it alone).
  noisy    — the same four words plus three of the language's ten commonest words in the dev split that the span
             lacks (how a question adds its own grammar): recall@1 / @5.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLORES = {"en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "pt": "por_Latn", "id": "ind_Latn", "de": "deu_Latn",
          "it": "ita_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "nl": "nld_Latn", "sw": "swh_Latn", "ha": "hau_Latn",
          "yo": "yor_Latn", "so": "som_Latn", "zu": "zul_Latn", "af": "afr_Latn", "am": "amh_Ethi", "ti": "tir_Ethi",
          "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "be": "bel_Cyrl", "hi": "hin_Deva", "ar": "arb_Arab", "bn": "ben_Beng",
          "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "th": "tha_Thai"}
CHARS = {"zh": 12, "ja": 12, "th": 16}  # unsegmented: span length in characters
OLD = ("en", "es", "fr", "pt", "id", "de", "it", "tr", "pl", "nl")
NER = {"ha": "hau.test.txt", "yo": "yor.test.txt", "sw": "swa.test.txt", "zu": "zul.test.txt", "am": "amh.test.txt"}
SPAN_LANGS = tuple(FLORES)
HOOKS = str.maketrans({"ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ƴ": "Y"})


def load_plugin():
    """The plugin as tests/conftest.py loads it: Hermes core from HERMES_AGENT_REPO (default ~/.hermes/hermes-agent),
    this tree as the package hermes_plugin_holographic, HERMES_HOME in a temporary directory."""
    agent = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
    sys.path.insert(0, str(agent))
    os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="african_probe_")
    name = "hermes_plugin_holographic"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    import importlib as il
    return il.import_module(f"{name}.text"), il.import_module(f"{name}.store"), il.import_module(f"{name}.retrieval")


def sentences(flores: Path, lang: str, split: str) -> list[str]:
    path = flores / split / f"{FLORES[lang]}.{split}"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ner_sentences(path: Path) -> list[list[tuple[str, str]]]:
    out, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            if cur:
                out.append(cur)
            cur = []
        elif len(parts) >= 2:
            cur.append((parts[0], parts[-1]))
    if cur:
        out.append(cur)
    return out


def detect(t, lang: str, sents: list[str]) -> dict:
    exact = tied = leak = undetermined = 0
    for s in sents:
        a = t.analyze(s)
        if not a.determined:
            undetermined += 1
            continue
        exact += a.langs == (lang,)
        tied += lang in a.langs
        leak += lang in OLD and any(x not in OLD for x in a.langs)
    n = len(sents)
    return {"exact": round(exact / n, 4), "tied": round(tied / n, 4), "undetermined": round(undetermined / n, 4),
            "leak": round(leak / n, 4)}


def detect_cyrillic(t, lang: str, sents: list[str]) -> dict:
    exact = tied = undetermined = leak = 0
    for s in sents:
        langs = getattr(t.analyze(s), "cyrillic", ("ru",))
        exact += langs == (lang,)
        tied += lang in langs
        undetermined += lang != "ru" and langs == ("ru",)
        leak += lang == "ru" and langs != ("ru",)
    n = len(sents)
    return {"exact": round(exact / n, 4), "tied": round(tied / n, 4), "undetermined": round(undetermined / n, 4),
            "leak": round(leak / n, 4)}


def dropped(t, sents: list[str]) -> float:
    total = stop = 0
    for s in sents:
        for w in t.analyze(s).words:
            total += 1
            stop += w.stop
    return round(stop / max(total, 1), 4)


def ner_dropped(t, sents: list[list[tuple[str, str]]]) -> dict:
    """Name tokens dropped as function words. "capital": only tokens written with a capital (a name word itself, not
    the "ta" of "Hukumar wasan motsa jiki ta ..."); Ethiopic has no case, so there every name token counts."""
    names = lost = caps = caps_lost = 0
    examples: dict[str, int] = {}
    for toks in sents:
        a = t.analyze(" ".join(tok for tok, _ in toks))
        stopped = {w.text for w in a.words if w.stop}
        for tok, tag in toks:
            if tag[:2] in ("B-", "I-") and any(c.isalpha() for c in tok):
                names += 1
                f = t.fold(tok.lower().strip(t.PUNCT))
                capital = tok[0].isupper() or not tok[0].isascii() and tok[0].lower() == tok[0].upper()
                caps += capital
                if f in stopped:
                    lost += 1
                    caps_lost += capital
                    if capital:
                        examples[f] = examples.get(f, 0) + 1
    top = sorted(examples.items(), key=lambda kv: -kv[1])[:12]
    return {"names": names, "dropped": round(lost / max(names, 1), 4), "capital": caps,
            "capital_dropped": round(caps_lost / max(caps, 1), 4), "top_capital": top}


def gate(store_mod, retr_mod, sents: list[str], common: list[str], lang: str = "") -> dict:
    """Automatic recall (FactRetriever.relevant over search): the store holds the first half of the sentences. Noisy
    span questions about 300 of them should recall their sentence ("hit"); the same kind of question about 300 sentences
    of the other half, which the store does not hold, should recall nothing ("false": share that recalled something)."""
    half = len(sents) // 2
    held, other = sents[:half], sents[half:]
    present = noisy_queries(held, common, lang)[:300]
    absent = noisy_queries(other, common, lang)[:300]
    with tempfile.TemporaryDirectory() as d:
        store = store_mod.MemoryStore(db_path=str(Path(d) / "m.db"))
        ids = []
        for f in held:
            try:
                ids.append(store.add_fact(f))
            except store_mod.DuplicateContentError as e:
                ids.append(e.fact_id)
        r = retr_mod.FactRetriever(store)
        def recalled(q):
            return [x["fact_id"] for x in r.relevant(q, r.search(q, limit=10, min_trust=0.0))]
        hit = sum(ids[i] in recalled(q) for i, q in present)
        false = sum(bool(recalled(q)) for _, q in absent)
        store.close() if hasattr(store, "close") else None
    return {"hit": round(hit / max(len(present), 1), 4), "false": round(false / max(len(absent), 1), 4),
            "n": [len(present), len(absent)]}


def retrieval(store_mod, retr_mod, facts: list[str], queries: list[tuple[int, str]]) -> dict:
    with tempfile.TemporaryDirectory() as d:
        store = store_mod.MemoryStore(db_path=str(Path(d) / "m.db"))
        ids = []
        for f in facts:
            try:
                ids.append(store.add_fact(f))
            except store_mod.DuplicateContentError as e:
                ids.append(e.fact_id)
        r = retr_mod.FactRetriever(store)
        at1 = at5 = 0
        for i, q in queries:
            got = [x["fact_id"] for x in r.search(q, limit=5, min_trust=0.0)]
            at1 += bool(got) and got[0] == ids[i]
            at5 += ids[i] in got
        store.close() if hasattr(store, "close") else None
    n = len(queries)
    return {"n": n, "r@1": round(at1 / max(n, 1), 4), "r@5": round(at5 / max(n, 1), 4)}


def hook_queries(sents: list[str]) -> list[tuple[int, str]]:
    out = []
    for i, s in enumerate(sents):
        words = s.split()
        hooked = [j for j, w in enumerate(words) if w != w.translate(HOOKS)]
        if not hooked:
            continue
        j = hooked[0]
        pick = words[max(0, j - 1): j + 3][:4]
        out.append((i, " ".join(pick).translate(HOOKS)))
    return out


def span_queries(sents: list[str], lang: str = "") -> list[tuple[int, str]]:
    out = []
    if lang in CHARS:
        n = CHARS[lang]
        for i, s in enumerate(sents):
            if len(s) >= 2 * n:
                mid = len(s) // 2 - n // 2
                out.append((i, s[mid: mid + n].strip()))
        return out
    for i, s in enumerate(sents):
        words = s.split()
        if len(words) >= 8:
            mid = len(words) // 2 - 2
            out.append((i, " ".join(words[mid: mid + 4])))
    return out


def common_words(flores: Path, lang: str) -> list[str]:
    """The ten commonest tokens of the dev split, as written (the measure does not use the plugin's lists); for an
    unsegmented language the ten commonest character pairs without punctuation."""
    counts: dict[str, int] = {}
    if lang in CHARS:
        for s in sentences(flores, lang, "dev"):
            for k in range(len(s) - 1):
                pair = s[k:k + 2]
                if all(c.isalpha() for c in pair):
                    counts[pair] = counts.get(pair, 0) + 1
        return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:10]]
    for s in sentences(flores, lang, "dev"):
        for w in s.split():
            w = w.strip(".,;:!?\"'()«»“”።፣")
            if w:
                counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:10]]


def noisy_queries(sents: list[str], common: list[str], lang: str = "") -> list[tuple[int, str]]:
    out = []
    if lang in CHARS:  # the pairs go in with spaces around them, inside and around the span
        for i, q in span_queries(sents, lang):
            extra = [w for w in common if w not in q][(i % 3):(i % 3) + 3]
            half = len(q) // 2
            out.append((i, " ".join([*extra[:1], q[:half], *extra[1:2], q[half:], *extra[2:3]])))
        return out
    for i, q in span_queries(sents):
        have = {w.lower() for w in q.split()}
        extra = [w for w in common if w.lower() not in have][(i % 3):(i % 3) + 3]
        out.append((i, " ".join([*extra[:1], *q.split()[:2], *extra[1:2], *q.split()[2:], *extra[2:3]])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--ner", type=Path)
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--only-gate", action="store_true", help="the gate task alone (plus detection)")
    args = ap.parse_args()
    t, store_mod, retr_mod = load_plugin()
    report: dict = {"split": args.split, "detect": {}, "dropped": {}, "ner": {}, "hooks": {}, "span": {}, "noisy": {}}
    for lang in FLORES:
        sents = sentences(args.flores, lang, args.split)
        if FLORES[lang].endswith("_Latn"):
            report["detect"][lang] = detect(t, lang, sents)
        elif FLORES[lang].endswith("_Cyrl"):
            report["detect"][lang] = detect_cyrillic(t, lang, sents)
        report["dropped"][lang] = dropped(t, sents)
    if args.ner:
        for lang, name in NER.items():
            report["ner"][lang] = ner_dropped(t, ner_sentences(args.ner / name))
    report["gate"] = {}
    if args.only_gate:
        for lang in SPAN_LANGS:
            report["gate"][lang] = gate(store_mod, retr_mod, sentences(args.flores, lang, args.split),
                                        common_words(args.flores, lang), lang)
    elif not args.skip_retrieval:
        t0 = time.time()
        ha = sentences(args.flores, "ha", args.split)
        report["hooks"]["ha"] = retrieval(store_mod, retr_mod, ha, hook_queries(ha))
        for lang in SPAN_LANGS:
            sents = sentences(args.flores, lang, args.split)
            report["span"][lang] = retrieval(store_mod, retr_mod, sents, span_queries(sents, lang))
            report["noisy"][lang] = retrieval(store_mod, retr_mod, sents,
                                              noisy_queries(sents, common_words(args.flores, lang), lang))
        report["retrieval_seconds"] = round(time.time() - t0, 1)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
