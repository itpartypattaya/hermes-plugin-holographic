"""Fixtures for tools/mcnemar_gate.py (item 2 at 14:01 + the fix list at 19:07).

The partial-loss case stands fully paired on its own: if branch B ever reverts to the question-level
`c == 0`, gate_only (which keeps the question but drops one target) would pass and the first test would
fail. Unpaired real rows and unpaired chit rows get their own tests (both block both branches), and a
file that cannot be compared honestly — duplicates, mixed runs, targets disagreeing with off — must
raise instead of reporting."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "mcnemar_gate.py"
_spec = importlib.util.spec_from_file_location("mcnemar_gate", _TOOL)
assert _spec and _spec.loader
mcnemar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcnemar)


def _row(mode, qid, keys, targets, *, hit5=True, confused=False, git_sha=None):
    row = {"set": "household", "lang": "en", "qid": qid, "mode": mode, "kind": "real",
           "hit5": hit5, "keys": keys, "targets": targets, "confused": confused,
           "out_n": len(keys)}
    if git_sha:
        row["git_sha"] = git_sha
    return row


def _chit(mode, qid):
    # the same qid across all modes: chit pairing is checked just like real
    return {"lang": "en", "qid": qid, "mode": mode, "kind": "chit", "keys": [], "targets": [],
            "hit5": None, "confused": False, "out_n": 0}


def test_partial_loss_fully_paired(tmp_path, capsys):
    """Partial loss — a purely paired case: it blocks ONLY l5, there is no unpaired here (item 5 19:07)."""
    rows = [
        # q1: two targets; gate_only keeps hit5 (t1 in top-5) but t2 is pushed to the SIXTH position —
        # question-level c == 0, yet the per-target loss exists.
        _row("off", "q1", ["t1", "t2", "f1", "f2", "f3"], ["t1", "t2"], confused=True),
        _row("gate_only", "q1", ["t1", "f1", "f2", "f3", "f4", "t2"], ["t1", "t2"]),
        _row("gate_min2", "q1", ["t1", "t2", "f1", "f2", "f3"], ["t1", "t2"]),
        _chit("off", "c1"), _chit("gate_only", "c1"), _chit("gate_min2", "c1"),
    ]
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out

    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    min2_line = next(line for line in out.splitlines() if line.startswith("gate_min2"))
    # the question is not lost (c == 0) but the target is lost — branch B is absent. If l5 reverted to c this test
    # would fail: pairing is complete, conf 0 < 1, neg are equal — B would appear.
    assert gate_line.split()[-1] == "-", gate_line
    # a control that the fixture can produce B at all (otherwise the test would only prove suppression)
    assert min2_line.split()[-1] == "B", min2_line
    assert "UNPAIRED" not in out


def test_unpaired_rows_fail_closed(tmp_path, capsys):
    """Complete pairing — otherwise B is never set, even when no losses are visible."""
    rows = [
        # all other branch B conditions hold: no losses (l5 = 0), confused reduced, neg equal —
        # the only remaining reason not to set B is incomplete pairing (UNPAIRED).
        _row("off", "q1", ["t1"], ["t1"], confused=True),
        _row("gate_only", "q1", ["t1"], ["t1"]),
        _row("gate_only", "q2", ["t2"], ["t2"]),  # only for gate_only
        _chit("off", "c1"), _chit("gate_only", "c1"),
    ]
    path = tmp_path / "unpaired.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out
    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    assert gate_line.split()[-1] == "-", gate_line
    assert "UNPAIRED" in out


def test_unpaired_chit_rows_block(tmp_path, capsys):
    """Real is fully paired, gate_only has no chit block of its own — neg would be undercounted (item 5 19:07)."""
    rows = [
        _row("off", "q1", ["t1"], ["t1"], confused=True),
        _row("gate_only", "q1", ["t1"], ["t1"]),
        _chit("off", "c1"),  # the gate_only chit is missing
    ]
    path = tmp_path / "unpaired_chit.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    mcnemar.report(str(path))
    out = capsys.readouterr().out
    gate_line = next(line for line in out.splitlines() if line.startswith("gate_only"))
    assert gate_line.split()[-1] == "-", gate_line
    assert "UNPAIRED" in out


def test_duplicate_rows_fail(tmp_path):
    rows = [
        _row("off", "q1", ["t1"], ["t1"]),
        _row("off", "q1", ["t1"], ["t1"]),  # a duplicate: confused/lost would double up
        _row("gate_only", "q1", ["t1"], ["t1"]),
    ]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        mcnemar.report(str(path))


def test_mixed_runs_fail(tmp_path):
    rows = [
        _row("off", "q1", ["t1"], ["t1"], git_sha="aaa1111"),
        _row("off", "q2", ["t2"], ["t2"], git_sha="bbb2222"),  # another run in the same file
        _row("gate_only", "q1", ["t1"], ["t1"], git_sha="aaa1111"),
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        mcnemar.report(str(path))


def test_targets_mismatch_fail(tmp_path):
    rows = [
        _row("off", "q1", ["t1", "t2"], ["t1", "t2"]),
        _row("gate_only", "q1", ["t1", "t2"], ["t1"]),  # a different label: the loss «disappears»
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit):
        mcnemar.report(str(path))


def _a_rows(gate_confused: int):
    """12 questions: off hits 6, gate hits all 12 -> b=6, c=0, p=0.031, pts=+5.0."""
    rows = []
    for i in range(1, 13):
        hit = i > 6
        keys = [f"t{i}"] if hit else [f"f{i}"]
        rows.append(_row("off", f"q{i}", keys, [f"t{i}"], hit5=hit,
                         confused=i in (11, 12), git_sha="abc1234"))
        rows.append(_row("gate_only", f"q{i}", [f"t{i}"], [f"t{i}"], hit5=True,
                         confused=i in range(13 - gate_confused, 13), git_sha="abc1234"))
    rows.append(_chit("off", "c1"))
    rows.append(_chit("gate_only", "c1"))
    return rows


def test_branch_a_needs_confused_guard(tmp_path, capsys):
    """Unprotected A (item 4 19:07): +5 points at p<0.05 yields A only when conf/noise is not above off."""
    path = tmp_path / "a_clean.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _a_rows(gate_confused=1)),
                    encoding="utf-8")
    mcnemar.report(str(path))
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("gate_only"))
    assert "A" in line.split()[-1], line  # conf 1 <= 2, l5 = 0 -> A (and B)

    path = tmp_path / "a_guarded.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _a_rows(gate_confused=3)),
                    encoding="utf-8")
    mcnemar.report(str(path))
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("gate_only"))
    assert line.split()[-1] == "-", line  # conf 3 > 2 -> neither A nor B, although the points and p are the same