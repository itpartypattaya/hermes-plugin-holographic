"""0.32.0 (absolute_dates, on by default): a relative day word the agent stores gets its day.

The requirement: "The telescope meeting is tomorrow at 10:00", stored on 27 November, must still say which day a
month later. Only fact_store writes in the current turn are changed (the agent sees what was added and can keep the
text as written); automatic extraction reads old messages again at compaction and session end, so a day computed
then would be wrong, and it is left alone.
"""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.dates import absolutize

DAY = date(2026, 11, 27)


@pytest.mark.parametrize("text, expected", [
    ("The meeting is tomorrow at 10:00", "The meeting is tomorrow (2026-11-28) at 10:00"),
    ("Встреча завтра в 10:00", "Встреча завтра (2026-11-28) в 10:00"),
    ("Сегодня Лена улетает, вчера был концерт",
     "Сегодня (2026-11-27) Лена улетает, вчера (2026-11-26) был концерт"),
    ("The parcel comes the day after tomorrow", "The parcel comes the day after tomorrow (2026-11-29)"),
    ("Отчёт сдали позавчера, созвон послезавтра", "Отчёт сдали позавчера (2026-11-25), созвон послезавтра (2026-11-29)"),
    ("Tonight the server reboots", "Tonight (2026-11-27) the server reboots"),
    ("YESTERDAY the backup failed", "YESTERDAY (2026-11-26) the backup failed"),
])
def test_a_relative_day_word_gets_its_day(text, expected):
    out, added = absolutize(text, DAY)
    assert out == expected
    assert added and all(" → 2026-11-" in a for a in added)


def test_month_and_year_boundaries():
    assert absolutize("Due tomorrow", date(2026, 11, 30))[0] == "Due tomorrow (2026-12-01)"
    assert absolutize("Сдали вчера", date(2027, 1, 1))[0] == "Сдали вчера (2026-12-31)"


@pytest.mark.parametrize("text", [
    "The invoice goes out 3 days after payment",        # counts from an event, not from today
    "Call Lena on Friday",                              # which Friday is unknown
    "Next week the office moves",
    "Сегодняшний отчёт готов",                          # part of a longer word
    "Today's standup is cancelled",                     # possessive: a day would split the phrase
    "The motto is «завтра будет лучше»",                # quoted
    'He always says "see you tomorrow"',
    "Run `make tomorrow` to build it",                  # code
    "Docs at https://example.org/today/notes",          # URL
    "The meeting is tomorrow (2026-11-28) at 10:00",    # already dated: storing it again adds nothing
    "Встреча завтра 2026-11-28 в 10:00",
])
def test_what_is_not_one_day_or_is_quoted_stays_as_written(text):
    assert absolutize(text, DAY) == (text, [])
    out, added = absolutize(text + " and the call is tomorrow", DAY)  # the same text with a word that does change
    assert out.endswith("tomorrow (2026-11-28)") and len(added) == 1


def test_an_unknown_day_adds_nothing():
    assert absolutize("The meeting is tomorrow", None) == ("The meeting is tomorrow", [])
    assert absolutize("The meeting is tomorrow", DAY)[0] == "The meeting is tomorrow (2026-11-28)"


# -- the provider ------------------------------------------------------------------------------------------------------

@pytest.fixture
def make(tmp_path, monkeypatch):
    import hermes_time

    made = []
    monkeypatch.setattr(hermes_time, "now", lambda: datetime(2026, 11, 27, 6, 30, tzinfo=timezone(timedelta(hours=7))))

    def _make(**config):
        p = HolographicMemoryProvider({"db_path": str(tmp_path / f"facts{len(made)}.db"), **config})
        p.initialize(session_id="s", platform="cli")
        made.append(p)
        return p

    yield _make
    for p in made:
        p.shutdown()


def store(p, **args):
    return json.loads(p.handle_tool_call("fact_store", args))


def content(p, fid):
    return p._store.read("SELECT content FROM facts WHERE fact_id = ?", (fid,))[0]["content"]


def test_add_stores_the_day_of_the_person_and_says_what_it_added(make):
    """06:30 in Bangkok is 23:30 UTC of the day before: the day is the person's (Hermes's clock), not UTC's."""
    p = make()
    out = store(p, action="add", content="The telescope meeting is tomorrow at 10:00", origin="said")
    assert out["status"] == "added"
    assert content(p, out["fact_id"]) == "The telescope meeting is tomorrow (2026-11-28) at 10:00"
    assert out["dates_added"] == ["tomorrow → 2026-11-28"]


def test_change_correct_and_update_date_their_new_text(make):
    p = make()
    fid = store(p, action="add", content="The telescope meeting is on Monday", origin="said")["fact_id"]
    assert store(p, action="change", fact_id=fid, content="The telescope meeting is today", origin="said")[
        "dates_added"] == ["today → 2026-11-27"]
    assert content(p, fid) == "The telescope meeting is today (2026-11-27)"
    store(p, action="correct", fact_id=fid, content="The telescope meeting was yesterday", origin="said")
    assert content(p, fid) == "The telescope meeting was yesterday (2026-11-26)"
    other = store(p, action="add", content="Lena flies to Porto on Monday", origin="said")["fact_id"]
    store(p, action="update", fact_id=other, content="Lena flies to Porto tomorrow")
    assert content(p, other) == "Lena flies to Porto tomorrow (2026-11-28)"


def test_dates_keep_stores_the_text_as_written(make):
    p = make()
    out = store(p, action="add", content="The telescope meeting is tomorrow at 10:00", origin="said", dates="keep")
    assert content(p, out["fact_id"]) == "The telescope meeting is tomorrow at 10:00"
    assert "dates_added" not in out
    dated = store(p, action="add", content="The dome opens tomorrow at noon", origin="said")
    assert content(p, dated["fact_id"]) == "The dome opens tomorrow (2026-11-28) at noon"  # without keep it is dated
    fixed = store(p, action="add", content="The dome opens tomorrow", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fixed, content="The dome opens tomorrow", dates="keep")
    assert content(p, fixed) == "The dome opens tomorrow"  # the way to take an added day back out


def test_absolute_dates_off_stores_the_text_as_written(make):
    p = make(absolute_dates="off")
    out = store(p, action="add", content="The telescope meeting is tomorrow at 10:00", origin="said")
    assert content(p, out["fact_id"]) == "The telescope meeting is tomorrow at 10:00"
    assert "dates_added" not in out
    on = make()
    dated = store(on, action="add", content="The telescope meeting is tomorrow at 10:00", origin="said")
    assert content(on, dated["fact_id"]) == "The telescope meeting is tomorrow (2026-11-28) at 10:00"


def test_the_same_text_on_the_next_day_is_a_new_claim_not_a_silent_duplicate(make, monkeypatch):
    import hermes_time

    p = make()
    first = store(p, action="add", content="The dome opens tomorrow", origin="said")["fact_id"]
    monkeypatch.setattr(hermes_time, "now", lambda: datetime(2026, 11, 28, 9, 0, tzinfo=timezone(timedelta(hours=7))))
    again = store(p, action="add", content="The dome opens tomorrow", origin="said")
    assert again["status"] == "added" and again["fact_id"] != first
    assert content(p, again["fact_id"]) == "The dome opens tomorrow (2026-11-29)"
    assert again.get("similar") or again.get("possible_conflicts"), "the earlier day is pointed out"
    same_day = store(p, action="add", content="The dome opens tomorrow", origin="said")
    assert same_day["status"] == "exists" and same_day["fact_id"] == again["fact_id"]


def test_a_forgotten_text_is_recognised_in_either_form(make):
    p = make()
    fid = store(p, action="add", content="The dome opens tomorrow", origin="said", dates="keep")["fact_id"]
    store(p, action="forget", fact_id=fid)
    back = store(p, action="add", content="The dome opens tomorrow", origin="said")
    assert content(p, back["fact_id"]) == "The dome opens tomorrow (2026-11-28)"  # the dated form was stored
    assert back.get("forgotten_before", {}).get("fact_id") == fid  # and the forgotten one, as written, still known
    dated = store(p, action="add", content="The roof opens tomorrow", origin="said")["fact_id"]
    store(p, action="forget", fact_id=dated)
    again = store(p, action="add", content="The roof opens tomorrow", origin="said")
    assert again.get("forgotten_before", {}).get("fact_id") == dated


def test_automatic_extraction_and_the_memory_mirror_keep_the_words(make):
    """Positive control in the same store: the same sentence through fact_store does get its day."""
    p = make(auto_extract="on", mirror_memory_writes="true")
    p.sync_turn("Я предпочитаю созвоны завтра утром, а не вечером.", "Понял.", session_id="s")
    p.on_memory_write("add", "memory", "Lena prefers calls tomorrow morning")
    texts = [r["content"] for r in p._store.read("SELECT content FROM facts")]
    assert texts, "extraction and the mirror stored something"
    assert all("(2026-" not in t for t in texts)
    out = store(p, action="add", content="Lena prefers calls tomorrow morning, not evening", origin="said")
    assert "(2026-11-28)" in content(p, out["fact_id"])


def test_without_the_core_clock_nothing_is_added(make, monkeypatch):
    import hermes_time

    p = make()
    before = store(p, action="add", content="The roof opens tomorrow", origin="said")
    assert content(p, before["fact_id"]) == "The roof opens tomorrow (2026-11-28)"
    monkeypatch.delattr(hermes_time, "now")
    out = store(p, action="add", content="The dome opens tomorrow", origin="said")
    assert content(p, out["fact_id"]) == "The dome opens tomorrow" and "dates_added" not in out
