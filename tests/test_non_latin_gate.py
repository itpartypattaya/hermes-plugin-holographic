"""Scripts without capitals and unsegmented scripts recall a fact by what a question is about, not by its grammar
(0.33.1).

- Hindi, Arabic, Hebrew, Bengali, Amharic have no capitals to tell a name by. One common word shared with several
  facts no longer recalls them, as in English ("What day is today?"); a word of a declared entity's name still does.
- Japanese, Korean and Thai are compared by character n-grams. A gram most facts hold (Japanese して, ます; Korean 있습,
  니다) weighs as little as a common word, so a message sharing only grammar with the store recalls less often; a
  question whose question words no fact holds still recalls its topic.

Measured on FLORES-200 devtest by tools/african_probe.py (gate)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider


def _store(tmp_path, facts):
    """facts: list of (text, declared entities, or None to let the store guess them as it does by default)."""
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "s.db"), "hrr_dim": 1024})
    provider.initialize(session_id="s")
    back = {}
    for i, (text, entities) in enumerate(facts):
        args = {"action": "add", "content": text}
        if entities is not None:
            args["entities"] = entities
        result = json.loads(provider.handle_tool_call("fact_store", args))
        back[result["fact_id"]] = i
    return provider, back


def _recalled(provider, back, query):
    provider.on_session_switch("q", reset=True)
    block = provider.prefetch(query)
    return {back[int(line.split()[1][1:])] for line in block.splitlines()[1:]} if block else set()


# (question sharing one common word with every fact, the facts); the question is about none of them
ONE_COMMON_WORD = [
    ("पानी कितनी देर उबालें?", [  # Hindi: how long to boil water? / garden watered, bucket, fridge
        ("बगीचे के पौधों को हर सुबह पानी दिया जाता है", []),
        ("कार धोने के लिए पानी की बाल्टी गैराज में है", []),
        ("फ्रिज में ठंडा पानी रखा है", [])]),
    ("كيف أحضر القهوة التركية؟", [  # Arabic: how to make Turkish coffee? / we drink, cupboard, machine broken
        ("نشرب القهوة كل صباح في المطبخ", []),
        ("القهوة في الخزانة العلوية", []),
        ("آلة القهوة معطلة منذ أسبوع", [])]),
]


@pytest.mark.parametrize("query,facts", ONE_COMMON_WORD)
def test_one_common_word_of_a_script_without_capitals_recalls_nothing(tmp_path, query, facts):
    provider, back = _store(tmp_path, facts)
    try:
        assert _recalled(provider, back, query) == set()
    finally:
        provider.shutdown()


# (question, facts, the fact that answers it): the shared word is a name the store knows (declared entity). Five
# facts about Daniel: with two, the entity's own recall (prefetch) brings them in whatever the gate says.
_DANIEL = ["דניאל"]
KNOWN_NAME = [
    ("मीरा क्या सीखती है?", [("मीरा मिट्टी के बर्तन बनाना सीखती है", ["मीरा"]),  # Hindi: what does Meera learn?
                             ("मीरा पुणे में रहती है", ["मीरा"])], 0),
    ("מה דניאל לומד?", [("דניאל משתתף בקורס ספרדית", _DANIEL),  # Hebrew: what is Daniel studying? / a Spanish
                        ("דניאל גר בתל אביב", _DANIEL),  # course, lives in Tel Aviv, black coffee, a bank, Eilat
                        ("דניאל אוהב קפה שחור", _DANIEL),
                        ("דניאל עובד בבנק", _DANIEL),
                        ("דניאל נוסע לאילת בקיץ", _DANIEL)], 0),
]


@pytest.mark.parametrize("query,facts,answer", KNOWN_NAME)
def test_a_known_name_in_a_script_without_capitals_still_recalls(tmp_path, query, facts, answer):
    provider, back = _store(tmp_path, facts)
    try:
        assert answer in _recalled(provider, back, query)
    finally:
        provider.shutdown()


def test_a_guessed_name_does_not_let_a_common_word_recall(tmp_path):
    """A quoted word becomes a guessed one-word entity («पानी», water), not a declared one: water stays a common
    word, and a question about boiling water recalls none of the facts that merely mention water."""
    provider, back = _store(tmp_path, [("«पानी» मेरा प्रोजेक्ट है", None),  # "Water" is my project
                                       ("बगीचे के पौधों को हर सुबह पानी दिया जाता है", None),
                                       ("फ्रिज में ठंडा पानी रखा है", None)])
    try:
        assert _recalled(provider, back, "पानी कब उबालें?") == set()  # when to boil water?
    finally:
        provider.shutdown()


# A store of everyday facts besides the ones a case is about: rent, insurance, a birthday, a lesson, a bank, a dog,
# a move, the dentist, a bike, jogging, a meeting, a passport.
OTHERS = {
    "ja": ["パタヤの家賃は毎月二万バーツです", "車の保険は来年三月に切れる", "息子の誕生日は九月十二日",
           "妻はタイ語を勉強中", "口座はバンコク銀行にある", "犬の名前はポチ", "チェンマイへの引っ越しは春の予定",
           "歯医者の予約は金曜日", "バイクはホンダのフォルツァ", "朝はジョギングをする", "会議は毎週月曜の十時",
           "パスポートの更新は二〇二八年"],
    "ko": ["파타야 집세는 매달 이만 바트다", "자동차 보험은 내년 삼월에 끝난다", "아들 생일은 구월 십이일이다",
           "아내는 태국어를 공부한다", "계좌는 방콕 은행에 있다", "강아지 이름은 뽀삐다", "치앙마이 이사는 봄 예정이다",
           "치과 예약은 금요일이다", "오토바이는 혼다 포르자다", "아침에는 조깅을 한다", "회의는 매주 월요일 열 시다",
           "여권 갱신은 2028년이다"],
}

# (message sharing only grammar with the facts, the facts, the language of the other facts)
GRAMMAR_ONLY = [
    ("今日は何をしていますか", [  # Japanese: what are you doing today? / coffee, company, piano (all "-te imasu")
        ("レオンは毎朝コーヒーを飲んでいます", []),
        ("ナターシャは人材会社を経営しています", []),
        ("子供たちは夕方にピアノを練習しています", [])], "ja"),
    ("지금 어디에 있습니까", [  # Korean: where are you now? / lives in Pattaya, runs a company, learns piano
        ("레온은 파타야에 살고 있습니다", []),
        ("나타샤는 회사를 운영하고 있습니다", []),
        ("아이들은 저녁에 피아노를 배우고 있습니다", [])], "ko"),
]


@pytest.mark.xfail(strict=True, reason=(
    "known limit: in a small store the grammar a message shares is the rarest thing it shares, and a gram no fact "
    "holds weighs as the rarest gram the message shares, so that the question words of a question about a topic every "
    "fact holds do not drown it (test_a_question_about_a_topic_every_fact_holds_recalls); measured with FLORES-200 "
    "facts, Korean grammar alone recalls nothing from about 30 facts on, and Japanese grammar still recalls some "
    "unrelated facts at 30; on the whole FLORES devtest store the gate (tools/african_probe.py) gives Japanese 46 %, "
    "Korean 38 % false recall"))
@pytest.mark.parametrize("query,facts,lang", GRAMMAR_ONLY)
def test_a_message_sharing_only_grammar_recalls_nothing(tmp_path, query, facts, lang):
    provider, back = _store(tmp_path, facts + [(text, []) for text in OTHERS[lang]])
    try:
        assert _recalled(provider, back, query) == set()
    finally:
        provider.shutdown()


QUESTIONS = [
    ("レオンは何を飲んでいますか", [("レオンは毎朝コーヒーを飲んでいます", []),  # Japanese: what does Leon drink?
                                  ("ナターシャは人材会社を経営しています", [])], 0),
    ("나타샤는 무슨 회사를 운영합니까", [("레온은 파타야에 살고 있습니다", []),  # Korean: what company does Natasha run?
                                       ("나타샤는 회사를 운영하고 있습니다", [])], 1),
]


@pytest.mark.parametrize("query,facts,answer", QUESTIONS)
def test_a_question_about_a_fact_still_recalls_it(tmp_path, query, facts, answer):
    provider, back = _store(tmp_path, facts)
    try:
        assert answer in _recalled(provider, back, query)
    finally:
        provider.shutdown()


# (question, facts that all hold its topic, the fact that answers it): the question words (어디, どこ) are in no fact
TOPIC_EVERY_FACT_HOLDS = [
    ("레온은 어디에 살아요?", ["레온은 파타야에 살고 있습니다", "레온은 커피를 좋아합니다",  # Korean: where does Leon
                               "레온은 회사를 운영합니다"], 0),  # live? / Pattaya, coffee, a company
    ("파타야 수영 어디서 배워요?", ["레온은 파타야에서 수영을 배운다", "아이들은 파타야 수영장에 간다",  # where to learn
                                   "파타야 수영 강사는 친절하다", "수영 수업은 파타야 해변에서 한다",  # swimming in
                                   "파타야에는 수영 학교가 많다"], 0),  # Pattaya? / five facts about it
]


@pytest.mark.parametrize("query,facts,answer", TOPIC_EVERY_FACT_HOLDS)
def test_a_question_about_a_topic_every_fact_holds_recalls(tmp_path, query, facts, answer):
    provider, back = _store(tmp_path, [(text, []) for text in facts])
    try:
        assert answer in _recalled(provider, back, query)
    finally:
        provider.shutdown()
