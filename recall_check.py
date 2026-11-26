"""The check behind embeddings.RECALL_CHECK_MARGIN (a measurement, issue #121): one function from a message and a fact
to yes, no or not_answered, with its raw score.

With RECALL_CHECK_MARGIN set, a fact that stands out at that lower margin but not at RECALL_MARGIN is offered on meaning
only on the check's yes. A no, a not_answered or a pair the screens keep at home leaves it out, so recall is what it is
without the value. One question a turn at most, about the closest such fact, and with rerank off only: with rerank on
the value changes nothing.

``CHECK`` lets a run swap the check (another model, another API): a callable ``(message, fact) -> (verdict, score)``.
None, the default, is ``cross_encoder`` on the rerank_* keys. ``ask`` runs either one behind the screens of the rerank
stage: nothing is sent when the message or the fact looks secret, or when the pair does not fit a 1024-token
cross-encoder (rerank.RERANK_MAX_QUERY_CHARS, rerank.RERANK_PAIR_CHARS).
"""
from __future__ import annotations

from . import rerank
from .screen import _secret_in, _sendable

YES, NO, NOT_ANSWERED = "yes", "no", "not_answered"
SKIPPED_SECRET, SKIPPED_LONG = "skipped_secret", "skipped_long"  # screened: no call, the fact stays out
CHECK = None


def cross_encoder(rescorer: rerank.Rescorer, message: str, fact: str) -> tuple[str, float | None]:
    """The default check: the cross-encoder at rerank_url scores one pair, the message as the query and the fact as
    the document; yes when the score is above the middle of rerank_scale (0 for logit, 0.5 for unit). The client's
    guards hold: one rerank_timeout, which fits the prefetch budget with the embedding call, and the pause after a
    failure. A call it does not make, or that fails, is not_answered."""
    ranked = rescorer.scores(message, [fact], single=True)
    if not ranked:
        return NOT_ANSWERED, None
    score = ranked[0][1]
    return (YES if score > (0.5 if rescorer.settings.scale == "unit" else 0.0) else NO), score


def ask(check, message: str, content: str, shown: str) -> tuple[str, float | None]:
    """*check*'s verdict and raw score on one fact: its whole *content* is screened, *shown* (the fact as recall
    shows it) is what is sent. A check that raises, or answers anything but yes or no, is not_answered."""
    if _secret_in(message) or not _sendable(content):
        return SKIPPED_SECRET, None
    if len(message) > rerank.RERANK_MAX_QUERY_CHARS or len(message) + len(shown) > rerank.RERANK_PAIR_CHARS:
        return SKIPPED_LONG, None
    try:
        verdict, score = check(message, shown)
    except Exception:  # noqa: BLE001 - a check that fails keeps recall as it was, like a failed call
        return NOT_ANSWERED, None
    return (verdict if verdict in (YES, NO) else NOT_ANSWERED), score
