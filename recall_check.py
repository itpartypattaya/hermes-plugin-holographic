"""The check behind embeddings.RECALL_CHECK_MARGIN (a measurement, issue #121): one function from a message and a fact
to yes, no or not_answered, with its raw score.

With RECALL_CHECK_MARGIN set, a fact that stands out at that lower margin but not at RECALL_MARGIN is offered on meaning
only on the check's yes. A no or a not_answered leaves it out, so recall is what it is without the value. One question
a turn at most, about the closest such fact.

``CHECK`` lets a run swap the check (another model, another API): a callable ``(message, fact) -> (verdict, score)``.
None, the default, is ``cross_encoder`` on the rerank_* keys.
"""
from __future__ import annotations

from . import rerank

YES, NO, NOT_ANSWERED = "yes", "no", "not_answered"
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
    """*check*'s verdict and raw score on one fact: *shown* (the fact as recall shows it) is what is sent."""
    return check(message, shown)
