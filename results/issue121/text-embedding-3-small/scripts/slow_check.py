"""The probe with rerank_timeout raised for the CPU checker of the #121 replication; nothing else changes.

rerank.Settings caps it at the prefetch budget minus the embedding timeout (7.5 - 3.0 = 4.5 s), the most the plugin
itself allows. The timeout decides only whether an answer arrives, not what it says.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance_probe as pp

_make = pp.make_provider


def make_provider(db, mode, emb, extra=None):
    return _make(db, mode, emb, {**(extra or {}), "rerank_timeout": "10"})


pp.make_provider = make_provider
pp.main(sys.argv[1:])
