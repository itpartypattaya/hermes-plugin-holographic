"""Loopback /v1/rerank for the #121 replication: Qwen3-Reranker-0.6B scored as in PR #135, section 5.

The score is logit("yes") - logit("no") at the last position, with the PR's instruction (instruction.txt), the model
card's pair format. Cohere-shaped answer, 127.0.0.1 only. This run: float32 with 4 threads on CPU (bf16 on CPU took
about 6 s a pair), about 1 s a call.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR, INSTRUCTION_FILE, PORT, DTYPE = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 2)
INSTRUCTION = open(INSTRUCTION_FILE, encoding="utf-8").read()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=getattr(torch, DTYPE)).eval()
YES, NO = tok.convert_tokens_to_ids(["yes", "no"])
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
PRE, SUF = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
MAX_LEN = getattr(model.config, "max_position_embeddings", 32768)


def score(query: str, document: str) -> float:
    text = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    ids = PRE + tok.encode(text, add_special_tokens=False) + SUF
    if len(ids) > MAX_LEN:
        return -100.0
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1).logits[0, -1].float()
    return (logits[YES] - logits[NO]).item()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/rerank":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        docs = body.get("documents") or []
        results = sorted(({"index": i, "relevance_score": score(body.get("query", ""), d)} for i, d in enumerate(docs)),
                         key=lambda r: -r["relevance_score"])[: int(body.get("top_n") or len(docs))]
        out = json.dumps({"model": body.get("model"), "results": results}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
