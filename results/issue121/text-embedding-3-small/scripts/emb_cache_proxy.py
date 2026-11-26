"""Loopback /v1/embeddings that caches OpenAI's vectors by (model, text), so every arm of the #121 run sees the same
vector for the same text. OpenAI's embeddings differ slightly between calls; on facts near the standout bar that is
enough to flip a row between two otherwise identical runs, which a paired gate on separately run arms cannot tell
from an effect. 127.0.0.1 only. The key comes from OPENAI_API_KEY and goes only to api.openai.com.

    python emb_cache_proxy.py <port> <cache.jsonl>
"""
import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, CACHE_FILE = int(sys.argv[1]), sys.argv[2]
UPSTREAM = "https://api.openai.com/v1/embeddings"
lock = threading.Lock()
cache: dict[tuple[str, str], list[float]] = {}
if os.path.exists(CACHE_FILE):
    for line in open(CACHE_FILE, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            cache[(r["model"], r["text"])] = r["vector"]
stats = {"hit": 0, "miss": 0}


def fetch(model, texts, extra, auth):
    body = json.dumps({"model": model, "input": texts, **extra}).encode()
    req = urllib.request.Request(UPSTREAM, data=body, method="POST",
                                 headers={"Authorization": auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])], data.get("usage")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.endswith("/embeddings"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model, inp = body["model"], body["input"]
        texts = [inp] if isinstance(inp, str) else list(inp)
        extra = {k: v for k, v in body.items() if k not in ("model", "input")}
        with lock:
            missing = [t for t in dict.fromkeys(texts) if (model, t) not in cache]
        usage = {"prompt_tokens": 0, "total_tokens": 0}
        if missing:
            try:
                vectors, usage = fetch(model, missing, extra, self.headers.get("Authorization") or
                                       f"Bearer {os.environ['OPENAI_API_KEY']}")
            except Exception as e:  # noqa: BLE001 - the client sees a failed call, as it would upstream
                self.send_error(502, str(e)[:200])
                return
            with lock, open(CACHE_FILE, "a", encoding="utf-8") as f:
                for t, v in zip(missing, vectors):
                    if (model, t) not in cache:
                        cache[(model, t)] = v
                        f.write(json.dumps({"model": model, "text": t, "vector": v}) + "\n")
        with lock:
            stats["miss"] += len(missing)
            stats["hit"] += len(texts) - len(missing)
            out = {"object": "list", "model": model, "usage": usage or {},
                   "data": [{"object": "embedding", "index": i, "embedding": cache[(model, t)]} for i, t in enumerate(texts)]}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        raw = json.dumps({**stats, "cached": len(cache)}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
