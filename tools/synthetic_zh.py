# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 liwangcc (https://github.com/liwangcc)
"""Synthetic Chinese-heavy retrieval set for issue #95 (keywords in a fact's vector).

Generic software-project knowledge, no personal data. 30 facts, 36 labelled
questions in four slices (direct / paraphrase / crosslingual / colloquial) and
12 noise rows (6 negative chit-chat + 6 in-domain no-answer). Fact ids are string
keys "1".."30" to match tools/issue95_keywords/synthetic_zh.json.

FACTS follows tools/contrast_zh.py ({key: (content, entities)}); entities are empty
and user-style tags are deliberately omitted — in both arms the tags field holds
only the keywords line, so the word side stays identical. QUERIES / CHITCHAT follow
tools/household.py: each record carries its targets, its slice, and its text under
its language key ("zh" or "en"); crosslingual questions are English text targeting
Chinese facts. CHITCHAT cat tells negative chit-chat from in-domain no-answer.

Keywords were written blind (a pass that saw only fact texts + the keywords field
description, no queries/results); information-blind but not a separate session.
See tools/issue95_keywords/README.md and the freeze record in the contribution.
"""

LANG = "synthetic_zh"  # the keywords file key; store_facts stores under this lang

FACTS = {
    "1": ("后端用 FastAPI 框架，异步处理请求。", []),
    "2": ("数据库是 PostgreSQL，用 SQLAlchemy 做 ORM。", []),
    "3": ("前端是 Vue 3 单页应用，Vite 构建。", []),
    "4": ("部署在 nginx 反向代理后面，gunicorn 起服务。", []),
    "5": ("用 Redis 做缓存和会话存储。", []),
    "6": ("任务队列用 Celery，broker 是 RabbitMQ。", []),
    "7": ("日志统一输出 JSON 格式，便于采集。", []),
    "8": ("单元测试用 pytest，覆盖率要求八成以上。", []),
    "9": ("CI 用 GitHub Actions，推 main 自动跑测试。", []),
    "10": ("配置用环境变量管理，密钥不进仓库。", []),
    "11": ("接口鉴权用 JWT，令牌一小时过期。", []),
    "12": ("文件上传走对象存储，本地只留临时副本。", []),
    "13": ("搜索用 Elasticsearch 做全文索引。", []),
    "14": ("监控用 Prometheus 加 Grafana 看板。", []),
    "15": ("容器化用 Docker，编排是 Kubernetes。", []),
    "16": ("数据库迁移用 Alembic 管理版本。", []),
    "17": ("跨域请求在网关层统一配置 CORS。", []),
    "18": ("限流用令牌桶算法，每秒一百次。", []),
    "19": ("错误响应统一 RFC 7807 problem details 格式。", []),
    "20": ("文档用 OpenAPI 自动生成 Swagger 页面。", []),
    "21": ("The deployment uses a blue-green strategy to avoid downtime.", []),
    "22": ("Backups run nightly and are restore-tested monthly.", []),
    "23": ("敏感字段在入库前做加密处理。", []),
    "24": ("灰度发布先放百分之五的流量。", []),
    "25": ("链路追踪用 OpenTelemetry 采集 span。", []),
    "26": ("静态资源上 CDN 加速。", []),
    "27": ("密码哈希用 argon2，不存明文。", []),
    "28": ("接口版本放在 URL 路径里，比如 /v1/。", []),
    "29": ("健康检查端点是 /healthz。", []),
    "30": ("批处理任务在低峰期凌晨跑。", []),
}

QUERIES = {
    "D1": {"targets": ['1'], "slice": "direct", "zh": "后端用什么框架"},
    "D2": {"targets": ['2'], "slice": "direct", "zh": "数据库是什么"},
    "D3": {"targets": ['3'], "slice": "direct", "zh": "前端用什么构建"},
    "D4": {"targets": ['4'], "slice": "direct", "zh": "怎么部署的"},
    "D5": {"targets": ['5'], "slice": "direct", "zh": "缓存用什么"},
    "D6": {"targets": ['8'], "slice": "direct", "zh": "单元测试用什么"},
    "D7": {"targets": ['11'], "slice": "direct", "zh": "鉴权怎么做"},
    "D8": {"targets": ['15'], "slice": "direct", "zh": "容器化用什么"},
    "P1": {"targets": ['1'], "slice": "paraphrase", "zh": "服务端接口是哪个库写的"},
    "P2": {"targets": ['2'], "slice": "paraphrase", "zh": "数据存在哪里"},
    "P3": {"targets": ['3'], "slice": "paraphrase", "zh": "页面是怎么打包出来的"},
    "P4": {"targets": ['4'], "slice": "paraphrase", "zh": "线上服务怎么跑起来的"},
    "P5": {"targets": ['5'], "slice": "paraphrase", "zh": "怎么让重复查询快一点"},
    "P6": {"targets": ['6'], "slice": "paraphrase", "zh": "后台异步任务怎么排队"},
    "P7": {"targets": ['8'], "slice": "paraphrase", "zh": "怎么保证代码质量"},
    "P8": {"targets": ['18'], "slice": "paraphrase", "zh": "怎么防止接口被刷"},
    "P9": {"targets": ['21'], "slice": "paraphrase", "zh": "上线怎么不停机"},
    "P10": {"targets": ['29'], "slice": "paraphrase", "zh": "怎么知道服务挂没挂"},
    "X1": {"targets": ['1'], "slice": "crosslingual", "en": "what web framework is the backend"},
    "X2": {"targets": ['2'], "slice": "crosslingual", "en": "which database is used"},
    "X3": {"targets": ['11'], "slice": "crosslingual", "en": "how is authentication done"},
    "X4": {"targets": ['5'], "slice": "crosslingual", "en": "what is used for caching"},
    "X5": {"targets": ['15'], "slice": "crosslingual", "en": "how are containers orchestrated"},
    "X6": {"targets": ['18'], "slice": "crosslingual", "en": "how is rate limiting implemented"},
    "X7": {"targets": ['27'], "slice": "crosslingual", "en": "how are passwords stored"},
    "X8": {"targets": ['29'], "slice": "crosslingual", "en": "what is the health check endpoint"},
    "C1": {"targets": ['1'], "slice": "colloquial", "zh": "后端那套是拿啥写的"},
    "C2": {"targets": ['2'], "slice": "colloquial", "zh": "数据搁哪儿存着"},
    "C3": {"targets": ['3'], "slice": "colloquial", "zh": "前端咋打包的"},
    "C4": {"targets": ['4'], "slice": "colloquial", "zh": "服务咋上线的"},
    "C5": {"targets": ['5'], "slice": "colloquial", "zh": "咋让查询别那么慢"},
    "C6": {"targets": ['18'], "slice": "colloquial", "zh": "咋知道接口被人猛刷"},
    "C7": {"targets": ['21'], "slice": "colloquial", "zh": "上线咋能不中断"},
    "C8": {"targets": ['27'], "slice": "colloquial", "zh": "密码咋存的不会泄露吧"},
    "C9": {"targets": ['29'], "slice": "colloquial", "zh": "咋看服务还活着没"},
    "C10": {"targets": ['19'], "slice": "colloquial", "zh": "出错返回长啥样"},
}

CHITCHAT = {
    "N1": {"zh": "今天天气怎么样", "cat": "negative"},
    "N2": {"zh": "推荐一家附近的火锅店", "cat": "negative"},
    "N3": {"zh": "帮我写一首关于春天的诗", "cat": "negative"},
    "N4": {"zh": "股票明天会涨吗", "cat": "negative"},
    "N5": {"zh": "怎么做红烧肉", "cat": "negative"},
    "N6": {"en": "hello how are you", "cat": "negative"},
    "NA1": {"zh": "gRPC 流式调用是怎么做的", "cat": "no_answer"},
    "NA2": {"zh": "WebSocket 长连接怎么保持", "cat": "no_answer"},
    "NA3": {"zh": "前端状态管理用什么", "cat": "no_answer"},
    "NA4": {"zh": "移动端适配怎么处理", "cat": "no_answer"},
    "NA5": {"en": "how is the GraphQL gateway configured", "cat": "no_answer"},
    "NA6": {"en": "which service mesh is used", "cat": "no_answer"},
}

def self_check() -> bool:
    """Every target key exists; slice/chit counts; each record has a language text."""
    errs = []
    keys = set(FACTS)
    if len(FACTS) != 30:
        errs.append(f"expected 30 facts, got {len(FACTS)}")
    want_slice = {"direct": 8, "paraphrase": 10, "crosslingual": 8, "colloquial": 10}
    got = {}
    for qid, q in QUERIES.items():
        if not (("zh" in q) or ("en" in q)):
            errs.append(f"{qid}: no language text")
        if not all(t in keys for t in q["targets"]):
            errs.append(f"{qid}: unknown target {[t for t in q['targets'] if t not in keys]}")
        if not q["targets"]:
            errs.append(f"{qid}: labelled query with empty targets")
        got[q["slice"]] = got.get(q["slice"], 0) + 1
    if got != want_slice:
        errs.append(f"slice counts {got} != {want_slice}")
    cats = {}
    for hid, h in CHITCHAT.items():
        if not (("zh" in h) or ("en" in h)):
            errs.append(f"{hid}: no language text")
        cats[h["cat"]] = cats.get(h["cat"], 0) + 1
    if cats != {"negative": 6, "no_answer": 6}:
        errs.append(f"chit cats {cats} != negative 6 / no_answer 6")
    if errs:
        print("synthetic_zh self_check FAILED:")
        for e in errs:
            print("  -", e)
        return False
    print(f"synthetic_zh self_check OK: {len(FACTS)} facts, {len(QUERIES)} labelled "
          f"({got}), {len(CHITCHAT)} noise ({cats})")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
