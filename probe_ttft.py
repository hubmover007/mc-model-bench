#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_ttft.py —— 单模型「首包 / 首 token / 首内容」延迟探针
================================================================

用途
----
针对某个渠道上的某个模型（默认 easyrouter 上的 gemini-3.1-pro-preview），
用标准库 http.client 精确拆解一次流式请求的**每个阶段耗时**，并做多次采样统计：

    DNS 解析 → TCP 建连 → TLS 握手 → 请求发送 → 响应头(首字节)
             → 首个 SSE 包 → 首个 token(可能是 reasoning) → 首个 content(用户可见首字) → 结束

为什么不用 requests
-------------------
requests 无法给出 DNS/TCP/TLS 与「响应头到达」的精确时间点，也无法保证 `read1`
级别的首包检测（缓冲/解压会推迟首个 chunk）。本脚本只用标准库，零外部依赖。

「首包」的三种口径（本脚本全部输出，避免混用）
------------------------------------------------
1. `ttfb_headers_ms`  首字节（HTTP 响应头到达）—— 网络+网关受理，不含模型推理
2. `first_token_ms`   首个 token 到达，热口径（不含建连）—— 业界常说的 TTFT
3. `first_content_ms` 首个 **content** 到达 —— 用户真正看到文字的首字延迟
   （推理模型会先吐 reasoning_content，二者可能相差几十秒，必须分开看）

用法
----
    # 0) 先看有哪些内置用例
    python probe_ttft.py --list-cases

    # 1) 最常用：测 easyrouter 上 gemini-3.1-pro-preview，20 次采样（前 2 次预热不计）
    export EASYROUTER_API_KEY=sk-xxx        # Windows: $set env:EASYROUTER_API_KEY="sk-xxx"
    python probe_ttft.py --model gemini-3.1-pro-preview --runs 20 --warmup 2

    # 2) 同时测短输入/长输入（看 prefill 对首包的影响）
    python probe_ttft.py --cases short,long --runs 20

    # 3) 连接复用 vs 新建连接 对比（判断握手占了多少首包时间）
    python probe_ttft.py --keepalive --runs 20

    # 4) 多模型基线对比（同口径、同时段才可比）
    python probe_ttft.py --models gemini-3.1-pro-preview,gpt-5.2,kimi-k3 --runs 10

    # 5) 验证脚本自身（不需要真实 key，先起本地 mock）
    python mock_server.py 18080
    python probe_ttft.py --base-url http://127.0.0.1:18080/v1 --api-key test --runs 5

输出
----
    output_ttft/ttft_<时间戳>/
        probe.json      # 每次采样的完整阶段数据（可用于二次分析）
        report.md       # 阶段分解表 + 分布统计 + 自动诊断 + 正反建议
        report.csv      # 明细表（方便贴 Excel）

退出码：0=至少一条成功；2=全部失败（便于 CI/定时任务判断）。
"""

import argparse
import csv
import datetime
import json
import os
import random
import socket
import ssl
import statistics
import sys
import time
import traceback

try:
    import http.client as httplib
except ImportError:  # pragma: no cover
    import httplib  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_URL = "https://easyrouter.io/v1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OUT = os.path.join(HERE, "output_ttft")
DEFAULT_CFG = os.path.join(HERE, "config", "providers.json")


# ---------------------------------------------------------------------------
# 内置用例：控制「输入长度」与「要求输出长度」两个变量
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = [
    "Distributed systems rely on consensus protocols to keep replicas consistent under network partitions.",
    "A write-ahead log records every mutation before it is applied, enabling crash recovery and replay.",
    "Connection pooling reduces handshake overhead by reusing established TCP and TLS sessions.",
    "Observability requires correlated metrics, traces and logs rather than any single signal alone.",
    "The scheduler balances queue depth against fairness, and long waits indicate insufficient capacity.",
    "Paged attention stores key-value blocks non-contiguously, cutting memory waste during long decoding.",
    "Backpressure protects downstream services by rejecting or delaying work when queues saturate.",
    "Idempotency keys let clients retry safely, because duplicate submissions map to a single effect.",
    "Cold starts dominate tail latency for serverless workloads that cannot keep instances warm.",
    "TLS session resumption and early data can save one round trip on subsequent connections.",
    "Speculative decoding trades extra compute for lower inter-token latency on small batches.",
    "Quorum reads and writes trade consistency for availability depending on the configured level.",
    "Rate limiting at the edge protects origins, but misconfigured limits surface as 429 bursts.",
    "Chunked transfer encoding lets servers stream responses before the total body size is known.",
    "A trace context propagated across services is what makes cross-boundary latency attributable.",
    "Compression buffers can delay the first byte, so interactive streams often disable gzip.",
]


def make_filler(n_chars, seed="ttft-filler"):
    """确定性生成约 n_chars 长度的英文技术材料（用于构造长输入）。"""
    rnd = random.Random(seed)
    parts, total = [], 0
    while total < n_chars:
        s = rnd.choice(_FILLER_SENTENCES)
        parts.append(s)
        total += len(s) + 1
    text = " ".join(parts)
    return text[:n_chars]


def _medium_case():
    body = make_filler(1800, seed="ttft-medium")
    return {
        "name": "中等输入(~500 tokens)/结构化作答",
        "prompt": "阅读下面的技术材料，然后用中文输出 3 条要点，每条不超过 20 字。\n\n"
                  "[材料开始]\n%s\n[材料结束]" % body,
        "max_tokens": 384,
    }


def _long_case():
    body = make_filler(16000, seed="ttft-long")
    return {
        "name": "长输入(~4K tokens)/单句作答",
        "prompt": "以下是一份技术材料，请通读后回答问题。\n\n[材料开始]\n%s\n[材料结束]\n\n"
                  "问题：以上材料主要讨论的主题是什么？用一句话回答（不超过 30 字）。" % body,
        "max_tokens": 256,
    }


CASES = {
    "short": {
        "name": "短输入/单句作答",
        "prompt": "用一句话解释什么是 HTTP/2 多路复用（不超过 40 字）。",
        "max_tokens": 256,
    },
    "medium": _medium_case(),
    "long": _long_case(),
    "longout": {
        "name": "短输入/长输出(~600字)",
        "prompt": "请写一篇约 600 字的技术说明，介绍 HTTP/2 与 HTTP/3 的差异与取舍。",
        "max_tokens": 1600,
    },
}


# ---------------------------------------------------------------------------
# 带阶段计时的 HTTP 连接（仅标准库，精确到 DNS/TCP/TLS）
# ---------------------------------------------------------------------------

class PhaseConn(httplib.HTTPConnection):
    """覆盖 connect()，分别记录 DNS / TCP / TLS 三段耗时。支持 http 与 https。"""

    def __init__(self, host, port=None, timeout=60, use_tls=True, insecure=False):
        super().__init__(host, port, timeout=timeout)
        self.use_tls = use_tls
        self.insecure = insecure
        self.phases = {"dns_ms": None, "tcp_ms": None, "tls_ms": None}

    def connect(self):  # noqa: C901 - 明确拆段，保持线性
        t0 = time.perf_counter()
        infos = socket.getaddrinfo(self.host, self.port, 0, socket.SOCK_STREAM)
        self.phases["dns_ms"] = (time.perf_counter() - t0) * 1000.0

        family, socktype, proto, _canon, sockaddr = infos[0]
        t1 = time.perf_counter()
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(self.timeout)
        sock.connect(sockaddr)
        self.phases["tcp_ms"] = (time.perf_counter() - t1) * 1000.0

        if self.use_tls:
            t2 = time.perf_counter()
            ctx = ssl._create_unverified_context() if self.insecure else ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=self.host)
            self.phases["tls_ms"] = (time.perf_counter() - t2) * 1000.0

        self.sock = sock


class ApiError(Exception):
    def __init__(self, message, kind="api_error", status=None, retryable=False, retry_after=None, extra=None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        # 失败也保留阶段耗时（例如 401 很快、429 很慢，本身是有价值的首包信号）
        self.extra = extra or {}


# ---------------------------------------------------------------------------
# 请求构造
# ---------------------------------------------------------------------------

def build_headers(api_key, stream=True, accept_encoding="identity"):
    h = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
        "Accept-Encoding": accept_encoding,
        "User-Agent": "mc-model-bench/ttft-probe",
        "Connection": "keep-alive" if stream else "close",
    }
    if stream:
        h["Accept"] = "text/event-stream"
    return h


def build_body(model, case, args):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "stream": bool(args.stream),
        "max_tokens": int(args.max_tokens or case.get("max_tokens", 256)),
    }
    if not args.no_temperature:
        body["temperature"] = args.temperature
    if args.extra_body:
        try:
            body.update(json.loads(args.extra_body))
        except ValueError:
            sys.exit("--extra-body 不是合法 JSON：%s" % args.extra_body)
    return body


# ---------------------------------------------------------------------------
# 单次探测：核心计时逻辑
# ---------------------------------------------------------------------------

def probe_once(conn, args, model, case):
    """在给定连接上发一次流式请求，返回阶段耗时字典（失败抛 ApiError）。"""
    t0 = time.perf_counter()

    def _new_conn():
        c = PhaseConn(args.host, args.port, timeout=args.timeout,
                      use_tls=(args.scheme == "https"), insecure=args.insecure)
        try:
            c.connect()
        except (OSError, ssl.SSLError) as e:
            raise ApiError("建连失败（DNS/TCP/TLS）: %s" % e, kind="connect_error", retryable=True)
        return c

    if conn is None:
        conn = _new_conn()
        conn_started = {k: v for k, v in conn.phases.items()}
        reused = False
    elif conn.sock is None:
        # keep-alive 场景下服务端早已断开，需要重连：单独记一次建连耗时
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        conn = _new_conn()
        conn_started = {k: v for k, v in conn.phases.items()}
        reused = False
    else:
        conn_started = {"dns_ms": 0.0, "tcp_ms": 0.0, "tls_ms": 0.0}
        reused = True

    body = build_body(model, case, args)
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = build_headers(args.api_key, stream=args.stream, accept_encoding=args.accept_encoding)
    path = args.path if args.path.startswith("/") else "/" + args.path

    send_started = time.perf_counter()  # noqa: F841 - 保留便于排障时打印
    try:
        conn.request("POST", path, body=payload, headers=headers)
    except (httplib.HTTPException, OSError) as e:
        raise ApiError("发送请求失败: %s" % e, kind="send_error", retryable=True)
    t_sent = time.perf_counter()

    try:
        resp = conn.getresponse()
    except (httplib.HTTPException, OSError) as e:
        raise ApiError("读取响应头失败: %s" % e, kind="timeout_or_network", retryable=True)
    t_hdr = time.perf_counter()

    if resp.status != 200:
        detail = ""
        try:
            detail = resp.read(2000).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        ra = resp.getheader("Retry-After")
        retry_after = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else None
        kind = "rate_limit" if resp.status == 429 else "http_error"
        raise ApiError("HTTP %d: %s" % (resp.status, detail[:300]), kind=kind, status=resp.status,
                       retryable=(resp.status >= 500 or resp.status in (408, 429)),
                       retry_after=retry_after,
                       extra={"conn_phases": conn_started, "t0": t0, "t_sent": t_sent, "t_hdr": t_hdr,
                              "reused_connection": reused, "response_headers": dict(resp.getheaders())})

    record = {
        "ok": True,
        "status": resp.status,
        "headers": dict(resp.getheaders()),
        "content_type": resp.getheader("Content-Type", ""),
        "transfer_encoding": resp.getheader("Transfer-Encoding", ""),
        "will_close": bool(getattr(resp, "will_close", False)),
        "t0": t0,
        "t_sent": t_sent,
        "t_hdr": t_hdr,
        "t_first_sse": None,
        "t_first_token": None,
        "t_first_reasoning": None,
        "t_first_content": None,
        "t_done": None,
        "conn_phases": conn_started,
        "reused_connection": reused,
        "will_close": None,
        "usage": {},
        "finish_reason": None,
        "sse_events": 0,
        "sse_issues": [],
        "reasoning_chars": 0,
        "content_chars": 0,
        "content_preview": "",
        "reasoning_preview": "",
    }

    if not args.stream:
        # 非流式对照：整包返回，「首包」= 完整响应体到达
        try:
            raw = resp.read()
        except (httplib.HTTPException, OSError) as e:
            raise ApiError("读取响应体失败: %s" % e, kind="timeout_or_network", retryable=True)
        t_done = time.perf_counter()
        record["t_done"] = t_done
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            raise ApiError("响应不是合法 JSON（前 300 字节：%r）" % raw[:300], kind="bad_json", retryable=False)
        ch = (data.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        record["content_chars"] = len(content)
        record["reasoning_chars"] = len(reasoning)
        record["content_preview"] = content[:160]
        record["reasoning_preview"] = reasoning[:160]
        record["finish_reason"] = ch.get("finish_reason")
        record["usage"] = data.get("usage") or {}
        record["t_first_sse"] = t_done
        record["t_first_token"] = t_done
        record["t_first_content"] = t_done if content else None
        return record, conn

    # ---- 流式：逐次 read1，保证首个 chunk 的到达时刻不被缓冲推迟 ----
    buf = b""
    text_parts, reasoning_parts = [], []
    done = False
    rd = getattr(resp, "read1", None) or resp.read
    try:
        while not done:
            chunk = rd(8192)
            if not chunk:
                break
            now = time.perf_counter()
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith(b"data:"):
                    # 兼容 event: / id: / 注释行，仅记录异常形态
                    if line.startswith(b":") or line.startswith(b"event:") or line.startswith(b"id:"):
                        continue
                    record["sse_issues"].append({"kind": "non_data_line", "line": line[:120].decode("utf-8", "replace")})
                    continue
                payload_line = line[5:].strip()
                if payload_line == b"[DONE]":
                    if record["t_done"] is None:
                        record["t_done"] = now
                    done = True
                    break
                try:
                    evt = json.loads(payload_line.decode("utf-8", "replace"))
                except ValueError as e:
                    record["sse_issues"].append({"kind": "bad_json", "err": str(e)[:120],
                                                 "line": payload_line[:120].decode("utf-8", "replace")})
                    continue

                if event_is_first_data(record, now):
                    record["t_first_sse"] = now

                usage = evt.get("usage")
                if usage:
                    record["usage"] = usage

                choices = evt.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    record["finish_reason"] = ch["finish_reason"]

                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                content = delta.get("content") or ""
                if reasoning:
                    if record["t_first_token"] is None:
                        record["t_first_token"] = now
                    if record["t_first_reasoning"] is None:
                        record["t_first_reasoning"] = now
                    reasoning_parts.append(reasoning)
                if content:
                    if record["t_first_token"] is None:
                        record["t_first_token"] = now
                    if record["t_first_content"] is None:
                        record["t_first_content"] = now
                    text_parts.append(content)
    except (httplib.HTTPException, OSError) as e:
        raise ApiError("流式读取中断: %s" % e, kind="stream_broken", retryable=True)
    finally:
        try:
            resp.read()  # 排空剩余字节，保证 keep-alive 连接可复用
        except Exception:  # noqa: BLE001
            pass

    t_end = time.perf_counter()
    if record["t_done"] is None:
        record["t_done"] = t_end
    record["sse_events"] = record["sse_events"] or 0
    content = "".join(text_parts)
    reasoning = "".join(reasoning_parts)
    record["content_chars"] = len(content)
    record["reasoning_chars"] = len(reasoning)
    record["content_preview"] = content[:160]
    record["reasoning_preview"] = reasoning[:160]
    return record, conn


_counted = {"n": 0}


def event_is_first_data(record, _now):
    """首个 data: 行只计一次（用 sse 事件计数避免重复置位）。"""
    record["sse_events"] += 1
    return record["sse_events"] == 1


# ---------------------------------------------------------------------------
# 计时换算
# ---------------------------------------------------------------------------

def to_timings(rec):
    def rel(t):
        return None if t is None else round((t - rec["t_sent"]) * 1000.0, 1)

    def cold(t):
        return None if t is None else round((t - rec["t0"]) * 1000.0, 1)

    cp = rec.get("conn_phases") or {}
    dns, tcp, tls = cp.get("dns_ms"), cp.get("tcp_ms"), cp.get("tls_ms")
    connect = None
    if dns is not None:
        connect = round((dns or 0) + (tcp or 0) + (tls or 0), 1)
    return {
        "dns_ms": None if dns is None else round(dns, 1),
        "tcp_ms": None if tcp is None else round(tcp, 1),
        "tls_ms": None if tls is None else round(tls, 1),
        "connect_ms": connect,
        "send_ms": round((rec["t_sent"] - rec["t0"]) * 1000.0 - (connect or 0), 1)
        if connect is not None else None,
        "ttfb_headers_ms": round((rec["t_hdr"] - rec["t_sent"]) * 1000.0, 1),
        "first_sse_ms": rel(rec["t_first_sse"]),
        "first_token_ms": rel(rec["t_first_token"]),
        "first_content_ms": rel(rec["t_first_content"]),
        "e2e_ms": rel(rec["t_done"]),
        "cold_first_token_ms": cold(rec["t_first_token"]),
        "total_ms": cold(rec["t_done"]),
        "reasoning_gap_ms": (round((rec["t_first_content"] - rec["t_first_token"]) * 1000.0, 1)
                             if rec["t_first_content"] and rec["t_first_token"] else None),
    }


def error_timings(extra):
    """失败样本也保留「建连 + 响应头」耗时：401 很快 / 429 很慢，本身都是首包信号。"""
    if not extra:
        return {}
    cp = extra.get("conn_phases") or {}
    dns, tcp, tls = cp.get("dns_ms"), cp.get("tcp_ms"), cp.get("tls_ms")
    connect = None
    if dns is not None:
        connect = round((dns or 0) + (tcp or 0) + (tls or 0), 1)
    out = {"connect_ms": connect}
    t_sent, t_hdr = extra.get("t_sent"), extra.get("t_hdr")
    if t_sent is not None and t_hdr is not None:
        out["ttfb_headers_ms"] = round((t_hdr - t_sent) * 1000.0, 1)
    t0 = extra.get("t0")
    if t0 is not None and t_hdr is not None:
        out["total_ms"] = round((t_hdr - t0) * 1000.0, 1)
    return out


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stat_block(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": round(min(vals), 1),
        "p50": round(pct(vals, 0.50), 1),
        "p90": round(pct(vals, 0.90), 1),
        "p95": round(pct(vals, 0.95), 1),
        "p99": round(pct(vals, 0.99), 1),
        "max": round(max(vals), 1),
        "mean": round(statistics.mean(vals), 1),
        "std": round(statistics.pstdev(vals), 1) if len(vals) > 1 else 0.0,
        "jitter_p95_minus_p50": round(pct(vals, 0.95) - pct(vals, 0.50), 1),
    }


TIMING_KEYS = ["dns_ms", "tcp_ms", "tls_ms", "connect_ms", "send_ms", "ttfb_headers_ms",
               "first_sse_ms", "first_token_ms", "first_content_ms", "reasoning_gap_ms",
               "e2e_ms", "cold_first_token_ms", "total_ms"]

TIMING_LABELS = {
    "dns_ms": "DNS解析",
    "tcp_ms": "TCP建连",
    "tls_ms": "TLS握手",
    "connect_ms": "建连合计",
    "send_ms": "请求发送",
    "ttfb_headers_ms": "首字节(响应头)",
    "first_sse_ms": "首个SSE包",
    "first_token_ms": "首token(热)",
    "first_content_ms": "首content(热)",
    "reasoning_gap_ms": "思考→内容间隔",
    "e2e_ms": "流结束(热)",
    "cold_first_token_ms": "首token(冷,含建连)",
    "total_ms": "总耗时(冷)",
}


# ---------------------------------------------------------------------------
# 采样主流程
# ---------------------------------------------------------------------------

def resolve_target(args):
    """确定 base_url / api_key：命令行 > 环境变量 > config/providers.json。"""
    base_url = args.base_url
    api_key = args.api_key

    cfg = {}
    if os.path.exists(DEFAULT_CFG):
        try:
            with open(DEFAULT_CFG, encoding="utf-8") as f:
                cfg = json.load(f)
        except ValueError:
            cfg = {}

    if not base_url and cfg:
        for ch in cfg.get("channels") or []:
            if ch.get("id") == args.channel and ch.get("base_url"):
                base_url = ch["base_url"]
                if not api_key and ch.get("api_key"):
                    api_key = ch["api_key"]
                if not api_key and ch.get("api_key_env"):
                    api_key = os.environ.get(ch["api_key_env"], "")
                break
    base_url = base_url or DEFAULT_BASE_URL
    if not api_key:
        api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        pass  # 允许无 key 的 mock/本地服务；真实渠道会返回 401 并被记录
    return base_url.rstrip("/"), api_key


def parse_target(base_url):
    from urllib.parse import urlsplit
    u = urlsplit(base_url)
    scheme = u.scheme or "https"
    host = u.hostname or "localhost"
    port = u.port or (443 if scheme == "https" else 80)
    path_prefix = u.path.rstrip("/")
    return scheme, host, port, path_prefix


def run_samples(args, model, case_key, case):
    """对某个模型 × 用例采样 runs 次（含 warmup、重试），返回 records。"""
    records = []
    conn = None
    attempts = 0
    tries = 0
    while tries < args.runs:
        tries += 1
        attempt = 0
        while True:
            attempt += 1
            attempts += 1
            started = time.perf_counter()
            try:
                rec, conn = probe_once(conn if args.keepalive else None, args, model, case)
                rec["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                rec["attempt"] = attempt
                rec["model"] = model
                rec["case"] = case_key
                rec["case_name"] = case["name"]
                rec["run"] = tries
                rec["keepalive"] = bool(args.keepalive)
                rec["warmup"] = tries <= args.warmup
                rec["timings"] = to_timings(rec)
                records.append(rec)
                break
            except ApiError as e:
                can_retry = e.retryable and attempt <= args.retries
                if can_retry:
                    wait = e.retry_after if e.retry_after is not None else (5.0 if e.status == 429 else 1.5 * attempt)
                    print("    ! %s（%s），%.1fs 后重试..." % (e, "限流" if e.status == 429 else "可重试", wait))
                    time.sleep(min(wait, 30.0))
                    conn = None  # 失败后连接状态不可信，强制重建
                    continue
                rec = {
                    "ok": False, "error": {"type": e.kind, "status": e.status, "message": " ".join(str(e).split())[:400]},
                    "model": model, "case": case_key, "case_name": case["name"], "run": tries,
                    "attempt": attempt, "keepalive": bool(args.keepalive), "warmup": tries <= args.warmup,
                    "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "conn_phases": e.extra.get("conn_phases") or {},
                    "reused_connection": bool(e.extra.get("reused_connection")),
                    "timings": error_timings(e.extra),
                }
                records.append(rec)
                conn = None
                break
        if not args.keepalive:
            pass
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    return records


# ---------------------------------------------------------------------------
# 输出：控制台 / JSON / CSV / Markdown
# ---------------------------------------------------------------------------

def fmt(v, unit=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%.1f" % v) + unit
    return str(v) + unit


def summarize(records):
    """按 (model, case) 汇总统计。"""
    groups = {}
    for r in records:
        if r.get("warmup"):
            continue
        key = (r["model"], r["case"])
        g = groups.setdefault(key, {"records": [], "errors": 0, "total": 0})
        g["total"] += 1
        if r.get("ok"):
            g["records"].append(r)
        else:
            g["errors"] += 1
    out = {}
    for key, g in groups.items():
        stats = {}
        for k in TIMING_KEYS:
            stats[k] = stat_block([(r.get("timings") or {}).get(k) for r in g["records"]])
        ok_recs = g["records"]
        cached_hits = sum(1 for r in ok_recs if _cached(r) > 0)
        truncated = sum(1 for r in ok_recs if r.get("finish_reason") == "length")
        out[key] = {
            "stats": stats,
            "total": g["total"],
            "ok": len(ok_recs),
            "errors": g["errors"],
            "cached_hits": cached_hits,
            "truncated": truncated,
            "reconnects": sum(1 for r in ok_recs if not r.get("reused_connection")),
            "usage_samples": [r.get("usage") or {} for r in ok_recs],
            "sse_issues": sum(len(r.get("sse_issues") or []) for r in ok_recs),
        }
    return out


def _cached(rec):
    u = rec.get("usage") or {}
    ctd = u.get("prompt_tokens_details") or {}
    v = ctd.get("cached_tokens")
    if v is None:
        v = u.get("cached_tokens")
    return v or 0


def print_console(summary, args):
    print("\n" + "=" * 108)
    print("  首包延迟探针结果（单位 ms，warmup 不计入统计）")
    print("=" * 108)
    for (model, case_key), s in summary.items():
        st = s["stats"]
        print("\n· %s  |  用例 %s（%s）  |  成功 %d/%d  失败 %d  缓存命中 %d  截断 %d"
              % (model, case_key, CASES.get(case_key, {}).get("name", ""),
                 s["ok"], s["total"], s["errors"], s["cached_hits"], s["truncated"]))
        print("  %-20s %10s %10s %10s %10s %10s" % ("阶段", "P50", "P90", "P95", "均值", "最大"))
        print("  " + "-" * 76)
        for k in ["connect_ms", "ttfb_headers_ms", "first_sse_ms", "first_token_ms",
                  "first_content_ms", "reasoning_gap_ms", "e2e_ms"]:
            b = st.get(k)
            if not b:
                continue
            print("  %-20s %10.1f %10.1f %10.1f %10.1f %10.1f"
                  % (TIMING_LABELS[k], b["p50"], b["p90"], b["p95"], b["mean"], b["max"]))
        cold = st.get("cold_first_token_ms")
        if cold:
            print("  %-20s %10.1f %10.1f %10.1f %10.1f %10.1f"
                  % ("首token(冷,含建连)", cold["p50"], cold["p90"], cold["p95"], cold["mean"], cold["max"]))
    print("\n完整明细与建议见 report.md / probe.json / report.csv")


def render_markdown(summary, records, args, env):
    lines = []
    lines.append("# 首包延迟探针报告（%s）\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("## 一、测试配置\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 目标渠道 / Base URL | `%s` |" % args.base_url)
    lines.append("| 模型 | `%s` |" % ", ".join(args.models))
    lines.append("| 用例 | %s |" % ", ".join("%s（%s）" % (k, CASES[k]["name"]) for k in args.cases))
    lines.append("| 采样次数 | %d 次/用例（预热 %d 次不计） |" % (args.runs, args.warmup))
    lines.append("| 连接模式 | %s |" % ("复用长连接（keep-alive）" if args.keepalive else "每次新建连接（冷连接）"))
    lines.append("| 流式 | %s |" % ("是（SSE）" if args.stream else "否（非流式对照，首包=整包返回）"))
    lines.append("| Accept-Encoding | `%s` |" % args.accept_encoding)
    lines.append("| 超时 / 重试 | %ss / %d 次 |" % (args.timeout, args.retries))
    lines.append("| temperature | %s |" % ("不传" if args.no_temperature else args.temperature))
    lines.append("| 执行主机 | %s / %s / Python %s |" % (env.get("hostname"), env.get("os"), env.get("python")))
    lines.append("")

    lines.append("## 二、「首包」三种口径（先对齐定义，避免结论打架）\n")
    lines.append("| 口径 | 字段 | 含义 | 适用场景 |")
    lines.append("|---|---|---|---|")
    lines.append("| 首字节 | `ttfb_headers_ms` | 请求发出 → HTTP 响应头到达 | 判断网关/排队/网络，不含模型推理 |")
    lines.append("| 首 token（业界 TTFT） | `first_token_ms` | 请求发出 → 第一个 delta（可能是 reasoning）到达 | 对比模型/渠道的服务端响应速度 |")
    lines.append("| 首 content（用户可见首字） | `first_content_ms` | 请求发出 → 第一个 `content` 到达 | 真实用户体验；推理模型可能比首 token 晚很多 |")
    lines.append("")
    lines.append("> 冷口径 `cold_first_token_ms` = 含 DNS/TCP/TLS 建连的首 token；热口径不含建连，二者差值即建连开销。\n")

    lines.append("## 三、阶段分解与分布统计（单位 ms）\n")
    for (model, case_key), s in summary.items():
        st = s["stats"]
        lines.append("### %s ｜ 用例 `%s`：%s\n" % (model, case_key, CASES.get(case_key, {}).get("name", "")))
        lines.append("成功 %d/%d，失败 %d，缓存命中 %d 次，输出被截断 %d 次，SSE 异常 %d 处。\n"
                     % (s["ok"], s["total"], s["errors"], s["cached_hits"], s["truncated"], s["sse_issues"]))
        lines.append("| 阶段 | P50 | P90 | P95 | P99 | 均值 | 最小 | 最大 | 抖动(P95-P50) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for k in TIMING_KEYS:
            b = st.get(k)
            if not b:
                continue
            lines.append("| %s `%s` | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
                         % (TIMING_LABELS[k], k, b["p50"], b["p90"], b["p95"], b["p99"],
                            b["mean"], b["min"], b["max"], b["jitter_p95_minus_p50"]))
        lines.append("")

    lines.append("## 四、逐次采样明细\n")
    lines.append("| 模型 | 用例 | 序号 | 首字节 | 首SSE | 首token | 首content | 结束 | prompt_tok | cached_tok | finish |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        if r.get("warmup"):
            continue
        t = r.get("timings") or {}
        u = r.get("usage") or {}
        if not r.get("ok"):
            lines.append("| %s | %s | %d | 失败：%s | | | | | | | |"
                         % (r["model"], r["case"], r["run"], (r.get("error") or {}).get("message", "")[:60]))
            continue
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (r["model"], r["case"], r["run"],
                        fmt(t.get("ttfb_headers_ms")), fmt(t.get("first_sse_ms")),
                        fmt(t.get("first_token_ms")), fmt(t.get("first_content_ms")),
                        fmt(t.get("e2e_ms")), u.get("prompt_tokens", "-"), _cached(r),
                        r.get("finish_reason")))
    lines.append("")

    lines.append("## 五、自动诊断（基于本次实测数据）\n")
    for (model, case_key), s in summary.items():
        st = s["stats"]
        lines.append("### %s ｜ `%s`\n" % (model, case_key))
        for d in diagnose(s, st):
            lines.append("- %s" % d)
        lines.append("")

    lines.append("## 六、方法论：正反建议\n")
    lines.append(FIXED_ADVICE)
    return "\n".join(lines)


def diagnose(s, st):
    """把实测数据翻译成可执行判断，避免只给数字不给结论。

    所有结论都带「绝对阈值」保护：本地/mock 等毫秒级场景不会输出无意义的百分比结论。
    """
    out = []
    ttfb = st.get("ttfb_headers_ms")
    ft = st.get("first_token_ms")
    fc = st.get("first_content_ms")
    conn = st.get("connect_ms")
    cold = st.get("cold_first_token_ms")
    gap = st.get("reasoning_gap_ms")

    if s["errors"]:
        out.append("❌ 有 %d/%d 次失败：先排除限流/超时/鉴权，再谈首包，否则样本不可比。" % (s["errors"], s["total"]))
    if ttfb and ft and dt_ms(ttfb["p50"]) >= 200:
        if ttfb["p50"] > ft["p50"] * 0.8:
            out.append("⚠️ 首字节（%.0fms）已占首 token（%.0fms）的 %.0f%%，瓶颈在网关/排队或网络，而非模型推理本身。"
                       % (ttfb["p50"], ft["p50"], 100.0 * ttfb["p50"] / max(ft["p50"], 1)))
        else:
            out.append("✅ 首字节仅 %.0fms、首 token %.0fms：网关受理很快，主要耗时在模型侧 prefill/排队（模型侧 %.0fms）。"
                       % (ttfb["p50"], ft["p50"], ft["p50"] - ttfb["p50"]))
    if ft and fc:
        ratio = fc["p50"] / max(ft["p50"], 1)
        diff = fc["p50"] - ft["p50"]
        if ratio >= 1.5 and diff >= 500:
            out.append("⚠️ 首 content（%.1fs）是首 token（%.1fs）的 %.1f 倍：该模型先吐 reasoning，"
                       "用户可见首字要等 %.1fs。做「首包」验收必须明确取哪个口径，否则结论会差 %.1fs。"
                       % (fc["p50"] / 1000.0, ft["p50"] / 1000.0, ratio, fc["p50"] / 1000.0, diff / 1000.0))
        elif diff >= 100:
            out.append("ℹ️ 首 content 比首 token 晚 %.0fms（存在短暂思考期），"
                       "首包验收若指「用户可见首字」应以首 content 为准。" % diff)
        else:
            out.append("✅ 首 content 与首 token 接近（%.0f vs %.0fms），该用例未见明显「思考期」，"
                       "首包可直接代表用户可见首字。" % (fc["p50"], ft["p50"]))
    if gap and gap["p50"] > 3000:
        out.append("ℹ️ 思考→首个可见内容间隔中位数 %.1fs（P95 %.1fs）：属于推理型模型的正常形态，"
                   "但对「首字即响应」类产品是硬伤，建议产品侧加进度占位或先流式输出思考摘要。"
                   % (gap["p50"] / 1000.0, gap["p95"] / 1000.0))
    if conn and cold and cold["p50"] >= 200:
        share = 100.0 * conn["p50"] / max(cold["p50"], 1)
        out.append("ℹ️ 建连（DNS+TCP+TLS）P50 %.0fms，占冷口径首 token 的 %.1f%%：%s"
                   % (conn["p50"], share,
                      "占比偏高，生产端务必复用连接，否则每次请求都白付这笔开销。" if share >= 15
                      else "占比可接受，但长期高频调用仍建议复用连接。"))
    if ft and ft["p50"] >= 100:
        j = ft["jitter_p95_minus_p50"]
        out.append("ℹ️ 首 token 抖动 P95-P50 = %.0fms，P95/P50 = %.2f：%s"
                   % (j, ft["p95"] / max(ft["p50"], 1),
                      "抖动大，单次测量无意义，验收应看 P95 且至少 20 次采样。" if j > ft["p50"] * 0.5
                      else "抖动可控，P50 可作为常规参考。"))
    if s["cached_hits"]:
        out.append("⚠️ %d/%d 次出现 cached_tokens>0：命中前缀缓存会让首包「虚低」，"
                   "若要测真实冷启动首包，应换 prompt（可用 --fresh-prompt）或单独剔除这些样本。"
                   % (s["cached_hits"], s["ok"]))
    if s["truncated"]:
        out.append("⚠️ %d 次输出被 max_tokens 截断：若思考很长会吃掉额度导致看不到 content，"
                   "测「首 content」时请调大 --max-tokens。" % s["truncated"])
    if ft and ft["p95"] > 8000:
        out.append("❌ 首 token P95 = %.1fs，已超出常见交互可接受阈值（一般 3~5s），"
                   "若用于线上产品需确认是否为 preview 后端资源不足或跨区路由。"
                   % (ft["p95"] / 1000.0))
    elif ft and ft["p50"] >= 100:
        out.append("✅ 首 token P50 = %.0fms、P95 = %.0fms，处于可接受区间。"
                   % (ft["p50"], ft["p95"]))
    elif ft:
        out.append("ℹ️ 首 token P50 仅 %.1fms（亚百毫秒级，本地/mock 或极近端场景）："
                   "该结果只能验证采集链路正确，不能外推公网渠道表现。" % ft["p50"])
    if not out:
        out.append("✅ 各阶段均处于亚百毫秒级（本地/mock 或极近端场景），"
                   "该结果只能验证采集链路正确，不能外推公网渠道表现。")
    return out


def dt_ms(v):
    return v or 0


FIXED_ADVICE = """
### 6.1 这个测法的「正」面（为什么值得做）

1. **能定位瓶颈归属**：把 DNS/TCP/TLS/响应头/首 token 拆开，可以立刻回答「慢在网络、慢在网关排队、还是慢在模型 prefill」——这三类的优化动作完全不同（换线路 / 换渠道 / 换模型或缩短输入）。
2. **区分「首 token」与「首 content」**：对推理模型（如 gemini-3.1-pro-preview 这类 preview 推理模型）二者可能差几十秒。只测一个口径，要么高估体验（只看思考首包），要么误判模型不行（只看可见首字）。
3. **冷热两套口径同时给**：`cold_first_token_ms` 含建连、`first_token_ms` 不含，差值就是连接复用能省下的钱，能直接支撑「上 keep-alive / 连接池」的决策。
4. **看分布而不是看单次**：P50/P90/P95/P99 + 抖动，避免被一次 800ms 的「幸运样本」误导。首包是典型长尾指标，平均值几乎没有意义。
5. **暴露缓存与截断污染**：`cached_tokens>0` 会让首包虚低，`max_tokens` 太小会让 content 永远等不到——脚本会标注，避免得出反向结论。
6. **零依赖、可移植**：纯标准库，EC2、本地、容器里都能直接跑，不需要装 requests，也不受代理库行为影响。

### 6.2 这个测法的「反」面（必须在报告里写清的局限）

1. **单点单机，不代表全链路**：测的是「你的机器 → 该渠道」的首包，跨区、跨运营商、走代理、TLS 中间盒都会显著改变结果；对最终用户只能作参考基线。
2. **只测首包，不测吞吐**：首包快不等于生成快。必须配合 `tokens/s`、`e2e` 一起看，否则会出现「首包 300ms 但吐 20 分钟」的误判。
3. **preview 模型本身不稳定**：`*-preview` 通常指向灰度/共享后端，容量与版本随时变化；单日数据不能当长期结论，需连续多天、多时段复测。
4. **路由/渠道差异会被归因到模型**：同一模型不同渠道首包可能差 10 倍（历史数据里 glm 在两条渠道上 E2E 差 4.5 倍）。结论必须写成「模型 @ 渠道」，不能只写模型名。
5. **HTTP/1.1 单连接测法**：脚本用标准库 `http.client`（HTTP/1.1），无法测 HTTP/2 多路复用/QUIC 下的表现；若生产走 HTTP/2，首包基线需另外用 HTTP/2 客户端测。
6. **`Accept-Encoding` 会改变首包**：带 gzip 时部分网关会攒包再压，首个 chunk 被推迟。脚本默认 `identity`（不压缩，最接近真实首字），但生产若开 gzip，实际首包会与本次结果不同。
7. **小样本仍会骗人**：`--runs 5` 得到的 P95 基本没有统计意义；要下结论建议 ≥20 次，且预热样本必须剔除（脚本已默认剔除，但仍需注意首个用例的首批样本受冷启动影响）。

### 6.3 落地建议（按优先级）

1. **先定验收口径**：产品说「首包」通常指**首 content**；技术侧 SLA 建议同时约定 `first_token_p95 ≤ 3s` 与 `first_content_p95 ≤ 8s`（按业务可调），并写清是否含建连。
2. **采样规范**：每用例 ≥20 次、预热 2 次剔除、跨 2 个时段（闲时/忙时）各测一轮；报告只引用 P50/P95，不引用均值。
3. **固定对照基线**：至少同测一个已知稳定的模型（如 gpt-5.2 或 kimi-k3）作为「本机→本渠道」的网络基线，用差值判断 gemini 是否真的慢。
4. **分输入长度测**：首包对输入长度敏感（prefill），至少跑 `short` 与 `long` 两个用例，确认首包随输入 token 的线性增长斜率是否正常。
5. **缓存隔离**：若渠道有前缀缓存，测冷首包必须每次换 prompt（脚本支持 `--fresh-prompt` 追加随机后缀），否则测到的是缓存命中速度。
6. **上线前灰度验证**：把首包 P95 纳入定时探测（本脚本可直接进 cron），持续观察 preview 版本变更带来的跳变。
7. **不要只看首包做选型**：选型 = 首包(P95) + 生成速度 + 质量得分 + 长上下文，缺一不可；首包只解决「响应快不快」，解决不了「答得对不对」。
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def collect_env():
    import platform
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "timezone": time.strftime("%Z"),
    }


def main():
    global _FILLER_SENTENCES  # noqa: PLW0603

    ap = argparse.ArgumentParser(
        description="单模型首包/首 token/首内容延迟探针（标准库实现，零依赖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：python probe_ttft.py --model gemini-3.1-pro-preview --runs 20 --cases short,long",
    )
    ap.add_argument("--base-url", default="", help="渠道 Base URL（默认读 config/providers.json 的 easyrouter，兜底 %s）" % DEFAULT_BASE_URL)
    ap.add_argument("--channel", default="easyrouter", help="从 config/providers.json 取该渠道的 base_url/api_key（默认 easyrouter）")
    ap.add_argument("--api-key", default="", help="API Key（默认读 --api-key-env 环境变量，或 providers.json）")
    ap.add_argument("--api-key-env", default="EASYROUTER_API_KEY", help="API Key 环境变量名（默认 EASYROUTER_API_KEY）")
    ap.add_argument("--path", default="/chat/completions", help="请求路径（默认 /chat/completions）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="单模型（默认 %s）" % DEFAULT_MODEL)
    ap.add_argument("--models", default="", help="逗号分隔多模型对比（优先于 --model）")
    ap.add_argument("--cases", default="short,long", help="逗号分隔用例：%s（默认 short,long）" % ",".join(CASES))
    ap.add_argument("--runs", type=int, default=10, help="每个用例采样次数（默认 10，建议 ≥20）")
    ap.add_argument("--warmup", type=int, default=2, help="预热次数，不计入统计（默认 2）")
    ap.add_argument("--keepalive", action="store_true", help="复用长连接（默认每次新建连接，测冷连接首包）")
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="非流式对照（首包=整包返回）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-temperature", action="store_true", help="不传 temperature（部分推理模型拒绝该参数）")
    ap.add_argument("--max-tokens", type=int, default=0, help="覆盖用例的 max_tokens（0=用用例自带值）")
    ap.add_argument("--accept-encoding", default="identity", choices=["identity", "gzip", "gzip, deflate"],
                    help="默认 identity（避免压缩缓冲推迟首包）；gzip 用于观察网关攒包影响")
    ap.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒数（默认 120）")
    ap.add_argument("--retries", type=int, default=2, help="失败重试次数（429/5xx/超时可重试，默认 2）")
    ap.add_argument("--insecure", action="store_true", help="跳过 TLS 证书校验（仅排障用）")
    ap.add_argument("--extra-body", default="", help="追加进请求体的 JSON，如 '{\"reasoning_effort\":\"low\"}'")
    ap.add_argument("--fresh-prompt", action="store_true", help="每次请求在 prompt 末尾追加随机串，规避前缀缓存")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出根目录（默认 output_ttft）")
    ap.add_argument("--no-ts", action="store_true", help="不加时间戳子目录")
    ap.add_argument("--list-cases", action="store_true", help="列出内置用例后退出")
    args = ap.parse_args()

    if args.list_cases:
        print("内置用例：")
        for k, c in CASES.items():
            print("  %-9s %-30s max_tokens=%-5d prompt≈%d 字符" % (k, c["name"], c.get("max_tokens"), len(c["prompt"])))
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [args.model]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    for c in cases:
        if c not in CASES:
            sys.exit("未知用例 %s（可用：%s）" % (c, ",".join(CASES)))
    args.models = models
    args.cases = cases

    base_url, api_key = resolve_target(args)
    scheme, host, port, path_prefix = parse_target(base_url)
    args.base_url = base_url
    args.api_key = api_key
    args.scheme = scheme
    args.host = host
    args.port = port
    if path_prefix:
        args.path = path_prefix + (args.path if args.path.startswith("/") else "/" + args.path)

    if not api_key:
        print("[提示] 未找到 API Key：依次检查 --api-key、环境变量 %s、config/providers.json 的 %s 渠道。"
              % (args.api_key_env, args.channel))

    print("目标：%s  |  模型：%s  |  用例：%s  |  采样：%d 次/用例（预热 %d）  |  连接：%s"
          % (base_url, ",".join(models), ",".join(cases), args.runs, args.warmup,
             "复用" if args.keepalive else "新建"))

    all_records = []
    for model in models:
        for case_key in cases:
            case = dict(CASES[case_key])
            if args.fresh_prompt:
                case["prompt"] = case["prompt"] + "\n\n[req-%s]" % os.urandom(4).hex()
            print("\n>> 采样 %s ｜ %s" % (model, case["name"]))
            recs = run_samples(args, model, case_key, case)
            for r in recs:
                if r.get("warmup"):
                    status = "ok" if r.get("ok") else "fail"
                    print("   warmup #%d %s" % (r["run"], status))
                    continue
                t = r.get("timings") or {}
                if r.get("ok"):
                    print("   #%-2d 首字节 %7s  首SSE %7s  首token %7s  首content %7s  结束 %8s  %s"
                          % (r["run"], fmt(t.get("ttfb_headers_ms")), fmt(t.get("first_sse_ms")),
                             fmt(t.get("first_token_ms")), fmt(t.get("first_content_ms")),
                             fmt(t.get("e2e_ms")),
                             ("cached=%s" % _cached(r)) if _cached(r) else ""))
                else:
                    msg = (r.get("error") or {}).get("message", "")
                    print("   #%-2d 失败：%s" % (r["run"], " ".join(msg.split())[:120]))
            all_records.extend(recs)

    summary = summarize(all_records)
    print_console(summary, args)

    out_dir = args.out if args.no_ts else os.path.join(args.out, "ttft_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "probe.json"), "w", encoding="utf-8") as f:
        json.dump({
            "args": {k: v for k, v in vars(args).items() if k != "api_key"},
            "environment": collect_env(),
            "summary": {"%s|%s" % k: v for k, v in summary.items()},
            "records": [{k: v for k, v in r.items() if k not in ("t0", "t_sent", "t_hdr", "t_first_sse",
                                                                  "t_first_token", "t_first_reasoning",
                                                                  "t_first_content", "t_done")}
                        for r in all_records],
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(summary, all_records, args, collect_env()))

    with open(os.path.join(out_dir, "report.csv"), "w", encoding="utf-8-sig", newline="") as f:
        cols = ["model", "case", "run", "ok", "keepalive", "warmup"] + TIMING_KEYS + \
               ["prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason", "error"]
        w = csv.writer(f)
        w.writerow(cols)
        for r in all_records:
            t = r.get("timings") or {}
            u = r.get("usage") or {}
            w.writerow([r.get("model"), r.get("case"), r.get("run"), r.get("ok"), r.get("keepalive"),
                        r.get("warmup")] + [t.get(k) for k in TIMING_KEYS] +
                       [u.get("prompt_tokens"), u.get("completion_tokens"), _cached(r),
                        r.get("finish_reason"), json.dumps(r.get("error"), ensure_ascii=False) if r.get("error") else ""])

    print("\n结果已写入：%s" % os.path.abspath(out_dir))
    print("  probe.json / report.md / report.csv")
    ok_any = any(r.get("ok") for r in all_records if not r.get("warmup"))
    return 0 if ok_any else 2


if __name__ == "__main__":
    sys.exit(main())
