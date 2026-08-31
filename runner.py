#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多渠道统一模型能力测试（统一请求构造 / 变量控制 / 指标采集 / 结果输出）
================================================================

与旧版单文件脚本 kimi_k3_easyrouter_vs_mayi.py 相比的核心改进：
  1. 配置外置（config/providers.json，模板 config/providers-template.json），只换 base_url / api_key / model，其余参数全渠道共用；
  2. 测试用例独立成 JSON（test_cases/*.json），并支持从 HuggingFace 下载抽取（download_datasets.py）；
  3. 统一请求构造函数 build_request_body，长输入用「用例ID」种子确定性生成，prompt 全渠道一致；
  4. 用例按固定顺序逐条依次发给所有渠道，发送时间戳写入 order.json；
  5. 推理模型适配：单独采集 reasoning_content / reasoning_tokens，缓存命中 cached_tokens；
  6. 输出原始 JSON + 汇总 JSON + Markdown 对比表，配合 Excel 报告（export_report.py）。

四层用例：performance（性能基准）/ compatibility（功能兼容）/ quality（质量，来自 HF）/
          long_context（长上下文专项）

用法：
    python runner.py --list-cases
    python runner.py --dry-run
    python runner.py --providers easyrouter,mayi
    python runner.py --layers performance,quality,long_context
    python runner.py --max-cases 5 --verbose
依赖：pip install requests
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import re
import socket
import statistics
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("缺少依赖 requests，请先执行：pip install requests")

LAYERS = ("performance", "compatibility", "quality", "long_context")

# 各测试层的用途说明（写入报告，便于理解实验设计）
LAYER_DESC = {
    "performance": "性能基准层：短/长输入 × 短/长输出 四象限，测 TTFT、生成速度、端到端延迟、chunk 分布",
    "compatibility": "功能兼容层：SSE 格式 / stop / max_tokens / temperature / Function Calling / JSON mode / 多轮 / system 提示词",
    "quality": "质量层：数学推理(GSM8K) + 幻觉事实性(TruthfulQA)，judge 判分正确率",
    "long_context": "长上下文专项：大海捞针 10K/32K/64K/128K，输入按 target_tokens 生成，输出 max_tokens=4096",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROVIDERS_FILE = os.path.join(HERE, "config", "providers.json")
DEFAULT_CASES_DIR = os.path.join(HERE, "test_cases")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output")


def log(msg, verbose=True):
    if verbose:
        print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def stable_seed(s):
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def estimate_tokens(text):
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef")
    return cjk + (len(text) - cjk) // 4


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    if f + 1 >= len(s):
        return round(s[f], 2)
    return round(s[f] + (s[f + 1] - s[f]) * (k - f), 2)


def dist_stats(values):
    if not values:
        return None
    avg = statistics.mean(values)
    try:
        std = statistics.pstdev(values)
    except Exception:
        std = 0.0
    return {"count": len(values), "min": round(min(values), 2), "avg": round(avg, 2),
            "max": round(max(values), 2), "std": round(std, 2)}


def total_memory_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        pass
    try:  # Windows 兜底
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return round(st.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass
    return None


def env_info(env_tag=""):
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "memory_gb": total_memory_gb(),
        "python": platform.python_version(),
        "tz": time.strftime("%Z %z"),
        "env_tag": env_tag or "(未标注)",
        "note": "所有渠道在同一台机器上顺序执行，保证网络与算力环境一致",
    }


# ---------------------------------------------------------------------------
# 确定性填充文本（保证所有渠道收到完全一致的 prompt）
# ---------------------------------------------------------------------------

_ZH_SENTENCES = [
    "数字孪生技术正在重塑制造业的生产管理方式。", "深度学习模型在自然语言处理任务上取得了显著进展。",
    "大语言模型的核心能力来源于海量语料上的自监督预训练。", "检索增强生成可以有效缓解模型幻觉问题并提升事实准确性。",
    "多模态模型能够同时理解文本、图像、音频与视频信息。", "强化学习从人类反馈中优化模型行为，使其更符合用户预期。",
    "推理时的思维链提示能够显著提升复杂数学与逻辑任务的准确率。", "向量数据库是构建企业级知识库问答系统的重要基础设施。",
    "提示词工程强调用清晰、结构化的指令约束模型的输出格式。", "模型量化技术可以将大模型的显存占用降低数倍。",
    "分布式训练框架需要在通信带宽与计算效率之间取得平衡。", "上下文窗口的大小决定了模型一次能够处理的信息量上限。",
    "流式输出技术显著降低了用户感知的首字延迟。", "函数调用能力让语言模型可以接入外部工具与业务系统。",
    "结构化输出能力保证了模型结果可以被程序直接解析。", "温度参数控制着模型生成结果的随机性与多样性。",
    "注意力机制让模型能够在长文本中定位关键信息。", "知识蒸馏通过教师模型指导学生模型实现能力迁移。",
    "模型评估需要同时关注正确率、延迟、成本与稳定性。", "数据清洗与去重是训练高质量模型的前置条件。",
    "部署推理服务时需要考虑吞吐量、并发与弹性伸缩。", "缓存命中率直接影响线上推理服务的响应速度。",
    "长文本摘要任务对模型的定位与压缩能力提出了更高要求。", "对话系统的多轮一致性依赖对历史上下文的精确建模。",
    "安全对齐训练能够降低模型输出有害内容的概率。", "系统提示词为模型定义了角色、边界与输出规范。",
    "工具调用协议通常采用 JSON 格式描述函数签名与参数。", "评测基准的设计需要覆盖多种难度与场景以反映真实能力。",
    "模型微调可以在少量标注数据上快速适配垂直领域。", "上下文压缩技术可以在保留关键信息的同时降低输入开销。",
]


def build_filler(target_chars, seed):
    rng = random.Random(seed)
    parts, total, pi = [], 0, 0
    while total < target_chars:
        pi += 1
        n = rng.randint(5, 9)
        parts.append("第%d段。" % pi + "".join(rng.choice(_ZH_SENTENCES) for _ in range(n)) + "\n")
        total += len(parts[-1])
    return "".join(parts)[:target_chars]


def build_messages(case):
    msgs = []
    if case.get("system"):
        msgs.append({"role": "system", "content": case["system"]})
    for h in case.get("history", []):
        msgs.append({"role": h["role"], "content": h["content"]})
    prompt = case["prompt"]
    if case.get("filler_tokens"):
        filler = build_filler(int(float(case["filler_tokens"]) * 1.2), stable_seed(case["id"]))
        prompt = prompt.replace("{filler}", filler)
    if case.get("needle"):
        base = build_filler(int(float(case["target_tokens"]) * 1.2), stable_seed(case["id"]))
        at = int(float(case["target_tokens"]) * 1.2 * float(case["needle"]["depth"]))
        text = base[:at] + "\n" + case["needle"]["text"] + "\n" + base[at:]
        prompt = prompt.replace("{needle_text}", text).replace("{question}", case["question"])
    msgs.append({"role": "user", "content": prompt})
    return msgs


# ---------------------------------------------------------------------------
# 统一请求构造函数（所有渠道共用同一套参数，只换 base_url / api_key）
# ---------------------------------------------------------------------------

def build_request_body(provider, case, shared_params, reasoning_max_tokens):
    body = {"model": provider["model"], "messages": build_messages(case)}
    if case.get("stream"):
        body["stream"] = True
        if provider.get("supports", {}).get("stream_usage", True):
            body["stream_options"] = {"include_usage": True}

    supports_temp = provider.get("supports", {}).get("temperature", True)
    force = case.get("force_params", False)
    case_temp = case.get("temperature")
    if case_temp is not None and (force or supports_temp):
        body["temperature"] = case_temp
    elif case_temp is None and "temperature" in shared_params and supports_temp:
        body["temperature"] = shared_params["temperature"]

    max_tokens = case.get("max_tokens")
    if max_tokens is None:
        max_tokens = shared_params.get("max_tokens")
    # 推理模型：除非用例显式 force_params（如 max_tokens 约束测试），否则至少给 reasoning_max_tokens，
    # 避免 reasoning 耗尽内容空间导致输出被截断（对应旧脚本 REASONING_MAX_TOKENS=4096 的做法）
    if provider.get("reasoning") and not force:
        max_tokens = max(max_tokens or 0, int(reasoning_max_tokens or 4096))
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    # top_p 默认不传（部分模型如 kimi-k3 只接受特定值，统一不传最稳）；仅当用例或模型显式配置时才传
    if "top_p" in case:
        body["top_p"] = case["top_p"]
    elif provider.get("top_p") is not None:
        body["top_p"] = provider["top_p"]

    for key in ("stop", "response_format", "tools", "tool_choice", "seed", "n", "logprobs"):
        if key in case:
            body[key] = case[key]
        elif key in shared_params:
            body[key] = shared_params[key]
    return body


def resolve_api_key(provider, environ=os.environ):
    key = provider.get("api_key") or ""
    env_name = provider.get("api_key_env") or ""
    if env_name:
        key = environ.get(env_name, "") or key
    return key


# ---------------------------------------------------------------------------
# 请求执行与指标采集（含 reasoning_content / cached_tokens）
# ---------------------------------------------------------------------------

class ApiError(Exception):
    def __init__(self, message, kind="api_error", http_status=None, retryable=False):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status
        self.retryable = retryable


def extract_usage_fields(usage):
    """统一抽取 usage 相关字段，兼容不同渠道的字段位置差异。"""
    out = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
           "reasoning_tokens": None, "cached_tokens": None, "cache_write_tokens": None}
    if not usage:
        return out
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    out["total_tokens"] = usage.get("total_tokens")
    ctd = usage.get("completion_tokens_details") or {}
    out["reasoning_tokens"] = ctd.get("reasoning_tokens")
    if out["reasoning_tokens"] is None:
        out["reasoning_tokens"] = usage.get("reasoning_tokens")  # 部分渠道放 usage 顶层
    ptd = usage.get("prompt_tokens_details") or {}
    out["cached_tokens"] = ptd.get("cached_tokens")
    if out["cached_tokens"] is None:
        out["cached_tokens"] = usage.get("cached_tokens")
    out["cache_write_tokens"] = ptd.get("cache_write_tokens")
    if out["cache_write_tokens"] is None:
        out["cache_write_tokens"] = usage.get("cache_write_tokens")
    return out


def compute_metrics(timeline, reasoning_timeline, full_text, reasoning_text, usage, finish_reason,
                    e2e_ms, stream, status_code):
    u = extract_usage_fields(usage)
    m = {"e2e_ms": round(e2e_ms, 1), "ttft_ms": None, "content_ttft_ms": None, "reasoning_ttft_ms": None,
         "generation_ms": None, "tokens_per_sec": None, "chunk_count": 0, "reasoning_chunk_count": len(reasoning_timeline),
         "chunk_size": None, "chunk_interval_ms": None,
         "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"], "total_tokens": u["total_tokens"],
         "reasoning_tokens": u["reasoning_tokens"], "cached_tokens": u["cached_tokens"], "cache_write_tokens": u["cache_write_tokens"],
         "content_chars": len(full_text), "reasoning_chars": len(reasoning_text),
         "finish_reason": finish_reason, "http_status": status_code}

    all_events = sorted([(c["t_ms"], "content") for c in timeline] + [(c["t_ms"], "reasoning") for c in reasoning_timeline])
    if stream and all_events:
        first_t = all_events[0][0]
        m["ttft_ms"] = round(first_t, 1)
        m["generation_ms"] = round(all_events[-1][0] - first_t, 1)
        for t, kind in all_events:
            if kind == "content" and m["content_ttft_ms"] is None:
                m["content_ttft_ms"] = round(t, 1)
            if kind == "reasoning" and m["reasoning_ttft_ms"] is None:
                m["reasoning_ttft_ms"] = round(t, 1)
        if timeline:
            sizes = [c["len"] for c in timeline]
            ts = [c["t_ms"] for c in timeline]
            intervals = [round(b - a, 2) for a, b in zip(ts, ts[1:])]
            m["chunk_count"] = len(timeline)
            m["chunk_size"] = dist_stats(sizes)
            m["chunk_interval_ms"] = dist_stats(intervals)

    comp = m["completion_tokens"]
    if comp is None:
        comp = max(1, estimate_tokens(full_text + reasoning_text))
    if stream and m["generation_ms"]:
        m["tokens_per_sec"] = round(comp / (m["generation_ms"] / 1000.0), 2)
    elif not stream and e2e_ms:
        m["tokens_per_sec"] = round(comp / (e2e_ms / 1000.0), 2)
    return m


def run_stream(url, headers, body, timeout):
    started = time.monotonic()
    timeline, reasoning_timeline, texts, reasoning_texts = [], [], [], []
    tool_calls, usage, finish_reason, sse_issues, done_seen, status_code = [], None, None, [], False, None
    resp = None
    try:
        resp = requests.post(url, headers=headers, json=body, stream=True, timeout=timeout)
        status_code = resp.status_code
        if status_code != 200:
            detail = (resp.text or "")[:500]
            raise ApiError("HTTP %d: %s" % (status_code, detail), http_status=status_code, retryable=status_code >= 500)
        ctype = resp.headers.get("Content-Type", "")
        if "charset" not in ctype.lower():
            resp.encoding = "utf-8"  # 部分渠道 SSE 不带 charset，回退 UTF-8 避免中文损坏
        for raw in resp.iter_lines(decode_unicode=True):
            now_ms = (time.monotonic() - started) * 1000.0
            if not raw:
                continue
            if not raw.startswith("data:"):
                sse_issues.append({"kind": "non_data_line", "line": raw[:200]})
                continue
            payload = raw[5:].strip()
            if payload == "[DONE]":
                done_seen = True
                break
            try:
                evt = json.loads(payload)
            except Exception as e:
                sse_issues.append({"kind": "bad_json", "line": payload[:200], "err": str(e)[:200]})
                continue
            choices = evt.get("choices") or []
            if not choices:
                if evt.get("usage"):
                    usage = evt["usage"]
                continue
            ch = choices[0]
            delta = ch.get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if content:
                timeline.append({"t_ms": round(now_ms, 2), "len": len(content), "delta": content})
                texts.append(content)
            if reasoning:
                reasoning_timeline.append({"t_ms": round(now_ms, 2), "len": len(reasoning), "delta": reasoning})
                reasoning_texts.append(reasoning)
            if delta.get("tool_calls"):
                tool_calls.append(delta["tool_calls"])
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
            if evt.get("usage"):
                usage = evt["usage"]
    except requests.exceptions.RequestException as e:
        kind = "timeout" if isinstance(e, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout)) else "network"
        raise ApiError("%s: %s" % (kind, e), kind=kind, retryable=True)
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
    e2e_ms = (time.monotonic() - started) * 1000.0
    full_text = "".join(texts)
    reasoning_text = "".join(reasoning_texts)
    metrics = compute_metrics(timeline, reasoning_timeline, full_text, reasoning_text, usage, finish_reason, e2e_ms, True, status_code)
    return {"metrics": metrics, "output": {"text": full_text, "reasoning": reasoning_text, "tool_calls": tool_calls},
            "sse": {"done_seen": done_seen, "issues": sse_issues, "timeline": timeline},
            "usage_source": "stream_final" if usage else "estimated"}


def run_nonstream(url, headers, body, timeout):
    started = time.monotonic()
    status_code, usage, finish_reason = None, None, None
    resp = None
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        status_code = resp.status_code
        if status_code != 200:
            detail = (resp.text or "")[:500]
            raise ApiError("HTTP %d: %s" % (status_code, detail), http_status=status_code, retryable=status_code >= 500)
        try:
            data = resp.json()
        except Exception as e:
            raise ApiError("响应非合法 JSON: %s | 原文: %s" % (e, (resp.text or "")[:200]), kind="bad_json")
        choices = data.get("choices") or []
        msg = choices[0].get("message", {}) if choices else {}
        usage = data.get("usage")
        finish_reason = choices[0].get("finish_reason") if choices else None
        full_text = msg.get("content") or ""
        reasoning_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        tool_calls = msg.get("tool_calls") or []
    except requests.exceptions.RequestException as e:
        kind = "timeout" if isinstance(e, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout)) else "network"
        raise ApiError("%s: %s" % (kind, e), kind=kind, retryable=True)
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
    e2e_ms = (time.monotonic() - started) * 1000.0
    metrics = compute_metrics([], [], full_text, reasoning_text, usage, finish_reason, e2e_ms, False, status_code)
    return {"metrics": metrics, "output": {"text": full_text, "reasoning": reasoning_text, "tool_calls": tool_calls},
            "sse": None, "usage_source": "response" if usage else "estimated"}


# ---------------------------------------------------------------------------
# 质量判分（GSM8K 精确匹配 / TruthfulQA 命中）
# ---------------------------------------------------------------------------

_PUNCT = "，。！？、；：\"'（）《》【】,.!?;:()[]"


def normalize(s):
    s = (s or "").lower().strip()
    for ch in _PUNCT:
        s = s.replace(ch, " ")
    return re.sub(r"\s+", " ", s).strip()


def judge_answer(case, output):
    judge = case.get("judge")
    ref = case.get("reference")
    if judge == "gsm8k_exact":
        nums = re.findall(r"[-+]?\d[\d,]*", output or "")
        if not nums:
            return False, "输出中无数字，无法判分"
        last = nums[-1].replace(",", "")
        try:
            ok = int(last) == int(str(ref).replace(",", ""))
        except ValueError:
            return False, "数字解析失败：输出=%s 参考=%s" % (last, ref)
        return ok, "末位数字=%s，参考=%s" % (last, ref)
    if judge == "truthfulqa_any":
        refs = ref if isinstance(ref, list) else [ref]
        norm_out = normalize(output or "")
        hits = [r for r in refs if r and normalize(r) in norm_out]
        return bool(hits), ("命中标准答案: %s" % hits[:2]) if hits else "未命中任何标准答案"
    if judge == "truthfulqa_contains":
        return normalize(str(ref)) in normalize(output or ""), "参考=%s" % ref
    return True, "无判分规则（judge=%s）" % judge


# ---------------------------------------------------------------------------
# 兼容性 / 质量 / 长上下文校验（记录每个失败项的具体表现）
# ---------------------------------------------------------------------------

def validate_case(case, result):
    feature = case.get("compat_feature")
    checks = []
    out = (result.get("output") or {}).get("text") or ""
    metrics = result.get("metrics") or {}

    if result.get("error"):
        checks.append({"name": "request_error", "severity": "fail",
                       "detail": "请求失败: %s | %s" % (result["error"].get("type"), result["error"].get("message", "")[:200])})
        return {"passed": False, "checks": checks}

    if case.get("judge"):
        ok, detail = judge_answer(case, out)
        checks.append({"name": "judge_%s" % case.get("judge"), "severity": "pass" if ok else "fail", "detail": detail})
        return {"passed": ok, "checks": checks}

    if feature == "stream_sse":
        issues = (result.get("sse") or {}).get("issues") or []
        done_seen = bool((result.get("sse") or {}).get("done_seen"))
        checks.append({"name": "SSE_chunk_parse", "severity": "fail" if issues else "pass",
                       "detail": ("非法行/坏JSON: " + json.dumps(issues[:3], ensure_ascii=False)) if issues else "所有行均为合法 data: JSON"})
        checks.append({"name": "DONE_marker", "severity": "warn" if not done_seen else "pass",
                       "detail": "已收到 data: [DONE]" if done_seen else "未收到 data: [DONE]（部分代理渠道不发送）"})
        checks.append({"name": "delta_accumulate", "severity": "pass", "detail": "流式内容拼接长度 %d 字符" % len(out)})
    elif feature == "stop_param":
        contains = "STOP" in out
        checks.append({"name": "stop_token_not_in_output", "severity": "fail" if contains else "pass",
                       "detail": ("输出中出现 STOP：%s" % out[:100]) if contains else "输出中不含 STOP 停止符"})
        checks.append({"name": "generation_halted", "severity": "pass",
                       "detail": "输出长度 %d，finish_reason=%s" % (len(out), metrics.get("finish_reason"))})
    elif feature == "max_tokens":
        cap = case.get("max_tokens")
        comp = metrics.get("completion_tokens")
        ok = comp is not None and comp <= cap + 10
        checks.append({"name": "completion_within_cap", "severity": "pass" if ok else "fail",
                       "detail": "completion_tokens=%s，max_tokens=%d，finish_reason=%s" % (comp, cap, metrics.get("finish_reason"))})
    elif feature == "temperature_accept":
        checks.append({"name": "temperature_accepted", "severity": "pass",
                       "detail": "渠道接受 temperature=0 参数"})
    elif feature == "temperature_zero":
        outs = [o.get("output", {}).get("text", "") for o in result.get("run_details", [])]
        if len(outs) < 2:
            checks.append({"name": "deterministic", "severity": "warn",
                           "detail": "单次模式：仅采样 1 次，未做一致性双跑"})
        else:
            identical = len(set(outs)) <= 1
            checks.append({"name": "deterministic", "severity": "pass" if identical else "warn",
                           "detail": "两次输出完全一致" if identical else "两次输出不一致（长度 %s），部分渠道在 temperature=0 下仍非确定性" % [len(o) for o in outs]})
    elif feature == "function_calling":
        tcs = (result.get("output") or {}).get("tool_calls") or []
        names = [tc.get("function", {}).get("name") for tc in tcs if tc.get("function")]
        ok = len(names) > 0
        checks.append({"name": "tool_calls_present", "severity": "pass" if ok else "fail",
                       "detail": ("未返回 tool_calls，模型直接回复：" + out[:120]) if not ok else "返回 tool_calls: %s" % names})
        if ok:
            for tc in tcs:
                fn = tc.get("function", {})
                args = fn.get("arguments", "") or ""
                try:
                    json.loads(args)
                    checks.append({"name": "arguments_json_%s" % fn.get("name"), "severity": "pass", "detail": "arguments 为合法 JSON"})
                except Exception as e:
                    checks.append({"name": "arguments_json_%s" % fn.get("name"), "severity": "fail", "detail": "arguments 非 JSON: %s" % args[:100]})
    elif feature == "json_mode":
        try:
            obj = json.loads(out)
            missing = [k for k in case.get("expect_keys", []) if k not in obj]
            checks.append({"name": "valid_json", "severity": "pass", "detail": "输出为合法 JSON"})
            checks.append({"name": "keys_present", "severity": "pass" if not missing else "fail",
                           "detail": ("缺少字段: %s" % missing) if missing else "字段齐全: %s" % case.get("expect_keys")})
        except Exception as e:
            checks.append({"name": "valid_json", "severity": "fail", "detail": "输出非 JSON（%s）：%s" % (e, out[:200])})
    elif feature == "multi_turn":
        kw = case.get("expect_contains", "")
        checks.append({"name": "answer_mentions", "severity": "pass" if kw and kw in out else "fail",
                       "detail": ("回答包含“%s”" % kw) if kw and kw in out else ("未包含“%s”，实际：%s" % (kw, out[:120]))})
    elif feature == "system_prompt":
        prefix = case.get("expect_prefix", "")
        checks.append({"name": "prefix", "severity": "pass" if prefix and out.startswith(prefix) else "fail",
                       "detail": ("以“%s”开头" % prefix) if prefix and out.startswith(prefix) else "未以“%s”开头，实际：%s" % (prefix, out[:60])})

    if case.get("needle"):
        kw = case.get("expect_contains", "")
        found = kw and kw in out
        checks.append({"name": "needle_retrieval", "severity": "pass" if found else "fail",
                       "detail": ("检索到“%s”" % kw) if found else ("未检索到“%s”，实际：%s" % (kw, out[:120]))})

    failed = [c for c in checks if c.get("severity") == "fail"]
    return {"passed": not failed, "checks": checks}


# ---------------------------------------------------------------------------
# 单条用例执行（重试 / temperature=0 双跑 / 跳过逻辑）
# ---------------------------------------------------------------------------

def run_case(provider, case, cfg, order_index, once=False):
    nominal = int(provider.get("nominal_context_tokens") or 0)
    need = int(case.get("target_tokens") or 0)
    if nominal and need and need > nominal:
        return {"skipped": True, "skip_reason": "标称上下文 %d < 用例需求 %d，跳过" % (nominal, need)}

    body = build_request_body(provider, case, cfg.get("shared_request_params", {}), cfg.get("reasoning_max_tokens", 4096))
    url = provider["base_url"].rstrip("/") + "/chat/completions"
    key = resolve_api_key(provider)
    if not key:
        print("[警告] 渠道 %s 未配置 api_key，请求可能 401（请在 config 该渠道的 api_key 字段填 key，或设置对应环境变量）" % provider["name"])
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    timeouts = cfg.get("timeouts", {})
    read_timeout = timeouts.get("read_seconds_long_context", 600) if need else timeouts.get("read_seconds", 180)
    timeout = (timeouts.get("connect_seconds", 10), read_timeout)

    # 单次模式：每条用例只跑 1 次（跳过 temperature=0 一致性双跑）
    n_runs = 1 if once else int(case.get("runs", 1))
    result, attempts, retries = None, [], int(cfg.get("retries", 0))
    for attempt in range(1 + retries):
        try:
            runs = [run_stream(url, headers, body, timeout) if case.get("stream") else run_nonstream(url, headers, body, timeout)
                    for _ in range(n_runs)]
            result = runs[0]
            if n_runs > 1:
                result["run_details"] = [{"metrics": r["metrics"], "output": r["output"]} for r in runs]
            result["attempt"] = attempt + 1
            break
        except ApiError as e:
            attempts.append({"attempt": attempt + 1, "kind": e.kind, "http_status": e.http_status, "message": str(e)[:300]})
            if e.retryable and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            result = {"error": {"type": e.kind, "http_status": e.http_status, "message": str(e)[:500], "attempts": attempts}}
            break

    result["validation"] = validate_case(case, result)
    result["order_index"] = order_index
    return result


# ---------------------------------------------------------------------------
# 聚合与评分（与 Excel 模板公式保持一致）
# ---------------------------------------------------------------------------

def agg_mean(values):
    return round(statistics.mean(values), 2) if values else None


def aggregate(rows, providers):
    out = {}
    for p in providers:
        out[p["id"]] = {
            "provider": p,
            "performance": {"cases": 0, "ok": 0, "errors": {}, "ttft_ms": [], "e2e_ms": [], "speed": [],
                            "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens_sum": 0},
            "compatibility": {"total": 0, "passed": 0, "features": {}},
            "quality": {"total": 0, "passed": 0, "by_category": {}},
            "long_context": {"total": 0, "run": 0, "passed": 0, "skipped": 0, "by_case": {}},
            "errors": [],
        }
    for r in rows:
        pid = r["meta"]["provider_id"]
        agg = out[pid]
        layer = r["meta"]["layer"]
        cid = r["meta"]["case_id"]
        if r.get("skipped"):
            if layer == "long_context":
                agg["long_context"]["total"] += 1
                agg["long_context"]["skipped"] += 1
                agg["long_context"]["by_case"][cid] = "跳过"
            continue
        if r.get("error"):
            agg["errors"].append({"case_id": cid, "type": r["error"]["type"], "message": r["error"]["message"]})
            if layer == "performance":
                agg["performance"]["cases"] += 1
                agg["performance"]["errors"][r["error"]["type"]] = agg["performance"]["errors"].get(r["error"]["type"], 0) + 1
            elif layer == "compatibility":
                agg["compatibility"]["total"] += 1
                f = agg["compatibility"]["features"].setdefault(r["meta"].get("compat_feature") or "other", {"total": 0, "passed": 0})
                f["total"] += 1
            elif layer == "quality":
                agg["quality"]["total"] += 1
            elif layer == "long_context":
                agg["long_context"]["total"] += 1
                agg["long_context"]["run"] += 1
                agg["long_context"]["by_case"][cid] = "错误"
            continue
        m = r.get("metrics") or {}
        if layer == "performance":
            agg["performance"]["cases"] += 1
            agg["performance"]["ok"] += 1
            if m.get("ttft_ms") is not None:
                agg["performance"]["ttft_ms"].append(m["ttft_ms"])
            if m.get("e2e_ms") is not None:
                agg["performance"]["e2e_ms"].append(m["e2e_ms"])
            if m.get("tokens_per_sec") is not None:
                agg["performance"]["speed"].append(m["tokens_per_sec"])
            agg["performance"]["completion_tokens"] += m.get("completion_tokens") or 0
            agg["performance"]["reasoning_tokens"] += m.get("reasoning_tokens") or 0
            agg["performance"]["cached_tokens_sum"] += m.get("cached_tokens") or 0
        elif layer == "compatibility":
            agg["compatibility"]["total"] += 1
            passed = bool((r.get("validation") or {}).get("passed"))
            if passed:
                agg["compatibility"]["passed"] += 1
            f = agg["compatibility"]["features"].setdefault(r["meta"].get("compat_feature") or "other", {"total": 0, "passed": 0})
            f["total"] += 1
            f["passed"] += int(passed)
        elif layer == "quality":
            agg["quality"]["total"] += 1
            passed = bool((r.get("validation") or {}).get("passed"))
            if passed:
                agg["quality"]["passed"] += 1
            cat = r["meta"].get("category") or "other"
            c = agg["quality"]["by_category"].setdefault(cat, {"total": 0, "passed": 0})
            c["total"] += 1
            c["passed"] += int(passed)
        elif layer == "long_context":
            agg["long_context"]["total"] += 1
            agg["long_context"]["run"] += 1
            passed = bool((r.get("validation") or {}).get("passed"))
            agg["long_context"]["passed"] += int(passed)
            agg["long_context"]["by_case"][cid] = "是" if passed else "否"
    return out


def compute_scores(agg_by_provider, weights):
    w = {"performance": 0.3, "compatibility": 0.3, "quality": 0.2, "long_context": 0.2}
    w.update(weights or {})
    pids = list(agg_by_provider.keys())
    ttft_avg = {pid: agg_mean(agg_by_provider[pid]["performance"]["ttft_ms"]) for pid in pids}
    speed_avg = {pid: agg_mean(agg_by_provider[pid]["performance"]["speed"]) for pid in pids}
    e2e_avg = {pid: agg_mean(agg_by_provider[pid]["performance"]["e2e_ms"]) for pid in pids}
    min_ttft = min([v for v in ttft_avg.values() if v], default=None)
    max_speed = max([v for v in speed_avg.values() if v], default=None)
    min_e2e = min([v for v in e2e_avg.values() if v], default=None)

    scores = {}
    for pid in pids:
        perf, comp, q, lc = (agg_by_provider[pid]["performance"], agg_by_provider[pid]["compatibility"],
                             agg_by_provider[pid]["quality"], agg_by_provider[pid]["long_context"])
        sub = {}
        if min_ttft and ttft_avg[pid]:
            sub["ttft"] = round(100 * min_ttft / ttft_avg[pid], 1)
        if max_speed and speed_avg[pid]:
            sub["speed"] = round(100 * speed_avg[pid] / max_speed, 1)
        if min_e2e and e2e_avg[pid]:
            sub["e2e"] = round(100 * min_e2e / e2e_avg[pid], 1)
        perf_score = round(statistics.mean(sub.values()), 1) if sub else None
        compat_score = round(100.0 * comp["passed"] / comp["total"], 1) if comp["total"] else None
        quality_score = round(100.0 * q["passed"] / q["total"], 1) if q["total"] else None
        lc_score = round(100.0 * lc["passed"] / lc["run"], 1) if lc["run"] else None
        parts = []
        for score, weight in ((perf_score, w["performance"]), (compat_score, w["compatibility"]),
                              (quality_score, w["quality"]), (lc_score, w["long_context"])):
            if score is not None:
                parts.append(score * weight)
        scores[pid] = {"sub": sub, "performance": perf_score, "compatibility": compat_score,
                       "quality": quality_score, "long_context": lc_score, "total": round(sum(parts), 1) if parts else None, "weights": w}
    return scores


# ---------------------------------------------------------------------------
# Markdown 汇总表
# ---------------------------------------------------------------------------

def build_test_config(cfg, cases):
    """汇总本次实验的参数配置（共享参数/推理/超时/重试/权重/渠道/模型/用例设计），写入报告。"""
    layers = {}
    for c in cases:
        layers[c["layer"]] = layers.get(c["layer"], 0) + 1
    return {
        "shared_request_params": cfg.get("shared_request_params", {}),
        "top_p_note": "默认不传（仅用例或模型显式配置时才传）",
        "reasoning_max_tokens": cfg.get("reasoning_max_tokens", 4096),
        "timeouts": cfg.get("timeouts", {}),
        "retries": cfg.get("retries", 0),
        "score_weights": cfg.get("score_weights", {}),
        "channels": [{"id": ch["id"], "name": ch.get("name", ch["id"]), "base_url": ch.get("base_url", "")}
                     for ch in cfg.get("channels", [])],
        "models": [{"id": m["id"], "aliases": m.get("aliases", {}), "reasoning": bool(m.get("reasoning")),
                    "context": m.get("context"), "supports": m.get("supports", {})}
                   for m in cfg.get("models", [])],
        "layers": {k: {"count": v, "desc": LAYER_DESC.get(k, "")} for k, v in layers.items()},
    }


def render_markdown(agg_by_provider, scores, providers, env=None, test_config=None):
    lines = ["# 多渠道模型测试汇总对比\n"]
    w = scores[providers[0]["id"]]["weights"]
    lines.append("> 总评分 = 性能×%.1f + 兼容×%.1f + 质量×%.1f + 长上下文×%.1f（各项满分 100）。\n"
                 % (w["performance"], w["compatibility"], w["quality"], w["long_context"]))

    if env:
        lines.append("\n## 0. 测试环境\n")
        lines.append("| 项目 | 值 |")
        lines.append("|---|---|")
        for k, v in env.items():
            lines.append("| %s | %s |" % (k, v))
        lines.append("")

    if test_config:
        tc = test_config
        sp = tc.get("shared_request_params") or {}
        to = tc.get("timeouts") or {}
        w = tc.get("score_weights") or {}
        lines.append("\n## 1. 测试参数配置\n")
        lines.append("### 1.1 共享请求参数\n")
        lines.append("| 参数 | 值 |")
        lines.append("|---|---|")
        lines.append("| temperature | %s |" % (sp.get("temperature", "不传")))
        lines.append("| top_p | %s |" % (sp.get("top_p", tc.get("top_p_note", "不传"))))
        lines.append("")
        lines.append("### 1.2 推理 / 超时 / 重试\n")
        lines.append("| 参数 | 值 |")
        lines.append("|---|---|")
        lines.append("| reasoning_max_tokens（推理模型下限） | %s |" % tc.get("reasoning_max_tokens"))
        lines.append("| 连接超时 / 读超时 / 长上下文读超时 | %ss / %ss / %ss |" % (
            to.get("connect_seconds", 10), to.get("read_seconds", 180), to.get("read_seconds_long_context", 600)))
        lines.append("| 失败重试次数 | %s |" % tc.get("retries"))
        lines.append("")
        lines.append("### 1.3 评分权重\n")
        lines.append("| 维度 | 权重 |")
        lines.append("|---|---|")
        for k, v in w.items():
            lines.append("| %s | %s |" % (k, v))
        lines.append("")
        lines.append("### 1.4 渠道\n")
        lines.append("| id | 名称 | base_url |")
        lines.append("|---|---|---|")
        for ch in tc.get("channels", []):
            lines.append("| %s | %s | %s |" % (ch.get("id"), ch.get("name"), ch.get("base_url")))
        lines.append("")
        lines.append("### 1.5 模型\n")
        lines.append("| id | 各渠道调用名(aliases) | 推理模型 | 标称上下文 | supports |")
        lines.append("|---|---|---|---|---|")
        for m in tc.get("models", []):
            lines.append("| %s | %s | %s | %s | %s |" % (
                m.get("id"), json.dumps(m.get("aliases", {}), ensure_ascii=False),
                "是" if m.get("reasoning") else "否", m.get("context"),
                json.dumps(m.get("supports", {}), ensure_ascii=False)))
        lines.append("")
        lines.append("## 2. 测试用例设计\n")
        lines.append("| 层 | 用例数 | 目的 |")
        lines.append("|---|---|---|")
        for layer, info in test_config.get("layers", {}).items():
            lines.append("| %s | %d | %s |" % (layer, info.get("count", 0), info.get("desc", "")))
        lines.append("")

    lines.append("\n## 3. 性能基准层\n")
    lines.append("| 模型@渠道 | 用例 | 成功 | 错误 | TTFT均值(ms) | TTFT P50 | TTFT P95 | 速度(tok/s) | E2E均值(ms) | 推理tokens | 推理占比 | 缓存tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in providers:
        a = agg_by_provider[p["id"]]["performance"]
        ratio = (a["reasoning_tokens"] / a["completion_tokens"] * 100) if a["completion_tokens"] else None
        ratio_str = ("%.0f%%" % ratio) if ratio is not None else "-"
        lines.append("| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            p["name"], a["cases"], a["ok"], sum(a["errors"].values()),
            agg_mean(a["ttft_ms"]), percentile(a["ttft_ms"], 0.5), percentile(a["ttft_ms"], 0.95),
            agg_mean(a["speed"]), agg_mean(a["e2e_ms"]), a["reasoning_tokens"], ratio_str, a["cached_tokens_sum"]))

    lines.append("\n## 4. 功能兼容层（通过/总数）\n")
    feats = []
    for p in providers:
        for feat in agg_by_provider[p["id"]]["compatibility"]["features"]:
            if feat not in feats:
                feats.append(feat)
    lines.append("| 模型@渠道 | 通过率 | " + " | ".join(feats) + " |")
    lines.append("|" + "---|" * (len(feats) + 2))
    for p in providers:
        c = agg_by_provider[p["id"]]["compatibility"]
        rate = ("%.1f%%" % (100.0 * c["passed"] / c["total"])) if c["total"] else "-"
        cells = []
        for feat in feats:
            f = c["features"].get(feat, {"total": 0, "passed": 0})
            cells.append("%d/%d" % (f["passed"], f["total"]) if f["total"] else "-")
        lines.append("| %s | %s | %s |" % (p["name"], rate, " | ".join(cells)))

    lines.append("\n## 5. 质量层（HuggingFace 数据，判分）\n")
    cats = []
    for p in providers:
        for cat in agg_by_provider[p["id"]]["quality"]["by_category"]:
            if cat not in cats:
                cats.append(cat)
    lines.append("| 模型@渠道 | 正确率 | " + " | ".join(cats) + " |")
    lines.append("|" + "---|" * (len(cats) + 2))
    for p in providers:
        q = agg_by_provider[p["id"]]["quality"]
        rate = ("%.1f%%" % (100.0 * q["passed"] / q["total"])) if q["total"] else "-"
        cells = []
        for cat in cats:
            c = q["by_category"].get(cat, {"total": 0, "passed": 0})
            cells.append("%d/%d" % (c["passed"], c["total"]) if c["total"] else "-")
        lines.append("| %s | %s | %s |" % (p["name"], rate, " | ".join(cells)))

    lines.append("\n## 6. 长上下文专项（大海捞针）\n")
    lc_cases = []
    for p in providers:
        for cid in agg_by_provider[p["id"]]["long_context"]["by_case"]:
            if cid not in lc_cases:
                lc_cases.append(cid)
    lines.append("| 模型@渠道 | 检索率 | " + " | ".join(lc_cases) + " |")
    lines.append("|" + "---|" * (len(lc_cases) + 2))
    for p in providers:
        l = agg_by_provider[p["id"]]["long_context"]
        rate = ("%.1f%%" % (100.0 * l["passed"] / l["run"])) if l["run"] else "-"
        cells = [l["by_case"].get(cid, "-") for cid in lc_cases]
        lines.append("| %s | %s | %s |" % (p["name"], rate, " | ".join(cells)))

    lines.append("\n## 7. 综合评分（满分 100）\n")
    lines.append("| 模型@渠道 | 性能 | 兼容 | 质量 | 长上下文 | 总评分 |")
    lines.append("|---|---|---|---|---|---|")
    for p in providers:
        s = scores[p["id"]]
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            p["name"],
            s["performance"] if s["performance"] is not None else "-",
            s["compatibility"] if s["compatibility"] is not None else "-",
            s["quality"] if s["quality"] is not None else "-",
            s["long_context"] if s["long_context"] is not None else "-",
            s["total"] if s["total"] is not None else "-"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_combos(cfg):
    """把 channels × models 展开成「模型@渠道」组合（每个组合等价于旧版一个 provider）。"""
    channels = cfg.get("channels", [])
    models = cfg.get("models", [])
    combos = []
    for m in models:
        aliases = m.get("aliases") or {}
        for ch in channels:
            alias = aliases.get(ch["id"])
            if alias is None:
                continue  # 该模型不挂在该渠道上
            model_supports = m.get("supports") or {}
            channel_supports = ch.get("supports") or {}
            combos.append({
                "id": "%s__%s" % (m["id"], ch["id"]),
                "model_id": m["id"],
                "model_name": m.get("name", m["id"]),
                "channel_id": ch["id"],
                "channel_name": ch.get("name", ch["id"]),
                "name": "%s @ %s" % (m.get("name", m["id"]), ch.get("name", ch["id"])),
                "model": alias,
                "base_url": ch["base_url"],
                "api_key_env": ch.get("api_key_env", ""),
                "api_key": ch.get("api_key", ""),
                "reasoning": bool(m.get("reasoning", False)),
                "nominal_context_tokens": int(m.get("context", 0) or 0),
                "supports": {**model_supports, **channel_supports},
                "notes": ch.get("notes", ""),
            })
    return combos


def load_sample_case(cases_dir):
    """示例模式复用性能层第 1 条「短输入短输出」用例（perf_sis_01）。"""
    path = os.path.join(cases_dir, "performance.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            perf = json.load(f)
        for c in perf:
            if c.get("category") == "短输入短输出":
                c = dict(c)
                c["layer"] = "sample"  # 标记为示例层，避免与性能层混淆
                return c
    # 兜底：找不到性能层用例时用固定 smoke 提示词
    return {
        "id": "sample_smoke", "name": "示例-连通性", "category": "sample", "layer": "sample",
        "prompt": "请用一句话介绍你自己（不超过 50 字）。", "max_tokens": 200, "stream": True,
    }


def load_config(args):
    if not os.path.exists(args.providers_file):
        sys.exit("配置文件不存在：%s\n请先复制 config/providers-template.json 为 config/providers.json，并填写 api key / 模型调用名。" % args.providers_file)
    with open(args.providers_file, encoding="utf-8") as f:
        cfg = json.load(f)
    # 兼容旧格式（providers 直接是组合）与新版（channels + models）
    if "channels" in cfg or "models" in cfg:
        providers = build_combos(cfg)
    else:
        providers = list(cfg.get("providers", []))
    cases = []
    for layer in args.layers.split(","):
        layer = layer.strip()
        path = os.path.join(args.cases_dir, "%s.json" % layer)
        if not os.path.exists(path):
            print("警告：找不到用例文件 %s，跳过该层" % path)
            continue
        with open(path, encoding="utf-8") as f:
            for c in json.load(f):
                c["layer"] = layer
                cases.append(c)
    return cfg, cases, providers


def render_sample_table(rows):
    print("\n" + "=" * 100)
    print("  示例结果：每个「模型 × 渠道」组合跑 1 条示例")
    print("=" * 100)
    print("%-18s %-10s %-10s %-10s %-12s %-8s %-8s %s" % (
        "模型", "渠道", "TTFT(ms)", "E2E(ms)", "速度(tok/s)", "内容字数", "推理字数", "状态"))
    print("-" * 100)
    for r in rows:
        m = r.get("metrics") or {}
        mid = r["meta"].get("model_id", "")
        chn = r["meta"].get("channel_name", "")
        if r.get("error"):
            print("%-18s %-10s %-10s %-10s %-12s %-8s %-8s ❌ %s" % (
                mid, chn, "-", "-", "-", "-", "-", r["error"].get("type", "error")))
        elif r.get("skipped"):
            print("%-18s %-10s %-10s %-10s %-12s %-8s %-8s ⏭ 跳过" % (mid, chn, "-", "-", "-", "-", "-"))
        else:
            print("%-18s %-10s %-10s %-10s %-12s %-8s %-8s ✅" % (
                mid, chn, m.get("ttft_ms") or "-", m.get("e2e_ms") or "-",
                m.get("tokens_per_sec") or "-", m.get("content_chars") or 0, m.get("reasoning_chars") or 0))
    print("\n--- 各组合输出预览 ---")
    for r in rows:
        mid = r["meta"].get("model_id", "")
        chn = r["meta"].get("channel_name", "")
        out = (r.get("output") or {}).get("text") or ""
        if r.get("error"):
            print("  [%s @ %s] 错误：%s" % (mid, chn, r["error"].get("message", "")[:120]))
        elif r.get("skipped"):
            print("  [%s @ %s] 跳过：%s" % (mid, chn, r.get("skip_reason", "")))
        else:
            print("  [%s @ %s] %s" % (mid, chn, (out or "(空)")[:80].replace("\n", " ")))


def run_sample(providers, cfg, out_dir, sample_case, env_tag=""):
    print("[示例模式] 复用性能层第 1 条「短输入短输出」用例（%s），每个「模型×渠道」组合只跑 1 次，共 %d 次请求" % (
        sample_case["id"], len(providers)))
    env = env_info(env_tag)
    rows = []
    for p in providers:
        log("sample -> %s @ %s" % (p["model_id"], p["channel_name"]))
        result = run_case(p, sample_case, cfg, 0, once=True)
        result.setdefault("meta", {})
        result["meta"].update({
            "layer": "sample", "case_id": sample_case["id"], "case_name": sample_case.get("name", ""),
            "category": sample_case.get("category", ""), "provider_id": p["id"], "provider_name": p["name"],
            "model_id": p["model_id"], "model_name": p["model_name"],
            "channel_id": p["channel_id"], "channel_name": p["channel_name"],
            "model": p["model"], "base_url": p["base_url"],
        })
        result["environment"] = env
        layer_dir = os.path.join(out_dir, "raw", "sample")
        os.makedirs(layer_dir, exist_ok=True)
        with open(os.path.join(layer_dir, "%s__%s.json" % (p["id"], sample_case["id"])), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        rows.append(result)
    render_sample_table(rows)
    summary = {
        "mode": "sample", "sample_case": sample_case,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": env,
        "rows": [{
            "model_id": r["meta"]["model_id"], "channel_id": r["meta"]["channel_id"],
            "channel_name": r["meta"]["channel_name"], "base_url": r["meta"]["base_url"],
            "model": r["meta"]["model"], "error": r.get("error"), "skipped": r.get("skipped"),
            "metrics": r.get("metrics"), "output_preview": ((r.get("output") or {}).get("text") or "")[:200],
        } for r in rows],
    }
    with open(os.path.join(out_dir, "sample_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n示例结果已写入：%s" % os.path.join(os.path.abspath(out_dir), "sample_summary.json"))

    # 自动生成示例 Excel 报告
    try:
        from export_report import generate_sample_report
        rp = generate_sample_report(out_dir)
        if rp:
            print("示例报告已生成：%s" % os.path.abspath(rp))
    except Exception as e:
        print("[提示] 自动生成示例 Excel 失败：%s" % e)


# ---------------------------------------------------------------------------
# 缓存命中专项测试
# ---------------------------------------------------------------------------

CACHE_PREFIX_TOKENS = 2000
CACHE_RUNS = 5  # 连打次数：run1 建立缓存(cache_write)，run2~5 测命中


def _cached(v):
    """cached_tokens null→0 归一化：None 按未命中(0)处理。"""
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def classify_cache(same_prompt, prefix_cached, baseline_cached, repeat_hit, prefix_hit):
    """缓存类型自动归类（字段可信度前置）。"""
    pr = _cached(prefix_cached)
    bl = _cached(baseline_cached)
    # 1. 字段可信度：基线>0 说明「完全不同前缀」也报命中，cached 字段语义异常，不可信
    if bl > 0:
        return "字段不可信"
    # 2. 前缀复用命中（cached>0 且 >基线）且重复也命中 → 真前缀缓存
    if prefix_hit and repeat_hit > 0:
        return "前缀缓存"
    # 3. 重复命中、但前缀复用未命中 → 精确匹配缓存（换问题整体 miss）
    if repeat_hit > 0 and not prefix_hit:
        return "精确匹配缓存(无前缀复用)"
    # 4. 连打完全没命中、换问题反而命中 → 反常，需复测/疑似路由漂移
    if prefix_hit and repeat_hit == 0:
        return "反常(连打miss但前缀命中)"
    return "无缓存"


def render_cache_table(rows):
    print("\n" + "=" * 140)
    print("  缓存命中测试结果（前缀 %d token，连打 %d 次：run1 建缓存，run2~%d 测命中）" % (
        CACHE_PREFIX_TOKENS, CACHE_RUNS, CACHE_RUNS))
    print("=" * 140)
    print("%-14s %-9s %-9s %-9s %-14s %-11s %-11s %-9s %-9s %s" % (
        "模型", "渠道", "首写tokens", "重复命中", "cached(2~N)", "前缀cached", "基线cached", "前缀复用", "session", "缓存类型"))
    print("-" * 140)
    for r in rows:
        sp = r.get("same_prompt", [])
        write_tk = r.get("cache_write_tokens")
        write_disp = "-" if write_tk is None else str(write_tk)
        hit_vals = [_cached(s.get("cached_tokens")) for s in sp[1:]]
        uniq = sorted(set(hit_vals))
        cached_disp = str(uniq[0]) if len(uniq) == 1 else " / ".join(str(v) for v in hit_vals)
        pr = _cached(r.get("prefix_reuse", {}).get("cached_tokens"))
        bl = _cached(r.get("baseline", {}).get("cached_tokens"))
        repeat_hits = sum(1 for c in hit_vals if c > 0)
        repeat_disp = "%d/%d" % (repeat_hits, CACHE_RUNS - 1)
        sess = r.get("session_id_supported")
        sess_disp = "是" if sess is True else ("否" if sess is False else "未验证")
        print("%-14s %-9s %-9s %-9s %-14s %-11s %-11s %-9s %-9s %s" % (
            r.get("model_id"), r.get("channel_name"), write_disp, repeat_disp, cached_disp,
            pr, bl, "✅" if r.get("prefix_hit") else "❌", sess_disp, r.get("cache_type", "-")))
    print("\n说明：首写=run1 的 cache_write_tokens(建立缓存)；重复命中=run2~%d 中 cached>0 的次数/%d；"
          "cached 为 null 按 0(未命中)归一；前缀复用=前缀cached>0 且 >基线(增量)；基线>0 → 字段不可信；"
          "session=该渠道是否接受 session_id(否=自动去 session 重试)。\n" % (CACHE_RUNS, CACHE_RUNS - 1))


def run_cache(providers, cfg, out_dir, env_tag=""):
    print("[缓存测试] 每个「模型×渠道」组合：长前缀连打 %d 次(run1 建缓存, run2~%d 测命中) + 前缀复用 + 基线，共 %d 个组合" % (
        CACHE_RUNS, CACHE_RUNS, len(providers)))
    env = env_info(env_tag)
    prefix_a = build_filler(int(CACHE_PREFIX_TOKENS * 1.2), stable_seed("cache_prefix_a"))
    prefix_b = build_filler(int(CACHE_PREFIX_TOKENS * 1.2), stable_seed("cache_prefix_b"))
    q1 = "请用一句话总结以上内容的核心观点。"
    q2 = "以上内容主要讨论了哪些技术？请列举三项。"
    to = cfg.get("timeouts", {})
    timeout = (to.get("connect_seconds", 10), to.get("read_seconds", 180))

    rows = []
    for p in providers:
        entry = {"model_id": p["model_id"], "channel_id": p["channel_id"], "channel_name": p["channel_name"],
                 "model": p["model"], "base_url": p["base_url"]}
        # session 设计：按「渠道×模型」分；连打+前缀复用共用一个 session（共享长前缀，粘住同一节点），基线独立 session
        base_session = "%s_%s_cache" % (p["channel_id"], p["model_id"])
        session_shared = base_session + "_shared"
        session_baseline = base_session + "_baseline"
        session_supported = None  # None=未探测 / True=支持 / False=不支持(自动 fallback)

        def call(p, prefix, question, session_id=None):
            nonlocal session_supported
            case = {"id": "cache_tmp", "prompt": prefix + "\n\n" + question, "stream": True, "max_tokens": 512}
            body = build_request_body(p, case, cfg.get("shared_request_params", {}), cfg.get("reasoning_max_tokens", 16384))
            url = p["base_url"].rstrip("/") + "/chat/completions"
            headers = {"Authorization": "Bearer " + resolve_api_key(p), "Content-Type": "application/json"}
            if session_id and session_supported is not False:
                body["session_id"] = session_id
            try:
                r = run_stream(url, headers, body, timeout)
                if session_id and session_supported is None:
                    session_supported = True  # 带 session 请求成功 → 支持
                return r
            except ApiError as e:
                if session_id and session_supported is None and e.http_status in (400, 422):
                    session_supported = False  # session_id 不被支持，fallback 重试
                    body.pop("session_id", None)
                    try:
                        return run_stream(url, headers, body, timeout)
                    except ApiError as e2:
                        return {"error": {"type": e2.kind, "message": str(e2)[:200]}}
                return {"error": {"type": e.kind, "message": str(e)[:200]}}

        # 用例1：相同 prompt 连打（run1 建立缓存，run2~N 测命中）
        same = []
        for i in range(CACHE_RUNS):
            r = call(p, prefix_a, q1, session_shared)
            m = (r.get("metrics") or {}) if not r.get("error") else {}
            same.append({"run": i + 1, "ttft_ms": m.get("ttft_ms"), "e2e_ms": m.get("e2e_ms"),
                         "cached_tokens": m.get("cached_tokens"), "cache_write_tokens": m.get("cache_write_tokens"),
                         "prompt_tokens": m.get("prompt_tokens"), "error": r.get("error")})
            time.sleep(1)
        entry["same_prompt"] = same
        # 用例2：前缀复用（同 session、同前缀、换问题）
        r2 = call(p, prefix_a, q2, session_shared)
        m2 = (r2.get("metrics") or {}) if not r2.get("error") else {}
        entry["prefix_reuse"] = {"cached_tokens": m2.get("cached_tokens"), "cache_write_tokens": m2.get("cache_write_tokens"),
                                 "prompt_tokens": m2.get("prompt_tokens"), "ttft_ms": m2.get("ttft_ms"), "error": r2.get("error")}
        # 用例3：基线（独立 session、不同前缀，应不命中）
        r3 = call(p, prefix_b, q1, session_baseline)
        m3 = (r3.get("metrics") or {}) if not r3.get("error") else {}
        entry["baseline"] = {"cached_tokens": m3.get("cached_tokens"), "cache_write_tokens": m3.get("cache_write_tokens"),
                             "prompt_tokens": m3.get("prompt_tokens"), "ttft_ms": m3.get("ttft_ms"), "error": r3.get("error")}
        # —— 指标判定：null→0 归一化后拆分 repeat_hit / prefix_hit ——
        hit_vals = [_cached(s.get("cached_tokens")) for s in same[1:]]  # run2~N 的命中值
        repeat_hits = sum(1 for c in hit_vals if c > 0)
        repeat_hit = repeat_hits / float(max(1, CACHE_RUNS - 1))  # 0.0 ~ 1.0
        pr = _cached(entry["prefix_reuse"].get("cached_tokens"))
        bl = _cached(entry["baseline"].get("cached_tokens"))
        prefix_hit = pr > 0 and pr > bl                     # 增量命中：>0 且 > 基线
        entry["session_id_supported"] = session_supported   # 待验证项：EasyRouter 是否支持 session_id
        entry["cache_write_tokens"] = same[0].get("cache_write_tokens") if same else None  # run1 首次写入
        entry["repeat_hit"] = repeat_hit
        entry["prefix_hit"] = prefix_hit
        entry["hit"] = repeat_hit > 0
        entry["cache_type"] = classify_cache(same, pr, bl, repeat_hit, prefix_hit)
        entry["cached_trusted"] = entry["cache_type"] != "字段不可信"
        rows.append(entry)
        layer_dir = os.path.join(out_dir, "raw", "cache")
        os.makedirs(layer_dir, exist_ok=True)
        with open(os.path.join(layer_dir, "%s__cache.json" % p["id"]), "w", encoding="utf-8") as f:
            json.dump({"meta": {"provider_id": p["id"], "provider_name": p["name"], "model_id": p["model_id"],
                                "channel_id": p["channel_id"], "channel_name": p["channel_name"],
                                "model": p["model"], "base_url": p["base_url"]},
                       "environment": env, "cache_result": entry}, f, ensure_ascii=False, indent=2)

    render_cache_table(rows)
    summary = {
        "mode": "cache", "prefix_tokens": CACHE_PREFIX_TOKENS, "runs": CACHE_RUNS,
        "null_normalized": True,
        "session_design": "连打+前缀复用共用 session（{渠道}_{模型}_cache_shared），基线独立 session；不支持 session_id 时自动 fallback",
        "metrics": {
            "cache_write_tokens": "run1 首次写入(建立缓存)的 token 数",
            "repeat_hit": "same_prompt run2~%d 中 cached_tokens>0 的次数 / %d" % (CACHE_RUNS, CACHE_RUNS - 1),
            "prefix_hit": "prefix_reuse.cached_tokens>0 且 > baseline（增量命中）",
        },
        "cache_types": ["字段不可信", "前缀缓存", "精确匹配缓存(无前缀复用)", "反常(连打miss但前缀命中)", "无缓存"],
        "environment": env, "rows": rows,
    }
    with open(os.path.join(out_dir, "cache_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("缓存结果已写入：%s（将合并进最终 report.xlsx）" % os.path.join(os.path.abspath(out_dir), "cache_summary.json"))


def main():
    ap = argparse.ArgumentParser(description="多渠道统一模型能力测试")
    ap.add_argument("--providers-file", default=DEFAULT_PROVIDERS_FILE)
    ap.add_argument("--cases-dir", default=DEFAULT_CASES_DIR)
    ap.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--models", default="", help="逗号分隔模型 id，默认全部")
    ap.add_argument("--channels", default="", help="逗号分隔渠道 id，默认全部")
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--per-layer", type=int, default=0, help="每层只保留最后 N 条用例（快速跑子集，如 --per-layer 4）")
    ap.add_argument("--cache", action="store_true", help="缓存命中专项测试（长前缀连打 + 前缀复用 + 基线）")
    ap.add_argument("--parallel", action="store_true", help="渠道级并行：每个「模型×渠道」组合串行，不同组合并行（会轻微影响性能数据精度）")
    ap.add_argument("--sample", action="store_true", help="示例模式：每个「模型×渠道」组合只跑 1 条示例")
    ap.add_argument("--once", action="store_true", help="单次模式：每条用例只跑 1 次、不做重试，快速预览结果")
    ap.add_argument("--list-cases", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-timeline", action="store_true", help="原始 JSON 不保存逐 chunk 时间线")
    ap.add_argument("--env-tag", default="", help="测试环境标注，写入报告（如：本地笔记本 / EC2 g5.xlarge）")
    ap.add_argument("--no-ts", action="store_true", help="结果直接写到 --out 目录（不加时间戳子目录，会覆盖历史）")
    args = ap.parse_args()

    cfg, cases, providers = load_config(args)
    if not providers:
        sys.exit("没有可用组合，请检查 config/providers.json 的 channels/models 与 --models/--channels")
    if args.models:
        ids = set(args.models.split(","))
        providers = [p for p in providers if p["model_id"] in ids]
        if not providers:
            sys.exit("--models 指定的模型 id 不存在")
    if args.channels:
        ids = set(args.channels.split(","))
        providers = [p for p in providers if p["channel_id"] in ids]
        if not providers:
            sys.exit("--channels 指定的渠道 id 不存在")
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if args.per_layer > 0:
        # 每层只保留最后 N 条用例（快速子集，避免重复测试）
        by_layer = {}
        for c in cases:
            by_layer.setdefault(c["layer"], []).append(c)
        cases = []
        for layer in LAYERS:
            cases.extend(by_layer.get(layer, [])[-args.per_layer:])

    once = args.once
    if once:
        cfg["retries"] = 0  # 单次模式不重试
        print("[单次模式] 每条用例只执行 1 次、不做重试，快速预览结果")

    # 每次运行结果写入带时间戳的子目录，避免覆盖历史结果（--no-ts 可关闭）
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out if args.no_ts else os.path.join(args.out, "run_" + run_id)
    if not args.dry_run and not args.list_cases:
        os.makedirs(out_dir, exist_ok=True)
        print("本次结果输出目录：%s" % os.path.abspath(out_dir))

    # 缓存命中专项测试（附加模式：先跑缓存，再继续常规测试，最终合并进同一份 report.xlsx）
    if args.cache:
        run_cache(providers, cfg, out_dir, args.env_tag)

    # 示例模式：复用性能层第 1 条用例，每个组合只跑 1 次，跑完即返回
    if args.sample:
        run_sample(providers, cfg, out_dir, load_sample_case(args.cases_dir), args.env_tag)
        return

    if args.list_cases:
        print("共 %d 条用例：" % len(cases))
        for c in cases:
            extra = " | target=%s" % c.get("target_tokens") if c.get("target_tokens") else ""
            extra += " | filler=%s" % c.get("filler_tokens") if c.get("filler_tokens") else ""
            extra += " | judge=%s" % c.get("judge") if c.get("judge") else ""
            print("  [%s] %-18s %s | stream=%s | max_tokens=%s%s" % (
                c["layer"], c["id"], c.get("name", ""), c.get("stream"), c.get("max_tokens"), extra))
        return

    total_requests = (len(cases) * len(providers)) if once else (sum(int(c.get("runs", 1)) for c in cases) * len(providers))
    print("渠道：%d 个 | 用例：%d 条 | 预计请求：%d 次%s" % (
        len(providers), len(cases), total_requests, "" if once else "（含 temperature=0 双跑）"))

    env = env_info(args.env_tag)
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    rows, order_log, order_index, interrupted = [], [], 0, False
    started_wall = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def execute_one(p, case):
        """执行单条用例并落盘，返回 result。"""
        sent_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = run_case(p, case, cfg, 0, once=once)
        result.setdefault("meta", {})
        result["meta"].update({
            "layer": case["layer"], "case_id": case["id"], "case_name": case.get("name", ""),
            "category": case.get("category", ""), "compat_feature": case.get("compat_feature"),
            "judge": case.get("judge"), "reference": case.get("reference"), "target_tokens": case.get("target_tokens"),
            "depth": (case.get("needle") or {}).get("depth"),
            "provider_id": p["id"], "provider_name": p["name"], "model": p["model"],
            "base_url": p["base_url"], "sent_at": sent_at, "wall_started": started_wall,
        })
        result["environment"] = env
        if args.no_timeline and result.get("sse"):
            result["sse"]["timeline"] = None
        layer_dir = os.path.join(raw_dir, case["layer"])
        os.makedirs(layer_dir, exist_ok=True)
        with open(os.path.join(layer_dir, "%s__%s.json" % (p["id"], case["id"])), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def record(p, case, result):
        nonlocal order_index
        order_index += 1
        order_log.append({"order": order_index, "case_id": case["id"], "provider_id": p["id"],
                          "provider_name": p["name"], "sent_at": result["meta"]["sent_at"]})
        rows.append(result)
        m = result.get("metrics") or {}
        if result.get("skipped"):
            log("  -> 跳过：%s" % result.get("skip_reason"))
        elif result.get("error"):
            log("  -> 错误：%s | %s" % (result["error"]["type"], result["error"]["message"][:120]))
        else:
            log("  -> ok | ttft=%s ms | e2e=%s ms | %s tok/s | 内容 %d 字 | 推理 %d 字 | chunk=%s" % (
                m.get("ttft_ms"), m.get("e2e_ms"), m.get("tokens_per_sec"),
                m.get("content_chars"), m.get("reasoning_chars"), m.get("chunk_count")))

    try:
        if args.dry_run:
            for ci, case in enumerate(cases, 1):
                for p in providers:
                    order_index += 1
                    body = build_request_body(p, case, cfg.get("shared_request_params", {}), cfg.get("reasoning_max_tokens", 4096))
                    log("[dry-run] #%d %s -> %s | 输入估算 %d token | stream=%s | max_tokens=%s" % (
                        order_index, case["id"], p["id"], estimate_tokens(body["messages"][-1]["content"]),
                        case.get("stream"), body.get("max_tokens")))
        elif args.parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(providers)) as ex:
                for ci, case in enumerate(cases, 1):
                    log("[%d/%d] %s（并行 %d 个组合）" % (ci, len(cases), case["id"], len(providers)))
                    futures = {ex.submit(execute_one, p, case): p for p in providers}
                    for fut in futures:
                        p = futures[fut]
                        result = fut.result()
                        record(p, case, result)
        else:
            for ci, case in enumerate(cases, 1):
                for p in providers:
                    log("[%d/%d] %s -> %s" % (ci, len(cases), case["id"], p["name"]))
                    result = execute_one(p, case)
                    record(p, case, result)
    except KeyboardInterrupt:
        interrupted = True
        print("\n被中断，已保留已完成部分的原始数据，正在输出部分汇总……")

    if args.dry_run:
        return

    with open(os.path.join(out_dir, "order.json"), "w", encoding="utf-8") as f:
        json.dump({"started_at": started_wall, "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "order": order_log}, f, ensure_ascii=False, indent=2)

    agg = aggregate(rows, providers)
    for a in agg.values():  # 补算推理占比（思考 token 占总输出比例）
        ct = a["performance"]["completion_tokens"]
        rt = a["performance"]["reasoning_tokens"]
        a["performance"]["reasoning_ratio"] = round(rt / ct * 100, 1) if ct else None
    scores = compute_scores(agg, cfg.get("score_weights", {}))
    test_config = build_test_config(cfg, cases)
    summary = {"environment": env, "started_at": started_wall,
               "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "interrupted": interrupted,
               "score_weights": cfg.get("score_weights", {}), "shared_request_params": cfg.get("shared_request_params", {}),
               "test_config": test_config,
               "providers": [{k: v for k, v in p.items() if k != "api_key"} for p in providers],
               "aggregate": agg, "scores": scores}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = render_markdown(agg, scores, providers, env, test_config)
    with open(os.path.join(out_dir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print("\n结果已写入：%s" % os.path.abspath(out_dir))

    # 自动生成 Excel 报告（与 export_report.py 同一套逻辑）
    try:
        from export_report import generate_report
        template = os.path.join(HERE, "report_template.xlsx")
        rp = generate_report(out_dir, template, os.path.join(out_dir, "report.xlsx"))
        print("报告已生成：%s" % os.path.abspath(rp))
    except Exception as e:
        print("[提示] 自动生成 Excel 报告失败：%s（可手动运行 python export_report.py --out %s）" % (e, os.path.abspath(out_dir)))


if __name__ == "__main__":
    main()
