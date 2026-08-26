#!/usr/bin/env python3
"""
kimi-k3 双平台对比：EasyRouter vs Mayi

对比维度：
  1. 性能 —— TTFT / 生成速度 / 端到端延迟（多次取均值与 p50）
  2. Token 开销 —— prompt_tokens / completion_tokens / reasoning_tokens
  3. 功能兼容性 —— system message / 多轮 / stop / max_tokens / function calling
  4. 输出质量 —— 数学推理 / 代码生成 / 指令遵循
  5. 长文本能力 —— 长上下文输入处理
  6. 推理深度与模型指纹 —— reasoning 过程对比
  7. 幻觉检测 —— 事实性问题验证
  8. 复杂指令遵循 —— JSON / 多语言 / 字数限制
  9. 输出一致性 —— 同 prompt 多次调用
  10. 缓存机制 —— cache 命中率 / 时效 / 前缀匹配

用法:
  python3 kimi_k3_easyrouter_vs_mayi.py
  python3 kimi_k3_easyrouter_vs_mayi.py --only perf tokens compat
  python3 kimi_k3_easyrouter_vs_mayi.py --output result_k3.json
"""

import time
import json
import argparse
import statistics
import difflib
from datetime import datetime

import openai

# 逻辑模型名 -> 各平台实际调用的 model 名称
# 两平台目前都用不带时间戳的 kimi-k3
MODEL_ALIASES = {
    "kimi-k3": {
        "easyrouter": "kimi-k3",
        "mayi": "kimi-k3",
    },
}
MODELS = list(MODEL_ALIASES)

PLATFORMS = {
    "easyrouter": {
        "label": "EasyRouter",
        "base_url": "https://easyrouter.io/v1",
        "api_key": "xxx",
    },
    "mayi": {
        "label": "Mayi",
        "base_url": "https://maas-api-ga.computrix.ai/v1",
        "api_key": "xxx",
    },
}

REASONING_MAX_TOKENS = 4096
TEMPERATURE_SUPPORTED = False  # kimi-k3 不支持自定义 temperature，默认不传

MODEL = MODELS[0]  # 逻辑模型名，运行时由 main() 覆盖为当前测试的模型


def model_for(platform):
    """返回某个逻辑模型在指定平台上实际调用的 model 名称"""
    return MODEL_ALIASES[MODEL][platform]

# ========== 测试用例 ==========

QUALITY_CASES = [
    {
        "name": "数学推理",
        "messages": [{"role": "user", "content": "一个水池有两个进水管和一个出水管。单开A管6小时注满，单开B管8小时注满，单开出水管12小时放完。三管同时打开，多少小时注满？给出推理过程。"}],
        "max_tokens": 4096,
    },
    {
        "name": "代码生成",
        "messages": [{"role": "user", "content": "用 Python 写一个函数求最大子数组和，要求带注释、复杂度分析和测试用例。"}],
        "max_tokens": 4096,
    },
    {
        "name": "指令遵循",
        "messages": [{"role": "user", "content": "请列出5个中国城市名称，要求：每个城市名后跟一个英文逗号；全部写在同一行；不要任何多余文字、标点、解释；不要编号。"}],
        "max_tokens": 4096,
    },
]

HALLUCINATION_CASES = [
    {
        "name": "历史事实",
        "messages": [{"role": "user", "content": "中华人民共和国是哪一年成立的？只回答年份数字。"}],
        "expected_contains": "1949",
        "max_tokens": 512,
    },
    {
        "name": "科学常识",
        "messages": [{"role": "user", "content": "水的化学式是什么？只回答化学式。"}],
        "expected_contains": "H2O",
        "max_tokens": 512,
    },
    {
        "name": "数学计算",
        "messages": [{"role": "user", "content": "123 × 456 = ? 只回答数字结果。"}],
        "expected_contains": "56088",
        "max_tokens": 512,
    },
    {
        "name": "地理知识",
        "messages": [{"role": "user", "content": "日本的首都是哪里？只回答城市名。"}],
        "expected_contains": "东京",
        "max_tokens": 512,
    },
    {
        "name": "虚构知识检测",
        "messages": [{"role": "user", "content": "请介绍一下中国科学家张伟明在2019年获得诺贝尔物理学奖的研究成果。如果此信息不准确请指出。"}],
        "expected_contains": "不",
        "max_tokens": 1024,
    },
]

COMPLEX_INSTRUCTION_CASES = [
    {
        "name": "JSON 格式 + 字段约束",
        "messages": [{"role": "user", "content": "请用 JSON 格式输出3本中国古典小说的信息，要求：1) 每本包含 title、author、dynasty 三个字段；2) dynasty 用朝代全称；3) 只输出 JSON，不要任何解释文字；4) JSON 必须是合法的，可以被解析。"}],
        "max_tokens": 2048,
        "validate": "json",
    },
    {
        "name": "多语言混合 + 格式约束",
        "messages": [{"role": "user", "content": "请列出3个编程语言，每个用以下格式输出一行：\n[英文名] | [中文名] | [诞生年份] | [创始人英文名]\n不要任何标题、编号、解释，只输出3行数据。"}],
        "max_tokens": 1024,
        "validate": "lines_3",
    },
    {
        "name": "字数限制 + 内容约束",
        "messages": [{"role": "user", "content": "用恰好20个汉字描述春天的景色。要求：1) 恰好20个汉字；2) 不含标点符号；3) 不含英文和数字；4) 是一个完整的句子。"}],
        "max_tokens": 1024,
        "validate": "char_count_20",
    },
]

FINGERPRINT_CASES = [
    {
        "name": "简单数学推理",
        "messages": [{"role": "user", "content": "如果一个正方形的面积是64平方厘米，它的周长是多少厘米？请一步一步推理。"}],
        "max_tokens": 2048,
    },
    {
        "name": "逻辑推理",
        "messages": [{"role": "user", "content": "所有的猫都是动物，所有的动物都会死，小花是一只猫。请问小花会死吗？请一步一步推理。"}],
        "max_tokens": 2048,
    },
]


_CURRENT_PLATFORM = None  # 当前正在使用的平台 key，由 get_client 设置，build_kwargs 读取以选择正确的 model 名


def get_client(platform):
    global _CURRENT_PLATFORM
    _CURRENT_PLATFORM = platform
    p = PLATFORMS[platform]
    return openai.OpenAI(base_url=p["base_url"], api_key=p["api_key"])


def build_kwargs(messages, max_tokens, **extra):
    model_name = MODEL_ALIASES[MODEL].get(_CURRENT_PLATFORM, MODEL) if _CURRENT_PLATFORM else MODEL
    kwargs = {"model": model_name, "messages": messages, "max_tokens": max_tokens}
    if TEMPERATURE_SUPPORTED:
        kwargs["temperature"] = 0
    kwargs.update(extra)
    return kwargs


def stream_call(client, messages, max_tokens):
    """流式调用，返回性能指标 + 内容"""
    start = time.time()
    first_token = None
    content_chunks, reasoning_chunks = [], []
    usage = None

    kwargs = build_kwargs(messages, max_tokens, stream=True,
                          stream_options={"include_usage": True})
    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("stream_options", None)
        stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if first_token is None:
                first_token = time.time()
            content_chunks.append(delta.content)
        r = getattr(delta, "reasoning_content", None)
        if r:
            if first_token is None:
                first_token = time.time()
            reasoning_chunks.append(r)

    end = time.time()
    chunk_count = len(content_chunks) + len(reasoning_chunks)
    gen_time = (end - first_token) if first_token else 0
    return {
        "content": "".join(content_chunks),
        "reasoning": "".join(reasoning_chunks),
        "ttft_ms": (first_token - start) * 1000 if first_token else None,
        "total_ms": (end - start) * 1000,
        "chunks": chunk_count,
        "content_chunks": len(content_chunks),
        "reasoning_chunks": len(reasoning_chunks),
        "tps": chunk_count / gen_time if gen_time > 0 else 0,
        "usage": usage,
    }


def sync_call(client, messages, max_tokens):
    """非流式调用"""
    resp = client.chat.completions.create(
        **build_kwargs(messages, max_tokens))
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": resp.usage,
        "finish_reason": resp.choices[0].finish_reason,
    }


# ========== 1. 性能对比 ==========

def compare_performance(rounds):
    print_header("1. 性能对比")
    summary = {}

    for platform in PLATFORMS:
        label = PLATFORMS[platform]["label"]
        client = get_client(platform)
        ttfts, tpss, totals, errors = [], [], [], []

        for i in range(rounds):
            try:
                r = stream_call(client,
                                [{"role": "user", "content": f"Write a short sentence about the number {i+1}."}],
                                REASONING_MAX_TOKENS)
                if r["ttft_ms"]:
                    ttfts.append(r["ttft_ms"])
                    tpss.append(r["tps"])
                    totals.append(r["total_ms"])
            except Exception as e:
                errors.append(str(e))

        summary[platform] = {
            "ttft_avg": statistics.mean(ttfts) if ttfts else None,
            "ttft_p50": statistics.median(ttfts) if ttfts else None,
            "tps_avg": statistics.mean(tpss) if tpss else None,
            "total_avg": statistics.mean(totals) if totals else None,
            "ok": len(ttfts),
            "errors": errors,
        }

    rows = [
        ("TTFT avg (ms)", "ttft_avg", "{:.0f}", "lower"),
        ("TTFT p50 (ms)", "ttft_p50", "{:.0f}", "lower"),
        ("生成速度 (chunk/s)", "tps_avg", "{:.1f}", "higher"),
        ("端到端 avg (ms)", "total_avg", "{:.0f}", "lower"),
    ]
    print_table(rows, summary, rounds)
    return summary


# ========== 2. Token 开销对比 ==========

def compare_token_overhead():
    print_header("2. Token 开销对比 (prompt='Hello world')")
    result = {}

    for platform in PLATFORMS:
        label = PLATFORMS[platform]["label"]
        client = get_client(platform)
        try:
            resp = client.chat.completions.create(
                **build_kwargs([{"role": "user", "content": "Hello world"}], REASONING_MAX_TOKENS))
            u = resp.usage
            details = getattr(u, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            # 兼容 Mayi 的 reasoning_tokens 在 usage 顶层的情况
            if reasoning_tokens is None and hasattr(u, "reasoning_tokens"):
                reasoning_tokens = u.reasoning_tokens
            result[platform] = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "content": resp.choices[0].message.content,
            }
        except Exception as e:
            result[platform] = {"error": str(e)}

    for platform, r in result.items():
        label = PLATFORMS[platform]["label"]
        if "error" in r:
            print(f"  {label:<16} ❌ {r['error']}")
            continue
        print(f"  {label:<16} prompt={r['prompt_tokens']:<5} completion={r['completion_tokens']:<5} "
              f"reasoning={r['reasoning_tokens']}")
        print(f"  {'':<16} 回复: {r['content'][:70]!r}")

    ps = [r.get("prompt_tokens") for r in result.values() if "error" not in r]
    if ps and min(ps) > 0 and max(ps) / min(ps) >= 2:
        print(f"\n  ⚠️  prompt_tokens 差异 {max(ps)/min(ps):.1f}x —— 高的一侧很可能注入了隐藏 system prompt")
    elif ps and len(set(ps)) > 1:
        print(f"\n  ℹ️  prompt_tokens 有差异: {ps}")
    return result


# ========== 3. 功能兼容性对比 ==========

def compare_compatibility():
    print_header("3. 功能兼容性对比")
    checks = {}

    for platform in PLATFORMS:
        client = get_client(platform)
        res = {}

        # system message
        try:
            r = client.chat.completions.create(**build_kwargs(
                [{"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
                 {"role": "user", "content": "Hello"}], REASONING_MAX_TOKENS))
            c = r.choices[0].message.content or ""
            res["system message"] = ("arr" in c.lower() or "pirate" in c.lower(), c[:40])
        except Exception as e:
            res["system message"] = (False, str(e)[:60])

        # 多轮对话
        try:
            r = client.chat.completions.create(**build_kwargs(
                [{"role": "user", "content": "My name is Alice."},
                 {"role": "assistant", "content": "Hello Alice!"},
                 {"role": "user", "content": "What is my name?"}], REASONING_MAX_TOKENS))
            c = r.choices[0].message.content or ""
            res["多轮上下文"] = ("alice" in c.lower(), c[:40])
        except Exception as e:
            res["多轮上下文"] = (False, str(e)[:60])

        # stop 参数
        try:
            r = client.chat.completions.create(**build_kwargs(
                [{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
                REASONING_MAX_TOKENS, stop=[","]))
            c = r.choices[0].message.content or ""
            res["stop 参数"] = ("," not in c, c[:40])
        except Exception as e:
            res["stop 参数"] = (False, str(e)[:60])

        # max_tokens 约束
        try:
            r = client.chat.completions.create(**build_kwargs(
                [{"role": "user", "content": "Write a very long essay about the history of the world."}], 200))
            n = r.usage.completion_tokens
            res["max_tokens 约束"] = (n <= 210, f"completion_tokens={n} (设 200)")
        except Exception as e:
            res["max_tokens 约束"] = (False, str(e)[:60])

        # function calling
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get the weather for a location",
            "parameters": {"type": "object",
                           "properties": {"location": {"type": "string", "description": "City name"}},
                           "required": ["location"]}}}]
        try:
            r = client.chat.completions.create(**build_kwargs(
                [{"role": "user", "content": "What's the weather in Beijing?"}],
                REASONING_MAX_TOKENS, tools=tools, tool_choice="auto"))
            tc = r.choices[0].message.tool_calls
            res["function calling"] = (bool(tc), tc[0].function.arguments if tc else "未触发")
        except Exception as e:
            res["function calling"] = (False, str(e)[:60])

        # temperature=0
        try:
            client.chat.completions.create(**build_kwargs(
                [{"role": "user", "content": "1+1=?"}], REASONING_MAX_TOKENS, temperature=0))
            res["temperature=0"] = (True, "接受")
        except Exception as e:
            res["temperature=0"] = (False, str(e)[:60])

        checks[platform] = res

    names = list(next(iter(checks.values())).keys())
    plats = list(checks)
    header_labels = [PLATFORMS[p]["label"][:24] for p in plats]
    print(f"\n  {'测试项':<18}" + "".join(f"│ {l:<26}" for l in header_labels))
    print(f"  {'─'*18}" + "".join("┼" + "─"*27 for _ in plats))
    for n in names:
        print(f"  {n:<18}", end="")
        for p in plats:
            ok, detail = checks[p][n]
            print(f"│ {'✅' if ok else '❌'} {detail[:23]:<23}", end="")
        print()
    return checks


# ========== 4. 输出质量并排对比 ==========

def compare_quality(cases):
    print_header("4. 输出质量并排对比")
    out = []

    for i, case in enumerate(cases, 1):
        print(f"\n{'█'*100}")
        print(f"  [{i}/{len(cases)}] {case['name']}")
        print(f"{'█'*100}")
        print(f"  📝 {case['messages'][-1]['content'][:160]}")

        entry = {"name": case["name"], "results": {}}
        for platform in PLATFORMS:
            label = PLATFORMS[platform]["label"]
            try:
                r = stream_call(get_client(platform), case["messages"], case["max_tokens"])
                perf = (f"TTFT={r['ttft_ms']:.0f}ms | {r['tps']:.1f}chunk/s | 总耗时={r['total_ms']:.0f}ms | "
                        f"content={r['content_chunks']} reasoning={r['reasoning_chunks']}")
                print(f"\n  ✅ [{label}] {perf}")
                print(f"  {'─'*90}")
                if r["reasoning"]:
                    print(f"  💭 reasoning {len(r['reasoning'])} 字: {r['reasoning'][:150]}...")
                body = r["content"]
                if len(body) > 1200:
                    body = body[:1200] + f"\n  ...(共 {len(r['content'])} 字，已截断)"
                for line in body.split("\n"):
                    print(f"  {line}")
                entry["results"][platform] = {
                    "ttft_ms": r["ttft_ms"], "tps": r["tps"], "total_ms": r["total_ms"],
                    "content": r["content"], "reasoning": r["reasoning"],
                    "reasoning_len": len(r["reasoning"]),
                }
            except Exception as e:
                print(f"\n  ❌ [{label}] {e}")
                entry["results"][platform] = {"error": str(e)}
        out.append(entry)
    return out


# ========== 5. 长文本能力对比 ==========

def compare_long_context():
    print_header("5. 长文本能力对比")
    long_text = ("人工智能（Artificial Intelligence）是计算机科学的一个重要分支，"
                 "它致力于研究如何使计算机能够模拟人类的智能行为。" * 100)

    messages = [
        {"role": "user", "content": f"请阅读以下文本并总结其核心主题（用一句话），然后统计文本中'人工智能'这个词出现了多少次：\n\n{long_text}"}
    ]

    results = {}
    for platform in PLATFORMS:
        label = PLATFORMS[platform]["label"]
        client = get_client(platform)
        try:
            start = time.time()
            r = sync_call(client, messages, 2048)
            elapsed = time.time() - start
            usage = r["usage"]
            results[platform] = {
                "content": r["content"],
                "reasoning_len": len(r["reasoning"]),
                "elapsed_s": elapsed,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "finish_reason": r["finish_reason"],
            }
            print(f"\n  ✅ [{label}] 耗时={elapsed:.1f}s, prompt_tokens={usage.prompt_tokens if usage else '?'}, "
                  f"completion={usage.completion_tokens if usage else '?'}, finish={r['finish_reason']}")
            print(f"     回复: {r['content'][:200]}")
        except Exception as e:
            results[platform] = {"error": str(e)}
            print(f"\n  ❌ [{label}] {e}")

    return results


# ========== 6. 推理深度与模型指纹对比 ==========

def compare_reasoning_fingerprint():
    print_header("6. 推理深度与模型指纹对比")
    results = {}

    for case in FINGERPRINT_CASES:
        print(f"\n  {'─'*90}")
        print(f"  🔬 {case['name']}: {case['messages'][-1]['content'][:80]}")
        print(f"  {'─'*90}")

        case_results = {}
        reasonings = {}
        for platform in PLATFORMS:
            label = PLATFORMS[platform]["label"]
            client = get_client(platform)
            try:
                r = sync_call(client, case["messages"], case["max_tokens"])
                case_results[platform] = {
                    "content": r["content"],
                    "reasoning": r["reasoning"],
                    "reasoning_len": len(r["reasoning"]),
                }
                reasonings[platform] = r["reasoning"]
                print(f"  [{label}] reasoning={len(r['reasoning'])}字, content={len(r['content'])}字")
                print(f"     💭 {r['reasoning'][:150]}...")
                print(f"     📝 {r['content'][:150]}")
            except Exception as e:
                case_results[platform] = {"error": str(e)}
                print(f"  [{label}] ❌ {e}")

        plats = [p for p in PLATFORMS if p in reasonings and reasonings[p]]
        if len(plats) >= 2:
            print(f"\n  📊 Reasoning 相似度:")
            for i in range(len(plats)):
                for j in range(i + 1, len(plats)):
                    l1 = PLATFORMS[plats[i]]["label"]
                    l2 = PLATFORMS[plats[j]]["label"]
                    ratio = difflib.SequenceMatcher(
                        None, reasonings[plats[i]], reasonings[plats[j]]).ratio()
                    print(f"     {l1} vs {l2}: {ratio:.1%}")
                    if ratio < 0.3:
                        print(f"     ⚠️  相似度很低，可能使用了不同的模型版本或配置")
                    elif ratio > 0.7:
                        print(f"     ✅ 相似度高，很可能是同一个模型")

        results[case["name"]] = case_results

    return results


# ========== 7. 幻觉检测对比 ==========

def compare_hallucination():
    print_header("7. 幻觉检测对比")
    results = {}

    plats = list(PLATFORMS)
    labels = [PLATFORMS[p]["label"][:26] for p in plats]
    print(f"\n  {'测试项':<16}" + "".join(f"│ {l:<28}" for l in labels))
    print(f"  {'─'*16}" + "".join("┼" + "─"*29 for _ in plats))

    for case in HALLUCINATION_CASES:
        case_results = {}
        print(f"  {case['name']:<16}", end="")

        for platform in PLATFORMS:
            client = get_client(platform)
            try:
                r = sync_call(client, case["messages"], case["max_tokens"])
                content = r["content"].strip()
                expected = case["expected_contains"]
                passed = expected.lower() in content.lower()
                case_results[platform] = {
                    "content": content,
                    "passed": passed,
                }
                status = "✅" if passed else "❌"
                display = content[:22].replace("\n", " ")
                print(f"│ {status} {display:<25}", end="")
            except Exception as e:
                case_results[platform] = {"error": str(e)}
                print(f"│ ❌ ERROR{'':<20}", end="")

        print()
        results[case["name"]] = case_results

    print(f"\n  {'─'*16}" + "".join("┼" + "─"*29 for _ in plats))
    print(f"  {'得分':<16}", end="")
    for platform in plats:
        score = sum(1 for c in results.values()
                    if platform in c and c[platform].get("passed", False))
        total = len(HALLUCINATION_CASES)
        print(f"│ {score}/{total}{'':<25}", end="")
    print()

    return results


# ========== 8. 复杂指令遵循对比 ==========

def compare_complex_instructions():
    print_header("8. 复杂指令遵循对比")
    results = {}

    for i, case in enumerate(COMPLEX_INSTRUCTION_CASES, 1):
        print(f"\n  {'█'*90}")
        print(f"  [{i}/{len(COMPLEX_INSTRUCTION_CASES)}] {case['name']}")
        print(f"  {'█'*90}")
        print(f"  📝 {case['messages'][-1]['content'][:140]}")

        case_results = {}
        for platform in PLATFORMS:
            label = PLATFORMS[platform]["label"]
            client = get_client(platform)
            try:
                r = sync_call(client, case["messages"], case["max_tokens"])
                content = r["content"].strip()
                validation = validate_output(content, case.get("validate"))
                case_results[platform] = {
                    "content": content,
                    "validation": validation,
                }
                status = "✅" if validation["passed"] else "❌"
                print(f"\n  {status} [{label}] {validation['reason']}")
                display = content[:300].replace("\n", "\n     ")
                print(f"     {display}")
            except Exception as e:
                case_results[platform] = {"error": str(e)}
                print(f"\n  ❌ [{label}] {e}")

        results[case["name"]] = case_results

    return results


def validate_output(content, validate_type):
    """验证模型输出是否符合约束"""
    if not validate_type:
        return {"passed": True, "reason": "无验证规则"}

    if validate_type == "json":
        import re
        clean = content
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if match:
            clean = match.group(1).strip()
        try:
            parsed = json.loads(clean)
            has_codeblock = match is not None
            if isinstance(parsed, list) and len(parsed) == 3:
                required_fields = {"title", "author", "dynasty"}
                for item in parsed:
                    if not required_fields.issubset(set(item.keys())):
                        return {"passed": False, "reason": f"缺少字段，实际: {list(item.keys())}"}
                reason = "合法 JSON，3 条记录，字段完整"
                if has_codeblock:
                    reason += "（⚠️ 带 markdown 代码块包裹）"
                return {"passed": True, "reason": reason}
            elif isinstance(parsed, list):
                return {"passed": False, "reason": f"JSON 合法但记录数={len(parsed)}，期望 3"}
            else:
                return {"passed": False, "reason": f"JSON 合法但不是数组: {type(parsed).__name__}"}
        except json.JSONDecodeError as e:
            return {"passed": False, "reason": f"JSON 解析失败: {e}"}

    elif validate_type == "lines_3":
        lines = [l for l in content.strip().split("\n") if l.strip()]
        if len(lines) == 3:
            all_pipe = all("|" in l for l in lines)
            return {"passed": all_pipe, "reason": f"3 行，管道分隔={'是' if all_pipe else '否'}"}
        return {"passed": False, "reason": f"行数={len(lines)}，期望 3"}

    elif validate_type == "char_count_20":
        chinese_chars = [c for c in content if '\u4e00' <= c <= '\u9fff']
        count = len(chinese_chars)
        has_punctuation = any(c in content for c in "，。！？、；：""''（）《》【】")
        has_english = any(c.isascii() and c.isalpha() for c in content)
        passed = count == 20 and not has_punctuation and not has_english
        reason = f"汉字数={count}"
        if has_punctuation:
            reason += ", 含标点"
        if has_english:
            reason += ", 含英文"
        if passed:
            reason += " ✓"
        return {"passed": passed, "reason": reason}

    return {"passed": True, "reason": "未知验证类型"}


# ========== 9. 一致性测试 ==========

def compare_consistency(rounds=3):
    print_header("9. 输出一致性对比 (同 prompt 多次调用)")
    prompt = "1+1等于几？只回答数字。"
    results = {}

    for platform in PLATFORMS:
        label = PLATFORMS[platform]["label"]
        client = get_client(platform)
        outputs = []
        for _ in range(rounds):
            try:
                r = sync_call(client, [{"role": "user", "content": prompt}], 512)
                outputs.append(r["content"].strip())
            except Exception as e:
                outputs.append(f"ERROR: {e}")
        unique = set(outputs)
        results[platform] = {
            "outputs": outputs,
            "unique_count": len(unique),
            "consistent": len(unique) == 1,
        }
        status = "✅" if len(unique) == 1 else "⚠️"
        print(f"  {status} [{label}] {rounds} 次回复中有 {len(unique)} 种不同答案: {list(unique)}")

    return results


# ========== 10. 缓存机制对比 ==========

def compare_cache():
    print_header("10. 缓存机制对比")
    results = {}

    long_system = ("你是一个专业的技术顾问，擅长云计算、人工智能和软件工程领域。"
                   "请用专业但易懂的方式回答问题。" * 5)

    for platform in PLATFORMS:
        label = PLATFORMS[platform]["label"]
        client = get_client(platform)
        platform_result = {}
        print(f"\n  --- [{label}] ---")

        # 10.1 Cache 命中率
        print(f"\n  [10.1] 相同 prompt 连续调用（检测 cache 命中）")
        messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "什么是 Kubernetes？用一句话回答。"}
        ]
        cache_hits = []
        ttfts = []
        for i in range(3):
            try:
                start = time.time()
                resp = client.chat.completions.create(
                    **build_kwargs(messages, 1024))
                elapsed = time.time() - start
                usage = resp.usage
                cached = 0
                if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                cache_hits.append(cached)
                ttfts.append(elapsed * 1000)
                print(f"    第{i+1}次: prompt={usage.prompt_tokens}, cached={cached}, "
                      f"耗时={elapsed*1000:.0f}ms")
            except Exception as e:
                print(f"    第{i+1}次: ❌ {e}")
                cache_hits.append(0)
                ttfts.append(None)

        platform_result["same_prompt"] = {
            "cached_tokens": cache_hits,
            "ttfts_ms": ttfts,
            "cache_hit_from_2nd": cache_hits[1] > 0 if len(cache_hits) > 1 else False,
        }
        if len(cache_hits) > 1 and cache_hits[1] > 0:
            print(f"    ✅ 第2次起命中 cache: {cache_hits[1]} tokens")
        else:
            print(f"    ℹ️  未检测到 cache 命中（cached_tokens: {cache_hits}）")

        # 10.2 前缀 Cache
        print(f"\n  [10.2] 前缀 Cache 测试（相同 system prompt，不同 user 问题）")
        try:
            client.chat.completions.create(**build_kwargs([
                {"role": "system", "content": long_system},
                {"role": "user", "content": "什么是 Docker？"}
            ], 512))
        except:
            pass
        time.sleep(1)

        prefix_messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "什么是微服务架构？用一句话回答。"}
        ]
        try:
            resp = client.chat.completions.create(
                **build_kwargs(prefix_messages, 1024))
            usage = resp.usage
            cached = 0
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0
            platform_result["prefix_cache"] = {
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": cached,
                "hit": cached > 0,
            }
            if cached > 0:
                print(f"    ✅ 前缀 Cache 命中: cached={cached}/{usage.prompt_tokens} tokens "
                      f"({cached/usage.prompt_tokens*100:.0f}%)")
            else:
                print(f"    ❌ 前缀 Cache 未命中: prompt={usage.prompt_tokens}, cached=0")
        except Exception as e:
            platform_result["prefix_cache"] = {"error": str(e)}
            print(f"    ❌ {e}")

        results[platform] = platform_result

    # 汇总
    print(f"\n  {'─'*90}")
    print(f"  📊 缓存能力汇总:")
    print(f"  {'─'*90}")
    for platform in PLATFORMS:
        label = PLATFORMS[platform]["label"]
        r = results.get(platform, {})
        same = r.get("same_prompt", {})
        prefix = r.get("prefix_cache", {})
        cache_2nd = same.get("cache_hit_from_2nd", False)
        prefix_hit = prefix.get("hit", False)
        print(f"  {label}:")
        print(f"    相同 prompt cache: {'✅' if cache_2nd else '❌'}")
        print(f"    前缀 cache:        {'✅' if prefix_hit else '❌'}")

    return results


# ========== 输出辅助 ==========

def print_header(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def print_table(rows, summary, rounds):
    plats = list(summary)
    labels = [PLATFORMS[p]["label"][:20] for p in plats]
    print(f"\n  ({rounds} 次请求取统计值)\n")
    print(f"  {'指标':<20}" + "".join(f"│ {l:<22}" for l in labels) + "│ 优势方")
    print(f"  {'─'*20}" + "".join("┼" + "─"*23 for _ in plats) + "┼" + "─"*20)
    for label, key, fmt, better in rows:
        vals = {p: summary[p][key] for p in plats}
        valid = {p: v for p, v in vals.items() if v is not None}
        win = None
        if len(valid) >= 2:
            win = (min if better == "lower" else max)(valid, key=valid.get)
        print(f"  {label:<20}", end="")
        for p in plats:
            v = vals[p]
            cell = fmt.format(v) if v is not None else "N/A"
            if p == win:
                cell += " ★"
            print(f"│ {cell:<22}", end="")
        win_label = PLATFORMS[win]["label"][:18] if win else "-"
        print(f"│ {win_label}")
    for p in plats:
        if summary[p]["errors"]:
            label = PLATFORMS[p]["label"]
            print(f"\n  ⚠️  {label} 失败 {len(summary[p]['errors'])} 次: {summary[p]['errors'][0][:100]}")


def run_for_model(model, run_modules, rounds):
    """针对单个模型运行所有选定的测试模块"""
    global MODEL
    MODEL = model
    aliases = MODEL_ALIASES[model]

    print(f"\n{'#'*100}")
    print(f"  模型: {model}  (EasyRouter vs Mayi)")
    for p, cfg in PLATFORMS.items():
        print(f"    {cfg['label']:<16} {cfg['base_url']}  实际调用模型名: {aliases[p]}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试模块: {', '.join(sorted(run_modules))}")
    print(f"{'#'*100}")

    data = {"model": model, "model_aliases": aliases, "time": datetime.now().isoformat(),
            "platforms": {k: v["label"] for k, v in PLATFORMS.items()}}

    if "perf" in run_modules:
        data["performance"] = compare_performance(rounds)
    if "tokens" in run_modules:
        data["token_overhead"] = compare_token_overhead()
    if "compat" in run_modules:
        data["compatibility"] = {p: {k: list(v) for k, v in r.items()}
                                 for p, r in compare_compatibility().items()}
    if "quality" in run_modules:
        data["quality"] = compare_quality(QUALITY_CASES)
    if "longctx" in run_modules:
        data["long_context"] = compare_long_context()
    if "fingerprint" in run_modules:
        data["fingerprint"] = compare_reasoning_fingerprint()
    if "hallucination" in run_modules:
        data["hallucination"] = compare_hallucination()
    if "complex" in run_modules:
        data["complex_instructions"] = compare_complex_instructions()
    if "consistency" in run_modules:
        data["consistency"] = compare_consistency(rounds=rounds)
    if "cache" in run_modules:
        data["cache"] = compare_cache()

    print(f"\n{'#'*100}")
    print(f"  模型 {model} 对比完成!")
    print(f"{'#'*100}\n")

    return data


def main():
    all_modules = ["perf", "tokens", "compat", "quality", "longctx",
                   "fingerprint", "hallucination", "complex", "consistency", "cache"]

    parser = argparse.ArgumentParser(
        description="kimi-k3 双平台对比 (EasyRouter vs Mayi)")
    parser.add_argument("--model", nargs="*", default=MODELS, choices=MODELS,
                       help="要测试的模型，默认: kimi-k3")
    parser.add_argument("--rounds", type=int, default=5, help="性能/一致性测试轮数")
    parser.add_argument("--skip", nargs="*", default=[], choices=all_modules,
                       help="跳过的模块")
    parser.add_argument("--only", nargs="*", default=None, choices=all_modules,
                       help="只运行指定模块")
    parser.add_argument("--output", default=None, help="结果保存为 JSON")
    args = parser.parse_args()

    if args.only:
        run_modules = set(args.only)
    else:
        run_modules = set(all_modules) - set(args.skip)

    all_data = {}
    for model in args.model:
        all_data[model] = run_for_model(model, run_modules, args.rounds)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  📄 已保存: {args.output}")


if __name__ == "__main__":
    main()
