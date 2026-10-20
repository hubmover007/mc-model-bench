#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 HuggingFace 下载并抽取评测数据，生成四层「统一测试文件」（test_cases/*.json）。

设计：
  - 使用 HF datasets-server 的公开 REST API（https://datasets-server.huggingface.co/rows），
    零依赖、无需 HF Token，任何机器都可运行；
  - 支持 --all 一键从 HuggingFace 重新拉取生成全部四层测试集：
      1) quality.json      <- openai/gsm8k + truthfulqa/truthful_qa(multiple_choice) + cais/mmlu
      2) performance.json  <- tatsu-lab/alpaca + abisee/cnn_dailymail（覆盖短/长输入 × 短/长输出四象限）
      3) compatibility.json<- rajpurkar/squad_v2 真实语料驱动的 5 项协议契约（stream_sse/function_calling/json_mode/system_prompt/stop_param）
      4) long_context.json <- 10K / 64K / 128K 三档大海捞针（错位深度 0.25 / 0.75 / 0.50）
  - 默认设置 offset>=35 跳过早期行，避免与历史旧题重合；
  - 原始数据缓存到 datasets/<name>.jsonl，便于审计与复现。

用法：
    python download_datasets.py --list
    python download_datasets.py --all                        # 重新从 HF 拉取生成全部 4 层测试集（共 20 条）
    python download_datasets.py                              # 仅重新拉取 quality.json（默认 8 条：3 math + 3 hallucination + 2 reasoning）
    python download_datasets.py --datasets gsm8k,truthfulqa,mmlu_cs,mmlu_logic --limit 5 --offset 40
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # datasets-server 每页最大 100 行


def _extract_gsm8k(row, min_steps=1):
    """GSM8K：question 为题面，answer 形如 '...#### 18'，支持按最少推理步数筛选。"""
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    steps = [ln for ln in answer.splitlines() if ln.strip() and not ln.strip().startswith("####")]
    if len(steps) < min_steps:
        return "", ""
    m = re.search(r"####\s*([-+]?[\d,]+)", answer)
    reference = (m.group(1).replace(",", "") if m else "")
    return question, reference


def _extract_truthfulqa_mc(row):
    """TruthfulQA(multiple_choice) MC1：1 个正确选项 + 若干错误选项。
    选项顺序按题面哈希做确定性打乱（数据集原序正确项在最前，不打乱会产生位置偏差），所有模型看到的 prompt 逐字一致。"""
    question = (row.get("question") or "").strip()
    targets = row.get("mc1_targets") or {}
    choices, labels = targets.get("choices") or [], targets.get("labels") or []
    if not question or not choices or len(choices) != len(labels) or sum(labels) != 1:
        return "", ""
    order = list(range(len(choices)))
    random.Random(int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)).shuffle(order)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = ["%s. %s" % (letters[i], str(choices[j]).strip()) for i, j in enumerate(order)]
    correct = letters[order.index(labels.index(1))]
    prompt = ("Answer the following multiple-choice question. "
              "Reply with ONLY the letter of the single correct option.\n\n"
              "Question: %s\n\n%s\n\nAnswer:" % (question, "\n".join(opts)))
    return prompt, correct


def _extract_mmlu(row):
    """MMLU 四选一标准题：choices 为 4 个选项列表，answer 为 0~3 整数索引。"""
    question = (row.get("question") or "").strip()
    choices = row.get("choices") or []
    ans_idx = row.get("answer")
    if not question or len(choices) != 4 or not isinstance(ans_idx, int) or not (0 <= ans_idx < 4):
        return "", ""
    letters = "ABCD"
    opts = ["%s. %s" % (letters[i], str(choices[i]).strip()) for i in range(4)]
    correct = letters[ans_idx]
    prompt = ("Answer the following multiple-choice question. "
              "Reply with ONLY the letter (A, B, C, or D) of the correct option.\n\n"
              "Question: %s\n\n%s\n\nAnswer:" % (question, "\n".join(opts)))
    return prompt, correct


def _extract_truthfulqa(row):
    """TruthfulQA(generation)：旧版生成式，仅保留兼容。"""
    question = (row.get("question") or "").strip()
    correct = row.get("correct_answers") or []
    if isinstance(correct, str):
        correct = [correct]
    return question, [str(c).strip() for c in correct if str(c).strip()]


# 数据集规格：dataset / config / split / 默认条数 / 默认起始偏移（避开旧题）
DATASET_SPECS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "name": "数学推理-GSM8K",
        "category": "math",
        "judge": "gsm8k_exact",
        "default_limit": 3,
        "default_offset": 35,
        "extract": lambda row: _extract_gsm8k(row, min_steps=4),
    },
    "truthfulqa": {
        "dataset": "truthfulqa/truthful_qa",
        "config": "multiple_choice",
        "split": "validation",
        "name": "事实抗幻觉-TruthfulQA-MC1",
        "category": "hallucination",
        "judge": "mc_letter",
        "default_limit": 3,
        "default_offset": 35,
        "extract": _extract_truthfulqa_mc,
    },
    "mmlu_cs": {
        "dataset": "cais/mmlu",
        "config": "high_school_computer_science",
        "split": "test",
        "name": "计算机推理-MMLU-CS",
        "category": "reasoning",
        "judge": "mc_letter",
        "default_limit": 1,
        "default_offset": 15,
        "extract": _extract_mmlu,
    },
    "mmlu_logic": {
        "dataset": "cais/mmlu",
        "config": "logical_fallacies",
        "split": "test",
        "name": "逻辑谬误识别-MMLU-Logic",
        "category": "reasoning",
        "judge": "mc_letter",
        "default_limit": 1,
        "default_offset": 15,
        "extract": _extract_mmlu,
    },
    "truthfulqa_gen": {  # 旧版生成式，逐字子串判分误判多，仅保留作对照，不建议用于评分
        "dataset": "truthfulqa/truthful_qa",
        "config": "generation",
        "split": "validation",
        "name": "幻觉检测-TruthfulQA(生成式-已弃用)",
        "category": "hallucination",
        "judge": "truthfulqa_any",
        "default_limit": 2,
        "default_offset": 35,
        "extract": _extract_truthfulqa,
    },
}


def http_get_json(url, timeout=60, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mc-model-bench/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError("请求失败 %s: %s" % (url, last))


def fetch_rows(dataset, config, split, offset=0, length=50):
    qs = urllib.parse.urlencode({
        "dataset": dataset, "config": config,
        "split": split, "offset": offset, "length": min(length, PAGE),
    })
    data = http_get_json("%s?%s" % (API, qs))
    return data.get("rows") or [], data.get("num_rows_total") or 0


def fetch_dataset(spec, limit, start_offset=0, key_tag=""):
    """分页下载并抽取，返回统一用例列表 + 原始行列表。"""
    cases, raw_rows = [], []
    offset = start_offset
    prefix = key_tag or spec["dataset"].split("/")[-1]
    while len(cases) < limit:
        rows, total = fetch_rows(spec["dataset"], spec["config"], spec["split"], offset=offset, length=PAGE)
        if not rows:
            break
        for r in rows:
            row = r.get("row") or {}
            row_idx = r.get("row_idx", offset)
            question, reference = spec["extract"](row)
            if not question or not reference:
                continue
            raw_rows.append({"row_idx": row_idx, **row})
            cases.append({
                "id": "quality_%s_%04d" % (prefix, row_idx),
                "name": "%s-#%d" % (spec["name"], row_idx),
                "category": spec["category"],
                "source": "%s/%s:%s#%d" % (spec["dataset"], spec["config"], spec["split"], row_idx),
                "prompt": question,
                "reference": reference,
                "judge": spec["judge"],
                "max_tokens": 4096 if spec["category"] == "math" else 1024,
                "stream": False,
            })
            if len(cases) >= limit:
                break
        offset += PAGE
        if offset >= total:
            break
    return cases, raw_rows


def build_performance_from_hf(raw_dir):
    """从 HF tatsu-lab/alpaca 与 abisee/cnn_dailymail 抽取真实语料构造性能四象限用例（4条）。"""
    print("[performance] 下载 tatsu-lab/alpaca 与 abisee/cnn_dailymail 构造四象限性能用例 ...")
    alpaca_rows, _ = fetch_rows("tatsu-lab/alpaca", "default", "train", offset=120, length=60)
    cnn_rows, _ = fetch_rows("abisee/cnn_dailymail", "3.0.0", "test", offset=50, length=20)

    short_q, long_q = None, None
    for r in alpaca_rows:
        row = r.get("row") or {}
        inst = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        if inp or not inst:
            continue
        if short_q is None and 25 <= len(inst) <= 90 and 40 <= len(out) <= 180:
            short_q = (r.get("row_idx"), inst, row)
        if long_q is None and 35 <= len(inst) <= 140 and len(out) >= 350:
            long_q = (r.get("row_idx"), inst, row)
        if short_q and long_q:
            break

    long_docs = []
    for r in cnn_rows:
        row = r.get("row") or {}
        art = (row.get("article") or "").strip()
        if len(art) >= 2500:
            long_docs.append((r.get("row_idx"), art, row))
        if len(long_docs) >= 2:
            break

    with open(os.path.join(raw_dir, "performance_hf_sources.jsonl"), "w", encoding="utf-8") as f:
        for tag, item in [("alpaca_short", short_q), ("alpaca_long", long_q)] + [("cnn_%d" % i, d) for i, d in enumerate(long_docs)]:
            if item:
                f.write(json.dumps({"tag": tag, "row_idx": item[0], "row": item[2]}, ensure_ascii=False) + "\n")

    sis_idx, sis_inst, _ = short_q
    sil_idx, sil_inst, _ = long_q
    lis_idx, lis_art, _ = long_docs[0]
    lil_idx, lil_art, _ = long_docs[1]

    return [
        {
            "id": "perf_sis_hf_%04d" % sis_idx,
            "name": "短输入短输出-Alpaca#%d" % sis_idx,
            "category": "短输入短输出",
            "source": "tatsu-lab/alpaca:train#%d" % sis_idx,
            "prompt": "%s\n\n(Reply concisely in 1-2 sentences, no more than 50 words.)" % sis_inst,
            "max_tokens": 200,
            "stream": True,
        },
        {
            "id": "perf_sil_hf_%04d" % sil_idx,
            "name": "短输入长输出-Alpaca#%d" % sil_idx,
            "category": "短输入长输出",
            "source": "tatsu-lab/alpaca:train#%d" % sil_idx,
            "prompt": "%s\n\nPlease write a comprehensive, structured explanation of at least 600 words covering key principles, concrete examples, and practical trade-offs." % sil_inst,
            "max_tokens": 1800,
            "stream": True,
        },
        {
            "id": "perf_lis_hf_%04d" % lis_idx,
            "name": "长输入短输出-CNN#%d" % lis_idx,
            "category": "长输入短输出",
            "source": "abisee/cnn_dailymail:test#%d" % lis_idx,
            "prompt": "Reference material:\n\n{filler}\n\nAdditional news report:\n%s\n\nTask: Summarize the core facts of the material above in no more than 3 bullet points (under 60 words total)." % lis_art[:2500],
            "filler_tokens": 2200,
            "max_tokens": 200,
            "stream": True,
        },
        {
            "id": "perf_lil_hf_%04d" % lil_idx,
            "name": "长输入长输出-CNN#%d" % lil_idx,
            "category": "长输入长输出",
            "source": "abisee/cnn_dailymail:test#%d" % lil_idx,
            "prompt": "Reference material:\n\n{filler}\n\nPrimary case report:\n%s\n\nTask: Write a detailed analytical briefing (around 700-900 words) synthesizing the background, key arguments, and implications from the documents above." % lil_art[:2500],
            "filler_tokens": 2500,
            "max_tokens": 2000,
            "stream": True,
        },
    ]


def build_compatibility_from_hf(raw_dir):
    """基于 HF 真实问答语料生成 5 项兼容性测试（stream_sse / function_calling / json_mode / system_prompt / stop_param）。"""
    print("[compatibility] 下载 rajpurkar/squad_v2 语料构造 5 项协议兼容性用例 ...")
    squad_rows, _ = fetch_rows("rajpurkar/squad_v2", "squad_v2", "validation", offset=80, length=10)
    sq = (squad_rows[0].get("row") or {}) if squad_rows else {}
    sq_idx = squad_rows[0].get("row_idx", 80) if squad_rows else 80
    sq_ctx = (sq.get("context") or "The Amazon rainforest covers much of northwestern Brazil and extends into Colombia, Peru and other South American countries.").strip()

    with open(os.path.join(raw_dir, "compatibility_hf_sources.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"dataset": "rajpurkar/squad_v2", "row_idx": sq_idx, "row": sq}, ensure_ascii=False) + "\n")

    return [
        {
            "id": "cmp_stream_hf",
            "name": "流式输出-SSE格式与DONE校验",
            "compat_feature": "stream_sse",
            "source": "rajpurkar/squad_v2:validation#%d" % sq_idx,
            "prompt": "Read the passage and summarize it in exactly three sentences:\n\n%s" % sq_ctx,
            "max_tokens": 240,
            "stream": True,
        },
        {
            "id": "cmp_tools_hf",
            "name": "Function Calling-工具选择与JSON参数",
            "compat_feature": "function_calling",
            "source": "hf-tool-contract",
            "prompt": "Please check the current flight status for flight MU5101 from Shanghai to Beijing on 2026-10-25 using the available tools.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_flight_status",
                        "description": "Query real-time flight schedule and status by flight number and date",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "flight_number": {"type": "string", "description": "IATA flight code, e.g. MU5101"},
                                "date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
                            },
                            "required": ["flight_number", "date"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_airport_weather",
                        "description": "Query airport weather conditions by city name",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                },
            ],
            "tool_choice": "auto",
            "max_tokens": 400,
            "stream": False,
        },
        {
            "id": "cmp_json_hf",
            "name": "结构化输出-JSON Mode契约",
            "compat_feature": "json_mode",
            "source": "rajpurkar/squad_v2:validation#%d" % sq_idx,
            "prompt": ("Extract key metadata from the following passage and return ONLY a valid raw JSON object "
                       "(do NOT wrap in markdown code fences) containing exactly these four keys: "
                       "\"topic\", \"key_entities\", \"region\", \"word_count_estimate\".\n\nPassage:\n%s" % sq_ctx),
            "response_format": {"type": "json_object"},
            "expect_keys": ["topic", "key_entities", "region", "word_count_estimate"],
            "max_tokens": 400,
            "stream": False,
        },
        {
            "id": "cmp_system_hf",
            "name": "System提示词严格前缀遵循",
            "compat_feature": "system_prompt",
            "source": "hf-system-contract",
            "system": "You are a strict compliance auditor. Every response you produce MUST begin with the exact tag [AUDIT-PASS] on the very first character, followed by a one-sentence answer.",
            "prompt": "What is the primary purpose of database indexing?",
            "expect_prefix": "[AUDIT-PASS]",
            "max_tokens": 200,
            "stream": False,
        },
        {
            "id": "cmp_stop_hf",
            "name": "stop停止符参数生效校验",
            "compat_feature": "stop_param",
            "source": "hf-stop-contract",
            "prompt": "Count upward line by line using the exact format 'ITEM-1', 'ITEM-2', 'ITEM-3', up to 'ITEM-10'. When you reach ITEM-5, output the token STOP immediately.",
            "stop": ["STOP"],
            "max_tokens": 200,
            "stream": True,
        },
    ]


def build_long_context_cases():
    """生成 10K / 64K / 128K 三档长上下文大海捞针用例（针深度 0.25 / 0.75 / 0.50，全新密钥不与旧题重合）。"""
    return [
        {
            "id": "lc_needle_10k",
            "name": "大海捞针-10K(前段depth=0.25)",
            "target_tokens": 10000,
            "needle": {
                "text": "【核心审计归档】生产数据库应急回滚凭证：RV94-71KP-3MX8。审批负责人：周明远。生效窗口：72小时。",
                "depth": 0.25,
            },
            "prompt": "请仔细阅读以下全部技术归档材料，然后回答问题。\n\n{needle_text}\n\n问题：{question}",
            "question": "以上全部材料中提到的「生产数据库应急回滚凭证」是什么？请只回答凭证代码本身，不要输出任何其他文字。",
            "expect_contains": "RV94-71KP-3MX8",
            "max_tokens": 4096,
            "stream": True,
        },
        {
            "id": "lc_needle_64k",
            "name": "大海捞针-64K(后段depth=0.75)",
            "target_tokens": 64000,
            "needle": {
                "text": "【核心审计归档】跨区灾备集群切换授权码：GT58-39WN-6LP2。审批负责人：陈静姝。生效窗口：48小时。",
                "depth": 0.75,
            },
            "prompt": "请仔细阅读以下全部技术归档材料，然后回答问题。\n\n{needle_text}\n\n问题：{question}",
            "question": "以上全部材料中提到的「跨区灾备集群切换授权码」是什么？请只回答授权码本身，不要输出任何其他文字。",
            "expect_contains": "GT58-39WN-6LP2",
            "max_tokens": 4096,
            "stream": True,
        },
        {
            "id": "lc_needle_128k",
            "name": "大海捞针-128K(中段depth=0.50)",
            "target_tokens": 128000,
            "needle": {
                "text": "【核心审计归档】零信任网关根密钥冷备提取码：HK27-84QD-9ZN5。审批负责人：林涛然。生效窗口：24小时。",
                "depth": 0.5,
            },
            "prompt": "请仔细阅读以下全部技术归档材料，然后回答问题。\n\n{needle_text}\n\n问题：{question}",
            "question": "以上全部材料中提到的「零信任网关根密钥冷备提取码」是什么？请只回答提取码本身，不要输出任何其他文字。",
            "expect_contains": "HK27-84QD-9ZN5",
            "max_tokens": 4096,
            "stream": True,
        },
    ]


def main():
    ap = argparse.ArgumentParser(description="从 HuggingFace 下载抽取评测数据，生成统一测试文件")
    ap.add_argument("--datasets", default="gsm8k,truthfulqa,mmlu_cs,mmlu_logic",
                    help="逗号分隔的数据集 key：%s" % ",".join(DATASET_SPECS))
    ap.add_argument("--limit", type=int, default=0, help="每个数据集抽取条数（0=使用各数据集推荐条数：3+3+1+1=8条）")
    ap.add_argument("--offset", type=int, default=-1, help="起始行偏移（-1=使用各数据集默认非重合偏移）")
    ap.add_argument("--all", action="store_true", help="同时从 HF 刷新生成 performance/compatibility/quality/long_context 全部四层")
    ap.add_argument("--out", default=os.path.join(HERE, "test_cases", "quality.json"))
    ap.add_argument("--cases-dir", default=os.path.join(HERE, "test_cases"))
    ap.add_argument("--raw-dir", default=os.path.join(HERE, "datasets"))
    ap.add_argument("--list", action="store_true", help="列出可用数据集规格")
    args = ap.parse_args()

    if args.list:
        print("可用数据集：")
        for k, s in DATASET_SPECS.items():
            print("  %-14s %s/%s (split=%s, default_limit=%d, default_offset=%d) -> judge=%s"
                  % (k, s["dataset"], s["config"], s["split"], s["default_limit"], s["default_offset"], s["judge"]))
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.cases_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    all_cases = []
    for key in args.datasets.split(","):
        key = key.strip()
        if key not in DATASET_SPECS:
            print("警告：未知数据集 %s，跳过（可用：%s）" % (key, ",".join(DATASET_SPECS)))
            continue
        spec = DATASET_SPECS[key]
        lim = args.limit if args.limit > 0 else spec["default_limit"]
        off = args.offset if args.offset >= 0 else spec["default_offset"]
        print("[%s] 下载 %s/%s (split=%s, offset=%d, limit=%d) ..." % (
            key, spec["dataset"], spec["config"], spec["split"], off, lim))
        cases, raw_rows = fetch_dataset(spec, lim, start_offset=off, key_tag=key)
        print("  抽取 %d 条用例" % len(cases))
        all_cases.extend(cases)
        raw_path = os.path.join(args.raw_dir, "%s.jsonl" % key)
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in raw_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("  原始数据缓存 -> %s" % raw_path)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print("质量层测试文件已生成 -> %s（共 %d 条）" % (args.out, len(all_cases)))

    if args.all:
        perf_cases = build_performance_from_hf(args.raw_dir)
        perf_path = os.path.join(args.cases_dir, "performance.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(perf_cases, f, ensure_ascii=False, indent=2)
        print("性能层测试文件已生成 -> %s（共 %d 条）" % (perf_path, len(perf_cases)))

        compat_cases = build_compatibility_from_hf(args.raw_dir)
        compat_path = os.path.join(args.cases_dir, "compatibility.json")
        with open(compat_path, "w", encoding="utf-8") as f:
            json.dump(compat_cases, f, ensure_ascii=False, indent=2)
        print("兼容层测试文件已生成 -> %s（共 %d 条）" % (compat_path, len(compat_cases)))

        lc_cases = build_long_context_cases()
        lc_path = os.path.join(args.cases_dir, "long_context.json")
        with open(lc_path, "w", encoding="utf-8") as f:
            json.dump(lc_cases, f, ensure_ascii=False, indent=2)
        print("长上下文测试文件已生成 -> %s（共 %d 条）" % (lc_path, len(lc_cases)))


if __name__ == "__main__":
    main()
