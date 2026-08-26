#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 HuggingFace 下载并抽取评测数据，生成「统一测试文件」（test_cases/quality.json）。

设计：
  - 使用 HF datasets-server 的公开 REST API（https://datasets-server.huggingface.co/rows），
    零依赖、无需 HF Token，任何机器都可运行；
  - 抽取结果按「统一用例格式」写入 test_cases/quality.json，供 runner.py 直接执行；
  - 同时把原始数据缓存到 datasets/<name>.jsonl，便于审计与复现。

内置数据集规格：
  - gsm8k      openai/gsm8k          数学推理（judge=gsm8k_exact，答案取 answer 中 "#### " 之后的数字）
  - truthfulqa truthfulqa/truthful_qa 幻觉/事实性（judge=truthfulqa_any，命中 correct_answers 任一条）

用法：
    python download_datasets.py --list
    python download_datasets.py                              # 默认 gsm8k+truthfulqa，各 50 条
    python download_datasets.py --datasets gsm8k --limit 100
    python download_datasets.py --out test_cases/quality.json --raw-dir datasets

依赖：仅标准库（urllib）。离线环境可用仓库自带 test_cases/quality.json 作为兜底样例。
"""

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # datasets-server 每页最大 100 行

# 数据集规格：dataset / config / split / 字段抽取器
DATASET_SPECS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "name": "数学推理-GSM8K",
        "category": "math",
        "judge": "gsm8k_exact",
        "extract": lambda row: _extract_gsm8k(row),
    },
    "truthfulqa": {
        "dataset": "truthfulqa/truthful_qa",
        "config": "generation",
        "split": "validation",
        "name": "幻觉检测-TruthfulQA",
        "category": "hallucination",
        "judge": "truthfulqa_any",
        "extract": lambda row: _extract_truthfulqa(row),
    },
}


def _extract_gsm8k(row):
    """GSM8K：question 为题面，answer 形如 '...#### 18'，答案取 #### 之后数字。"""
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    m = re.search(r"####\s*([-+]?[\d,]+)", answer)
    reference = (m.group(1).replace(",", "") if m else "")
    return question, reference


def _extract_truthfulqa(row):
    """TruthfulQA(generation)：question 为题面，correct_answers 为标准答案列表。"""
    question = (row.get("question") or "").strip()
    correct = row.get("correct_answers") or []
    if isinstance(correct, str):
        correct = [correct]
    return question, [str(c).strip() for c in correct if str(c).strip()]


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


def fetch_dataset(spec, limit):
    """分页下载并抽取，返回统一用例列表 + 原始行列表。"""
    cases, raw_rows = [], []
    offset = 0
    while len(cases) < limit:
        qs = urllib.parse.urlencode({
            "dataset": spec["dataset"], "config": spec["config"],
            "split": spec["split"], "offset": offset, "length": PAGE,
        })
        data = http_get_json("%s?%s" % (API, qs))
        rows = data.get("rows") or []
        if not rows:
            break
        for r in rows:
            row = r.get("row") or {}
            raw_rows.append(row)
            question, reference = spec["extract"](row)
            if not question or not reference:
                continue
            cases.append({
                "id": "quality_%s_%04d" % (spec["dataset"].split("/")[-1], r.get("row_idx", len(cases))),
                "name": "%s-%d" % (spec["name"], r.get("row_idx", len(cases))),
                "category": spec["category"],
                "prompt": question,
                "reference": reference,
                "judge": spec["judge"],
                "max_tokens": 4096 if spec["category"] == "math" else 512,
                "stream": False,
            })
            if len(cases) >= limit:
                break
        offset += PAGE
        if offset >= (data.get("num_rows_total") or 0):
            break
    return cases, raw_rows


def main():
    ap = argparse.ArgumentParser(description="从 HuggingFace 下载抽取评测数据，生成统一测试文件")
    ap.add_argument("--datasets", default="gsm8k,truthfulqa",
                    help="逗号分隔的数据集 key：%s" % ",".join(DATASET_SPECS))
    ap.add_argument("--limit", type=int, default=50, help="每个数据集抽取条数")
    ap.add_argument("--out", default=os.path.join(HERE, "test_cases", "quality.json"))
    ap.add_argument("--raw-dir", default=os.path.join(HERE, "datasets"))
    ap.add_argument("--list", action="store_true", help="列出可用数据集规格")
    args = ap.parse_args()

    if args.list:
        print("可用数据集：")
        for k, s in DATASET_SPECS.items():
            print("  %-12s %s/%s (config=%s, split=%s) -> judge=%s"
                  % (k, s["dataset"], s["config"], s["config"], s["split"], s["judge"]))
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    all_cases = []
    for key in args.datasets.split(","):
        key = key.strip()
        if key not in DATASET_SPECS:
            print("警告：未知数据集 %s，跳过（可用：%s）" % (key, ",".join(DATASET_SPECS)))
            continue
        spec = DATASET_SPECS[key]
        print("[%s] 下载 %s/%s (split=%s) ..." % (key, spec["dataset"], spec["config"], spec["split"]))
        cases, raw_rows = fetch_dataset(spec, args.limit)
        print("  抽取 %d 条用例" % len(cases))
        all_cases.extend(cases)
        raw_path = os.path.join(args.raw_dir, "%s.jsonl" % key)
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in raw_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("  原始数据缓存 -> %s" % raw_path)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    print("统一测试文件已生成 -> %s（共 %d 条）" % (args.out, len(all_cases)))


if __name__ == "__main__":
    main()
