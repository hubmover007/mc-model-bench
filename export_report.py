#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 runner.py 的运行结果（output/raw/*.json + output/summary.json）回填到报告模板，
生成一份可直接交付的对比报告 Excel。

用法：
    python export_report.py
    python export_report.py --out output --template report_template.xlsx --result report.xlsx
"""

import argparse
import datetime
import json
import os

import openpyxl
from openpyxl.styles import Alignment

HERE = os.path.dirname(os.path.abspath(__file__))


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    if f + 1 >= len(s):
        return round(s[f], 2)
    return round(s[f] + (s[f + 1] - s[f]) * (k - f), 2)


def load_raw_rows(raw_dir):
    rows = []
    for layer in ("performance", "compatibility", "quality", "long_context"):
        d = os.path.join(raw_dir, layer)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    r = json.load(f)
                r["_layer"] = layer
                rows.append(r)
    return rows


def build_detail_rows(rows):
    perf, compat, quality, lc = [], [], [], []
    perf_ttft = {}
    for r in rows:
        layer = r.get("_layer")
        meta = r.get("meta", {})
        m = r.get("metrics") or {}
        pid = meta.get("provider_id", "")
        if layer == "performance":
            if r.get("skipped"):
                perf.append([meta.get("case_id"), meta.get("category"), meta.get("provider_name"),
                             "", "", "", "", "", "", "", "", "", "", "", "skipped", r.get("skip_reason", "")])
            elif r.get("error"):
                perf.append([meta.get("case_id"), meta.get("category"), meta.get("provider_name"),
                             "", "", "", "", "", "", "", "", "", "", "", r["error"].get("type"), r["error"].get("message", "")])
            else:
                cint = m.get("chunk_interval_ms") or {}
                perf.append([meta.get("case_id"), meta.get("category"), meta.get("provider_name"),
                             m.get("ttft_ms"), m.get("content_ttft_ms"), m.get("e2e_ms"), m.get("generation_ms"),
                             m.get("completion_tokens"), m.get("tokens_per_sec"), m.get("reasoning_tokens"),
                             m.get("cached_tokens"), m.get("chunk_count"), cint.get("avg"), cint.get("std"), "", ""])
                if m.get("ttft_ms") is not None:
                    perf_ttft.setdefault(pid, []).append(m["ttft_ms"])
        elif layer == "compatibility":
            v = r.get("validation") or {}
            checks = v.get("checks") or []
            if r.get("error"):
                compat.append([meta.get("case_id"), meta.get("compat_feature"), meta.get("provider_name"),
                               "失败", "请求失败: %s" % r["error"].get("message", "")[:150], "", "请求失败"])
            else:
                detail = "；".join("%s：%s" % (c.get("name"), c.get("detail")) for c in checks if c.get("severity") in ("fail", "warn")) or "正常"
                compat.append([meta.get("case_id"), meta.get("compat_feature"), meta.get("provider_name"),
                               "通过" if v.get("passed") else "失败", detail, m.get("completion_tokens"), ""])
        elif layer == "quality":
            v = r.get("validation") or {}
            checks = v.get("checks") or []
            out = (r.get("output") or {}).get("text") or ""
            reference = json.dumps(meta.get("reference"), ensure_ascii=False) if meta.get("reference") else ""
            detail = "；".join("%s：%s" % (c.get("name"), c.get("detail")) for c in checks)
            quality.append([meta.get("case_id"), meta.get("category"), meta.get("provider_name"),
                            "通过" if v.get("passed") else "失败", detail, reference, out[:120], ""])
        elif layer == "long_context":
            v = r.get("validation") or {}
            checks = v.get("checks") or []
            out = (r.get("output") or {}).get("text") or ""
            if r.get("skipped"):
                lc.append([meta.get("case_id"), meta.get("provider_name"), meta.get("target_tokens"), "跳过",
                           meta.get("depth"), "", "", r.get("skip_reason", "")])
            elif r.get("error"):
                lc.append([meta.get("case_id"), meta.get("provider_name"), meta.get("target_tokens"), "错误",
                           meta.get("depth"), "", "", r["error"].get("message", "")[:150]])
            else:
                detail = "；".join("%s：%s" % (c.get("name"), c.get("detail")) for c in checks)
                lc.append([meta.get("case_id"), meta.get("provider_name"), meta.get("target_tokens"),
                           "是" if v.get("passed") else "否", meta.get("depth"), out[:80], m.get("completion_tokens"), detail])
    return perf, compat, quality, lc, perf_ttft


def fill_sheet(ws, rows, num_fmts):
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for i, rv in enumerate(rows, 2):
        for c, v in enumerate(rv, 1):
            if v == "":
                v = None
            cell = ws.cell(row=i, column=c, value=v)
            if num_fmts and c <= len(num_fmts) and num_fmts[c - 1]:
                cell.number_format = num_fmts[c - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--template", default=os.path.join(HERE, "report_template.xlsx"))
    ap.add_argument("--result", default="")
    args = ap.parse_args()

    summary_path = os.path.join(args.out, "summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit("未找到 %s，请先运行 runner.py" % summary_path)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    rows = load_raw_rows(os.path.join(args.out, "raw"))
    perf, compat, quality, lc, perf_ttft = build_detail_rows(rows)

    wb = openpyxl.load_workbook(args.template)
    fill_sheet(wb["性能明细"], perf, [None, None, None, "0.0", "0.0", "0.0", "0.0", "0", "0.00", "0", "0", "0", "0.0", "0.0", None, None])
    fill_sheet(wb["兼容性明细"], compat, [None, None, None, None, None, "0", None])
    fill_sheet(wb["质量明细"], quality, [None, None, None, None, None, None, None, None])
    fill_sheet(wb["长上下文明细"], lc, [None, None, "0", None, "0.00", None, "0", None])

    ws = wb["汇总表"]
    providers = summary.get("providers", [])
    for i in range(2, 22):
        ws.cell(row=i, column=1).value = None
        ws.cell(row=i, column=3).value = None
        ws.cell(row=i, column=4).value = None
    for i, p in enumerate(providers):
        row = i + 2
        ws.cell(row=row, column=1, value=p.get("name"))
        ttfts = perf_ttft.get(p.get("id"), [])
        ws.cell(row=row, column=3, value=percentile(ttfts, 0.5))
        ws.cell(row=row, column=4, value=percentile(ttfts, 0.95))

    ws2 = wb["评分模型"]
    for i in range(2, 22):
        ws2.cell(row=i, column=1).value = None
    for i, p in enumerate(providers):
        ws2.cell(row=i + 2, column=1, value=p.get("name"))

    ws0 = wb["说明"]
    ws0["A2"] = "本次运行信息"
    info = "运行时间：%s ~ %s\n主机：%s | 平台：%s\n渠道数：%d | 输出目录：%s\n数据来源：runner.py 原始结果，已回填至本报告。" % (
        summary.get("started_at", ""), summary.get("finished_at", ""),
        (summary.get("environment") or {}).get("hostname", ""),
        (summary.get("environment") or {}).get("platform", ""),
        len(providers), os.path.abspath(args.out))
    ws0.cell(row=3, column=2, value=info).alignment = Alignment(vertical="top", wrap_text=True)

    result_path = args.result or os.path.join(HERE, "report_%s.xlsx" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    wb.save(result_path)
    print("已生成报告：%s" % os.path.abspath(result_path))


if __name__ == "__main__":
    main()
