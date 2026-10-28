# -*- coding: utf-8 -*-
"""缓存专项（--cache-only 独立报告）单元测试：python -m unittest discover -s tests -v"""
import json
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runner  # noqa: E402

TMP = os.path.join(runner.HERE, "tests", "_tmp_cache_report")


def cache_summary():
    return {
        "mode": "cache",
        "prefix_tokens": 2000,
        "runs": 5,
        "environment": {"env_tag": "EC2", "hostname": "ip-172-31-34-48"},
        "rows": [{
            "model_id": "claude-opus-5-5", "channel_id": "easyrouter", "channel_name": "EasyRouter",
            "cache_write_tokens": None,
            "same_prompt": [{"run": i, "cached_tokens": 0} for i in range(1, 6)],
            "prefix_reuse": {"cached_tokens": 0},
            "baseline": {"cached_tokens": 0},
            "session_id_supported": None,
            "repeat_hit": 0.0, "prefix_hit": False, "hit": False,
            "cache_type": "无缓存",
        }],
    }


class BuildCacheReportTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(TMP, exist_ok=True)
        self.path = os.path.join(TMP, "cache_summary.json")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(cache_summary(), f, ensure_ascii=False)

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_generates_standalone_report(self):
        out = runner.build_cache_report(TMP)
        self.assertIsNotNone(out)
        md_path = os.path.join(TMP, "cache_table.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, encoding="utf-8") as f:
            md = f.read()
        self.assertIn("claude-opus-5-5", md)
        self.assertIn("无缓存", md)
        self.assertIn("前缀 2000 token", md)
        try:
            import openpyxl
        except ImportError:  # pragma: no cover - 环境无 openpyxl 时降级
            return
        xlsx = os.path.join(TMP, "cache_report.xlsx")
        self.assertTrue(os.path.exists(xlsx))
        wb = openpyxl.load_workbook(xlsx)
        self.assertEqual(wb.sheetnames, ["缓存命中"])
        self.assertEqual(wb["缓存命中"]["A2"].value, "claude-opus-5-5")

    def test_missing_cache_summary_returns_none(self):
        os.remove(self.path)
        self.assertIsNone(runner.build_cache_report(TMP))


class CacheCliArgsTest(unittest.TestCase):
    def test_cache_only_flag_exists(self):
        # 参数存在性是回归保护：--cache-only 曾被误以为默认开启
        import argparse
        import inspect
        src = inspect.getsource(runner.main)
        self.assertIn('"--cache-only"', src)
        self.assertIn('"--cache"', src)
        self.assertIsInstance(argparse.ArgumentParser(), argparse.ArgumentParser)


if __name__ == "__main__":
    unittest.main()
