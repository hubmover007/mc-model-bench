# -*- coding: utf-8 -*-
"""判分逻辑单元测试：python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import download_datasets as dd  # noqa: E402
import runner  # noqa: E402


def res(text="", reasoning="", finish="stop"):
    return {"output": {"text": text, "reasoning": reasoning}, "metrics": {"finish_reason": finish}}


class PickAnswerTest(unittest.TestCase):
    def test_content_preferred(self):
        self.assertEqual(runner.pick_answer(res("A", "thinking"))[1], "content")

    def test_fallback_to_reasoning_when_content_empty(self):
        # 0827 报告里 deepseek@EasyRouter 的真实形态：答案落进 reasoning，content 空，finish=stop
        text, src = runner.pick_answer(res("", "7C9F-42KX-8QZ1"))
        self.assertEqual((text, src), ("7C9F-42KX-8QZ1", "reasoning"))

    def test_truncated_not_credited(self):
        self.assertEqual(runner.pick_answer(res("", "long thinking...", finish="length"))[1], "truncated")

    def test_empty(self):
        self.assertEqual(runner.pick_answer(res())[1], "empty")


class Gsm8kTest(unittest.TestCase):
    case = {"judge": "gsm8k_exact", "reference": "18"}

    def ok(self, out):
        return runner.judge_answer(self.case, out)[0]

    def test_variants_pass(self):
        for out in ("18", "$18", "$18.00", "18.0", "The answer is 18 dollars.", "Final answer: $18",
                    "#### 18", "\\boxed{18}", "\\boxed{\\$18}", "She makes $18 every day."):
            self.assertTrue(self.ok(out), out)

    def test_wrong_fails(self):
        for out in ("17", "The answer is 20", "", "no number here"):
            self.assertFalse(self.ok(out), out)

    def test_boxed_beats_trailing_number(self):
        self.assertTrue(self.ok("\\boxed{18}\n(the price was $2 per egg)"))


class McLetterTest(unittest.TestCase):
    case = {"judge": "mc_letter", "reference": "C"}

    def ok(self, out):
        return runner.judge_answer(self.case, out)[0]

    def test_variants_pass(self):
        for out in ("C", "C.", "(C)", "**C**", " C \n", "Answer: C", "The answer is C.",
                    "\\boxed{C}", "C. The watermelon seeds pass through", "Reasoning...\nFinal answer: C"):
            self.assertTrue(self.ok(out), out)

    def test_wrong_fails(self):
        for out in ("A", "B.", "Answer: D", "", "I am not sure"):
            self.assertFalse(self.ok(out), out)

    def test_lowercase_article_not_a_letter(self):
        self.assertIsNone(runner.extract_choice_letter("The correct answer is a lie")[0])


class ValidateCaseTest(unittest.TestCase):
    def test_quality_reasoning_fallback_marks_misplaced(self):
        case = {"judge": "gsm8k_exact", "reference": "18"}
        v = runner.validate_case(case, res("", "#### 18"))
        self.assertTrue(v["passed"])
        self.assertTrue(v["misplaced_answer"])
        self.assertTrue(any(c["name"] == "answer_channel" for c in v["checks"]))

    def test_quality_truncated_fails(self):
        v = runner.validate_case({"judge": "gsm8k_exact", "reference": "18"}, res("", "18", finish="length"))
        self.assertFalse(v["passed"])
        self.assertFalse(v["misplaced_answer"])

    def test_needle_in_reasoning_passes_but_flagged(self):
        case = {"needle": {"text": "x"}, "expect_contains": "7C9F-42KX-8QZ1"}
        v = runner.validate_case(case, res("", "7C9F-42KX-8QZ1"))
        self.assertTrue(v["passed"])
        self.assertTrue(v["misplaced_answer"])

    def test_needle_normal(self):
        case = {"needle": {"text": "x"}, "expect_contains": "7C9F-42KX-8QZ1"}
        v = runner.validate_case(case, res("7C9F-42KX-8QZ1", "thinking"))
        self.assertTrue(v["passed"])
        self.assertFalse(v["misplaced_answer"])

    def test_needle_miss_fails(self):
        case = {"needle": {"text": "x"}, "expect_contains": "7C9F-42KX-8QZ1"}
        self.assertFalse(runner.validate_case(case, res("unknown"))["passed"])


class AggregateTest(unittest.TestCase):
    def test_misplaced_counted(self):
        prov = [{"id": "p1", "name": "P1"}]
        rows = [{"meta": {"provider_id": "p1", "layer": "quality", "case_id": "q1", "category": "math"},
                 "metrics": {}, "validation": {"passed": True, "misplaced_answer": True}},
                {"meta": {"provider_id": "p1", "layer": "quality", "case_id": "q2", "category": "math"},
                 "metrics": {}, "validation": {"passed": True, "misplaced_answer": False}}]
        agg = runner.aggregate(rows, prov)["p1"]
        self.assertEqual(agg["misplaced_answers"], 1)
        self.assertEqual((agg["quality"]["passed"], agg["quality"]["total"]), (2, 2))

    def test_retry_tracking_counted(self):
        prov = [{"id": "p1", "name": "P1"}]
        rows = [
            {
                "meta": {"provider_id": "p1", "layer": "performance", "case_id": "perf_1"},
                "attempt": 2,
                "retry_count": 1,
                "retry_history": [{"attempt": 1, "kind": "rate_limit", "http_status": 429, "elapsed_ms": 120.0}],
                "metrics": {"ttft_ms": 300.0, "content_ttft_ms": 500.0, "e2e_ms": 900.0, "total_wall_ms": 6020.0, "tokens_per_sec": 45.0},
                "validation": {"passed": True},
            },
            {
                "meta": {"provider_id": "p1", "layer": "compatibility", "case_id": "cmp_1", "compat_feature": "json_mode"},
                "attempt": 3,
                "retry_count": 2,
                "error": {"type": "http_error", "http_status": 502, "message": "Bad Gateway",
                          "attempts": [{"attempt": 1, "kind": "http_error", "http_status": 502},
                                       {"attempt": 2, "kind": "http_error", "http_status": 502}]},
                "validation": {"passed": False},
            },
        ]
        agg = runner.aggregate(rows, prov)["p1"]
        self.assertEqual(agg["retried_cases"], 1)
        self.assertEqual(agg["retry_count"], 3)
        self.assertEqual(agg["performance"]["retried_cases"], 1)
        self.assertEqual(agg["performance"]["retry_count"], 1)
        self.assertEqual(agg["performance"]["content_ttft_ms"], [500.0])

    def test_dynamic_layer_description(self):
        cases = [
            {"layer": "performance", "id": "p1", "category": "短输入短输出"},
            {"layer": "compatibility", "id": "c1", "compat_feature": "json_mode"},
            {"layer": "quality", "id": "q1", "category": "math", "judge": "gsm8k_exact"},
            {"layer": "long_context", "id": "lc1", "target_tokens": 64000, "needle": {"depth": 0.75}},
        ]
        tc = runner.build_test_config({"score_weights": {"performance": 0.3, "compatibility": 0.2, "quality": 0.3, "long_context": 0.2}}, cases)
        self.assertIn("64K@depth=0.75", tc["layers"]["long_context"]["desc"])
        self.assertIn("json_mode", tc["layers"]["compatibility"]["desc"])

    def test_auto_migrate_providers_json(self):
        import argparse
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pfile = os.path.join(tmp, "providers.json")
            old_cfg = {
                "reasoning_max_tokens": 131072,
                "score_weights": {"performance": 0.3, "compatibility": 0.3, "quality": 0.2, "long_context": 0.2},
                "channels": [{"id": "c1", "base_url": "http://localhost"}],
                "models": [{"id": "m1", "aliases": {"c1": "m1"}}],
            }
            with open(pfile, "w", encoding="utf-8") as f:
                json.dump(old_cfg, f)
            args = argparse.Namespace(providers_file=pfile, cases_dir=os.path.join(runner.HERE, "test_cases"), layers="performance")
            cfg, _, _ = runner.load_config(args)
            self.assertEqual(cfg["score_weights"], {"performance": 0.3, "compatibility": 0.2, "quality": 0.3, "long_context": 0.2})
            self.assertEqual(cfg["reasoning_max_tokens"], 16384)
            self.assertTrue(os.path.exists(pfile + ".bak"))
            with open(pfile, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["score_weights"]["quality"], 0.3)
            self.assertEqual(saved["score_weights"]["compatibility"], 0.2)
            self.assertEqual(saved["reasoning_max_tokens"], 16384)


class MmluExtractTest(unittest.TestCase):
    def test_extract_mmlu(self):
        row = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
        prompt, ref = dd._extract_mmlu(row)
        self.assertEqual(ref, "B")
        self.assertIn("B. 4", prompt)


class TruthfulQaMcTest(unittest.TestCase):
    row = {"question": "What happens if you eat watermelon seeds?",
           "mc1_targets": {"choices": ["Seeds pass through", "You grow a watermelon", "You die", "You get sick"],
                           "labels": [1, 0, 0, 0]}}

    def test_deterministic_and_correct_letter_maps_to_correct_text(self):
        p1, c1 = dd._extract_truthfulqa_mc(self.row)
        p2, c2 = dd._extract_truthfulqa_mc(self.row)
        self.assertEqual((p1, c1), (p2, c2))
        line = [ln for ln in p1.splitlines() if ln.startswith(c1 + ". ")][0]
        self.assertIn("Seeds pass through", line)

    def test_invalid_rows_skipped(self):
        bad = {"question": "q", "mc1_targets": {"choices": ["a", "b"], "labels": [1, 1]}}
        self.assertEqual(dd._extract_truthfulqa_mc(bad), ("", ""))


if __name__ == "__main__":
    unittest.main()
