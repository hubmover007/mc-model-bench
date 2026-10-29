# -*- coding: utf-8 -*-
"""缓存字段解析（Anthropic/Gemini 命名 + cache_control 断点）单元测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runner  # noqa: E402


class AnthropicNativeTest(unittest.TestCase):
    def test_build_body_with_breakpoint(self):
        b = runner.build_anthropic_message_body("claude-opus-5-5", "P" * 2000, "问题", 512)
        self.assertEqual(b["model"], "claude-opus-5-5")
        blocks = b["messages"][0]["content"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", blocks[1])
        self.assertEqual(blocks[1]["text"], "问题")
        self.assertEqual(b["max_tokens"], 512)

    def test_build_body_without_breakpoint(self):
        b = runner.build_anthropic_message_body("m", "P", "q", cache_control=False)
        self.assertNotIn("cache_control", b["messages"][0]["content"][0])

    def test_parse_usage_plain_fields(self):
        u = runner.parse_anthropic_usage({"input_tokens": 3062, "output_tokens": 20,
                                          "cache_creation_input_tokens": 2000,
                                          "cache_read_input_tokens": 0})
        self.assertEqual(u["prompt_tokens"], 3062)
        self.assertEqual(u["cache_write_tokens"], 2000)
        self.assertEqual(u["cached_tokens"], 0)

    def test_parse_usage_ttl_buckets(self):
        u = runner.parse_anthropic_usage({"input_tokens": 3000, "output_tokens": 5,
                                          "cache_creation": {"ephemeral_5m_input_tokens": 1800,
                                                             "ephemeral_1h_input_tokens": 200}})
        self.assertEqual(u["cache_write_tokens"], 2000)
        self.assertIsNone(u["cached_tokens"])

    def test_parse_usage_empty(self):
        self.assertIsNone(runner.parse_anthropic_usage(None)["cached_tokens"])

    def test_metrics_total_prompt_includes_cache(self):
        """原生 usage 的 input_tokens 只含未缓存部分，prompt_tokens 必须是三段之和。"""
        m = runner.anthropic_metrics({"input_tokens": 21, "cache_read_input_tokens": 3041,
                                      "cache_creation_input_tokens": 0, "output_tokens": 322}, e2e_ms=5231.5)
        self.assertEqual(m["prompt_tokens"], 3062)
        self.assertEqual(m["uncached_input_tokens"], 21)
        self.assertEqual(m["cached_tokens"], 3041)
        self.assertEqual(m["completion_tokens"], 322)
        self.assertEqual(m["e2e_ms"], 5231.5)

    def test_metrics_creation_counts_into_total(self):
        m = runner.anthropic_metrics({"input_tokens": 21, "cache_read_input_tokens": 0,
                                      "cache_creation": {"ephemeral_5m_input_tokens": 3041,
                                                         "ephemeral_1h_input_tokens": 0}})
        self.assertEqual(m["prompt_tokens"], 3062)
        self.assertEqual(m["cache_write_tokens"], 3041)


class ExtractUsageTest(unittest.TestCase):
    def test_openai_style(self):
        u = runner.extract_usage_fields({
            "prompt_tokens": 2000, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 1024},
        })
        self.assertEqual(u["cached_tokens"], 1024)
        self.assertIsNone(u["cache_write_tokens"])

    def test_anthropic_style(self):
        """easyrouter 等网关若原样透传 Anthropic usage，必须能识别 cache_read/cache_creation。"""
        u = runner.extract_usage_fields({
            "input_tokens": 2000, "output_tokens": 10,
            "cache_read_input_tokens": 1792, "cache_creation_input_tokens": 2000,
        })
        self.assertEqual(u["cached_tokens"], 1792)
        self.assertEqual(u["cache_write_tokens"], 2000)

    def test_gemini_style(self):
        u = runner.extract_usage_fields({"prompt_tokens": 2000, "cachedContentTokenCount": 1500})
        self.assertEqual(u["cached_tokens"], 1500)

    def test_input_tokens_details_style(self):
        u = runner.extract_usage_fields({"input_tokens_details": {"cached_tokens": 42}})
        self.assertEqual(u["cached_tokens"], 42)

    def test_missing_usage(self):
        u = runner.extract_usage_fields(None)
        self.assertIsNone(u["cached_tokens"])


class EasyRouterUsageTest(unittest.TestCase):
    """easyrouter 实际返回的 anthropic 语义 usage（2026-10-29 抓取）：缓存读写分别为 0。
    这类网关把 cached 读放进 prompt_tokens_details.cached_tokens，
    把 cache_creation 拆成 claude_cache_creation_{5m,1h}_tokens。"""

    USAGE = {
        "prompt_tokens": 3063, "completion_tokens": 104, "total_tokens": 3167,
        "usage_semantic": "openai", "usage_source": "anthropic",
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
        "input_tokens": 3063, "output_tokens": 0, "input_tokens_details": None,
        "claude_cache_creation_5_m_tokens": 0, "claude_cache_creation_1_h_tokens": 0,
    }

    def test_zero_is_not_none(self):
        u = runner.extract_usage_fields(self.USAGE)
        self.assertEqual(u["cached_tokens"], 0)
        self.assertEqual(u["cache_write_tokens"], 0)
        self.assertEqual(u["prompt_tokens"], 3063)

    def test_claude_creation_buckets_summed(self):
        usage = dict(self.USAGE, claude_cache_creation_5_m_tokens=1500,
                     claude_cache_creation_1_h_tokens=100)
        u = runner.extract_usage_fields(usage)
        self.assertEqual(u["cache_write_tokens"], 1600)
        self.assertEqual(u["cached_tokens"], 0)

    def test_real_read_field_beats_stale_zero_field(self):
        # 网关可能同时给 OpenAI 字段 0 和 Anthropic 真实读值 → 取 max，避免漏判命中
        usage = dict(self.USAGE)
        usage["prompt_tokens_details"] = {"cached_tokens": 0}
        usage["cache_read_input_tokens"] = 1792
        u = runner.extract_usage_fields(usage)
        self.assertEqual(u["cached_tokens"], 1792)

    def test_cache_field_paths(self):
        paths = runner.cache_field_paths(self.USAGE)
        self.assertIn("prompt_tokens_details.cached_tokens", paths)
        self.assertIn("claude_cache_creation_5_m_tokens", paths)
        self.assertIn("claude_cache_creation_1_h_tokens", paths)


class CacheFieldSeenTest(unittest.TestCase):
    def test_detects_nested(self):
        self.assertTrue(runner.usage_cache_field_seen(
            {"prompt_tokens_details": {"cached_tokens": 0}}))
        self.assertTrue(runner.usage_cache_field_seen({"cache_read_input_tokens": 0}))
        self.assertFalse(runner.usage_cache_field_seen({"prompt_tokens": 2000, "completion_tokens": 10}))
        self.assertFalse(runner.usage_cache_field_seen(None))


class AttachCacheControlTest(unittest.TestCase):
    def body(self, content):
        return {"model": "m", "messages": [{"role": "system", "content": "sys"},
                                           {"role": "user", "content": content}]}

    def test_rewrites_string_prompt(self):
        b = self.body("PREFIX\n\n问题")
        ok = runner.attach_cache_control(b, "PREFIX", "问题")
        self.assertTrue(ok)
        blocks = b["messages"][-1]["content"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["text"], "PREFIX")
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(blocks[1]["text"], "问题")
        self.assertNotIn("cache_control", blocks[1])
        # system 消息不受影响
        self.assertEqual(b["messages"][0]["content"], "sys")

    def test_already_blocks_untouched(self):
        b = self.body([{"type": "text", "text": "x"}])
        self.assertFalse(runner.attach_cache_control(b, "PREFIX", "问题"))

    def test_no_messages(self):
        self.assertFalse(runner.attach_cache_control({}, "P", "Q"))


if __name__ == "__main__":
    unittest.main()
