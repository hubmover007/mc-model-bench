# -*- coding: utf-8 -*-
"""缓存字段解析（Anthropic/Gemini 命名 + cache_control 断点）单元测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runner  # noqa: E402


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
