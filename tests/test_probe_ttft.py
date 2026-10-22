# -*- coding: utf-8 -*-
"""probe_ttft.py 单元测试：python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_ttft as pt  # noqa: E402


class StatsTest(unittest.TestCase):
    def test_pct_edges(self):
        self.assertIsNone(pt.pct([], 0.5))
        self.assertEqual(pt.pct([42.0], 0.95), 42.0)
        vals = [10, 20, 30, 40, 50]
        self.assertEqual(pt.pct(vals, 0.5), 30)
        self.assertAlmostEqual(pt.pct(vals, 0.95), 48.0)

    def test_stat_block(self):
        b = pt.stat_block([100, 200, 300, None])
        self.assertEqual(b["n"], 3)
        self.assertEqual(b["p50"], 200)
        self.assertEqual(b["max"], 300)
        self.assertEqual(b["jitter_p95_minus_p50"], round(pt.pct([100, 200, 300], 0.95) - 200, 1))
        self.assertIsNone(pt.stat_block([None, None]))


class TimingsTest(unittest.TestCase):
    def rec(self, base=1000.0):
        return {
            "t0": base,
            "t_sent": base + 0.1,
            "t_hdr": base + 0.2,
            "t_first_sse": base + 0.25,
            "t_first_token": base + 0.3,
            "t_first_content": base + 0.9,
            "t_done": base + 2.0,
            "conn_phases": {"dns_ms": 10.0, "tcp_ms": 20.0, "tls_ms": 30.0},
        }

    def test_to_timings_offsets(self):
        t = pt.to_timings(self.rec())
        self.assertEqual(t["connect_ms"], 60.0)
        self.assertAlmostEqual(t["ttfb_headers_ms"], 100.0, places=0)
        self.assertAlmostEqual(t["first_token_ms"], 200.0, places=0)
        self.assertAlmostEqual(t["first_content_ms"], 800.0, places=0)
        self.assertAlmostEqual(t["cold_first_token_ms"], 300.0, places=0)
        self.assertAlmostEqual(t["reasoning_gap_ms"], 600.0, places=0)

    def test_error_timings(self):
        et = pt.error_timings({"conn_phases": {"dns_ms": 5, "tcp_ms": 15, "tls_ms": 0},
                               "t0": 100.0, "t_sent": 100.1, "t_hdr": 100.4})
        self.assertEqual(et["connect_ms"], 20.0)
        self.assertAlmostEqual(et["ttfb_headers_ms"], 300.0, places=0)
        self.assertAlmostEqual(et["total_ms"], 400.0, places=0)
        self.assertEqual(pt.error_timings({}), {})


class DiagnoseTest(unittest.TestCase):
    def summary(self, **over):
        def block(v):
            return {"n": 20, "min": v, "p50": v, "p90": v, "p95": v, "p99": v, "max": v,
                    "mean": v, "std": 0.0, "jitter_p95_minus_p50": 0.0}

        st = {
            "ttfb_headers_ms": block(300.0),
            "first_token_ms": block(800.0),
            "first_content_ms": block(900.0),
            "connect_ms": block(60.0),
            "cold_first_token_ms": block(860.0),
            "reasoning_gap_ms": block(100.0),
        }
        st.update(over)
        return {"stats": st, "total": 20, "ok": 20, "errors": 0, "cached_hits": 0,
                "truncated": 0, "reconnects": 0, "sse_issues": 0}

    def test_gateway_bottleneck_detected(self):
        out = "\n".join(pt.diagnose(self.summary(), self.summary()["stats"]))
        self.assertIn("网关", out)

    def test_reasoning_gap_warning(self):
        s = self.summary(first_content_ms={"n": 20, "min": 3000.0, "p50": 3000.0, "p90": 3000.0,
                                           "p95": 3000.0, "p99": 3000.0, "max": 3000.0,
                                           "mean": 3000.0, "std": 0.0, "jitter_p95_minus_p50": 0.0})
        out = "\n".join(pt.diagnose(s, s["stats"]))
        self.assertIn("reasoning", out)

    def test_no_nonsense_on_local_tiny_latency(self):
        tiny = {"n": 5, "min": 1.0, "p50": 1.5, "p90": 2.0, "p95": 2.0, "p99": 2.0,
                "max": 2.0, "mean": 1.6, "std": 0.2, "jitter_p95_minus_p50": 0.5}
        s = self.summary(ttfb_headers_ms=tiny, first_token_ms=tiny, first_content_ms=tiny,
                         connect_ms=tiny, cold_first_token_ms=tiny, reasoning_gap_ms=tiny)
        out = "\n".join(pt.diagnose(s, s["stats"]))
        self.assertIn("亚百毫秒", out)
        self.assertNotIn("瓶颈在网关", out)

    def test_cache_hit_flagged(self):
        s = self.summary()
        s["cached_hits"] = 5
        out = "\n".join(pt.diagnose(s, s["stats"]))
        self.assertIn("cached_tokens", out)


if __name__ == "__main__":
    unittest.main()
