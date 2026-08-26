#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 OpenAI 兼容 mock 服务，用于在不消耗真实 API 额度的情况下验证 runner.py 全流程。
模拟一个「推理模型」：流式输出 reasoning_content + content，usage 带 reasoning_tokens / cached_tokens。

用法：
    python mock_server.py [端口]          # 默认 18080
    python runner.py --providers-file config/providers.mock.json --max-cases 3 --verbose
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("stream"):
            self._stream(body)
        else:
            self._json(body)

    def _compose(self, body):
        msgs = body.get("messages", [])
        last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        lu = last_user.lower()

        # 工具调用
        if body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_mock_1", "type": "function",
                "function": {"name": name, "arguments": json.dumps({"city": "北京"}, ensure_ascii=False)}}]}
        # JSON mode
        if body.get("response_format") and body["response_format"].get("type") == "json_object":
            return {"role": "assistant", "content": json.dumps(
                {"name": "张三", "age": 28, "city": "上海", "occupation": "工程师"}, ensure_ascii=False)}
        # 系统提示词
        sys_msgs = [m for m in msgs if m.get("role") == "system"]
        if sys_msgs and "[TRANSLATE]" in sys_msgs[0]["content"]:
            return {"role": "assistant", "content": "[TRANSLATE] Hello, world!"}
        # 多轮
        if "名字" in last_user or "猫" in last_user:
            return {"role": "assistant", "content": "它的名字是咪咪。"}
        # stop
        if "STOP" in last_user or "数到" in last_user:
            return {"role": "assistant", "content": "1\n2\n3\n4\n5"}
        # 质量判分：命中已知参考（仅用于冒烟验证 judge 流程）
        if "janet" in lu:
            return {"role": "assistant", "content": "18"}
        if "robe" in lu or "bolts" in lu:
            return {"role": "assistant", "content": "3"}
        if "josh" in lu:
            return {"role": "assistant", "content": "70000"}
        if "polar bear" in lu:
            return {"role": "assistant", "content": "Polar bears live in the Arctic."}
        if "watermelon" in lu:
            return {"role": "assistant", "content": "Nothing happens; the seeds pass through your body."}
        if "smallest country" in lu:
            return {"role": "assistant", "content": "Vatican City is the smallest country."}
        return {"role": "assistant", "content": "这是一条用于测试的固定回复内容，用于验证多渠道测试脚本的数据采集与指标计算是否正确。"}

    def _usage(self, content, reasoning):
        comp = (len(content or "") + len(reasoning or "")) // 2 or 1
        return {"prompt_tokens": 128, "completion_tokens": comp, "total_tokens": 128 + comp,
                "prompt_tokens_details": {"cached_tokens": 32},
                "completion_tokens_details": {"reasoning_tokens": len(reasoning or "") // 2}}

    def _json(self, body):
        msg = self._compose(body)
        content = msg.get("content") or ""
        reasoning = "好的，我先思考一下这个问题。"  # 模拟 reasoning_content
        resp = {"id": "mock-1", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "message": {**msg, "reasoning_content": reasoning}, "finish_reason": "stop"}],
                "usage": self._usage(content, reasoning)}
        data = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, body):
        msg = self._compose(body)
        content = msg.get("content") or ""
        reasoning = "好的，我先思考一下这个问题。"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(delta, finish=None):
            evt = {"id": "mock-1", "object": "chat.completion.chunk", "created": int(time.time()),
                   "model": body.get("model"), "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(("data: %s\n\n" % json.dumps(evt, ensure_ascii=False)).encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.005)

        step = 5
        for i in range(0, len(reasoning), step):
            emit({"reasoning_content": reasoning[i:i + step]})
        for i in range(0, len(content), step):
            emit({"content": content[i:i + step]})
        emit({}, "stop")
        self.wfile.write(("data: %s\n\n" % json.dumps({"usage": self._usage(content, reasoning)}, ensure_ascii=False)).encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
