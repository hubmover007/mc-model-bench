# mc-model-bench：多渠道统一模型能力测试

统一对比「**同一模型 × 不同渠道**」的服务能力：同一套参数、同一批用例、相同顺序依次打向多个渠道，自动记录
TTFT / chunk 分布 / 生成速度 / 端到端延迟 / usage（含 reasoning_tokens、cached_tokens）/ 错误，
输出原始 JSON + 汇总对比表 + 带评分公式的 Excel 报告。

> 之前做法见 `kimi_k3_easyrouter_vs_mayi.py`（单文件脚本，2 渠道硬编码对比）。差距分析见 `docs/差距解读.md`，
> 完整代码解读见 `docs/代码解读.md`。

## 目录结构

```
mc-model-bench/
├── config/
│   ├── providers-template.json   # 配置模板（channels × models，复制为 providers.json 后填写真实 key）
│   └── providers.json            # 本地真实配置（含 api key，gitignore，不入库）
│   └── providers.mock.json       # 本地 mock 自检配置
├── test_cases/
│   ├── performance.json         # 性能基准层（4 条：短/长输入 × 短/长输出 四象限，HF Alpaca+CNN 语料）
│   ├── compatibility.json       # 功能兼容层（5 条：stream_sse / function_calling / json_mode / system_prompt / stop_param）
│   ├── quality.json             # 质量评测层（8 条：3 GSM8K 多步数学 + 3 TruthfulQA MC1 + 2 MMLU CS/Logic）
│   └── long_context.json        # 长上下文专项（3 条：大海捞针 10K@0.25 / 64K@0.75 / 128K@0.50）
├── probe_ttft.py                # 首包/TTFT 专项探针（单模型分阶段延迟 + 正反建议）
├── download_datasets.py         # 从 HuggingFace 下载抽取四层评测用例（支持 --all 全量刷新）
├── runner.py                    # 统一测试执行器
├── gen_report_template.py       # 生成报告模板 Excel（含公式）
├── report_template.xlsx         # 报告模板
├── export_report.py             # 运行结果回填模板 → 最终报告
├── mock_server.py               # 本地 OpenAI 兼容 mock（自检，无需真实 key）
├── docs/
│   ├── 差距解读.md              # 现在做法 vs 之前做法
│   ├── 代码解读.md              # 测试内容 / 代码逻辑 / 输出结果
│   └── 首包延迟测试说明.md      # TTFT 三口径定义 / 指标字典 / 正反建议 / 常见坑
└── kimi_k3_easyrouter_vs_mayi.py  # 之前的单文件脚本（保留作参考）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 复制配置模板为本地配置（模板入库，真实 providers.json 不入库）
#    cp config/providers-template.json config/providers.json
#    channels 写渠道，models 写模型（aliases 映射各渠道实际调用名）
export EASYROUTER_API_KEY=sk-...     # Windows: $env:EASYROUTER_API_KEY="sk-..."
export MAYI_API_KEY=sk-...

# 2. 从 HuggingFace 下载抽取评测用例（已自带 20 条非重合评测集；如需重新拉取执行 --all）
python download_datasets.py --all
python download_datasets.py --datasets gsm8k,truthfulqa,mmlu_cs,mmlu_logic --offset 35

# 3. 执行（--env-tag 标注测试环境并写入报告，本地/EC2 各跑一次即可区分）
python runner.py --sample --env-tag "本地笔记本"     # 示例模式：复用性能层第1条，每个组合只跑1次（6次）
python runner.py --list-cases                       # 查看用例
python runner.py --dry-run                          # 只看请求计划
python runner.py --once --env-tag "本地笔记本"       # 单次模式：全量用例每条只跑 1 次、不重试
python runner.py --env-tag "EC2 g5.xlarge"          # 全量执行（在 EC2 上跑时标注 EC2 环境）
python runner.py --models kimi-k3 --channels easyrouter --layers performance,quality

# 4. 生成报告（自动定位 output/ 下最新一次完整运行目录 run_<时间戳>/）
python export_report.py --out output --result 测试报告.xlsx

# 每次运行结果都会写入带时间戳的子目录（不会覆盖历史）：
#   output/run_20260826_172743/{raw/, order.json, summary.json, summary_table.md}
#   需要直接写到固定目录可用 --no-ts；--sample 结果写 output/run_<ts>/sample_summary.json

# 5. 本地自检（不消耗真实额度）
python mock_server.py 18080
python runner.py --sample --providers-file config/providers.mock.json --out output_mock
python runner.py --providers-file config/providers.mock.json --out output_mock
```

## 变量控制

1. **统一请求构造函数**：`runner.build_request_body()` 一处生成，渠道间只替换 `base_url/api_key/model`。
2. **统一测试文件**：四层用例独立 JSON；质量层由 `download_datasets.py` 从 HuggingFace 下载抽取。
3. **确定性长输入**：填充文本用「用例ID」作种子生成，全渠道 prompt 逐字一致。
4. **相同顺序**：用例固定顺序逐条依次发给所有渠道，时间戳写入 `output/order.json`。
5. **环境一致**：同一台机器顺序执行，主机/平台信息写入每个结果文件。

## 评分公式与权重设计（满分 100，权重可改）

- **性能得分（默认权重 `0.30`，建议范围 `0.25~0.35`）** = 平均(TTFT得分, 速度得分, E2E得分)：TTFT/E2E 越低分越高、速度越高分越高（全渠道最优=100）；同时单独披露 `content_ttft_ms`（内容首字延迟）与重试发生次数。
- **质量得分（默认权重 `0.30`，建议范围 `0.25~0.35`）** = 判分正确数/总数 × 100（GSM8K 多步数值精确匹配 + TruthfulQA MC1 选择题字母匹配 + MMLU CS/Logic 选择题字母匹配；若 `content` 为空且未截断则自动回退读取 `reasoning` 并单独计数“答案落错通道”）。
- **兼容得分（默认权重 `0.20`，建议范围 `0.15~0.25`）** = 通过数/总数 × 100（覆盖 `stream_sse`、`function_calling`、`json_mode`、`system_prompt`、`stop_param` 5 项协议契约）。
- **长上下文得分（默认权重 `0.20`，建议范围 `0.15~0.25`）** = 检索成功数/实际执行数 × 100（覆盖 10K@0.25、64K@0.75、128K@0.50 三档大海捞针，超标称上下文自动跳过不计分母）。
- **总评分** = `性能×0.3 + 兼容×0.2 + 质量×0.3 + 长上下文×0.2`

详见 `docs/代码解读.md` 与 `docs/差距解读.md`。

## 首包延迟专项（probe_ttft.py）

只测「首包」，把一次流式请求拆成 **DNS / TCP / TLS / 响应头 / 首SSE / 首token / 首content / 结束**
各阶段，多次采样给 P50/P95，并自动区分推理模型的「思考期」，用于判断慢在网络、网关排队还是模型 prefill。
纯标准库实现，零依赖。

```bash
export EASYROUTER_API_KEY=sk-...
python probe_ttft.py --model gemini-3.1-pro-preview --runs 20 --warmup 2
python probe_ttft.py --cases short,long --keepalive --runs 20        # 长连接对比
python probe_ttft.py --models gemini-3.1-pro-preview,gpt-5.2 --runs 20  # 基线对比
python probe_ttft.py --list-cases                                    # 查看内置用例
```

三个「首包」口径（**不要混用**）：

| 口径 | 字段 | 用户视角 |
|---|---|---|
| 首字节 | `ttfb_headers_ms` | 网关受理快不快 |
| 首 token（业界 TTFT） | `first_token_ms` | 服务端 prefill/排队快不快 |
| 首 content | `first_content_ms` | **用户真正看到文字要等多久** |

输出 `output_ttft/ttft_<时间戳>/{report.md, probe.json, report.csv}`，其中 `report.md` 含
阶段分解表、逐次明细、**自动诊断**与**正反建议**。完整方法论见
[`docs/首包延迟测试说明.md`](docs/首包延迟测试说明.md)。
