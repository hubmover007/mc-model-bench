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
python runner.py --list-models                      # 查看 providers.json 里的「模型@渠道」组合与 id
python runner.py --dry-run                          # 只看请求计划
python runner.py --once --env-tag "本地笔记本"       # 单次模式：全量用例每条只跑 1 次、不重试
python runner.py --env-tag "EC2 g5.xlarge"          # 全量执行（在 EC2 上跑时标注 EC2 环境）
python runner.py --models kimi-k3 --channels easyrouter --layers performance,quality

# 4. 生成报告（自动定位 output/ 下最新一次完整运行目录 run_<时间戳>/）
python export_report.py --out output --result 测试报告.xlsx

# 4.1 缓存命中专项（可选，默认不跑；不加就**没有**「缓存命中」sheet）
python runner.py --env-tag "EC2" --cache            # 缓存 + 常规用例，合并进同一份 report.xlsx
python runner.py --cache-only                        # 只跑缓存（约 7 次请求/组合），出 cache_report.xlsx

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

## 缓存命中专项（--cache / --cache-only）

**默认不执行**：缓存测试会给每个「模型×渠道」组合多发 8 次请求（2000 token 长前缀连打 5 次 +
前缀复用 1 次 + 复用复测 1 次 + 基线 1 次），所以做成显式开关。不传就**不会**有缓存结论，`report.xlsx` 里也不会出现「缓存命中」sheet
（`说明` sheet 与 `summary_table.md` 第 8 节会标注「本次未启用缓存专项」）。

| 命令 | 行为 | 产出 |
|---|---|---|
| `python runner.py --cache` | 先跑缓存专项，再跑常规用例 | `report.xlsx` 多一个「缓存命中」sheet + `cache_summary.json` |
| `python runner.py --cache-only` | 只跑缓存，跳过 20 条常规用例（省额度，新跑一个模型时推荐） | `cache_report.xlsx`（单 sheet）+ `cache_table.md` |
| `python runner.py --cache-only --cache-control` | 长前缀以 content block + `cache_control: ephemeral` 发送（走 OpenAI 兼容 `/chat/completions`） | 同上（表里 `cache_control` 列为「是」） |
| `python runner.py --cache-only --cache-control --anthropic-native` | 同上，但走 **Anthropic 原生 `/messages`** 协议（easyrouter 支持），可真正验证断点是否生效 | 同上（协议列为「Anthropic 原生」） |

判定出的五种缓存类型：`前缀缓存` / `精确匹配缓存(无前缀复用)` / `反常(连打miss但前缀命中)` / `字段不可信` / `无缓存`。
表格里还有 **`缓存字段`**（渠道 usage 是否真的返回了 `cached_tokens` / `cache_read_input_tokens` / `cachedContentTokenCount`）
和 **`复用复测`**（同前缀 + 同新问题再发一次：首次 miss 但复测命中 → 缓存其实有效，只是网关按请求体路由到了不同上游节点）。

### 判定为「无缓存」时的三种可能（必须区分）

| 现象 | 原因 | 下一步 |
|---|---|---|
| 缓存字段 **未透传** | 渠道没返回任何缓存字段 → 测不到，不代表模型没有缓存能力 | 换直连官方 API，或换支持透传的渠道 |
| 透传但恒 0，且**没加** `--cache-control` | 请求方式没触发缓存：**Anthropic 系必须显式打 `cache_control` 断点**，纯文本 prompt 永远 0 | 加 `--cache-control` 重测 |
| 透传但恒 0，**已加** `--cache-control`（兼容层） | 网关收了参数但可能丢弃了断点 | 加 `--anthropic-native` 走原生 `/messages` 复测 |
| 透传但恒 0，**已加** `--cache-control --anthropic-native` | 原生也 0 → 上游未启用缓存 / 该模型不支持 prompt caching / 中转层剥离 | 找渠道方确认；用支持自动缓存的模型做阳性对照 |

`cache_summary.json` 会保存 run1 的**实发请求体** `request_body`，可直接确认断点有没有发出去；
`--cache-control` 若被网关 400/422 拒绝，会自动降级为纯文本重试并记 `cache_control_supported=false`。

> **实测结论（easyrouter / claude-opus-5-5，2026-10-29）**：OpenAI 兼容层 `--cache-control` → 全 0；
> 切到 `--anthropic-native` 后 run1 `cache_creation_input_tokens=3041`、run2~5 `cache_read_input_tokens=3041`
> （4/4 命中）→ **缓存能力本身没问题，是兼容层丢弃了 `cache_control` 断点**。原生模式下表里的
> `prompt_tokens` 是 `input_tokens + cache_read + cache_creation` 的总量（Anthropic 的 `input_tokens` 只含未缓存部分）。

> 对照建议：同一条缓存测试跑一个**支持自动前缀缓存的模型**（如 OpenAI 系）作为阳性对照。
> 它命中 → 说明测试链路没问题，问题在 claude/渠道；它也不命中 → 先查链路。

详见 [`docs/代码解读.md`](docs/代码解读.md) 第 3.6 节。

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
