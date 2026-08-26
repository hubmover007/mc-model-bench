# mc-model-bench：多渠道统一模型能力测试

统一对比「**同一模型 × 不同渠道**」的服务能力：同一套参数、同一批用例、相同顺序依次打向多个渠道，自动记录
TTFT / chunk 分布 / 生成速度 / 端到端延迟 / usage（含 reasoning_tokens、cached_tokens）/ 错误，
输出原始 JSON + 汇总对比表 + 带评分公式的 Excel 报告。

> 之前做法见 `kimi_k3_easyrouter_vs_mayi.py`（单文件脚本，2 渠道硬编码对比）。差距分析见 `docs/差距解读.md`，
> 完整代码解读见 `docs/代码解读.md`。

## 目录结构

```
mc-model-bench/
├── config/providers.json        # 渠道 + 模型配置（channels × models 分离，只换 base_url/api_key/model）
├── test_cases/
│   ├── performance.json         # 性能基准层（12 条，四象限）
│   ├── compatibility.json       # 功能兼容层（11 条）
│   ├── quality.json             # 质量层（GSM8K + TruthfulQA，来自 HF / 兜底样例）
│   └── long_context.json        # 长上下文专项（大海捞针 10K/32K/64K/128K）
├── download_datasets.py         # 从 HuggingFace 下载抽取质量层用例（创建代码下载）
├── runner.py                    # 统一测试执行器
├── gen_report_template.py       # 生成报告模板 Excel（含公式）
├── report_template.xlsx         # 报告模板
├── export_report.py             # 运行结果回填模板 → 最终报告
├── mock_server.py               # 本地 OpenAI 兼容 mock（自检，无需真实 key）
├── docs/
│   ├── 差距解读.md              # 现在做法 vs 之前做法
│   └── 代码解读.md              # 测试内容 / 代码逻辑 / 输出结果
└── kimi_k3_easyrouter_vs_mayi.py  # 之前的单文件脚本（保留作参考）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 配置渠道 + 模型 + api key（只改 base_url / api_key / model 三项）
#    config/providers.json：channels 写渠道，models 写模型（aliases 映射各渠道实际调用名）
export EASYROUTER_API_KEY=sk-...     # Windows: $env:EASYROUTER_API_KEY="sk-..."
export MAYI_API_KEY=sk-...

# 2. 下载抽取 HuggingFace 质量层用例（可选，离线用自带样例）
python download_datasets.py --datasets gsm8k,truthfulqa --limit 50

# 3. 执行（--env-tag 标注测试环境并写入报告，本地/EC2 各跑一次即可区分）
python runner.py --sample --env-tag "本地笔记本"     # 示例模式：复用性能层第1条，每个组合只跑1次（6次）
python runner.py --list-cases                       # 查看用例
python runner.py --dry-run                          # 只看请求计划
python runner.py --once --env-tag "本地笔记本"       # 单次模式：全量用例每条只跑 1 次、不重试
python runner.py --env-tag "EC2 g5.xlarge"          # 全量执行（在 EC2 上跑时标注 EC2 环境）
python runner.py --models kimi-k3 --channels easyrouter --layers performance,quality

# 4. 生成报告
python export_report.py --out output --result 测试报告.xlsx

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

## 评分公式（满分 100，权重可改）

- 性能得分 = 平均(TTFT得分, 速度得分, E2E得分)：TTFT/E2E 越低分越高、速度越高分越高（全渠道最优=100）
- 兼容得分 = 通过数/总数 × 100
- 质量得分 = 判分正确数/总数 × 100（GSM8K 数值精确匹配 / TruthfulQA 标准答案命中）
- 长上下文得分 = 检索成功数/实际执行数 × 100（跳过不计分母）
- 总评分 = 性能×0.3 + 兼容×0.3 + 质量×0.2 + 长上下文×0.2

详见 `docs/代码解读.md` 与 `docs/差距解读.md`。
