# Campus RAG：可评测的中文课程知识库问答

一个面向数据结构课程笔记的 RAG 学习项目：从 Markdown 文档中检索证据，再由 DeepSeek 生成带来源标注的中文回答。项目包含离线评测、混合检索、FastAPI 接口和中文网页。

## 项目亮点

- **可溯源回答**：每次回答都会返回对应的文档与章节证据。
- **混合检索**：结合 BGE 向量检索和 TF-IDF 关键词检索，兼顾语义表达与精确术语。
- **可评测**：提供检索评测集、答案质量评测和回归测试，而不是只看“感觉回答得不错”。
- **可运行网页**：FastAPI 提供 `/health`、`/retrieve`、`/answer` 接口和中文聊天界面。
- **CPU 可运行**：embedding 模型可在没有 GPU 的电脑上使用。

## 架构

```text
Markdown 课程笔记
      ↓ 切块（保留文件名与章节）
TF-IDF 检索 + BGE 向量检索
      ↓ RRF 融合与代码片段降权
前 k 条证据
      ↓
DeepSeek（仅依据证据回答）
      ↓
中文网页：答案 + 可展开证据
```

## 快速开始

要求：Python 3.10+。

```powershell
git clone https://github.com/yueranzhang16-pixel/campus-rag.git
cd campus-rag
python -m pip install -e .
```

先构建本地索引：

```powershell
python -m campus_rag.cli index --docs data/docs --output data/index.json
python -m campus_rag.cli embedding-index --docs data/docs --output data/embedding_index.json
```

仅检索证据（不调用大模型、不产生 API 费用）：

```powershell
python -m campus_rag.cli hybrid-query --index data/embedding_index.json --lexical-index data/index.json --question "线性探测再散列的增量序列是什么？"
```

设置 DeepSeek API Key 并启动网页：

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
$env:PYTHONPATH="src"
python -m uvicorn campus_rag.api:app --host 127.0.0.1 --port 8010 --reload
```

打开 [http://127.0.0.1:8010/](http://127.0.0.1:8010/) 使用中文问答页面。第一次提问会加载 embedding 模型，稍慢是正常的。

> 不要把 API Key 写入代码或提交到 GitHub；本项目已在 `.gitignore` 中忽略 `.env`。

## 质量评测

| 内容 | 命令 | 当前结果 |
| --- | --- | --- |
| 混合检索 | `hybrid-eval` | `recall@3 = 100% (10/10)` |
| 答案回归 | `answer-eval`（3 道已修复失败题） | `100% (3/3)` |
| 自动测试 | `python -m unittest discover -s tests -v` | `22 passed` |

运行检索评测：

```powershell
python -m campus_rag.cli hybrid-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/answer_eval_cases_day2.json --top-k 3 --report reports/hybrid.json
```

运行 DeepSeek 答案评测（每题会产生一次 API 调用费用）：

```powershell
python -m campus_rag.cli answer-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/answer_eval_cases_day2.json --report reports/answer_quality.json
```

每次增删或修改 `data/docs/` 的资料后，都必须重新构建两个索引；否则系统可能检索到旧资料。

## 接口

| 接口 | 作用 | 是否调用 DeepSeek |
| --- | --- | --- |
| `GET /health` | 检查服务是否启动 | 否 |
| `POST /retrieve` | 返回混合检索证据 | 否 |
| `POST /answer` | 返回回答与对应证据 | 是 |
| `GET /history` | 返回最近的本地问答记录 | 否 |
| `POST /feedback` | 记录“有帮助 / 有问题”反馈 | 否 |

开发调试接口可打开 [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)。

每次成功调用 `/answer` 后，系统会在本机 `logs/answer_history.jsonl` 追加记录问题、回答、来源和耗时。该目录已被 Git 忽略，可用于分析真实失败案例，不会上传到 GitHub。

网页中每条回答下方都有“有帮助 / 有问题”按钮；反馈会保存到本机 `logs/feedback.jsonl`，并通过回答编号关联对应的问答记录。

## 项目结构

```text
data/docs/                           # 课程 Markdown 笔记
data/answer_eval_cases_day2.json     # 第 2 天 10 题答案评测集
src/campus_rag/                      # 检索、生成、API、网页代码
tests/                               # 20 个自动测试
README.md                            # 项目说明
```

## 后续方向

- 扩充到更多真实课程资料和 20+ 道评测题。
- 记录问答日志、延迟和 API 成本。
- 加入用户反馈，持续沉淀失败案例并回归测试。

## 反馈闭环

每条回答都可以点赞或点踩。点踩时可选择“资料缺失、答非所问、内容错误、表达不清”，并可留下补充说明；这些记录只保存到本地 `logs/feedback.jsonl`，不会提交到 GitHub。

查看近期的反馈统计与待复盘问题（不调用 DeepSeek、不产生 API 费用）：

```powershell
$env:PYTHONPATH="src"
python -m campus_rag.cli feedback-report --history logs/answer_history.jsonl --feedback logs/feedback.jsonl --report reports/feedback_report.json
```

报告会把点踩记录关联回原问题、回答和检索来源。处理顺序是：先看“资料缺失”并补文档，再处理“答非所问”并检查检索，最后将修复过的问题加入评测集，避免后续改动导致回退。

## 请求追踪

从当前版本开始，每次 `/answer` 都会产生一个 `trace_id`（当前与 `answer_id` 相同），并在本地问答日志中记录：检索证据的排名与分数、检索器名称、模型名、提示词版本、索引哈希、模型返回的 token 用量（若 API 提供）和总耗时。

查看本地 Trace 概览（不调用 DeepSeek）：

```powershell
$env:PYTHONPATH="src"
python -m campus_rag.cli trace-report --history logs/answer_history.jsonl --report reports/trace_report.json
```

它会显示平均耗时、P95 耗时、最慢请求和缺少追踪字段的旧记录。一次回答出现问题时，优先按 Trace 判断：没有正确证据是检索或资料问题；证据正确但回答错误是生成或提示词问题；耗时异常则检查模型调用或向量模型首次加载。

## LLM 裁判评测

`judge-eval` 会对每个评测题执行两步：先生成带引用的回答，再让独立的裁判调用只根据“问题、回答、检索证据”评分。评分采用固定 1–5 量表：忠实于证据（45%）、回答相关性（25%）、引用支撑（20%）和资料不足时的拒答正确性（10%）。裁判必须先写明评分依据，再给出分数。

```powershell
$env:PYTHONPATH="src"
python -m campus_rag.cli judge-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/judge_eval_cases_day3.json --report reports/judge_eval_day3.json
```

当前评测集有 5 题，因此该命令会产生约 10 次 DeepSeek 调用（每题 1 次生成 + 1 次裁判）。可以用 `--judge-model` 指定不同于回答模型的裁判模型，以减轻“模型给自己高分”的偏差。

裁判分数不是事实真相：低置信度、低分或与人工反馈冲突的案例必须人工复核。每次修改裁判提示词都要保留版本号并重跑同一评测集，才能比较结果。

## Reranker 重排序实验

可选的生产检索链路为“混合召回 → Cross-Encoder 重排序”。混合检索先尽量找全候选片段；Reranker 再同时阅读“问题 + 候选片段”，调整候选的先后顺序。它通常能提升 `recall@1`，但会增加 CPU 延迟，因此必须和未重排版本在同一评测集上对照。

本地已缓存模型时，下面两条命令都不会调用 DeepSeek，也不会产生 API 费用：

```powershell
$env:PYTHONPATH="src"
python -m campus_rag.cli hybrid-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/answer_eval_cases_day2.json --top-k 1 --report reports/hybrid_top1.json
python -m campus_rag.cli hybrid-rerank-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/answer_eval_cases_day2.json --top-k 1 --candidate-k 10 --report reports/hybrid_rerank_top1.json
```

再对比两个报告中的 `recall@1`、`total_latency_ms` 和每题 `latency_ms`。若正确率没有提升，或 CPU 延迟明显不可接受，就不要把 Reranker 接入在线问答链路。

## 第 8～10 天工程交付

- 新增 16 道困难检索题，覆盖改写问法、公式、边界条件、定义、细节与概念对比；评测报告会按题型分别统计。
- 增加证据充分性门槛：首条证据得分不足时直接回复“资料不足，无法确认”，不调用 DeepSeek；英文专有术语在首条证据中直接出现时可作为窄范围例外。
- 增加 `abstention-eval`，同时报告拒答准确率、precision 和 recall。

第 8、9 天的真实结果及完整架构说明见 [docs/architecture.md](docs/architecture.md)。

```powershell
$env:PYTHONPATH="src"
python -m campus_rag.cli hybrid-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/retrieval_eval_cases_day8.json --top-k 1 --report reports/day8_hard_retrieval_top1.json
python -m campus_rag.cli abstention-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/abstention_eval_cases_day9.json --top-k 3 --report reports/day9_abstention.json
python -m unittest discover -s tests -v
```

## Parent-Child RAG

索引仍以段落级“子块”进行检索，以保持术语命中的精度；命中后，系统会一并传给模型该子块所在章节中的相邻正文（父级上下文）。这避免了只检索到“差异：”等标题、却遗漏后续定义或列表的碎片化问题。

每次升级文档切分算法或修改 `data/docs/` 后，都必须重建两套索引：

```powershell
$env:PYTHONPATH="src"
python -m campus_rag.cli index --docs data/docs --output data/index.json
python -m campus_rag.cli embedding-index --docs data/docs --output data/embedding_index.json
```

重建后需要重启服务，或在使用 `uvicorn --reload` 时等待它自动重载。网页的“查看检索证据”会显示命中片段及其相邻上下文。
