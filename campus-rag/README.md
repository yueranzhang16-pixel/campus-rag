# Campus RAG: 可评测校园知识库问答原型

这是一个可评测的 RAG 学习项目。它先把检索链路、来源引用和离线评测做对，再逐步加入 embedding、混合检索、重排序和大模型生成。

> 当前版本使用基于 token 的 TF-IDF 检索，因此输出的是最相关证据，而不是编造式回答。这是有意的：先为之后的 LLM 方案建立可复现的质量基线。

## 快速开始

要求：Python 3.10+。

```powershell
cd campus-rag
python -m campus_rag.cli index --docs data/docs --output data/index.json
python -m campus_rag.cli query --index data/index.json --question "借书期限是多久？"
python -m campus_rag.cli eval --index data/index.json --cases data/eval_cases.json --report reports/baseline.json
python -m campus_rag.cli embedding-index --docs data/docs --output data/embedding_index.json
python -m campus_rag.cli embedding-eval --index data/embedding_index.json --cases data/eval_cases_targeted.json --report reports/embedding.json
python -m campus_rag.cli hybrid-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/eval_cases_targeted.json --report reports/hybrid.json
python -m campus_rag.cli rerank-eval --index data/embedding_index.json --cases data/eval_cases_targeted.json --report reports/reranked.json
python -m unittest discover -s tests -v
```

## 项目结构

```text
data/docs/          # 可替换为课程、规章或实验室资料
data/eval_cases.json # 人工标注的问答评测集
src/campus_rag/     # 索引、检索、CLI
tests/              # 回归测试
```

## 质量指标

`eval` 会输出 `recall@k`：标准答案所需来源是否出现在前 k 个检索结果中。传入 `--report` 后会保存逐题的预期来源、实际来源、命中状态和延迟。提交任何检索改动前，都应保存该报告，用相同评测集比较指标、查询延迟和失败案例。

每次新增、删除或修改 `data/docs/` 中的资料后，都必须重新生成 `data/index.json` 和 `data/embedding_index.json`；否则系统可能会检索到已经删除的旧资料。第 2 天的 10 道有效评测题保存在 `data/answer_eval_cases_day2.json`。

## 下一步（按顺序）

1. 用自己的真实资料替换示例文档，并扩充到至少 20 条评测问题。
2. 加入 BGE 或其他 embedding，实现与本基线的 `recall@k` 对比。
3. 加混合检索和 reranker，并将最佳证据交给本地或 API 大模型生成答案。
4. 用 FastAPI 暴露接口，记录延迟、错误和 token 成本。

## 架构

```text
文档 (.md/.txt) -> 按标题/段落切块 -> 附加文件名与章节上下文 -> TF-IDF 索引 -> top-k 证据
                                   ^                   |
                                   |--- 离线评测集 ------|
```

不要把 API Key 放进仓库；使用环境变量，并在 `.gitignore` 中忽略 `.env`。

## 语义检索（CPU 可运行）

`embedding-index` 使用 `BAAI/bge-small-zh-v1.5` 将每个片段编码为向量；`embedding-query` 比较问题向量和文档向量的余弦相似度。首次运行会下载开源模型，之后查询与评测强制使用本地缓存。它不调用付费 API，也不会训练模型。

请始终同时运行 TF-IDF 的 `eval` 和 embedding 的 `embedding-eval`，再根据同一份评测集中的 `recall@1`、`recall@3`、延迟和失败案例决定是否采用新方案。

## DeepSeek 证据式回答

先在 PowerShell 仅为当前窗口设置密钥（不要把密钥写入仓库）：

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
python -m campus_rag.cli answer --index data/embedding_index.json --question "什么是顺序表？"
python -m campus_rag.cli answer-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/answer_eval_cases.json --report reports/answer_quality.json
```

`answer` 和 `answer-eval` 先把本地 embedding 检索与 TF-IDF 关键词检索合并，再调用 DeepSeek 生成答案；提示词要求模型只能依据检索到的资料回答，并在结论后标注文件来源。`answer-eval` 会调用 API 共一次/题，因此会产生少量费用。

默认直连 DeepSeek，避免开发环境中失效的代理变量影响请求。只有你确实需要自己的网络代理时，才设置 `$env:DEEPSEEK_USE_PROXY="1"`。

## 两阶段检索：embedding + reranker

`rerank-eval` 先用 embedding 召回前 10 个候选片段，再由 `BAAI/bge-reranker-base` 对“问题—片段”逐对评分，输出最终前 k 个结果。首次下载需显式加 `--allow-download`；后续默认强制离线加载。它通常更准，但在 CPU 上明显更慢，因此必须和 embedding 基线一起测量，而不是默认采用。

## 混合检索：embedding + TF-IDF

`hybrid-query`、`hybrid-eval` 和 DeepSeek 回答链路都使用混合检索：语义向量检索擅长理解近义表达，TF-IDF 擅长抓住“线性探测再散列”这类精确术语。两边各召回一批候选，再用 RRF（倒数排名融合）合并排序。因此它不需要额外下载模型，也能补足纯 embedding 在术语题上的漏检。

```powershell
python -m campus_rag.cli hybrid-query --index data/embedding_index.json --lexical-index data/index.json --question "线性探测再散列的增量序列是什么？"
```

## 评测集格式

每题至少包含一个有效来源；若同一问题确实可由多份资料正确回答，可用 `expected_sources` 标注全部合法来源，避免将正确的交叉引用误判成失败。

```json
{
  "question": "树有哪些基本遍历方式？",
  "expected_sources": ["树.md", "数据结构.md"]
}
```

## FastAPI 服务

先生成本地索引，并在当前 PowerShell 设置 DeepSeek 密钥。然后启动服务：

```powershell
$env:PYTHONPATH="src"
uvicorn campus_rag.api:app --reload
```

打开 `http://127.0.0.1:8010/`，进入中文问答页面；它会显示回答和可展开的检索证据。开发调试时，也可以打开 `http://127.0.0.1:8010/docs` 调用接口。接口名、说明和参数已是中文；Swagger 自带的少量操作按钮仍会显示英文：

- `GET /health`：服务是否正常启动；不加载模型，不调用 API。
- `POST /retrieve`：只返回混合检索到的证据；不调用 DeepSeek。
- `POST /answer`：返回 DeepSeek 答案和它使用的证据；会产生 API 费用。

第一次访问 `/retrieve` 或 `/answer` 会加载本地 embedding 模型，稍慢是正常的；同一服务进程后的请求会复用模型和索引。
