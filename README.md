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
| 自动测试 | `python -m unittest discover -s tests -v` | `20 passed` |

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

开发调试接口可打开 [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)。

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
