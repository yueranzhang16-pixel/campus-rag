# Campus RAG 架构与实验说明

## 1. 系统目标

Campus RAG 是一个面向数据结构课程笔记的中文问答系统。它不把模型当作知识来源：先从本地 Markdown 中检索证据，再让 DeepSeek 仅依据证据回答，并返回来源。项目重点是可验证的检索、拒答和评测闭环。

## 2. 在线链路

```text
用户问题
  │
  ├─ BGE 向量检索（语义召回）
  ├─ TF-IDF 检索（精确术语召回）
  │
  └─ RRF 融合 + 代码片段降权
           │
           ├─ 子块命中 + 父级相邻上下文扩展
           │
           └─ 证据充分性门槛
                 ├─ 分数足够，或英文专有术语在首条证据中直接出现
                 │      └─ DeepSeek 生成带来源的回答 → 校验来源是否来自本次证据
                 └─ 证据不足
                        └─ 直接返回“资料不足，无法确认”（不调用 API）
```

`Reranker` 已实现为可选实验链路：它在混合召回后用 Cross-Encoder 重排候选。当前机器没有其约 1.1 GB 权重，因此没有把它接入默认在线服务；必须先在困难集上证明收益，再决定是否下载和上线。

## 3. 为什么使用父子分块

检索单元是较短的段落子块，保证术语命中精确；交给生成模型时再附带所在章节的相邻父级正文。这样既不会把整篇笔记塞进提示词，也能避免只命中“差异：”标题、遗漏后续定义的碎片化问题。

## 4. 评测闭环

| 层级 | 评测内容 | 当前结果 |
| --- | --- | --- |
| 检索基线 | 10 道基础题 `recall@1` | 100% (10/10) |
| 困难检索集 | 16 道改写、公式、边界和对比题 `recall@1` | 100% (16/16) |
| 拒答门槛 | 4 道资料内 + 3 道库外问题 | 100% (7/7)，拒答 precision/recall 均为 100% |
| 自动回归测试 | API、检索、生成、评测与重排序模块 | 49 passed |

第 11～20 天继续补充了来源校验、检索诊断、稳态性能基准、`/ready` 与 GitHub Actions CI。当前自动回归测试为 **49 passed**；在本机以 16 道困难题、重复 2 次测得混合检索稳态中位延迟为 **148.135 ms**，P95 为 **161.930 ms**（不含 DeepSeek）。

这些结果只针对当前 6 份课程笔记与小规模人工标注集，不代表通用 RAG 能力。下一次增加资料、修改切分策略、改提示词或换模型后，都应重跑同一套评测，比较回归而不是凭主观感受判断。

## 5. 可复现实验

```powershell
$env:PYTHONPATH="src"

# 第 8 天：困难检索集（不调用 DeepSeek）
python -m campus_rag.cli hybrid-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/retrieval_eval_cases_day8.json --top-k 1 --report reports/day8_hard_retrieval_top1.json

# 第 9 天：资料外问题拒答（不调用 DeepSeek）
python -m campus_rag.cli abstention-eval --index data/embedding_index.json --lexical-index data/index.json --cases data/abstention_eval_cases_day9.json --top-k 3 --report reports/day9_abstention.json

# 自动回归测试（不调用 DeepSeek）
python -m unittest discover -s tests -v

# 第 13～15 天：检查一次查询的后端差异（不调用 DeepSeek）
python -m campus_rag.cli diagnose-query --index data/embedding_index.json --lexical-index data/index.json --question "B 树和 B+ 树有什么区别？" --report reports/diagnosis.json

# 第 19 天：测量预热后的检索延迟（不调用 DeepSeek）
python -m campus_rag.cli benchmark --index data/embedding_index.json --lexical-index data/index.json --cases data/retrieval_eval_cases_day8.json --top-k 3 --repeats 2 --report reports/benchmark.json
```

## 6. 当前边界与下一步

- 评测集规模仍小：应持续把用户点踩、错误回答和课程新增内容转成测试样本。
- 当前门槛由已知题与库外题校准；课程领域变化后需要重新校准，不应当作固定真理。
- Reranker、Query Rewrite、引用粒度和多文档推理是后续实验方向；每次只改变一个变量，并记录准确率、延迟和 API 成本。
- `citation_valid` 只验证来源标签是否来自本次检索，不能证明每一句话都被完整支持；高风险或重要结论仍应人工复核。
