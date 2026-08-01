from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .generation import DeepSeekGenerator
from .retrieval import SearchResult

JUDGE_PROMPT_VERSION = "grounded-rubric-v1"
CRITERIA = {
    "faithfulness": {"weight": 0.45, "description": "所有实质性结论都必须由给定证据支持，不能引入外部知识或与证据矛盾。"},
    "relevance": {"weight": 0.25, "description": "直接回答原问题；不因回答更长而加分。"},
    "citation_support": {"weight": 0.20, "description": "引用存在，且引用的文件能够支撑相应结论。"},
    "abstention": {"weight": 0.10, "description": "证据不足时明确说明资料不足；证据充分时不应无理由拒答。"},
}


def build_judge_messages(question: str, answer: str, evidence: list[SearchResult]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[来源：{item.source}｜章节：{item.context}]\n{item.text}" for item in evidence
    )
    criteria_text = "\n".join(
        f"- {name}（权重 {item['weight']}）：{item['description']}" for name, item in CRITERIA.items()
    )
    return [
        {
            "role": "system",
            "content": (
                "你是严格的课程知识库问答评测员。只能把‘检索证据’当作事实依据。"
                "先在 justification 中指出支持或不支持评分的具体证据，再评分；不要奖励篇幅、语气自信或位置。"
                "四项评分均为整数 1–5：1=严重不满足，3=部分满足，5=完全满足。"
                "仅输出 JSON，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：\n{question}\n\n回答：\n{answer}\n\n检索证据：\n{passages}\n\n"
                f"评分标准：\n{criteria_text}\n\n"
                "输出严格符合以下结构："
                '{"criteria":[{"name":"faithfulness","score":1,"justification":"先给证据理由","improvement":"具体改进"}],'
                '"confidence":0.0,"summary":"简短总结"}。'
                "criteria 必须恰好包含 faithfulness、relevance、citation_support、abstention 四项。"
            ),
        },
    ]


def parse_judgement(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("裁判没有返回有效 JSON") from exc
    criteria = result.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("裁判结果缺少 criteria")
    by_name = {item.get("name"): item for item in criteria if isinstance(item, dict)}
    if set(by_name) != set(CRITERIA):
        raise ValueError("裁判结果的评分维度不完整")
    normalized = []
    for name in CRITERIA:
        item = by_name[name]
        score = item.get("score")
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"裁判的 {name} 分数必须是 1–5 的整数")
        normalized.append(
            {
                "name": name,
                "score": score,
                "weight": CRITERIA[name]["weight"],
                "justification": str(item.get("justification", "")).strip(),
                "improvement": str(item.get("improvement", "")).strip(),
            }
        )
    confidence = result.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    weighted_score = sum(item["score"] * item["weight"] for item in normalized)
    return {
        "criteria": normalized,
        "weighted_score": round(weighted_score, 3),
        "confidence": max(0.0, min(float(confidence), 0.99)),
        "summary": str(result.get("summary", "")).strip(),
    }


@dataclass
class GroundedAnswerJudge:
    client: DeepSeekGenerator

    @classmethod
    def from_environment(cls, model: str) -> "GroundedAnswerJudge":
        return cls(client=DeepSeekGenerator.from_environment(model=model))

    def evaluate(self, question: str, answer: str, evidence: list[SearchResult]) -> dict:
        generated = self.client.complete(build_judge_messages(question, answer, evidence), temperature=0.0)
        result = parse_judgement(generated.content)
        result["judge_model"] = self.client.model
        result["judge_prompt_version"] = JUDGE_PROMPT_VERSION
        result["judge_usage"] = generated.usage
        return result
