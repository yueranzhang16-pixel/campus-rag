from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .retrieval import SearchResult

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
PROMPT_VERSION = "grounded-v1"


@dataclass
class GeneratedAnswer:
    content: str
    usage: dict[str, int]


def build_messages(question: str, evidence: list[SearchResult]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[来源：{item.source}｜章节：{item.context}]\n{item.text}" for item in evidence
    )
    return [
        {
            "role": "system",
            "content": (
                "你是课程知识库助手。只能依据给出的资料回答；资料不足时明确说“资料不足，无法确认”。"
                "回答简洁、准确；每个结论后以 [文件名] 标注来源。不要编造来源或外部知识。"
            ),
        },
        {"role": "user", "content": f"问题：{question}\n\n资料：\n{passages}"},
    ]


def check_answer(answer: str, expected_terms: list[str], expected_sources: list[str]) -> dict:
    """Cheap deterministic checks for regression tests; not a replacement for human review."""
    missing_terms = [term for term in expected_terms if term not in answer]
    cited_sources = [
        source
        for source in expected_sources
        if re.search(rf"\[(?:来源：)?{re.escape(source)}(?:\]|｜|/)", answer)
    ]
    return {
        "terms_pass": not missing_terms,
        "citation_pass": bool(cited_sources),
        "missing_terms": missing_terms,
        "cited_sources": cited_sources,
    }


@dataclass
class DeepSeekGenerator:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_environment(cls, model: str = DEFAULT_MODEL) -> "DeepSeekGenerator":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY。请在 PowerShell 中先设置该环境变量。")
        return cls(api_key=api_key, model=model)

    def answer(self, question: str, evidence: list[SearchResult]) -> str:
        return self.answer_with_usage(question, evidence).content

    def answer_with_usage(self, question: str, evidence: list[SearchResult]) -> GeneratedAnswer:
        payload = json.dumps(
            {"model": self.model, "messages": build_messages(question, evidence), "temperature": 0.2},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            # Some development environments inject an unusable localhost proxy.
            # Direct API access is the safe default; opt into a user-managed proxy
            # only when DEEPSEEK_USE_PROXY=1 is explicitly set.
            opener = urlopen if os.environ.get("DEEPSEEK_USE_PROXY") == "1" else build_opener(ProxyHandler({})).open
            with opener(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API 请求失败（HTTP {exc.code}）：{detail}") from exc
        usage = data.get("usage") or {}
        return GeneratedAnswer(
            content=data["choices"][0]["message"]["content"],
            usage={key: value for key, value in usage.items() if isinstance(value, int)},
        )
