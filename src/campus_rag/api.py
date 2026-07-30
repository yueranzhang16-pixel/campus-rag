from __future__ import annotations

import os
import time
from functools import cached_property
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cli import load_embedding_index, load_index
from .generation import DEFAULT_MODEL, DeepSeekGenerator
from .history import AnswerHistory
from .hybrid import HybridRetriever
from .retrieval import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_INDEX = PROJECT_ROOT / "data" / "embedding_index.json"
DEFAULT_LEXICAL_INDEX = PROJECT_ROOT / "data" / "index.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "logs" / "answer_history.jsonl"


class Retriever(Protocol):
    def search(self, question: str, top_k: int) -> list[SearchResult]: ...


class Generator(Protocol):
    def answer(self, question: str, evidence: list[SearchResult]) -> str: ...


class Evidence(BaseModel):
    source: str = Field(description="证据所在的 Markdown 文件名")
    context: str = Field(description="证据所在的文档章节")
    score: float = Field(description="混合检索分数，仅用于排序比较")
    text: str = Field(description="原始证据片段")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="要向知识库提出的问题", examples=["什么是顺序表？"])
    top_k: int = Field(default=3, ge=1, le=5, description="返回前几条证据，默认 3 条")


class RetrievalResponse(BaseModel):
    evidence: list[Evidence] = Field(description="按相关性排序的证据列表")


class AnswerResponse(RetrievalResponse):
    answer: str = Field(description="DeepSeek 基于证据生成的回答，包含来源标注")


class HistoryResponse(BaseModel):
    records: list[dict] = Field(description="最近的本地问答记录，按时间从新到旧排序")


class CampusRagService:
    """Lazily loads local indexes and the API client once per server process."""

    def __init__(
        self,
        embedding_index: Path = DEFAULT_EMBEDDING_INDEX,
        lexical_index: Path = DEFAULT_LEXICAL_INDEX,
        model: str = DEFAULT_MODEL,
        history_path: Path | None = DEFAULT_HISTORY_PATH,
    ):
        self.embedding_index = embedding_index
        self.lexical_index = lexical_index
        self.model = model
        self.history = AnswerHistory(history_path) if history_path else None

    @cached_property
    def retriever(self) -> HybridRetriever:
        if not self.embedding_index.is_file() or not self.lexical_index.is_file():
            raise FileNotFoundError(
                "未找到索引文件。请先运行 index 和 embedding-index 命令生成 data/index.json 与 data/embedding_index.json。"
            )
        return HybridRetriever(load_embedding_index(self.embedding_index), load_index(self.lexical_index))

    @cached_property
    def generator(self) -> DeepSeekGenerator:
        return DeepSeekGenerator.from_environment(model=self.model)

    def retrieve(self, question: str, top_k: int) -> list[SearchResult]:
        return self.retriever.search(question, top_k)

    def answer(self, question: str, top_k: int) -> tuple[str, list[SearchResult]]:
        started = time.perf_counter()
        evidence = self.retrieve(question, top_k)
        answer = self.generator.answer(question, evidence)
        if self.history:
            try:
                self.history.append(question, answer, evidence, (time.perf_counter() - started) * 1000)
            except OSError:
                pass
        return answer, evidence


def evidence_response(evidence: list[SearchResult]) -> list[Evidence]:
    return [Evidence(**item.__dict__) for item in evidence]


def create_app(service: CampusRagService | None = None) -> FastAPI:
    service = service or CampusRagService(
        embedding_index=Path(os.environ.get("CAMPUS_RAG_EMBEDDING_INDEX", DEFAULT_EMBEDDING_INDEX)),
        lexical_index=Path(os.environ.get("CAMPUS_RAG_LEXICAL_INDEX", DEFAULT_LEXICAL_INDEX)),
    )
    app = FastAPI(
        title="校园知识库问答 API",
        summary="基于混合检索与 DeepSeek 的可溯源问答服务",
        description=(
            "先从本地 Markdown 知识库中检索证据，再由 DeepSeek 根据证据回答。"
            "`/retrieve` 不调用大模型；`/answer` 会调用 DeepSeek 并产生少量 API 费用。"
            "问答记录仅保存在本机 logs/ 目录，不会上传。"
        ),
        version="0.1.0",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", summary="检查服务状态", description="确认 API 服务已成功启动；不会加载模型，也不会调用 DeepSeek。")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/history", response_model=HistoryResponse, summary="查看最近问答记录")
    def history(limit: int = Query(default=20, ge=1, le=50, description="返回最近几条本地记录")) -> HistoryResponse:
        return HistoryResponse(records=service.history.latest(limit) if service.history else [])

    @app.post(
        "/retrieve",
        response_model=RetrievalResponse,
        summary="检索相关证据",
        description="使用 embedding 与 TF-IDF 混合检索，返回资料片段；不调用 DeepSeek。",
    )
    def retrieve(request: QuestionRequest) -> RetrievalResponse:
        try:
            evidence = service.retrieve(request.question, request.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return RetrievalResponse(evidence=evidence_response(evidence))

    @app.post(
        "/answer",
        response_model=AnswerResponse,
        summary="根据证据生成回答",
        description="先检索证据，再调用 DeepSeek 生成带来源标注的回答；会产生 API 费用。",
    )
    def answer(request: QuestionRequest) -> AnswerResponse:
        try:
            text, evidence = service.answer(request.question, request.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AnswerResponse(answer=text, evidence=evidence_response(evidence))

    return app


app = create_app()
