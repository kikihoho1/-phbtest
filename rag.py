#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag.py – FAISS + Gemini 기반 RAG 챗봇 (Elasticsearch 제거)
2025‑06‑13
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()  # GOOGLE_API_KEY 등 환경변수 로드

# 필수 라이브러리
try:
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    sys.stderr.write(
        f"라이브러리 누락: {e}\n"
        "pip install faiss-cpu sentence-transformers langchain-google-genai python-dotenv\n"
    )
    sys.exit(1)

# ────────────────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "company_rules.faiss"
ARTICLES_DATA_PATH = "company_rules.json"

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
CROSS_ENCODER_MODEL = "kiyoungkim/ko-kce-cross-encoder"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TOP_K_SEARCH = 20
TOP_K_RERANK = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# 데이터 모델
# ────────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class LegalArticle:
    law_name: str
    law_id: str
    article_number: str
    paragraph_number: str
    content: str
    references: List[str]
    chapter: str = ""
    section: str = ""
    promulgation_date: str = ""
    enforcement_date: str = ""
    subparagraph_number: str = ""
    item_number: str = ""
    subsection: str = ""
    page_number: int = 0
    confidence_score: float = 1.0
    original_text: str = ""
    _ref_cache: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ref_cache = f"{self.law_name} {self.article_number} {self.paragraph_number}".strip()

    def get_full_reference(self) -> str:
        return self._ref_cache


# ────────────────────────────────────────────────────────────────────────────────
# 검색 (FAISS 전용)
# ────────────────────────────────────────────────────────────────────────────────
class VectorSearch:
    def __init__(self, index_path: str, model_name: str = EMBEDDING_MODEL_NAME):
        self.idx = faiss.read_index(index_path)
        self.embedder = SentenceTransformer(model_name)

    def search(self, query: str, k: int = TOP_K_SEARCH) -> List[Tuple[int, float]]:
        emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        scores, idxs = self.idx.search(emb, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]


# ────────────────────────────────────────────────────────────────────────────────
# 재순위화 (선택)
# ────────────────────────────────────────────────────────────────────────────────
class CrossReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        try:
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None
            logger.warning("Cross‑Encoder 로드 실패 → rerank 생략")

    def rerank(self, query: str, articles: List[LegalArticle], idxs: List[int], k: int = TOP_K_RERANK) -> List[int]:
        if not self.model:
            return idxs[:k]
        pairs = [(query, articles[i].content) for i in idxs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:k]]


# ────────────────────────────────────────────────────────────────────────────────
# LLM 응답 (Gemini)
# ────────────────────────────────────────────────────────────────────────────────
class AnswerGenerator:
    def __init__(self, api_key: Optional[str] = GOOGLE_API_KEY, model_name: str = "gemini-1.5-pro-latest"):
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            try:
                self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
                self.use_api = True
            except Exception as e:
                logger.warning(f"Gemini 초기화 실패: {e}")
                self.use_api = False
        else:
            logger.warning("GOOGLE_API_KEY 없음 – 템플릿 모드")
            self.use_api = False

    def generate(self, query: str, articles: List[LegalArticle]) -> str:
        if not articles:
            return "관련 규정/법령을 찾지 못했습니다."

        cites = "\n".join(f"- **{a.get_full_reference()}**: {a.content}" for a in articles)

        if not self.use_api:
            return f"[템플릿]\n{cites}"

        prompt = (
            "당신은 대한민국 법률 전문가 AI입니다.\n"
            "다음 [근거]만을 활용해 질문에 답하세요.\n\n"
            f"[질문]\n{query}\n\n[근거]\n{cites}"
        )
        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                return result.content.strip()
            return str(result).strip()
        except Exception as e:
            logger.error(f"Gemini 오류: {e}")
            return "[Gemini 호출 실패]"


# ────────────────────────────────────────────────────────────────────────────────
# RAG 시스템
# ────────────────────────────────────────────────────────────────────────────────
class LegalRAG:
    def __init__(self) -> None:
        if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(ARTICLES_DATA_PATH)):
            raise FileNotFoundError("index.py를 먼저 실행해 FAISS/JSON을 생성하세요.")

        # LegalArticle에 정의된 필드만 추출해서 객체 생성
        self.articles: List[LegalArticle] = [
            LegalArticle(**{k: v for k, v in d.items() if k in LegalArticle.__dataclass_fields__})
            for d in json.load(open(ARTICLES_DATA_PATH, encoding="utf-8"))
        ]

        self.searcher = VectorSearch(FAISS_INDEX_PATH)
        self.reranker = CrossReranker()
        self.generator = AnswerGenerator()

        # 참조 확대용 맵
        self._map: Dict[str, LegalArticle] = {a.get_full_reference(): a for a in self.articles}

    # (선택) 내부 참조 조항 확장
    def _expand(self, selected: List[LegalArticle]) -> List[LegalArticle]:
        queue = list(selected)
        out = {a.get_full_reference(): a for a in selected}
        while queue:
            a = queue.pop()
            for ref in a.references:
                key = f"{a.law_name} {ref}"
                if key in self._map and key not in out:
                    out[key] = self._map[key]
                    queue.append(self._map[key])
        return list(out.values())

    def ask(self, question: str) -> str:
        idxs = [idx for idx, _ in self.searcher.search(question, TOP_K_SEARCH)]
        idxs = self.reranker.rerank(question, self.articles, idxs)
        answer_articles = self._expand([self.articles[i] for i in idxs])
        return self.generator.generate(question, answer_articles)


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("🏛️  법령 RAG 챗봇 (ES 없음) – quit/exit로 종료")
    rag = LegalRAG()
    while True:
        q = input("\n질문> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print("\n[답변]\n" + rag.ask(q))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료합니다.")
