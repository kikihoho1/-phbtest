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
from collections import Counter, defaultdict
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re

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

# 간단한 속어/비격식 → 공식 용어 매핑
SYNONYM_MAP: Dict[str, str] = {
    "투잡": "겸직",
    "겹치기": "겸직",
    "성매매": "성매매",  # 동의어 예시 보존
    "뇌물": "금품수수",
}

def rewrite_query(text: str) -> str:
    out = text
    for k, v in SYNONYM_MAP.items():
        out = out.replace(k, v)
    return out

def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[\w]+", text.lower()) if t]

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


class BM25Search:
    """매우 단순한 BM25 구현"""

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs_len = []
        self.term_freqs: List[Counter] = []
        df: Counter = Counter()
        for text in docs:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.docs_len.append(len(tokens))
            df.update(tf.keys())
        self.avgdl = sum(self.docs_len) / max(1, len(self.docs_len))
        self.idf = {t: math.log((len(self.term_freqs) - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

    def search(self, query: str, k: int = TOP_K_SEARCH) -> List[int]:
        q_tokens = tokenize(query)
        scores = []
        for idx, tf in enumerate(self.term_freqs):
            score = 0.0
            dl = self.docs_len[idx]
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1) / denom)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:k]]


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
        self.bm25 = BM25Search([a.content for a in self.articles])
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
        rewritten = rewrite_query(question)
        v_idxs = [idx for idx, _ in self.searcher.search(rewritten, TOP_K_SEARCH)]
        b_idxs = self.bm25.search(rewritten, TOP_K_SEARCH)
        # 벡터 + BM25 결과 병합 (중복 제거, 순서 유지)
        idxs = []
        for i in v_idxs + b_idxs:
            if i not in idxs:
                idxs.append(i)
        idxs = self.reranker.rerank(rewritten, self.articles, idxs)
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
