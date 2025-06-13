#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
index.py – 대한민국 법령 PDF → FAISS 인덱싱 파이프라인
(Elasticsearch 제외 버전, 2025‑06‑13)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# 설정 상수
# ────────────────────────────────────────────────────────────────────────────────
PDF_INPUT_DIR = "pdf_input"
FAISS_INDEX_PATH = "company_rules.faiss"
ARTICLES_DATA_PATH = "company_rules.json"

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 32

# ────────────────────────────────────────────────────────────────────────────────
# 로깅
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("legal_indexer.log", encoding="utf-8")],
)
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
    subparagraph_number: str
    item_number: str
    content: str
    chapter: str
    section: str
    subsection: str
    promulgation_date: str
    enforcement_date: str
    references: List[str]
    original_text: str
    page_number: int
    confidence_score: float
    _full_ref_cache: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._full_ref_cache = " ".join(
            filter(
                bool,
                [
                    self.law_name,
                    self.article_number,
                    self.paragraph_number,
                    self.subparagraph_number,
                    self.item_number,
                ],
            )
        )

    def get_full_reference(self) -> str:
        return self._full_ref_cache

    def get_hierarchical_structure(self) -> str:
        bits = []
        if self.chapter:
            bits.append(self.chapter)
        if self.section:
            bits.append(self.section)
        if self.subsection:
            bits.append(self.subsection)
        return " > ".join(bits)


# ────────────────────────────────────────────────────────────────────────────────
# 한국 법령 텍스트 파서 (원본과 동일)
# ────────────────────────────────────────────────────────────────────────────────
class KoreanLegalTextParser:
    """한국 법령 조문/항/호/목 단위 파서"""

    _PATTERNS = {
        "chapter": re.compile(r"제\s*(\d+)\s*(?:편|장)\s+(.+?)(?=\n|$)", re.M),
        "section": re.compile(r"제\s*(\d+)\s*절\s+(.+?)(?=\n|$)", re.M),
        "subsection": re.compile(r"제\s*(\d+)\s*관\s+(.+?)(?=\n|$)", re.M),
        "article": re.compile(
            r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\(\s*(.+?)\s*\))?\s*(.+?)(?=제\s*\d+\s*조|$)",
            re.S,
        ),
        "paragraph_num": re.compile(r"([①-⑳])\s*(.+?)(?=[①-⑳]|$)", re.S),
        "subparagraph": re.compile(r"(\d+)\.\s*(.+?)(?=\d+\.|$)", re.S),
        "item": re.compile(r"([가-힣])\.\s*(.+?)(?=[가-힣]\.|$)", re.S),
        "references": re.compile(
            r"제\s*(\d+(?:\s*의\s*\d+)?)\s*조(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?"
        ),
        "date": re.compile(r"(\d{4})[년.]\s*(\d{1,2})[월.]\s*(\d{1,2})일?"),
        "law_name": re.compile(r"^(.+?)(?:법|령|규칙|고시|훈령|예규|지침)", re.M),
    }

    _PARA_MAP = {s: f"제{i}항" for i, s in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", start=1)}

    # ---- 파싱 엔트리 ---------------------------------------------------------
    def parse_pdf_text(self, text: str, *, law_id: str = "", page_number: int = 1) -> List[LegalArticle]:
        meta = self._extract_metadata(text)
        structure_positions = self._extract_structure_positions(text)
        articles: List[LegalArticle] = []

        for art_match in self._PATTERNS["article"].finditer(text):
            article_num, art_sub, art_title, art_body = art_match.groups(default="")
            full_article_no = f"제{article_num}조" + (f"의{art_sub}" if art_sub else "")
            ctx = self._context_at(art_match.start(), structure_positions)

            for parsed in self._parse_article_body(art_body):
                refs = self._extract_references(parsed["content"])
                articles.append(
                    LegalArticle(
                        law_name=meta["law_name"] or law_id,
                        law_id=law_id,
                        article_number=full_article_no,
                        paragraph_number=parsed["paragraph"],
                        subparagraph_number=parsed["subpara"],
                        item_number=parsed["item"],
                        content=parsed["content"],
                        chapter=ctx["chapter"],
                        section=ctx["section"],
                        subsection=ctx["subsection"],
                        promulgation_date=meta["promulgation_date"],
                        enforcement_date=meta["enforcement_date"],
                        references=refs,
                        original_text=art_body,
                        page_number=page_number,
                        confidence_score=parsed["score"],
                    )
                )
        return articles

    # ---- 내부 헬퍼 (원본과 동일, 생략 없이 유지) ------------------------------
    def _extract_metadata(self, text: str) -> Dict[str, str]:
        m = self._PATTERNS["law_name"].search(text[:1000])
        law_name = m.group(1).strip() if m else ""
        dates = self._PATTERNS["date"].findall(text[:2000])
        promul = "-".join(dates[0]) if dates else ""
        enf = "-".join(dates[1]) if len(dates) > 1 else ""
        return {"law_name": law_name, "promulgation_date": promul, "enforcement_date": enf}

    def _extract_structure_positions(self, text: str) -> Dict[str, List[Tuple[int, str]]]:
        out: Dict[str, List[Tuple[int, str]]] = {"chapter": [], "section": [], "subsection": []}
        for key in ("chapter", "section", "subsection"):
            for m in self._PATTERNS[key].finditer(text):
                pos = m.start()
                label = "편" if key == "chapter" else "절" if key == "section" else "관"
                out[key].append((pos, f"제{m.group(1)}{label} {m.group(2)}"))
            out[key].sort(key=lambda x: x[0])
        return out

    def _context_at(self, pos: int, positions: Dict[str, List[Tuple[int, str]]]) -> Dict[str, str]:
        ctx = {"chapter": "", "section": "", "subsection": ""}
        for key, lst in positions.items():
            prev = [title for p, title in lst if p <= pos]
            ctx[key] = prev[-1] if prev else ""
        return ctx

    def _parse_article_body(self, body: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        para_ms = list(self._PATTERNS["paragraph_num"].finditer(body))
        if not para_ms:  # 항이 하나뿐
            return [{"paragraph": "제1항", "subpara": "", "item": "", "content": body.strip(), "score": 0.8}]

        for i, pm in enumerate(para_ms):
            start, end = pm.end(), (para_ms[i + 1].start() if i + 1 < len(para_ms) else len(body))
            para_text = body[start:end].strip()
            para_no = self._PARA_MAP.get(pm.group(1), f"제{i+1}항")
            sub_ms = list(self._PATTERNS["subparagraph"].finditer(para_text))
            if not sub_ms:
                results.append({"paragraph": para_no, "subpara": "", "item": "", "content": para_text, "score": 0.9})
                continue
            for j, sm in enumerate(sub_ms):
                sstart = sm.end()
                send = sub_ms[j + 1].start() if j + 1 < len(sub_ms) else len(para_text)
                stext = para_text[sstart:send].strip()
                sub_no = f"제{sm.group(1)}호"
                itm_ms = list(self._PATTERNS["item"].finditer(stext))
                if not itm_ms:
                    results.append(
                        {"paragraph": para_no, "subpara": sub_no, "item": "", "content": stext, "score": 0.95}
                    )
                else:
                    for im in itm_ms:
                        results.append(
                            {
                                "paragraph": para_no,
                                "subpara": sub_no,
                                "item": f"{im.group(1)}목",
                                "content": im.group(2).strip(),
                                "score": 1.0,
                            }
                        )
        return results

    def _extract_references(self, content: str) -> List[str]:
        refs = []
        for m in self._PATTERNS["references"].finditer(content):
            art, para, sub = m.groups(default="")
            ref = f"제{art}조" + (f" 제{para}항" if para else "") + (f" 제{sub}호" if sub else "")
            refs.append(ref)
        return sorted(set(refs))


# ────────────────────────────────────────────────────────────────────────────────
# PDF 로더
# ────────────────────────────────────────────────────────────────────────────────
class PDFLegalDocumentLoader:
    def __init__(self) -> None:
        self.parser = KoreanLegalTextParser()

    @staticmethod
    def _clean(txt: str) -> str:
        txt = re.sub(r"\n\s*\n", "\n", txt)
        txt = re.sub(r"\s+", " ", txt)
        txt = re.sub(r"-\s*\d+\s*-", " ", txt)  # 페이지 번호 제거
        return txt.strip()

    def extract(self, pdf_path: str) -> List[Tuple[str, int]]:
        try:
            doc = fitz.open(pdf_path)
            out = [(self._clean(doc.load_page(p).get_text()), p + 1) for p in range(doc.page_count)]
            doc.close()
            return [(t, n) for t, n in out if t]
        except Exception as e:
            logger.error(f"PDF 읽기 오류: {e}")
            return []

    def load(self, pdf_dir: str) -> List[LegalArticle]:
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"PDF 없음 – '{pdf_dir}' 대신 샘플 텍스트 사용")
            return self._sample()

        arts: List[LegalArticle] = []
        for pdf in tqdm(pdf_files, desc="PDF 파싱"):
            joined = "\n".join(t for t, _ in self.extract(str(pdf)))
            if not joined:
                logger.warning(f"{pdf.name} 텍스트 추출 실패")
                continue
            arts.extend(self.parser.parse_pdf_text(joined, law_id=pdf.stem))
        return arts

    def _sample(self) -> List[LegalArticle]:
        sample = "민법\n제750조(불법행위) 고의 또는 과실로... 책임이 있다."
        return self.parser.parse_pdf_text(sample, law_id="CIVIL_LAW")


# ────────────────────────────────────────────────────────────────────────────────
# 인덱싱 파이프라인 (ES 제거)
# ────────────────────────────────────────────────────────────────────────────────
class LegalDocumentIndexer:
    def __init__(self) -> None:
        self.articles: List[LegalArticle] = []
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def build(self, pdf_dir: str = PDF_INPUT_DIR) -> None:
        logger.info("=== 인덱싱 시작 (ES 없음) ===")
        # 1) PDF → 조문
        self.articles = PDFLegalDocumentLoader().load(pdf_dir)
        if not self.articles:
            raise RuntimeError("추출된 조문이 없습니다.")

        # 2) 임베딩
        embeddings = self.model.encode(
            [a.content for a in self.articles],
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        faiss.normalize_L2(embeddings)

        # 3) FAISS 저장
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        idx.add(embeddings)
        faiss.write_index(idx, FAISS_INDEX_PATH)
        logger.info(f"FAISS 저장 완료 → {FAISS_INDEX_PATH}")

        # 4) 메타 JSON 저장
        with open(ARTICLES_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in self.articles], f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 저장 완료 → {ARTICLES_DATA_PATH}")
        logger.info("=== 인덱싱 종료 ===")


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="법령 PDF → FAISS 인덱서 (ES 제외)")
    p.add_argument("mode", choices=["index"])
    p.add_argument("--pdf-dir", default=PDF_INPUT_DIR)
    args = p.parse_args()
    if args.mode == "index":
        LegalDocumentIndexer().build(args.pdf_dir)


if __name__ == "__main__":
    main()
