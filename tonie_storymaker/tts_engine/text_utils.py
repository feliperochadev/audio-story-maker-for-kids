from __future__ import annotations
import re
from typing import List

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("…", "...")
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = re.split(r"(?<=[,;:])\s+|(?<=—)\s+|(?<=-)\s+", sentence)
    clauses = [c.strip() for c in clauses if c.strip()]
    chunks: List[str] = []
    cur = ""
    for clause in clauses:
        if len(cur) + len(clause) + 1 <= max_chars:
            cur = (cur + " " + clause).strip() if cur else clause
        else:
            if cur:
                chunks.append(cur)
                cur = ""
            if len(clause) <= max_chars:
                cur = clause
            else:
                words = clause.split()
                for word in words:
                    if len(cur) + len(word) + 1 <= max_chars:
                        cur = (cur + " " + word).strip() if cur else word
                    else:
                        if cur:
                            chunks.append(cur)
                        cur = word
    if cur:
        chunks.append(cur)
    return chunks

def split_into_chapters(
    text: str,
    max_chars: int = 2500,
    chapter_keywords: List[str] | None = None,
) -> List[str]:
    """
    Chapter-based split:
    - start a new chapter on headings like "Capítulo", "Capitulo", or "Chapter"
    - keep all text under that heading together
    """
    text = clean_text(text)
    lines = [line.strip() for line in text.split("\n")]
    chapters: List[str] = []
    current: List[str] = []
    preface: List[str] = []
    preface_sep = "..."
    seen_chapter = False

    def flush():
        if current:
            chapters.append(" ".join(current).strip())
            current.clear()

    keywords = [kw.strip() for kw in (chapter_keywords or []) if kw.strip()]
    pattern = None
    if keywords:
        escaped = [re.escape(k) for k in keywords]
        pattern = re.compile(rf"(?i)^({'|'.join(escaped)})\b")

    for line in lines:
        if not line:
            continue
        if pattern and pattern.match(line):
            if not seen_chapter and preface:
                current.extend(preface)
                current.append(preface_sep)
                preface.clear()
            flush()
            current.append(line)
            seen_chapter = True
            continue
        if not seen_chapter:
            preface.append(line)
        else:
            current.append(line)
    flush()

    if not chapters:
        combined = " ".join(preface).strip() if preface else ""
        return [combined or text] if text else []
    return chapters


def split_for_tts(text: str, max_chars: int) -> List[str]:
    text = clean_text(text)
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: List[str] = []
    cur = ""
    for sentence in sentences:
        for piece in _split_long_sentence(sentence, max_chars):
            if len(cur) + len(piece) + 1 <= max_chars:
                cur = (cur + " " + piece).strip() if cur else piece
            else:
                if cur:
                    chunks.append(cur)
                cur = piece
    if cur:
        chunks.append(cur)
    return chunks
