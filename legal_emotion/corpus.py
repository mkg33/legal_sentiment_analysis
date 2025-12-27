from __future__ import annotations
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
from transformers import PreTrainedTokenizerBase

_PARA_SPLIT_RE = re.compile('\\n\\s*\\n+')
_DATE_RE = re.compile('^\\d{4}-\\d{2}-\\d{2}$')
_ICJ_HEADER_RE = re.compile(
    '\\b(international court of justice|court internationale de justice|reports of judgments|advisory opinions and orders|recueil des arrets|avis consultatifs|ordonnances?|case concerning|application of|order of|ordonnance du|judgment of|judgement of|table of contents)\\b',
    re.IGNORECASE,
)
_ICJ_CASE_MARKER_RE = re.compile(
    '\\b(v\\.|c\\.)\\b',
    re.IGNORECASE,
)
_ROMAN_RE = re.compile(
    '^(?=[ivxlcdm]+$)[ivxlcdm]+$',
    re.IGNORECASE,
)
_TOC_LINE_RE = re.compile(
    '(?:\\.{2,}|\\s+)(\\d{1,4}|[ivxlcdm]{1,8})\\s*$',
    re.IGNORECASE,
)
_PAGE_LABEL_RE = re.compile(
    '^(page|pp?\\.?)\\s*\\d{1,4}$',
    re.IGNORECASE,
)


def _strip_diacritics(text: str) -> str:
    norm = unicodedata.normalize(
        'NFKD',
        text,
    )
    return ''.join((ch for ch in norm if not unicodedata.combining(ch)))


def _normalise_icj_line(line: str) -> str:
    if not line:
        return ''
    clean = _strip_diacritics(line)
    clean = re.sub(
        '\\s+',
        ' ',
        clean,
    ).strip().lower()
    return clean


def _upper_ratio(line: str) -> float:
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return 0.0
    return float(sum((1 for ch in letters if ch.isupper()))
        / float(len(letters)))


def _is_page_number(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.isdigit():
        return True
    if _ROMAN_RE.match(stripped) and len(stripped) <= 8:
        return True
    return False


def _toc_like_line(line: str) -> bool:
    if _TOC_LINE_RE.search(line):
        return True
    if '..' in line and line.rstrip().endswith(('.', ')')):
        return True
    if re.match(
        '^\\d{1,3}[\\.\\)]\\s+',
        line,
    ):
        return True
    return False


def strip_icj_boilerplate(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    normalised = [
        _normalise_icj_line(line) for line in lines
    ]
    counts = Counter((n for n in normalised if n and len(n) >= 6))
    repeated = {n for n, c in counts.items() if c >= 3}
    out_lines: List[str] = []
    in_toc = False
    toc_blank = 0
    early_caps_limit = 80
    for idx, (line, norm) in enumerate(zip(
            lines,
            normalised,
        )):
        stripped = line.strip()
        if not stripped:
            if in_toc:
                toc_blank += 1
                if toc_blank >= 2:
                    in_toc = False
            out_lines.append(line)
            continue
        toc_blank = 0
        if _is_page_number(stripped):
            continue
        if _PAGE_LABEL_RE.match(stripped):
            continue
        alpha_count = sum((1 for ch in line if ch.isalpha()))
        upper_ratio = _upper_ratio(line)
        is_caps = upper_ratio >= 0.7 and alpha_count >= 6
        if norm.startswith('table of contents'):
            in_toc = True
            continue
        if in_toc:
            if _toc_like_line(stripped):
                continue
            in_toc = False
        skip_early_caps = idx < early_caps_limit and is_caps
        skip_repeat = is_caps and norm in repeated
        skip_header = is_caps and (
            _ICJ_HEADER_RE.search(norm)
            or _ICJ_CASE_MARKER_RE.search(norm)
        )
        skip_title = is_caps and re.match(
            '^\\d{1,4}\\s+[A-Z]',
            stripped,
        )
        if (
            skip_early_caps
            or skip_repeat
            or skip_header
            or skip_title
        ):
            continue
        out_lines.append(line)
    return '\n'.join(out_lines)


def iter_text_paths(
    input_path: str | Path,
    *,
    recursive: bool = True,
    suffix: str = '.txt',
) -> Iterator[Path]:
    path = Path(input_path)
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError('error: FileNotFoundError')
    glob = '**/*' if recursive else '*'
    for p in sorted(path.glob(glob)):
        if p.is_file() and p.suffix.lower() == suffix:
            yield p


def read_text(
    path: Path,
    *,
    strip_icj: Optional[bool] = None,
) -> str:
    text = path.read_text(
        encoding='utf-8',
        errors='ignore',
    )
    if strip_icj is None:
        if path.stem.startswith('ICJ_'):
            return strip_icj_boilerplate(text)
    elif strip_icj:
        return strip_icj_boilerplate(text)
    return text


def split_paragraphs(text: str) -> List[str]:
    paras = [
        p.strip()
        for p in _PARA_SPLIT_RE.split(text)
        if p.strip()
    ]
    if paras:
        return paras
    stripped = text.strip()
    return [stripped] if stripped else []


def chunk_paragraphs(
    paragraphs: List[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    stride: int = 0,
) -> List[str]:
    if max_length <= 0:
        raise ValueError('error: ValueError')
    if stride < 0:
        raise ValueError('error: ValueError')
    special = tokenizer.num_special_tokens_to_add(pair=False)
    max_body = max(
        1,
        max_length - special,
    )
    out: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            out.append('\n\n'.join(buf).strip())
            buf = []
            buf_len = 0

    for para in paragraphs:
        ids = tokenizer(
            para,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )['input_ids']
        para_len = len(ids)
        if para_len <= max_body:
            if buf and buf_len + para_len > max_body:
                flush()
            buf.append(para)
            buf_len += para_len
            continue
        flush()
        if para_len == 0:
            continue
        step = (
            max(
                1,
                max_body - stride,
            )
            if stride > 0
            else max_body
        )
        for start in range(
            0,
            para_len,
            step,
        ):
            chunk_ids = ids[start : start + max_body]
            if not chunk_ids:
                break
            out.append(tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                ).strip())
            if start + max_body >= para_len:
                break
    flush()
    return [c for c in out if c]


@dataclass(frozen=True)
class TextChunk:
    text: str
    body_tokens: int
    new_body_tokens: int


def chunk_paragraphs_with_token_counts(
    paragraphs: List[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    stride: int = 0,
) -> List[TextChunk]:
    if max_length <= 0:
        raise ValueError('error: ValueError')
    if stride < 0:
        raise ValueError('error: ValueError')
    special = tokenizer.num_special_tokens_to_add(pair=False)
    max_body = max(
        1,
        max_length - special,
    )
    out: List[TextChunk] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        text = '\n\n'.join(buf).strip()
        buf = []
        buf_len = 0
        if not text:
            return
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )['input_ids']
        body_len = int(len(ids))
        if body_len <= 0:
            return
        out.append(TextChunk(
                text=text,
                body_tokens=body_len,
                new_body_tokens=body_len,
            ))

    for para in paragraphs:
        ids = tokenizer(
            para,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )['input_ids']
        para_len = int(len(ids))
        if para_len <= max_body:
            if buf and buf_len + para_len > max_body:
                flush()
            buf.append(para)
            buf_len += para_len
            continue
        flush()
        if para_len == 0:
            continue
        step = (
            max(
                1,
                max_body - stride,
            )
            if stride > 0
            else max_body
        )
        overlap = max_body - step
        for start in range(
            0,
            para_len,
            step,
        ):
            chunk_ids = ids[start : start + max_body]
            if not chunk_ids:
                break
            text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
            ).strip()
            body_len = int(len(chunk_ids))
            if not text or body_len <= 0:
                continue
            if start == 0:
                new_len = body_len
            else:
                new_len = max(
                    1,
                    body_len - overlap,
                )
            out.append(TextChunk(
                    text=text,
                    body_tokens=body_len,
                    new_body_tokens=int(new_len),
                ))
            if start + max_body >= para_len:
                break
    flush()
    return out


def parse_icj_meta(path: Path) -> Dict[str, str]:
    stem = path.stem
    parts = stem.split('_')
    meta: Dict[str, str] = {'stem': stem}
    if len(parts) >= 2 and parts[0] == 'ICJ':
        meta['source'] = 'ICJ'
        meta['case_id'] = parts[1]
        for i, p in enumerate(parts):
            if _DATE_RE.match(p):
                meta['date'] = p
                if i + 1 < len(parts):
                    meta['doc_type'] = parts[i + 1]
                break
        return meta
    try:
        parts_path = path.parts
        i = parts_path.index('crystal_ball_data')
    except ValueError:
        return meta
    meta['source'] = 'crystal_ball'
    if i + 1 < len(parts_path):
        meta['split'] = parts_path[i + 1]
    if i + 2 < len(parts_path) and parts_path[
        i + 2
    ].startswith('Article'):
        meta['article'] = parts_path[i + 2]
    outcome = None
    if i + 3 < len(parts_path):
        cand = parts_path[i + 3]
        if cand in {'violation', 'non-violation', 'both'}:
            outcome = cand
    if (
        outcome is None
        and meta.get('split') == 'test_violations'
    ):
        outcome = 'violation'
    if outcome:
        meta['outcome'] = outcome
    return meta
