#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import (
    iter_text_paths,
    parse_icj_meta,
    read_text,
)

SYSTEM_PROMPT_TEXT_V1 = 'You are a legal text section extractor for International Court of Justice (ICJ) documents.\n\nGoal (very important):\nWe want to measure the *tonality of the Court\'s reasoning itself* (the legal analysis/merits reasoning), NOT the tonality of other parts such as factual background, procedural history, or descriptions of alleged suffering.\nWe are also interested in what an individual judge has to say in opinions/declarations when it contains legal reasoning or evaluative discussion (not procedural logistics).\n\nYou will receive ONE document chunk at a time. The chunk may contain:\n- cover pages, citations, composition of the Court, table of contents\n- procedural history and orders\n- factual background (often sentiment-laden but not the Court\'s reasoning)\n- party submissions/arguments\n- the Court\'s legal reasoning/analysis (what we want)\n- separate/dissenting/individual opinions and declarations (judge-authored reasoning; not majority reasoning, but still relevant)\n\nYour task:\nExtract ONLY the passages that are the Court\'s own legal reasoning/analysis in this chunk (especially merits reasoning: applying law to facts, assessing evidence, weighing arguments, stating findings/holdings with reasons).\n\nInclude (when present in the chunk):\n- sections where the Court explains why it reaches conclusions (e.g., "The Court considers...", "The Court notes...", "The Court recalls...", "The Court finds/holds...")\n- legal analysis on jurisdiction/admissibility/merits (as reasoning, not procedural scheduling)\n- the Court\'s discussion/evaluation of arguments and evidence\n\nExclude (do NOT output):\n- purely factual narration/chronology (even if emotional)\n- procedural logistics (time-limits, filings, appointments, composition of the bench)\n- party submissions quoted or summarized without the Court\'s own evaluative reasoning\n- annexes/appendices, tables, indexes, citations lists\n- do not exclude judge-authored opinions/declarations (e.g., "INDIVIDUAL OPINION", "SEPARATE OPINION", "DISSENTING OPINION", "DECLARATION"): include them as reasoning text\n\nCRITICAL OUTPUT RULE:\n- Your assistant *content* MUST be JSON only (no prose, no lists, no code fences).\n- The "text" you output MUST be extracted from DOCUMENT_TEXT (do not paraphrase, do not summarize, do not add new words).\n- The source text may contain OCR/page artifacts (page numbers, running headers/footers, form-feed artifacts) and line-break hyphenation (e.g. "unilat-\\neral").\n  - You MAY keep these artifacts as-is OR delete obvious headers/footers/page numbers/form-feed artifacts.\n  - You MAY join line-break hyphenation into the intended word (e.g. "unilat-\\neral" -> "unilateral").\n  - Do NOT otherwise rewrite the Court\'s words.\n- Do NOT insert ellipses ("..." or "...") or any other markers to indicate omitted material. If you omit material, simply omit it.\n- Do NOT stitch together fragments by adding your own connector words; only output copied passages from DOCUMENT_TEXT.\n- If you output multiple spans, output them separated by a blank line; each span must be copied from DOCUMENT_TEXT (optionally with the limited cleanup rules above).\n- You are allowed to omit any material you do not want to include.\n- Your output will be validated against DOCUMENT_TEXT for very high overlap (~=98%+ character match to the source); if too much differs, it will be rejected and you will be asked again.\n- If this chunk contains no Court-reasoning passages, return NONE.\n\nMini examples (illustrative):\n- BAD (paraphrase): "The Court basically says Albania accepted jurisdiction."\n- GOOD (extract): copy the Court\'s sentences from DOCUMENT_TEXT, optionally removing page headers.\n\nOutput format (STRICT):\n- Return a single JSON object with exactly ONE top-level key: "EXTRACT" or "NONE".\n- Under that key, return an object with EXACTLY these fields (in this order):\n  - "file": string (exact file path as provided)\n  - "chunk": string (exact chunk id as provided, like "2/5")\n  - "text": string\n- If top-level key is "NONE", "text" must be the empty string "".\n- Do not add any other keys. Output JSON only.\n'
SYSTEM_PROMPT_RANGES_V1 = 'You are a legal text section extractor for International Court of Justice (ICJ) documents.\n\nGoal (very important):\nWe want to measure the *tonality of the Court\'s reasoning itself* (the legal analysis/merits reasoning), NOT the tonality of other parts such as factual background, procedural history, or descriptions of alleged suffering.\nWe are also interested in what an individual judge has to say in opinions/declarations when it contains legal reasoning or evaluative discussion (not procedural logistics).\n\nYou will receive ONE document chunk at a time. Instead of raw text, you will receive LINE-NUMBERED lines in this exact format:\n  NNNNN|<line text>\nWhere NNNNN is a 1-based line number *within this chunk* (not global to the full document).\n\nYour task:\nSelect ONLY the line ranges that correspond to the Court\'s own legal reasoning/analysis in this chunk (especially merits reasoning: applying law to facts, assessing evidence, weighing arguments, stating findings/holdings with reasons).\n\nInclude (when present):\n- "The Court considers...", "The Court notes...", "The Court recalls...", "The Court finds/holds..."\n- legal analysis on jurisdiction/admissibility/merits (as reasoning, not procedural scheduling)\n- the Court\'s evaluation of arguments and evidence\n- judge-authored reasoning in "INDIVIDUAL OPINION", "SEPARATE OPINION", "DISSENTING OPINION", and "DECLARATION" sections\n\nExclude:\n- purely factual narration/chronology (even if emotional)\n- procedural logistics (time-limits, filings, appointments, composition of the bench)\n- party submissions/arguments without the Court\'s evaluative reasoning\n- annexes/appendices, tables, indexes, citation lists\n\nCRITICAL OUTPUT RULE:\n- Your assistant *content* MUST be JSON only (no prose, no code fences).\n- Do NOT copy/paste the text. Only return line ranges.\n- Ranges must refer to the line numbers shown in this chunk (1-based, inclusive).\n\nOutput format (STRICT):\n- Return a single JSON object with exactly ONE top-level key: "EXTRACT" or "NONE".\n- Under that key, return an object with EXACTLY these fields (in this order):\n  - "file": string (exact file path as provided)\n  - "chunk": string (exact chunk id as provided, like "2/5")\n  - "ranges": array of 2-element arrays [start_line, end_line], sorted ascending\n    - Example: [[12, 34], [80, 115]]\n    - Use the smallest number of ranges that cover the reasoning.\n    - Do not overlap ranges.\n- If top-level key is "NONE", "ranges" must be [].\n- Do not add any other keys. Output JSON only.\n'
_EXTRACT_KEYS = {'EXTRACT', 'NONE'}


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: Optional[str]
    finish_reason: Optional[str]
    raw: Dict[str, Any]


def _canon_key_from_path(path: Path) -> str:
    root = _ROOT.resolve()
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        rel = resolved.relative_to(root)
        return rel.as_posix()
    except Exception:
        return str(resolved)


def _compact_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(',', ':'),
    )


def _log(msg: str) -> None:
    print(
        msg,
        file=sys.stderr,
        flush=True,
    )


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith('```'):
        return t
    lines = t.splitlines()
    if lines and lines[0].lstrip().startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith('```'):
        lines = lines[:-1]
    return '\n'.join(lines).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(
            obj,
            dict,
        ):
            return obj
    except Exception:
        pass
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start >= 0 and end > start:
        snippet = cleaned[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(
            obj,
            dict,
        ):
            return obj
    raise ValueError('error: ValueError')


def _validate_response_text(
    *,
    obj: Dict[str, Any],
    expected_file: str,
    expected_chunk: str,
) -> Tuple[str, str]:
    if not isinstance(
        obj,
        dict,
    ) or len(obj) != 1:
        raise ValueError('error: ValueError')
    top_key = next(iter(obj.keys()))
    if top_key not in _EXTRACT_KEYS:
        raise ValueError('error: ValueError')
    inner = obj.get(top_key)
    if not isinstance(
        inner,
        dict,
    ):
        raise ValueError('error: ValueError')
    expected_inner_keys = {'file', 'chunk', 'text'}
    if set(inner.keys()) != expected_inner_keys:
        raise ValueError('error: ValueError')
    if inner.get('file') != expected_file:
        raise ValueError('error: ValueError')
    if inner.get('chunk') != expected_chunk:
        raise ValueError('error: ValueError')
    text = inner.get('text')
    if not isinstance(
        text,
        str,
    ):
        raise ValueError('error: ValueError')
    if top_key == 'NONE' and text != '':
        raise ValueError('error: ValueError')
    return (top_key, text)


def _normalise_ranges(ranges: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for r in ranges:
        if not isinstance(
            r,
            (list, tuple),
        ) or len(r) != 2:
            raise ValueError('error: ValueError')
        (
            a,
            b,
        ) = r
        try:
            start = int(a)
            end = int(b)
        except Exception as exc:
            raise ValueError('error: ValueError') from exc
        out.append((start, end))
    out.sort(key=lambda t: (t[0], t[1]))
    merged: List[Tuple[int, int]] = []
    for start, end in out:
        if not merged:
            merged.append((start, end))
            continue
        (
            ps,
            pe,
        ) = merged[-1]
        if start <= pe + 1:
            merged[-1] = (ps, max(
                pe,
                end,
            ))
        else:
            merged.append((start, end))
    return merged


def _validate_response_ranges(
    *,
    obj: Dict[str, Any],
    expected_file: str,
    expected_chunk: str,
    max_lines: int,
) -> Tuple[str, List[Tuple[int, int]]]:
    if not isinstance(
        obj,
        dict,
    ) or len(obj) != 1:
        raise ValueError('error: ValueError')
    top_key = next(iter(obj.keys()))
    if top_key not in _EXTRACT_KEYS:
        raise ValueError('error: ValueError')
    inner = obj.get(top_key)
    if not isinstance(
        inner,
        dict,
    ):
        raise ValueError('error: ValueError')
    expected_inner_keys = {'file', 'chunk', 'ranges'}
    if set(inner.keys()) != expected_inner_keys:
        raise ValueError('error: ValueError')
    if inner.get('file') != expected_file:
        raise ValueError('error: ValueError')
    if inner.get('chunk') != expected_chunk:
        raise ValueError('error: ValueError')
    ranges = inner.get('ranges')
    if not isinstance(
        ranges,
        list,
    ):
        raise ValueError('error: ValueError')
    if top_key == 'NONE':
        if ranges != []:
            raise ValueError('error: ValueError')
        return (top_key, [])
    if not ranges:
        raise ValueError('error: ValueError')
    norm = _normalise_ranges(ranges)
    if len(norm) > 200:
        raise ValueError('error: ValueError')
    for start, end in norm:
        if start < 1 or end < 1:
            raise ValueError('error: ValueError')
        if start > end:
            raise ValueError('error: ValueError')
        if end > int(max_lines):
            raise ValueError('error: ValueError')
    return (top_key, norm)


def _openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    referer: Optional[str],
    title: Optional[str],
    debug_http: bool,
    debug_http_max_chars: int,
    quiet: bool,
) -> ChatResult:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    if referer:
        headers['HTTP-Referer'] = str(referer)
    if title:
        headers['X-Title'] = str(title)
    payload = {
        'model': str(model),
        'messages': messages,
        'temperature': float(temperature),
    }
    if int(max_tokens) > 0:
        payload['max_tokens'] = int(max_tokens)
    url = str(base_url).rstrip('/') + '/chat/completions'
    body_bytes = _compact_json(payload).encode('utf-8')
    if debug_http and (not quiet):
        _log('http post')
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers=headers,
        method='POST',
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(
        req,
        timeout=int(timeout_s),
    ) as resp:
        status = int(getattr(
                resp,
                'status',
                resp.getcode(),
            ))
        resp_headers = {
            k: v for k, v in resp.headers.items()
        }
        raw = resp.read()
        body = raw.decode(
            'utf-8',
            errors='replace',
        )
    if debug_http and (not quiet):
        elapsed = time.monotonic() - t0
        interesting = {}
        for k in (
            'x-request-id',
            'x-ratelimit-limit',
            'x-ratelimit-remaining',
            'x-ratelimit-reset',
            'retry-after',
            'openai-processing-ms',
        ):
            for hk, hv in resp_headers.items():
                if hk.lower() == k:
                    interesting[hk] = hv
        _log(f'http {status}')
        if int(debug_http_max_chars) <= 0:
            snippet = body
        else:
            snippet = (
                body
                if len(body) <= int(debug_http_max_chars)
                else body[: int(debug_http_max_chars)] + '...'
            )
        _log('server json')
    data = json.loads(body)
    choices = data.get('choices')
    if not isinstance(
        choices,
        list,
    ) or not choices:
        raise ValueError('error: ValueError')
    choice0 = (
        choices[0] if isinstance(
            choices[0],
            dict,
        ) else None
    )
    msg = (
        choice0.get('message')
        if isinstance(
            choice0,
            dict,
        )
        else None
    )
    if not isinstance(
        msg,
        dict,
    ):
        raise ValueError('error: ValueError')
    content = msg.get('content')
    if not isinstance(
        content,
        str,
    ):
        raise ValueError('error: ValueError')
    reasoning = msg.get('reasoning')
    if not isinstance(
        reasoning,
        str,
    ):
        reasoning = None
    finish_reason = (
        choice0.get('finish_reason')
        if isinstance(
            choice0,
            dict,
        )
        else None
    )
    if not isinstance(
        finish_reason,
        str,
    ):
        finish_reason = None
    return ChatResult(
        content=content,
        reasoning=reasoning,
        finish_reason=finish_reason,
        raw=data,
    )


def _chunk_text(
    text: str,
    *,
    chunk_chars: int,
) -> List[str]:
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    buf: List[str] = []
    n = 0
    for line in lines:
        if buf and n + len(line) > chunk_chars:
            chunks.append(''.join(buf))
            buf = [line]
            n = len(line)
        else:
            buf.append(line)
            n += len(line)
    if buf:
        chunks.append(''.join(buf))
    return chunks


def _normalise_ws(text: str) -> str:
    s = unicodedata.normalize(
        'NFKC',
        str(text),
    )
    s = s.translate(str.maketrans({
                '"': '"',
                '"': '"',
                '"': '"',
                '"': '"',
                ''': "'",
                ''': "'",
                ''': "'",
                ''': "'",
                '-': '-',
                '-': '-',
                '-': '-',
                '-': '-',
                '\xad': '',
            }))
    return re.sub(
        '\\s+',
        ' ',
        s,
    ).strip()


def _split_blocks(text: str) -> List[str]:
    return [
        b.strip()
        for b in re.split(
            '\\n\\s*\\n+',
            text,
        )
        if b.strip()
    ]


def _make_anchors(block_norm: str) -> List[str]:
    s = str(block_norm).lstrip()
    anchors: List[str] = []
    anchors.extend([s[:80], s[:60], s[:40]])
    words = s.split()
    if len(words) >= 8:
        anchors.append(' '.join(words[:8]))
    if len(words) >= 12:
        anchors.append(' '.join(words[:12]))
    if len(s) > 160:
        mid = len(s) // 2
        anchors.append(s[mid : mid + 80])
    uniq: List[str] = []
    seen = set()
    for a in anchors:
        a = a.strip()
        if len(a) < 20:
            continue
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
    return uniq


def _coverage_and_end(
    *,
    short: str,
    long: str,
) -> Tuple[float, int]:
    if not short:
        return (1.0, 0)
    sm = SequenceMatcher(
        None,
        short,
        long,
        autojunk=False,
    )
    blocks = sm.get_matching_blocks()
    matched = sum((b.size for b in blocks))
    cov = float(matched) / float(max(
            1,
            len(short),
        ))
    end = 0
    for b in blocks:
        if b.size <= 0:
            continue
        end = max(
            end,
            int(b.b + b.size),
        )
    return (cov, end)


def _best_partial_ratio_start(
    *,
    short: str,
    long: str,
) -> Tuple[float, int]:
    if not short or not long or len(short) > len(long):
        return (0.0, 0)
    m = SequenceMatcher(
        None,
        short,
        long,
        autojunk=False,
    )
    best = 0.0
    best_start = 0
    for b in m.get_matching_blocks():
        start = int(b.b - b.a)
        if start < 0:
            start = 0
        substr = long[start : start + len(short)]
        r = SequenceMatcher(
            None,
            short,
            substr,
            autojunk=False,
        ).ratio()
        if r > best:
            best = float(r)
            best_start = start
            if best >= 0.999:
                break
    return (best, best_start)


def _is_subset_of_chunk(
    *,
    chunk_text: str,
    extracted_text: str,
    min_ratio: float,
    window_margin_chars: int,
    max_anchor_occurrences: int,
) -> Tuple[bool, str]:
    if not extracted_text.strip():
        return (True, 'empty')
    chunk_norm = _normalise_ws(chunk_text)
    pos = 0
    for i, block in enumerate(
        _split_blocks(extracted_text),
        1,
    ):
        block_norm = _normalise_ws(block)
        if not block_norm:
            continue
        rem = chunk_norm[pos:]
        idx = rem.find(block_norm)
        if idx >= 0:
            pos = pos + idx + len(block_norm)
            continue
        anchors = _make_anchors(block_norm)
        starts: List[int] = []
        for anchor in anchors:
            search_from = 0
            while len(starts) < int(max_anchor_occurrences):
                j = rem.find(
                    anchor,
                    search_from,
                )
                if j < 0:
                    break
                starts.append(int(j))
                search_from = int(j) + 1
            if starts:
                break
        found = False
        best_cov = 0.0
        best_start = 0
        best_end = 0
        for j in sorted(set(starts))[
            : int(max_anchor_occurrences)
        ]:
            window = rem[
                j : j
                + len(block_norm)
                + int(window_margin_chars)
            ]
            (
                cov,
                end,
            ) = _coverage_and_end(
                short=block_norm,
                long=window,
            )
            if cov > best_cov:
                best_cov = cov
                best_start = j
                best_end = j + end
            if cov >= float(min_ratio):
                pos = pos + j + end
                found = True
                break
        if not found:
            (
                ratio,
                j,
            ) = _best_partial_ratio_start(
                short=block_norm,
                long=rem,
            )
            window = rem[
                j : j
                + len(block_norm)
                + int(window_margin_chars)
            ]
            (
                cov,
                end,
            ) = _coverage_and_end(
                short=block_norm,
                long=window,
            )
            if cov >= float(min_ratio):
                pos = pos + j + end
                found = True
            else:
                best_cov = max(
                    best_cov,
                    cov,
                )
                if best_cov == cov:
                    best_start = j
                    best_end = j + end
        if not found:
            preview = block_norm[:160] + (
                '...' if len(block_norm) > 160 else ''
            )
            return (
                False,
                f'block {i} match {best_cov:.3f} < {float(min_ratio):.3f}',
            )
    return (True, 'ok')


def _format_numbered_lines_for_prompt(text: str) -> Tuple[str, int]:
    lines = text.splitlines()
    width = max(
        5,
        len(str(max(
                    1,
                    len(lines),
                ))),
    )
    out_lines: List[str] = []
    for i, line in enumerate(
        lines,
        1,
    ):
        out_lines.append(f'{i:0{width}d}|{line}')
    return ('\n'.join(out_lines), len(lines))


def _build_user_prompt(
    *,
    file_path: str,
    meta: Dict[str, str],
    chunk_id: str,
    doc_text: str,
) -> str:
    meta_keep = {
        k: meta.get(k)
        for k in ('case_id', 'doc_type', 'date', 'stem')
        if meta.get(k)
    }
    return f'FILE: {file_path}\nMETA: {json.dumps(
        meta_keep,
        ensure_ascii=False,
    )}\nCHUNK: {chunk_id}\nDOCUMENT_TEXT:\n{doc_text}\n\nReturn STRICT JSON only, following the schema in the system prompt.'


def _build_user_prompt_ranges(
    *,
    file_path: str,
    meta: Dict[str, str],
    chunk_id: str,
    doc_text: str,
) -> str:
    meta_keep = {
        k: meta.get(k)
        for k in ('case_id', 'doc_type', 'date', 'stem')
        if meta.get(k)
    }
    (
        numbered,
        n_lines,
    ) = _format_numbered_lines_for_prompt(doc_text)
    return f'FILE: {file_path}\nMETA: {json.dumps(
        meta_keep,
        ensure_ascii=False,
    )}\nCHUNK: {chunk_id}\nLINE_COUNT: {n_lines}\nDOCUMENT_LINES (format NNNNN|text):\n{numbered}\n\nReturn STRICT JSON only, following the schema in the system prompt.'


def _out_path_for_input(
    *,
    in_path: Path,
    input_dir: Path,
    out_dir: Path,
) -> Path:
    try:
        rel = in_path.resolve().relative_to(input_dir.resolve())
    except Exception:
        rel = Path(in_path.name)
    stem = rel.stem
    suffix = rel.suffix or '.txt'
    out_name = f'{stem}_llm{suffix}'
    return out_dir / rel.parent / out_name


def main() -> int:
    ap = argparse.ArgumentParser(description="extract reasoning")
    ap.add_argument(
        '--input_dir',
        default=str(_ROOT / 'data' / 'EN_TXT_BEST_FULL'),
    )
    ap.add_argument(
        '--out_dir',
        default=str(_ROOT / 'dataset_llm'),
    )
    ap.add_argument(
        '--name_regex',
        default=None,
    )
    ap.add_argument(
        '--model',
        default='tngtech/deepseek-r1t2-chimera:free',
    )
    ap.add_argument(
        '--mode',
        choices=('ranges', 'text'),
        default='ranges',
    )
    ap.add_argument(
        '--base_url',
        default='https://openrouter.ai/api/v1',
    )
    ap.add_argument(
        '--api_key',
        default=None,
    )
    ap.add_argument(
        '--referer',
        default=None,
    )
    ap.add_argument(
        '--title',
        default='CLDS ICJ Reasoning Extraction',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--resume',
        action='store_true',
    )
    ap.add_argument(
        '--dry_run',
        action='store_true',
    )
    ap.add_argument(
        '--temperature',
        type=float,
        default=0.0,
    )
    ap.add_argument(
        '--max_tokens',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--timeout_s',
        type=int,
        default=180,
    )
    ap.add_argument(
        '--chunk_chars',
        type=int,
        default=25000,
    )
    ap.add_argument(
        '--min_subset_ratio',
        type=float,
        default=0.98,
    )
    ap.add_argument(
        '--subset_window_margin_chars',
        type=int,
        default=2000,
    )
    ap.add_argument(
        '--subset_max_anchor_occurrences',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--retries',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--retry_base_delay_s',
        type=float,
        default=2.0,
    )
    ap.add_argument(
        '--quiet',
        action='store_true',
    )
    ap.add_argument(
        '--log_skips',
        action='store_true',
    )
    ap.add_argument(
        '--echo_json',
        action='store_true',
    )
    ap.add_argument(
        '--echo_raw',
        action='store_true',
    )
    ap.add_argument(
        '--echo_raw_max_chars',
        type=int,
        default=2000,
    )
    ap.add_argument(
        '--debug_http',
        action='store_true',
    )
    ap.add_argument(
        '--debug_http_max_chars',
        type=int,
        default=2000,
    )
    opts = ap.parse_args()
    mode = str(opts.mode)
    system_prompt = (
        SYSTEM_PROMPT_RANGES_V1
        if mode == 'ranges'
        else SYSTEM_PROMPT_TEXT_V1
    )
    input_dir = Path(opts.input_dir)
    if not input_dir.is_absolute():
        input_dir = (_ROOT / input_dir).resolve()
    out_dir = Path(opts.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()
    paths = list(iter_text_paths(
            input_dir,
            recursive=True,
            suffix='.txt',
        ))
    if opts.name_regex:
        pat = re.compile(str(opts.name_regex))
        paths = [p for p in paths if pat.search(str(p))]
    if opts.limit is not None:
        paths = paths[: max(
            0,
            int(opts.limit),
        )]
    api_key = opts.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not opts.dry_run and (not api_key):
        raise RuntimeError('error: RuntimeError')
    total = len(paths)
    written = 0
    skipped = 0
    errors = 0
    skipped_existing: List[str] = []
    failed_files: List[Tuple[str, str]] = []
    for i, path in enumerate(
        paths,
        1,
    ):
        file_key = _canon_key_from_path(path)
        out_path = _out_path_for_input(
            in_path=path,
            input_dir=input_dir,
            out_dir=out_dir,
        )
        tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
        if opts.resume and out_path.exists():
            skipped += 1
            if opts.log_skips and (not opts.quiet):
                _log(f'skip {i}/{total}')
            skipped_existing.append(file_key)
            continue
        if not opts.quiet:
            _log(f'icj {i}/{total}')
        text = read_text(path)
        chunks = _chunk_text(
            text,
            chunk_chars=int(opts.chunk_chars),
        )
        n_chunks = len(chunks)
        meta = parse_icj_meta(path)
        extracted_parts: List[str] = []
        file_failed = False
        file_fail_reason: Optional[str] = None
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if tmp_path.exists():
            tmp_path.unlink()
        with tmp_path.open(
            'w',
            encoding='utf-8',
        ) as tmp_f:
            for ci, chunk_text in enumerate(
                chunks,
                1,
            ):
                chunk_id = f'{ci}/{n_chunks}'
                if mode == 'ranges':
                    user_prompt = _build_user_prompt_ranges(
                        file_path=file_key,
                        meta=meta,
                        chunk_id=chunk_id,
                        doc_text=chunk_text,
                    )
                else:
                    user_prompt = _build_user_prompt(
                        file_path=file_key,
                        meta=meta,
                        chunk_id=chunk_id,
                        doc_text=chunk_text,
                    )
                if opts.dry_run:
                    print('system prompt')
                    print(system_prompt)
                    print('user prompt')
                    print(user_prompt)
                    return 0
                base_messages: List[Dict[str, str]] = [
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': user_prompt,
                    },
                ]
                last_err: Optional[str] = None
                max_tokens_attempt = int(opts.max_tokens)
                for attempt in range(
                    1,
                    int(opts.retries) + 1,
                ):
                    messages = list(base_messages)
                    if (
                        attempt > 1
                        and last_err
                        and (
                            'subset check' in last_err
                            or 'No JSON detected in model output'
                            in last_err
                            or 'No valid JSON object found'
                            in last_err
                        )
                    ):
                        retry_rules = "- Output JSON only, matching the schema.\n- If EXTRACT, the 'text' must be copied from DOCUMENT_TEXT (no paraphrase, no new words).\n- You may keep or remove obvious page headers/footers/page numbers; you may dehyphenate line-break splits.\n- Do not add ellipses or connector words; omissions should be done by omission only.\n- The extract will be validated for very high overlap with DOCUMENT_TEXT (~=98%+).\n"
                        if mode == 'ranges':
                            retry_rules = '- Output JSON only, matching the schema.\n- Do NOT copy/paste text. Return line ranges only.\n- Use ONLY the line numbers shown in this chunk (1-based, inclusive).\n- Return ranges as [[start_line,end_line], ...], sorted, non-overlapping.\n'
                        messages.append({
                                'role': 'user',
                                'content': f'Your previous response was rejected by an automated verifier.\nREJECTION_REASON: {last_err}\n\nFix the issue and try again. Remember:\n{retry_rules}',
                            })
                    try:
                        out = _openrouter_chat(
                            api_key=str(api_key),
                            model=str(opts.model),
                            messages=messages,
                            base_url=str(opts.base_url),
                            temperature=float(opts.temperature),
                            max_tokens=int(max_tokens_attempt),
                            timeout_s=int(opts.timeout_s),
                            referer=opts.referer,
                            title=opts.title,
                            debug_http=bool(opts.debug_http),
                            debug_http_max_chars=int(opts.debug_http_max_chars),
                            quiet=bool(opts.quiet),
                        )
                        if opts.echo_raw and (
                            not opts.quiet
                        ):
                            cap = int(opts.echo_raw_max_chars)
                            content_text = (
                                out.content
                                if out.content
                                else '(empty)'
                            )
                            if (
                                cap > 0
                                and len(content_text) > cap
                            ):
                                content_text = (
                                    content_text[:cap] + '...'
                                )
                            _log('raw content')
                            if out.reasoning:
                                reasoning_text = (
                                    out.reasoning
                                )
                                if (
                                    cap > 0
                                    and len(reasoning_text)
                                    > cap
                                ):
                                    reasoning_text = (
                                        reasoning_text[:cap]
                                        + '...'
                                    )
                                _log('raw reasoning')
                        obj: Optional[Dict[str, Any]] = None
                        try:
                            obj = _extract_json(out.content)
                        except Exception:
                            if out.reasoning:
                                try:
                                    obj = _extract_json(out.reasoning)
                                except Exception:
                                    obj = None
                        if obj is None:
                            last_err = f'No JSON detected in model output (finish_reason={out.finish_reason!r}, content_chars={len(out.content)}, reasoning_chars={len(out.reasoning or '')})'
                            if (
                                out.finish_reason
                                == 'length'
                                and attempt
                                < int(opts.retries)
                            ):
                                max_tokens_attempt = int(min(
                                        max_tokens_attempt
                                        * 2,
                                        16000,
                                    ))
                                if not opts.quiet:
                                    _log(f'retry {chunk_id}')
                                continue
                            raise ValueError(last_err)
                        top_key = ''
                        extracted_text = ''
                        cleaned_obj: Dict[str, Any]
                        if mode == 'ranges':
                            chunk_lines = (
                                chunk_text.splitlines(keepends=True)
                            )
                            (
                                top_key,
                                ranges,
                            ) = (
                                _validate_response_ranges(
                                    obj=obj,
                                    expected_file=file_key,
                                    expected_chunk=chunk_id,
                                    max_lines=len(chunk_lines),
                                )
                            )
                            if top_key == 'EXTRACT':
                                blocks: List[str] = []
                                for start, end in ranges:
                                    block = ''.join(chunk_lines[
                                            start - 1 : end
                                        ]).rstrip('\n')
                                    if block.strip():
                                        blocks.append(block)
                                extracted_text = (
                                    '\n\n'.join(blocks).rstrip()
                                )
                                if (
                                    not extracted_text.strip()
                                ):
                                    raise ValueError('error: ValueError')
                            cleaned_obj = {
                                top_key: {
                                    'file': file_key,
                                    'chunk': chunk_id,
                                    'ranges': (
                                        [
                                            [s, e]
                                            for s, e in ranges
                                        ]
                                        if top_key
                                        == 'EXTRACT'
                                        else []
                                    ),
                                }
                            }
                        else:
                            (
                                top_key,
                                extracted_text,
                            ) = (
                                _validate_response_text(
                                    obj=obj,
                                    expected_file=file_key,
                                    expected_chunk=chunk_id,
                                )
                            )
                            if (
                                top_key == 'EXTRACT'
                                and extracted_text.strip()
                            ):
                                (
                                    ok_subset,
                                    subset_reason,
                                ) = (
                                    _is_subset_of_chunk(
                                        chunk_text=chunk_text,
                                        extracted_text=extracted_text,
                                        min_ratio=float(opts.min_subset_ratio),
                                        window_margin_chars=int(opts.subset_window_margin_chars),
                                        max_anchor_occurrences=int(opts.subset_max_anchor_occurrences),
                                    )
                                )
                                if not ok_subset:
                                    raise ValueError('error: ValueError')
                            cleaned_obj = {
                                top_key: {
                                    'file': file_key,
                                    'chunk': chunk_id,
                                    'text': (
                                        extracted_text
                                        if top_key
                                        == 'EXTRACT'
                                        else ''
                                    ),
                                }
                            }
                        line = _compact_json(cleaned_obj)
                        if opts.echo_json:
                            print(
                                line,
                                flush=True,
                            )
                        if (
                            top_key == 'EXTRACT'
                            and extracted_text.strip()
                        ):
                            if extracted_parts:
                                tmp_f.write('\n\n')
                            tmp_f.write(extracted_text.rstrip()
                                + '\n')
                            tmp_f.flush()
                            extracted_parts.append(extracted_text)
                        last_err = None
                        if not opts.quiet:
                            _log(f'ok {top_key} {chunk_id}')
                        break
                    except urllib.error.HTTPError as e:
                        body = ''
                        try:
                            body = e.read().decode(
                                'utf-8',
                                errors='replace',
                            )
                        except Exception:
                            body = ''
                        if opts.debug_http and (
                            not opts.quiet
                        ):
                            hdrs = (
                                {
                                    k: v
                                    for k, v in getattr(
                                        e,
                                        'headers',
                                        {},
                                    ).items()
                                }
                                if getattr(
                                    e,
                                    'headers',
                                    None,
                                )
                                else {}
                            )
                            interesting = {}
                            for k in (
                                'x-request-id',
                                'retry-after',
                            ):
                                for hk, hv in hdrs.items():
                                    if hk.lower() == k:
                                        interesting[hk] = hv
                            _log(f"http error {getattr(
                                    e,
                                    'code',
                                    '?',
                                )}")
                            if (
                                int(opts.debug_http_max_chars)
                                <= 0
                            ):
                                snippet = body
                            else:
                                snippet = (
                                    body
                                    if len(body)
                                    <= int(opts.debug_http_max_chars)
                                    else body[
                                        : int(opts.debug_http_max_chars)
                                    ]
                                    + '...'
                                )
                            _log('server error')
                        retryable = int(getattr(
                                e,
                                'code',
                                0,
                            )) in {429, 500, 502, 503, 504}
                        last_err = f'HTTPError {getattr(
                            e,
                            'code',
                            '?',
                        )}: {body[:300]}'
                        if not retryable or attempt >= int(opts.retries):
                            break
                    except Exception as e:
                        last_err = str(e)
                        if attempt >= int(opts.retries):
                            break
                    delay = float(opts.retry_base_delay_s) * 2.0 ** float(attempt - 1)
                    delay = delay * (
                        0.9 + 0.2 * random.random()
                    )
                    if not opts.quiet:
                        _log(f'retry {chunk_id} {attempt}/{opts.retries}')
                    time.sleep(delay)
                if last_err is not None:
                    errors += 1
                    file_failed = True
                    file_fail_reason = (
                        f'chunk {chunk_id}: {last_err}'
                    )
                    if not opts.quiet:
                        _log(f'error {file_key}')
                    break
                if file_failed:
                    break
        if file_failed:
            if not opts.quiet:
                _log(f'failed {file_key}')
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            failed_files.append((
                    file_key,
                    str(file_fail_reason or 'unknown error'),
                ))
            continue
        tmp_path.replace(out_path)
        written += 1
        if not opts.quiet:
            _log(f'wrote {out_path}')
    if not opts.quiet:
        _log(f'done {written} {len(failed_files)}')
        if skipped_existing:
            _log('skipped')
            for p in skipped_existing:
                _log(f'skip {p}')
        if failed_files:
            _log('failed')
            for p, reason in failed_files:
                _log(f'fail {p}')
    return 0 if errors == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
