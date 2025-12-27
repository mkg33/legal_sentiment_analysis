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
from legal_emotion.lexicon import load_stopwords
from legal_emotion.utils import load_config

SYSTEM_PROMPT_V1 = 'You are an expert annotator for emotion intensity in formal legal text (International Court of Justice decisions).\nYou must verify and, if necessary, correct the provided AUTOMATIC emotion scores so they match the document\'s *expressed emotional language / stance*.\n\nCRITICAL OUTPUT RULE: Your assistant *content* must be JSON only (no prose, no lists, no code fences). Do any thinking silently and then output the JSON.\n\nThe categories are:\n- anger: condemnation, indignation, outrage, blame, accusation (NOT mere disagreement stated neutrally).\n- fear: expressed fear, threat, intimidation, danger (NOT polite hedges like "I am afraid" or "for fear that").\n- joy: happiness, pleasure, satisfaction, celebration (rare in ICJ; do not force it).\n- sadness: regret, sorrow, grief, lament; formulaic "I regret that I cannot concur" is mild sadness/regret.\n- trust: confidence, endorsement/approval, reliance, good faith, assurances/undertakings.\n- disgust: revulsion/abhorrence; also strong moral disgust about "inhuman/degrading/cruel" treatment.\n- surprise: unexpectedness/astonishment/unforeseen (rare in ICJ; do not force it).\n- anticipation: expectation/hope about future outcomes; forward-looking concern/optimism.\n\nScoring scale:\n- IMPORTANT: keep the SAME scale as the provided AUTO_SCORES.\n- AUTO_SCORES come from a GoEmotions teacher: probability-like scores mapped into the 8 categories, then multiplied by 20 (score ~= 20 * p).\n- Each category score must be a number in [0, 20].\n- 0 means no expressed evidence in the provided text; 20 corresponds to maximal evidence (teacher p~=1).\n- Scores should be sparse and usually low for procedural/boilerplate documents.\n\nRules:\n1) Base your judgment ONLY on the provided document text (it may be truncated; do not assume missing content).\n2) Ignore purely procedural boilerplate and neutral legal reasoning when judging emotion.\n3) If a category is not present in the language, set it to 0.\n4) If the automatic scores are already meaningful, return AGREE and copy the scores EXACTLY as given.\n5) If not meaningful, return DISAGREE and provide a corrected full score set.\n\nOutput format (STRICT):\n- Return a single JSON object with exactly ONE top-level key: "AGREE" or "DISAGREE".\n- Under that key, return an object with EXACTLY these fields (in this order):\n  - "file": string (exact file path as provided)\n  - "scores": object mapping each emotion key to a number\n- The "scores" object must contain EXACTLY the provided emotion keys (no extras, no omissions).\n- Output JSON only. No markdown. No commentary. No extra keys.\n'
_ANCHOR_BY_EMOTION: Dict[str, re.Pattern[str]] = {
    'anger': re.compile(
        '\\b(condemn|condemned|condemnation|outrag\\w*|indign\\w*|deplor\\w*|barbar\\w*|atrocit\\w*)\\b',
        re.I,
    ),
    'fear': re.compile(
        '\\b(fear\\w*|afraid|terror\\w*|threat\\w*|intimidat\\w*|danger\\w*)\\b',
        re.I,
    ),
    'joy': re.compile(
        '\\b(pleased|happy|glad|delighted|welcome|with pleasure|a pleasure)\\b',
        re.I,
    ),
    'sadness': re.compile(
        '\\b(regret\\w*|sorrow\\w*|tragic\\w*|grief\\w*|mourn\\w*|lament\\w*)\\b',
        re.I,
    ),
    'trust': re.compile(
        '\\b(trust\\w*|confidence|good faith|assurance\\w*|undertaking\\w*)\\b',
        re.I,
    ),
    'disgust': re.compile(
        '\\b(degrading treatment|inhuman|humiliat\\w*|odious|detestable|abhorrent|repugnant|cruel\\w*)\\b',
        re.I,
    ),
    'surprise': re.compile(
        '\\b(surpris\\w*|astonish\\w*|unexpected\\w*|unforeseen\\w*)\\b',
        re.I,
    ),
    'anticipation': re.compile(
        '\\b(hope\\w*|expect\\w*|anticipat\\w*|look forward)\\b',
        re.I,
    ),
}
_ROOT_RESOLVED = _ROOT.resolve()


def _canon_key_from_path(path: Path) -> str:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        rel = resolved.relative_to(_ROOT_RESOLVED)
        return rel.as_posix()
    except Exception:
        return str(resolved)


def _canon_key_from_meta_path(path_str: str) -> str:
    p = Path(str(path_str))
    if not p.is_absolute():
        p = _ROOT / p
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    try:
        rel = resolved.relative_to(_ROOT_RESOLVED)
        return rel.as_posix()
    except Exception:
        return str(Path(path_str).as_posix())


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(
                obj,
                dict,
            ):
                yield obj


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


def _compact_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(',', ':'),
    )


def _default_scores_jsonl() -> Optional[Path]:
    out_root = _ROOT / 'outputs'
    if not out_root.exists():
        return None
    candidates: List[Path] = []
    for name in (
        'icj_scores_goemotions_stopphrases.jsonl',
        'icj_scores_goemotions.jsonl',
        'icj_scores.jsonl',
    ):
        candidates.extend(out_root.glob(f'**/{name}'))
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0]


def _format_scores_for_prompt(
    scores: Dict[str, float],
    emotions: Sequence[str],
) -> str:
    out: Dict[str, float] = {}
    for e in emotions:
        out[str(e)] = float(scores.get(
                str(e),
                0.0,
            ))
    return json.dumps(
        out,
        ensure_ascii=False,
    )


def _normalise_text(text: str) -> str:
    return re.sub(
        '\\s+',
        ' ',
        text,
    ).strip()


def _compile_stopphrase_patterns(stopwords_file: Optional[str]) -> List[re.Pattern[str]]:
    if not stopwords_file:
        return []
    try:
        stop = load_stopwords(stopwords_file)
    except Exception:
        return []
    phrases = [p for p in stop if ' ' in p]
    if not phrases:
        return []
    patterns: List[re.Pattern[str]] = []
    for phrase in sorted(
        phrases,
        key=len,
        reverse=True,
    ):
        toks = phrase.split()
        if len(toks) < 2:
            continue
        pat = (
            '\\b'
            + '\\s+'.join((re.escape(tok) for tok in toks))
            + '\\b'
        )
        patterns.append(re.compile(
                pat,
                flags=re.IGNORECASE,
            ))
    return patterns


def _resolve_stopwords_file(
    stopwords_file: Optional[str],
    cfg_path: Optional[str],
) -> Optional[str]:
    if stopwords_file:
        return str(stopwords_file)
    if cfg_path:
        setup = load_config(cfg_path)
        sw = getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        )
        if sw:
            return str(sw)
    default_path = _ROOT / 'data' / 'stopwords_legal_en.txt'
    if default_path.exists():
        return str(default_path)
    return None


def _snippets_around(
    pattern: re.Pattern[str],
    text: str,
    *,
    window: int,
    max_hits: int,
) -> List[str]:
    out: List[str] = []
    for m in pattern.finditer(text):
        if len(out) >= max_hits:
            break
        start = max(
            0,
            int(m.start()) - window,
        )
        end = min(
            len(text),
            int(m.end()) + window,
        )
        snippet = text[start:end].strip()
        if start > 0:
            snippet = '...' + snippet
        if end < len(text):
            snippet = snippet + '...'
        out.append(snippet)
    return out


def _truncate_with_anchors(
    *,
    text: str,
    emotions: Sequence[str],
    auto_scores: Dict[str, float],
    max_chars: int,
    head_chars: int,
    tail_chars: int,
    max_anchor_emotions: int,
    max_hits_per_emotion: int,
    window: int,
) -> Tuple[str, bool]:
    if max_chars <= 0:
        return (text, False)
    if len(text) <= max_chars:
        return (text, False)
    top = sorted(
        (
            (float(auto_scores.get(
                    e,
                    0.0,
                )), str(e))
            for e in emotions
        ),
        reverse=True,
    )
    top_feelings = [
        e
        for _, e in top[: max(
            1,
            int(max_anchor_emotions),
        )]
    ]
    snippets: List[str] = []
    for emo in top_feelings:
        pat = _ANCHOR_BY_EMOTION.get(str(emo))
        if pat is None:
            continue
        snippets.extend(_snippets_around(
                pat,
                text,
                window=int(window),
                max_hits=int(max_hits_per_emotion),
            ))
    uniq: List[str] = []
    seen = set()
    for s in snippets:
        k = s[:200]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    snippet_block = '\n\n'.join(uniq[
            : max(
                0,
                int(max_anchor_emotions)
                * int(max_hits_per_emotion),
            )
        ])
    prefix = '[TRUNCATED]\n\n[HEAD]\n'
    mid = '\n\n[ANCHOR_SNIPPETS]\n'
    suffix = '\n\n[TAIL]\n'
    max_chars_i = max(
        0,
        int(max_chars),
    )
    desired_head = max(
        0,
        int(head_chars),
    )
    desired_tail = max(
        0,
        int(tail_chars),
    )
    overhead = len(prefix) + len(mid) + len(suffix)
    if max_chars_i <= overhead:
        return (text[:max_chars_i].rstrip(), True)
    avail = max(
        0,
        max_chars_i - overhead,
    )
    snippet_cap = min(
        len(snippet_block),
        int(min(
                20000,
                avail // 4,
            )),
    )
    snippet = (
        snippet_block[:snippet_cap].rstrip()
        if snippet_cap > 0
        else ''
    )
    remaining = max(
        0,
        avail - len(snippet),
    )
    requested_total = desired_head + desired_tail
    if requested_total <= remaining:
        head_budget = min(
            desired_head,
            remaining,
        )
        tail_budget = min(
            desired_tail,
            max(
                0,
                remaining - head_budget,
            ),
        )
    elif requested_total > 0 and remaining > 0:
        head_budget = int(round(remaining
                * (
                    float(desired_head)
                    / float(requested_total)
                )))
        head_budget = max(
            0,
            min(
                head_budget,
                min(
                    desired_head,
                    remaining,
                ),
            ),
        )
        tail_budget = max(
            0,
            min(
                desired_tail,
                remaining - head_budget,
            ),
        )
        leftover = remaining - (head_budget + tail_budget)
        if leftover > 0:
            add = min(
                leftover,
                desired_head - head_budget,
            )
            head_budget += add
            leftover -= add
        if leftover > 0:
            add = min(
                leftover,
                desired_tail - tail_budget,
            )
            tail_budget += add
            leftover -= add
    else:
        head_budget = 0
        tail_budget = 0
    head = (
        text[:head_budget].rstrip()
        if head_budget > 0
        else ''
    )
    tail = (
        text[-tail_budget:].lstrip()
        if tail_budget > 0
        else ''
    )
    composed = f'{prefix}{head}{mid}{snippet}{suffix}{tail}'.rstrip()
    return (composed, True)


@dataclass(frozen=True)
class AutoScores:
    emotions: List[str]
    scores: Dict[str, float]


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: Optional[str]
    finish_reason: Optional[str]
    raw: Dict[str, Any]


def _load_auto_scores(scores_jsonl: Path) -> Tuple[List[str], Dict[str, AutoScores]]:
    first: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = []
    for row in _iter_jsonl(scores_jsonl):
        if first is None:
            first = row
        rows.append(row)
    if first is None:
        raise ValueError('error: ValueError')
    emotions = first.get('emotions')
    if not isinstance(
        emotions,
        list,
    ) or not emotions:
        raise ValueError('error: ValueError')
    emotions = [str(e) for e in emotions]
    n = len(emotions)
    by_path: Dict[str, AutoScores] = {}
    for row in rows:
        meta = (
            row.get('meta')
            if isinstance(
                row.get('meta'),
                dict,
            )
            else {}
        )
        p = meta.get('path')
        if not isinstance(
            p,
            str,
        ) or not p:
            continue
        key = _canon_key_from_meta_path(p)
        vec = row.get('pred_mixscaled_per_1k_words')
        if not isinstance(
            vec,
            list,
        ) or len(vec) != n:
            continue
        scores: Dict[str, float] = {}
        ok = True
        for emo, val in zip(
            emotions,
            vec,
        ):
            try:
                fv = float(val)
            except Exception:
                ok = False
                break
            scores[str(emo)] = max(
                0.0,
                min(
                    20.0,
                    fv,
                ),
            )
        if not ok:
            continue
        by_path[key] = AutoScores(
            emotions=list(emotions),
            scores=scores,
        )
    if not by_path:
        raise ValueError('error: ValueError')
    return (emotions, by_path)


def _load_processed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    processed: set[str] = set()
    with out_path.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(
                obj,
                dict,
            ) or len(obj) != 1:
                continue
            top_key = next(iter(obj.keys()))
            if top_key not in {'AGREE', 'DISAGREE'}:
                continue
            inner = obj.get(top_key)
            if not isinstance(
                inner,
                dict,
            ):
                continue
            file_ = inner.get('file')
            if isinstance(
                file_,
                str,
            ) and file_:
                processed.add(_canon_key_from_meta_path(file_))
    return processed


def _validate_response(
    *,
    obj: Dict[str, Any],
    expected_file: str,
    emotions: Sequence[str],
    auto_scores: Dict[str, float],
) -> Dict[str, Any]:
    if not isinstance(
        obj,
        dict,
    ) or len(obj) != 1:
        raise ValueError('error: ValueError')
    top_key = next(iter(obj.keys()))
    if top_key not in {'AGREE', 'DISAGREE'}:
        raise ValueError('error: ValueError')
    inner = obj.get(top_key)
    if not isinstance(
        inner,
        dict,
    ):
        raise ValueError('error: ValueError')
    if inner.get('file') != expected_file:
        raise ValueError('error: ValueError')
    scores = inner.get('scores')
    if not isinstance(
        scores,
        dict,
    ):
        raise ValueError('error: ValueError')
    missing = [e for e in emotions if e not in scores]
    extra = [
        k for k in scores.keys() if k not in set(emotions)
    ]
    if missing or extra:
        raise ValueError('error: ValueError')
    out_scores: Dict[str, float] = {}
    for e in emotions:
        try:
            v = float(scores[e])
        except Exception as exc:
            raise ValueError('error: ValueError') from exc
        if v < 0.0 or v > 20.0:
            raise ValueError('error: ValueError')
        out_scores[e] = float(v)
    if top_key == 'AGREE':
        tol = 0.001
        for e in emotions:
            base = float(auto_scores.get(
                    e,
                    0.0,
                ))
            if abs(out_scores[e] - base) > tol:
                raise ValueError('error: ValueError')
    return {
        top_key: {
            'file': expected_file,
            'scores': out_scores,
        }
    }


def _openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    base_url: str,
    temperature: float,
    max_tokens: Optional[int],
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
    if max_tokens is not None and int(max_tokens) > 0:
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


def _log(msg: str) -> None:
    print(
        msg,
        file=sys.stderr,
        flush=True,
    )


def _build_user_prompt(
    *,
    file_path: str,
    meta: Dict[str, str],
    emotions: Sequence[str],
    auto_scores: Dict[str, float],
    doc_text: str,
    truncated: bool,
) -> str:
    meta_keep = {
        k: meta.get(k)
        for k in ('case_id', 'doc_type', 'date', 'stem')
        if meta.get(k)
    }
    feelings_list = list(emotions)
    auto_str = _format_scores_for_prompt(
        auto_scores,
        feelings_list,
    )
    trunc_note = (
        'YES (do not assume missing content)'
        if truncated
        else 'NO'
    )
    return f'FILE: {file_path}\nMETA: {json.dumps(
        meta_keep,
        ensure_ascii=False,
    )}\nEMOTION_KEYS (fixed): {json.dumps(
        feelings_list,
        ensure_ascii=False,
    )}\nAUTO_SCORES (0-20 = 20*teacher p): {auto_str}\nTEXT_TRUNCATED: {trunc_note}\nDOCUMENT_TEXT:\n{doc_text}\n\nReturn STRICT JSON only, following the schema in the system prompt.'


def _get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='llm rescore icj')
    ap.add_argument(
        '--input_dir',
        default=str(_ROOT / 'data' / 'EN_TXT_BEST_FULL'),
    )
    ap.add_argument(
        '--scores_jsonl',
        default=None,
    )
    ap.add_argument(
        '--config',
        default=None,
    )
    ap.add_argument(
        '--stopwords_file',
        default=None,
    )
    ap.add_argument(
        '--out',
        default='openrouter_response.txt',
    )
    ap.add_argument(
        '--model',
        default='tngtech/deepseek-r1t2-chimera:free',
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
        default='CLDS ICJ Rescoring',
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
        default=1200,
    )
    ap.add_argument(
        '--timeout_s',
        type=int,
        default=120,
    )
    ap.add_argument(
        '--max_chars',
        type=int,
        default=600000,
    )
    ap.add_argument(
        '--head_chars',
        type=int,
        default=250000,
    )
    ap.add_argument(
        '--tail_chars',
        type=int,
        default=250000,
    )
    ap.add_argument(
        '--max_anchor_emotions',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--max_hits_per_emotion',
        type=int,
        default=2,
    )
    ap.add_argument(
        '--anchor_window',
        type=int,
        default=280,
    )
    ap.add_argument(
        '--retries',
        type=int,
        default=5,
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
    return ap.parse_args()


def _get_scores_jsonl(opts: argparse.Namespace) -> Path:
    scores_jsonl = (
        Path(opts.scores_jsonl)
        if opts.scores_jsonl
        else _default_scores_jsonl()
    )
    if scores_jsonl is None:
        raise FileNotFoundError('error: FileNotFoundError')
    if not scores_jsonl.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    return scores_jsonl


def _get_input_dir(input_dir: str) -> Path:
    p = Path(input_dir)
    if not p.is_absolute():
        p = (_ROOT / p).resolve()
    return p


def _get_paths(
    input_dir: Path,
    limit: Optional[int],
) -> List[Path]:
    paths = list(iter_text_paths(
            input_dir,
            recursive=True,
            suffix='.txt',
        ))
    if limit is not None:
        paths = paths[: max(
            0,
            int(limit),
        )]
    return paths


def _pick_auto_scores(
    scores_by_path: Dict[str, AutoScores],
    path: Path,
) -> Tuple[str, Optional[AutoScores]]:
    file_key = _canon_key_from_path(path)
    auto = scores_by_path.get(file_key)
    if auto is None:
        base = path.name
        fallback = next(
            (
                k
                for k in scores_by_path.keys()
                if Path(k).name == base
            ),
            None,
        )
        if fallback:
            file_key = str(fallback)
            auto = scores_by_path.get(fallback)
    return (file_key, auto)


def _clean_doc_text(
    path: Path,
    stopphrase_patterns: List[re.Pattern[str]],
) -> str:
    text = read_text(path)
    if stopphrase_patterns:
        for pat in stopphrase_patterns:
            text = pat.sub(
                ' ',
                text,
            )
    return _normalise_text(text)


def _make_doc_prompt(
    *,
    path: Path,
    file_key: str,
    emotions: Sequence[str],
    auto_scores: Dict[str, float],
    stopphrase_patterns: List[re.Pattern[str]],
    opts: argparse.Namespace,
) -> str:
    text = _clean_doc_text(
        path,
        stopphrase_patterns,
    )
    (
        trunc_text,
        was_truncated,
    ) = _truncate_with_anchors(
        text=text,
        emotions=emotions,
        auto_scores=auto_scores,
        max_chars=int(opts.max_chars),
        head_chars=int(opts.head_chars),
        tail_chars=int(opts.tail_chars),
        max_anchor_emotions=int(opts.max_anchor_emotions),
        max_hits_per_emotion=int(opts.max_hits_per_emotion),
        window=int(opts.anchor_window),
    )
    meta = parse_icj_meta(path)
    return _build_user_prompt(
        file_path=file_key,
        meta=meta,
        emotions=emotions,
        auto_scores=auto_scores,
        doc_text=trunc_text,
        truncated=was_truncated,
    )


def _get_api_key(opts: argparse.Namespace) -> Optional[str]:
    return opts.api_key or os.environ.get('OPENROUTER_API_KEY')


def main() -> int:
    opts = _get_args()
    scores_jsonl = _get_scores_jsonl(opts)
    (
        emotions,
        scores_by_path,
    ) = _load_auto_scores(scores_jsonl)
    stopwords_file = _resolve_stopwords_file(
        opts.stopwords_file,
        opts.config,
    )
    stopphrase_patterns = _compile_stopphrase_patterns(stopwords_file)
    out_path = Path(opts.out)
    processed = (
        _load_processed(out_path) if opts.resume else set()
    )
    input_dir = _get_input_dir(str(opts.input_dir))
    paths = _get_paths(
        input_dir,
        opts.limit,
    )
    api_key = _get_api_key(opts)
    if not opts.dry_run and (not api_key):
        raise RuntimeError('error: RuntimeError')
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    written = 0
    skipped = 0
    errors = 0
    total = len(paths)
    with out_path.open(
        'a',
        encoding='utf-8',
    ) as out_f:
        for i, path in enumerate(
            paths,
            1,
        ):
            (
                file_key,
                auto,
            ) = _pick_auto_scores(
                scores_by_path,
                path,
            )
            if processed and file_key in processed:
                skipped += 1
                if opts.log_skips and (not opts.quiet):
                    _log(f'skip {i}/{total}')
                continue
            if auto is None:
                err = {
                    'ERROR': {
                        'file': file_key,
                        'reason': 'missing auto scores for file',
                    }
                }
                out_f.write(_compact_json(err) + '\n')
                out_f.flush()
                errors += 1
                continue
            if not opts.quiet:
                _log(f'icj {i}/{total}')
            user_prompt = _make_doc_prompt(
                path=path,
                file_key=file_key,
                emotions=emotions,
                auto_scores=auto.scores,
                stopphrase_patterns=stopphrase_patterns,
                opts=opts,
            )
            if opts.dry_run:
                print('system prompt')
                print(SYSTEM_PROMPT_V1)
                print('user prompt')
                print(user_prompt)
                return 0
            messages: List[Dict[str, str]] = [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT_V1,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                },
            ]
            last_err: Optional[str] = None
            max_tokens_attempt: Optional[int]
            if int(opts.max_tokens) <= 0:
                max_tokens_attempt = None
            else:
                max_tokens_attempt = int(opts.max_tokens)
            for attempt in range(
                1,
                int(opts.retries) + 1,
            ):
                try:
                    out = _openrouter_chat(
                        api_key=str(api_key),
                        model=str(opts.model),
                        messages=messages,
                        base_url=str(opts.base_url),
                        temperature=float(opts.temperature),
                        max_tokens=max_tokens_attempt,
                        timeout_s=int(opts.timeout_s),
                        referer=opts.referer,
                        title=opts.title,
                        debug_http=bool(opts.debug_http),
                        debug_http_max_chars=int(opts.debug_http_max_chars),
                        quiet=bool(opts.quiet),
                    )
                    if opts.echo_raw and (not opts.quiet):
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
                            reasoning_text = out.reasoning
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
                            out.finish_reason == 'length'
                            and attempt < int(opts.retries)
                        ):
                            if (
                                max_tokens_attempt
                                is not None
                            ):
                                max_tokens_attempt = int(min(
                                        int(max_tokens_attempt)
                                        * 2,
                                        8000,
                                    ))
                                if not opts.quiet:
                                    _log('retry')
                                continue
                        raise ValueError(last_err)
                    cleaned = _validate_response(
                        obj=obj,
                        expected_file=file_key,
                        emotions=emotions,
                        auto_scores=auto.scores,
                    )
                    top_key = next(iter(cleaned.keys()))
                    line = _compact_json(cleaned)
                    out_f.write(line + '\n')
                    out_f.flush()
                    if opts.echo_json:
                        print(
                            line,
                            flush=True,
                        )
                    processed.add(file_key)
                    last_err = None
                    written += 1
                    if not opts.quiet:
                        _log(f'ok {top_key}')
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
                    if opts.debug_http and (not opts.quiet):
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
                    _log(f'retry {attempt}/{opts.retries}')
                time.sleep(min(
                        delay,
                        60.0,
                    ))
            if last_err is not None:
                err = {
                    'ERROR': {
                        'file': file_key,
                        'reason': last_err,
                    }
                }
                line = _compact_json(err)
                out_f.write(line + '\n')
                out_f.flush()
                if opts.echo_json:
                    print(
                        line,
                        flush=True,
                    )
                errors += 1
                if not opts.quiet:
                    _log(f'error {out_path}')
            if i % 20 == 0:
                time.sleep(0.2)
    if not opts.quiet:
        _log(f'done {written} {errors}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
