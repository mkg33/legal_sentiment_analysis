from __future__ import annotations
import html
import json
import math
import textwrap
import re
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


@dataclass(frozen=True)
class EmotionalDoc:
    emotion: str
    score_per_1k: float
    prob: Optional[float]
    total_per_1k: Optional[float]
    meta: Dict[str, Any]


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


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _as_int(x: Any) -> Optional[int]:
    if isinstance(
        x,
        bool,
    ):
        return None
    if isinstance(
        x,
        int,
    ):
        return int(x)
    if isinstance(
        x,
        float,
    ):
        if math.isnan(x) or math.isinf(x):
            return None
        return int(x)
    if isinstance(
        x,
        str,
    ):
        try:
            return int(x.strip())
        except Exception:
            try:
                return int(float(x.strip()))
            except Exception:
                return None
    return None


def _pick_vector(
    row: Dict[str, Any],
    key: str,
    n: int,
) -> Optional[List[float]]:
    v = row.get(key)
    if not isinstance(
        v,
        list,
    ) or len(v) != n:
        return None
    out: List[float] = []
    for item in v:
        fv = _as_float(item)
        out.append(0.0 if fv is None else fv)
    return out


def _read_excerpt(
    path: Path,
    *,
    excerpt_chars: int,
    anchor_re: Optional[re.Pattern[str]] = None,
    full_text: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    if full_text:
        try:
            text = path.read_text(
                encoding='utf-8',
                errors='ignore',
            )
        except Exception:
            return ''
        return (
            ' '.join(text.split())
            if collapse_whitespace
            else text
        )
    if excerpt_chars <= 0:
        return ''
    try:
        text = path.read_text(
            encoding='utf-8',
            errors='ignore',
        )
    except Exception:
        return ''
    text = (
        ' '.join(text.split())
        if collapse_whitespace
        else text
    )
    if anchor_re is not None:
        m = anchor_re.search(text)
        if m is not None:
            half = max(
                1,
                int(excerpt_chars) // 2,
            )
            start = max(
                0,
                int(m.start()) - half,
            )
            end = min(
                len(text),
                start + int(excerpt_chars),
            )
            snippet = text[start:end].strip()
            if start > 0:
                snippet = '...' + snippet
            if end < len(text):
                snippet = snippet + '...'
            return snippet
    if len(text) <= excerpt_chars:
        return text
    return text[:excerpt_chars].rstrip() + '...'


def _short_label(meta: Dict[str, Any]) -> str:
    if not isinstance(
        meta,
        dict,
    ):
        return ''
    case_id = str(meta.get('case_id') or '').strip()
    doc_type = str(meta.get('doc_type') or '').strip()
    date = str(meta.get('date') or '').strip()
    bits = [b for b in [case_id, doc_type, date] if b]
    return (
        ' * '.join(bits)
        if bits
        else str(meta.get('stem') or meta.get('path') or '')
    )


def write_most_emotional_docs_html_report(
    *,
    output_path: str | Path,
    scores_jsonl: str | Path,
    top_n: int = 10,
    excerpt_chars: int = 800,
    full_text: bool = False,
    collapse_whitespace: bool = True,
    pre_max_height_px: int = 220,
    title: str = 'Most Emotional Documents',
    manual_top_docs: str | Path | None = None,
    emotion_labels: Dict[str, str] | None = None,
) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    scores_p = Path(scores_jsonl)
    if not scores_p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    rows = _iter_jsonl(scores_p)
    first: Optional[Dict[str, Any]] = None
    all_rows: List[Dict[str, Any]] = []
    for row in rows:
        if first is None:
            first = row
        all_rows.append(row)
    if first is None:
        raise ValueError('error: ValueError')
    emotions: Sequence[str] = (
        first.get('emotions')
        if isinstance(
            first.get('emotions'),
            list,
        )
        else []
    )
    if not emotions:
        raise ValueError('error: ValueError')
    n = len(emotions)
    per_feeling: Dict[str, List[EmotionalDoc]] = {
        str(e): [] for e in emotions
    }

    def _score_per_1k(
        row: Dict[str, Any],
        emo_idx: int,
    ) -> Optional[float]:
        counts_cal = _pick_vector(
            row,
            'pred_counts_calibrated',
            n,
        )
        if counts_cal is not None:
            denom = _as_int(row.get('n_words')) or 0
            denom = max(
                1,
                denom,
            )
            return (
                float(counts_cal[emo_idx])
                / float(denom)
                * 1000.0
            )
        counts = _pick_vector(
            row,
            'pred_counts',
            n,
        )
        if counts is not None:
            denom = _as_int(row.get('n_words')) or 0
            denom = max(
                1,
                denom,
            )
            return (
                float(counts[emo_idx])
                / float(denom)
                * 1000.0
            )
        mix = _pick_vector(
            row,
            'pred_mixscaled_per_1k_words',
            n,
        )
        if mix is not None:
            return float(mix[emo_idx])
        dist_cal = _pick_vector(
            row,
            'pred_dist_calibrated',
            n,
        )
        if dist_cal is not None:
            return float(dist_cal[emo_idx])
        dist = _pick_vector(
            row,
            'pred_dist',
            n,
        )
        if dist is not None:
            return float(dist[emo_idx])
        return None

    def _push(
        bucket: List[EmotionalDoc],
        item: EmotionalDoc,
    ) -> None:
        bucket.append(item)
        bucket.sort(
            key=lambda x: x.score_per_1k,
            reverse=True,
        )
        if len(bucket) > int(top_n):
            del bucket[int(top_n) :]

    manual_map: Optional[Dict[str, Any]] = None
    manual_src: Optional[str] = None
    if manual_top_docs is not None:
        if isinstance(
            manual_top_docs,
            (str, Path),
        ):
            manual_p = Path(manual_top_docs)
            manual_src = str(manual_p)
            if not manual_p.exists():
                raise FileNotFoundError('error: FileNotFoundError')
            payload = json.loads(manual_p.read_text(encoding='utf-8'))
            manual_map = (
                payload
                if isinstance(
                    payload,
                    dict,
                )
                else None
            )
        elif isinstance(
            manual_top_docs,
            dict,
        ):
            manual_map = dict(manual_top_docs)
            manual_src = '(in-memory)'
        else:
            raise TypeError('error: TypeError')
    rows_by_path: Dict[str, Dict[str, Any]] = {}
    for row in all_rows:
        meta = (
            row.get('meta')
            if isinstance(
                row.get('meta'),
                dict,
            )
            else {}
        )
        p = meta.get('path')
        if isinstance(
            p,
            str,
        ) and p:
            rows_by_path[p] = row
    if manual_map is not None:
        for emo in emotions:
            bucket = per_feeling[str(emo)]
            raw_items = manual_map.get(str(emo))
            if not isinstance(
                raw_items,
                list,
            ):
                continue
            norm_paths: List[str] = []
            for it in raw_items:
                if isinstance(
                    it,
                    str,
                ) and it.strip():
                    norm_paths.append(it.strip())
                elif isinstance(
                    it,
                    dict,
                ):
                    p = it.get('path')
                    if isinstance(
                        p,
                        str,
                    ) and p.strip():
                        norm_paths.append(p.strip())
            for p in norm_paths[: int(top_n)]:
                row = rows_by_path.get(p)
                meta = (
                    row.get('meta')
                    if isinstance(
                        row,
                        dict,
                    )
                    and isinstance(
                        row.get('meta'),
                        dict,
                    )
                    else {'path': p}
                )
                emo_idx = int(emotions.index(str(emo)))
                score = (
                    _as_float(_score_per_1k(
                            row,
                            emo_idx,
                        ))
                    if isinstance(
                        row,
                        dict,
                    )
                    else None
                )
                odds_vec = (
                    _pick_vector(
                        row,
                        'pred_sigmoid',
                        n,
                    )
                    if isinstance(
                        row,
                        dict,
                    )
                    else None
                )
                total = (
                    _as_float(row.get('pred_counts_calibrated_per_1k_words'))
                    if isinstance(
                        row,
                        dict,
                    )
                    else None
                )
                if total is None and isinstance(
                    row,
                    dict,
                ):
                    total = _as_float(row.get('pred_per_1k_words'))
                bucket.append(EmotionalDoc(
                        emotion=str(emo),
                        score_per_1k=float(score or 0.0),
                        prob=(
                            odds_vec[emo_idx]
                            if odds_vec is not None
                            else None
                        ),
                        total_per_1k=total,
                        meta=dict(meta),
                    ))
    else:
        for row in all_rows:
            if row.get('low_emotion_signal') is True:
                continue
            odds_vec = _pick_vector(
                row,
                'pred_sigmoid',
                n,
            )
            total = _as_float(row.get('pred_counts_calibrated_per_1k_words'))
            if total is None:
                total = _as_float(row.get('pred_per_1k_words'))
            meta = (
                row.get('meta')
                if isinstance(
                    row.get('meta'),
                    dict,
                )
                else {}
            )
            for idx, emo in enumerate(emotions):
                score = _as_float(_score_per_1k(
                        row,
                        idx,
                    ))
                if score is None:
                    continue
                _push(
                    per_feeling[str(emo)],
                    EmotionalDoc(
                        emotion=str(emo),
                        score_per_1k=float(score),
                        prob=(
                            odds_vec[idx]
                            if odds_vec is not None
                            else None
                        ),
                        total_per_1k=total,
                        meta=dict(meta),
                    ),
                )
    anchor_by_feeling: Dict[str, re.Pattern[str]] = {
        'anger': re.compile(
            '\\\\b(condemn|condemned|condemnation|outrag\\\\w*|indign\\\\w*|deplor\\\\w*|barbar\\\\w*|atrocit\\\\w*)\\\\b',
            re.I,
        ),
        'fear': re.compile(
            '\\\\b(fear\\\\w*|afraid|terror\\\\w*|threat\\\\w*|intimidat\\\\w*|danger\\\\w*)\\\\b',
            re.I,
        ),
        'joy': re.compile(
            '\\\\b(pleased|happy|glad|delighted|welcome|with pleasure|a pleasure)\\\\b',
            re.I,
        ),
        'sadness': re.compile(
            '\\\\b(regret\\\\w*|sorrow\\\\w*|tragic\\\\w*|grief\\\\w*|mourn\\\\w*|lament\\\\w*|sad\\\\w*)\\\\b',
            re.I,
        ),
        'trust': re.compile(
            '\\\\b(trust\\\\w*|confidence|good faith|assurance\\\\w*|undertaking\\\\w*)\\\\b',
            re.I,
        ),
        'disgust': re.compile(
            '\\\\b(degrading treatment|inhuman|humiliat\\\\w*|odious|detestable|abhorrent|repugnant|cruel\\\\w*)\\\\b',
            re.I,
        ),
        'surprise': re.compile(
            '\\\\b(surpris\\\\w*|astonish\\\\w*|unexpected\\\\w*|unforeseen\\\\w*)\\\\b',
            re.I,
        ),
        'anticipation': re.compile(
            '\\\\b(hope\\\\w*|expect\\\\w*|anticipat\\\\w*|look forward)\\\\b',
            re.I,
        ),
    }
    excerpts: Dict[str, str] = {}
    for docs in per_feeling.values():
        for item in docs:
            p = Path(str(item.meta.get('path') or ''))
            key = str(p)
            if key in excerpts:
                continue
            excerpts[key] = _read_excerpt(
                p,
                excerpt_chars=int(excerpt_chars),
                anchor_re=anchor_by_feeling.get(str(item.emotion)),
                full_text=bool(full_text),
                collapse_whitespace=bool(collapse_whitespace),
            )

    def fmt(v: Optional[float]) -> str:
        if v is None:
            return '-'
        return f'{v:.4f}'

    blocks: List[str] = []
    method_label = (
        f'Manual top {int(top_n)} (curated) * source: {manual_src}'
        if manual_map is not None
        else f'Top {int(top_n)} by predicted intensity per 1k words'
    )
    for emo in emotions:
        emo_key = str(emo)
        disp = (
            str(emotion_labels.get(
                    emo_key,
                    emo_key,
                ))
            if isinstance(
                emotion_labels,
                dict,
            )
            else emo_key
        )
        docs = per_feeling.get(str(emo)) or []
        rows_html: List[str] = []
        for i, item in enumerate(
            docs,
            1,
        ):
            p = Path(str(item.meta.get('path') or ''))
            excerpt = excerpts.get(
                str(p),
                '',
            )
            label = _short_label(item.meta)
            rows_html.append(f'\n                <div class="doc">\n                  <div class="doc-top">\n                    <div class="rank">#{i}</div>\n                    <div class="label">{html.escape(label)}</div>\n                    <div class="pill">{html.escape(disp)}</div>\n                    <div class="pill">score/1k={html.escape(fmt(item.score_per_1k))}</div>\n                    <div class="pill">p={html.escape(fmt(item.prob))}</div>\n                    <div class="pill">total/1k={html.escape(fmt(item.total_per_1k))}</div>\n                  </div>\n                  <div class="path">{html.escape(str(p))}</div>\n                  <details class="excerpt">\n                    <summary>{('Full text' if full_text else 'Excerpt')}</summary>\n                    <pre>{(html.escape(excerpt) if excerpt else '(missing text file)')}</pre>\n                  </details>\n                </div>\n                ')
        inner = (
            '\n'.join(rows_html)
            if rows_html
            else "<div class='muted'>No documents found.</div>"
        )
        blocks.append(f'\n            <details class="emotion" open>\n              <summary>\n                <span class="emo">{html.escape(disp)}</span>\n                <span class="muted">{html.escape(method_label)}</span>\n              </summary>\n              {inner}\n            </details>\n            ')
    content = '\n'.join(blocks)
    doc = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{html.escape(title)}</title>
            <style>
              :root {{
                --ink: #0f172a;
                --muted: #64748b;
                --border: rgba(15, 23, 42, 0.10);
                --card: #ffffff;
                --bg-top: #fdf6ec;
                --bg-bottom: #eff2f6;
                --accent: #e07a5f;
              }}
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                color: var(--ink);
                background: linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
              }}
              header {{
                padding: 18px 22px 14px;
                background: radial-gradient(circle at top left, #ffe8d1 0%, #f0f4f8 60%);
                border-bottom: 1px solid var(--border);
              }}
              header h1 {{
                margin: 0;
                font-size: 20px;
                letter-spacing: 0.2px;
              }}
              header p {{
                margin: 6px 0 0;
                color: var(--muted);
                font-size: 13px;
                line-height: 1.4;
                max-width: 980px;
              }}
              main {{
                padding: 16px 22px 26px;
                max-width: 1100px;
                margin: 0 auto;
              }}
              .emotion {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                padding: 10px 12px;
                margin: 12px 0;
              }}
              .emotion summary {{
                cursor: pointer;
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 10px;
                padding: 6px 4px;
              }}
              .emo {{
                font-weight: 800;
                letter-spacing: 0.3px;
              }}
              .muted {{
                color: var(--muted);
                font-size: 12px;
              }}
              .doc {{
                margin-top: 10px;
                border-top: 1px dashed var(--border);
                padding-top: 10px;
              }}
              .doc-top {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
              }}
              .rank {{
                font-weight: 800;
                color: var(--accent);
              }}
              .label {{
                font-weight: 700;
              }}
              .pill {{
                font-size: 12px;
                padding: 3px 8px;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: rgba(224, 122, 95, 0.08);
              }}
              .path {{
                margin-top: 4px;
                color: var(--muted);
                font-size: 12px;
                word-break: break-all;
              }}
              .excerpt {{
                margin-top: 6px;
              }}
              .excerpt summary {{
                cursor: pointer;
                color: var(--muted);
                font-size: 12px;
              }}
              pre {{
                white-space: pre-wrap;
                font-size: 12px;
                line-height: 1.45;
                padding: 10px 12px;
                background: #f8fafc;
                border: 1px solid var(--border);
                border-radius: 10px;
                margin: 8px 0 0;
                max-height: {int(pre_max_height_px)}px;
                overflow: auto;
              }}
            </style>
          </head>
          <body>
            <header>
              <h1>{html.escape(title)}</h1>
            </header>
            <main>
              {content}
            </main>
          </body>
        </html>
        """)
    out_p.write_text(
        doc,
        encoding='utf-8',
    )
    return str(out_p)
