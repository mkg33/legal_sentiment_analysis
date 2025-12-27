#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
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


def _canon_path(path_str: str) -> str:
    p = Path(str(path_str))
    if not p.is_absolute():
        p = (_ROOT / p).resolve()
    try:
        rel = p.relative_to(_ROOT.resolve())
        return rel.as_posix()
    except Exception:
        return str(Path(str(path_str)).as_posix())


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
    return float(v)


def _normalise_dist(x: Sequence[float]) -> List[float]:
    vals = [max(
        0.0,
        float(v),
    ) for v in x]
    total = float(sum(vals))
    n = len(vals)
    if n <= 0:
        return []
    if total <= 1e-08:
        return [1.0 / float(n) for _ in range(n)]
    return [float(v) / total for v in vals]


def _entropy(dist: Sequence[float]) -> float:
    n = len(dist)
    if n <= 1:
        return 0.0
    p = [max(
        1e-08,
        float(v),
    ) for v in dist]
    h = -sum((v * math.log(v) for v in p))
    return float(h / max(
            math.log(float(n)),
            1e-08,
        ))


@dataclass(frozen=True)
class Adjudication:
    decision: str
    file_key: str
    scores: Optional[Dict[str, float]] = None
    raw: Optional[Dict[str, Any]] = None


def _load_adjudications(
    path: Path,
    emotions: Sequence[str],
) -> Tuple[Dict[str, Adjudication], Dict[str, int]]:
    latest: Dict[str, Adjudication] = {}
    stats = {
        'lines': 0,
        'parsed': 0,
        'agree': 0,
        'disagree': 0,
        'error': 0,
        'invalid': 0,
    }
    allowed = {'AGREE', 'DISAGREE', 'ERROR'}
    emo_keys = [str(e) for e in emotions]
    emo_set = set(emo_keys)
    with path.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            stats['lines'] += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                stats['invalid'] += 1
                continue
            if not isinstance(
                obj,
                dict,
            ) or len(obj) != 1:
                stats['invalid'] += 1
                continue
            decision = next(iter(obj.keys()))
            if decision not in allowed:
                stats['invalid'] += 1
                continue
            payload = obj.get(decision)
            if not isinstance(
                payload,
                dict,
            ):
                stats['invalid'] += 1
                continue
            file_raw = payload.get('file')
            if (
                not isinstance(
                    file_raw,
                    str,
                )
                or not file_raw.strip()
            ):
                stats['invalid'] += 1
                continue
            file_key = _canon_path(file_raw.strip())
            scores_map: Optional[Dict[str, float]] = None
            raw_scores = payload.get('scores')
            if decision in {'AGREE', 'DISAGREE'}:
                if not isinstance(
                    raw_scores,
                    dict,
                ):
                    stats['invalid'] += 1
                    continue
                if not emo_set.issubset(set(map(
                            str,
                            raw_scores.keys(),
                        ))):
                    stats['invalid'] += 1
                    continue
                scores_map = {}
                for k in emo_keys:
                    v = _as_float(raw_scores.get(k))
                    if v is None:
                        v = 0.0
                    scores_map[k] = float(min(
                            20.0,
                            max(
                                0.0,
                                v,
                            ),
                        ))
            latest[file_key] = Adjudication(
                decision=decision,
                file_key=file_key,
                scores=scores_map,
                raw=payload,
            )
            stats['parsed'] += 1
            if decision == 'AGREE':
                stats['agree'] += 1
            elif decision == 'DISAGREE':
                stats['disagree'] += 1
            else:
                stats['error'] += 1
    return (latest, stats)


def _infer_emotions(scores_jsonl: Path) -> List[str]:
    for row in _iter_jsonl(scores_jsonl):
        emotions = row.get('emotions')
        if (
            isinstance(
                emotions,
                list,
            )
            and emotions
            and all((isinstance(
                    e,
                    str,
                ) for e in emotions))
        ):
            return [str(e) for e in emotions]
    raise ValueError('error: ValueError')


def _apply_one(
    row: Dict[str, Any],
    *,
    emotions: Sequence[str],
    adjudication: Optional[Adjudication],
    scale_words: float = 20.0,
) -> Tuple[Dict[str, Any], str]:
    meta = row.get('meta')
    if not isinstance(
        meta,
        dict,
    ):
        meta = {}
    path_raw = meta.get('path')
    file_key = (
        _canon_path(path_raw)
        if isinstance(
            path_raw,
            str,
        )
        else ''
    )
    if (
        adjudication is None
        or adjudication.decision
        not in {'AGREE', 'DISAGREE'}
        or (not adjudication.scores)
    ):
        return (row, 'missing')
    if adjudication.file_key != file_key:
        return (row, 'mismatch')
    if adjudication.decision == 'AGREE':
        meta['openrouter_adjudication'] = 'AGREE'
        row['meta'] = meta
        return (row, 'agree')
    intensity_words = [
        float(adjudication.scores.get(
                str(e),
                0.0,
            ))
        for e in emotions
    ]
    probs = [
        min(
            1.0,
            max(
                0.0,
                float(v) / float(scale_words),
            ),
        )
        for v in intensity_words
    ]
    dist = _normalise_dist(probs)
    ent = _entropy(dist)
    n_words = int(row.get('n_words') or 0)
    n_tokens = int(row.get('n_tokens') or 0)
    denom_tokens = max(
        1,
        n_tokens,
    )
    ratio_words_tokens = (
        float(n_words) / float(denom_tokens)
        if n_words > 0
        else 0.0
    )
    intensity_tokens = [
        float(v) * ratio_words_tokens
        for v in intensity_words
    ]
    row['pred_sigmoid'] = probs
    row['pred_dist'] = dist
    row['pred_entropy'] = float(ent)
    row['pred_mixscaled_per_1k_words'] = intensity_words
    row['pred_mixscaled_per_1k_tokens'] = intensity_tokens
    row['pred_per_1k_words'] = float(sum(intensity_words))
    row['pred_per_1k_tokens'] = float(sum(intensity_tokens))
    row['emotion_signal_per_1k_words'] = float(sum(intensity_words))
    meta['openrouter_adjudication'] = 'DISAGREE'
    row['meta'] = meta
    return (row, 'disagree')


def main() -> int:
    ap = argparse.ArgumentParser(description='apply llm rescore')
    ap.add_argument(
        '--scores_jsonl',
        required=True,
        type=str,
    )
    ap.add_argument(
        '--openrouter_txt',
        required=True,
        type=str,
    )
    ap.add_argument(
        '--out',
        required=True,
        type=str,
    )
    ap.add_argument(
        '--scale_words',
        type=float,
        default=20.0,
    )
    opts = ap.parse_args()
    scores_p = Path(opts.scores_jsonl)
    out_p = Path(opts.out)
    adjud_p = Path(opts.openrouter_txt)
    if not scores_p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    if not adjud_p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    emotions = _infer_emotions(scores_p)
    (
        adjudications,
        adjud_stats,
    ) = _load_adjudications(
        adjud_p,
        emotions,
    )
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    counts = {
        'docs': 0,
        'agree': 0,
        'disagree': 0,
        'missing': 0,
        'mismatch': 0,
    }
    with scores_p.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as in_f, out_p.open(
        'w',
        encoding='utf-8',
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(
                row,
                dict,
            ):
                continue
            meta = (
                row.get('meta')
                if isinstance(
                    row.get('meta'),
                    dict,
                )
                else {}
            )
            path_raw = meta.get('path')
            file_key = (
                _canon_path(path_raw)
                if isinstance(
                    path_raw,
                    str,
                )
                else ''
            )
            adj = adjudications.get(file_key)
            (
                new_row,
                status,
            ) = _apply_one(
                row,
                emotions=emotions,
                adjudication=adj,
                scale_words=float(opts.scale_words),
            )
            counts['docs'] += 1
            if status in counts:
                counts[status] += 1
            out_f.write(json.dumps(
                    new_row,
                    ensure_ascii=False,
                )
                + '\n')
    summary = {
        'scores_jsonl': str(scores_p),
        'openrouter_txt': str(adjud_p),
        'out': str(out_p),
        'emotions': list(emotions),
        'scale_words': float(opts.scale_words),
        'adjudication_parse': adjud_stats,
        'apply': counts,
        'unique_adjudicated_files': int(len(adjudications)),
    }
    print(json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
