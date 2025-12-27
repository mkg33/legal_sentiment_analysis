#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(
    0,
    str(ROOT),
)
from legal_emotion.lexicon import (
    load_word_vad,
    tokenize as lex_tokenize,
)
from legal_emotion.token_compare import _vad_salience
from legal_emotion.utils import load_config


def _find_intensity_file(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.rglob('*.txt'))
    if not candidates:
        return None
    for cand in candidates:
        if (
            cand.name
            == 'NRC-Emotion-Intensity-Lexicon-v1.txt'
        ):
            return cand
    for cand in candidates:
        name = cand.name.lower()
        if (
            'intensity' in name
            and 'forvariouslanguages' not in name
        ):
            return cand
    return candidates[0]


def _load_intensity(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            term_raw = parts[0].strip()
            try:
                score = float(parts[2])
            except Exception:
                continue
            term = ' '.join(lex_tokenize(term_raw))
            if not term:
                continue
            out[term] = max(
                out.get(
                    term,
                    0.0,
                ),
                float(score),
            )
    return out


def _load_vad_salience(
    path: Optional[Path],
    *,
    scale: Optional[str],
) -> Dict[str, float]:
    if not path:
        return {}
    vad = load_word_vad(
        str(path),
        vad_scale=scale,
    )
    out: Dict[str, float] = {}
    for term, vec in vad.items():
        key = ' '.join(lex_tokenize(term))
        if not key:
            continue
        v = torch.tensor(
            [float(vec[0]), float(vec[1]), float(vec[2])],
            dtype=torch.float,
        )
        out[key] = float(_vad_salience(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='build term weights')
    ap.add_argument(
        '--config',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--intensity_path',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--vad_path',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--out',
        type=str,
        default='data/derived/token_term_weights.json',
    )
    ap.add_argument(
        '--intensity_weight',
        type=float,
        default=0.6,
    )
    ap.add_argument(
        '--salience_weight',
        type=float,
        default=0.4,
    )
    ap.add_argument(
        '--min_weight',
        type=float,
        default=0.05,
    )
    opts = ap.parse_args()
    setup = load_config(opts.config)
    intensity_path = (
        Path(opts.intensity_path)
        if opts.intensity_path
        else None
    )
    if intensity_path is None:
        intensity_path = Path('data/external/NRC-Emotion-Intensity-Lexicon-v1')
        if not intensity_path.exists():
            intensity_path = Path('data/NRC-Emotion-Intensity-Lexicon-v1')
    vad_path = (
        Path(opts.vad_path)
        if opts.vad_path
        else Path(getattr(
                setup,
                'vad_lexicon_path',
                None,
            )
            or 'data/NRC-VAD-Lexicon-v2.1')
    )
    vad_scale = getattr(
        setup,
        'word_vad_scale',
        None,
    )
    intensity_file = (
        _find_intensity_file(intensity_path)
        if intensity_path
        else None
    )
    if (
        intensity_file is None
        or not intensity_file.exists()
    ):
        raise SystemExit('error: SystemExit')
    intensity = _load_intensity(intensity_file)
    salience = _load_vad_salience(
        vad_path if vad_path.exists() else None,
        scale=vad_scale,
    )
    w_int = float(opts.intensity_weight)
    w_sal = float(opts.salience_weight)
    denom = w_int + w_sal
    if denom <= 0:
        denom = 1.0
    weights: Dict[str, float] = {}
    for term in set(intensity.keys()) | set(salience.keys()):
        v_int = float(intensity.get(
                term,
                0.0,
            ))
        v_sal = float(salience.get(
                term,
                0.0,
            ))
        if v_int <= 0.0 and v_sal <= 0.0:
            continue
        w = (w_int * v_int + w_sal * v_sal) / denom
        if w < float(opts.min_weight):
            continue
        weights[term] = float(min(
                1.0,
                max(
                    0.0,
                    w,
                ),
            ))
    vals = list(weights.values())
    meta = {
        'source': {
            'intensity': str(intensity_file),
            'vad': (
                str(vad_path) if vad_path.exists() else None
            ),
        },
        'intensity_weight': w_int,
        'salience_weight': w_sal,
        'min_weight': float(opts.min_weight),
        'n_terms': int(len(weights)),
        'min': float(min(vals)) if vals else None,
        'max': float(max(vals)) if vals else None,
        'mean': (
            float(statistics.mean(vals)) if vals else None
        ),
        'median': (
            float(statistics.median(vals)) if vals else None
        ),
    }
    out_path = Path(opts.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = {
        '__meta__': meta,
        'weights': weights,
    }
    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    print(json.dumps(
            {
                'out': str(out_path),
                'n_terms': int(len(weights)),
                'meta': meta,
            },
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
