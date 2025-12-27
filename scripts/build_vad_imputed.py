#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(
    0,
    str(ROOT),
)
from legal_emotion.lexicon import (
    load_lexicon,
    load_word_vad,
    tokenize as lex_tokenize,
)
from legal_emotion.token_compare import _embed_terms
from legal_emotion.utils import get_device, load_config


def _canonical(term: str) -> str:
    return ' '.join(lex_tokenize(term))


def _unique_terms(terms: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for t in terms:
        key = _canonical(t)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_term_weights_terms(path: Optional[str | Path]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding='utf-8'))
    data = payload
    if isinstance(
        payload,
        dict,
    ) and 'weights' in payload:
        data = payload['weights']
    if isinstance(
        data,
        dict,
    ):
        return list(data.keys())
    if isinstance(
        data,
        list,
    ):
        out = []
        for entry in data:
            if isinstance(
                entry,
                dict,
            ):
                term = (
                    entry.get('term')
                    or entry.get('token')
                    or entry.get('word')
                )
                if term:
                    out.append(str(term))
            elif isinstance(
                entry,
                (list, tuple),
            ) and entry:
                out.append(str(entry[0]))
        return out
    return []


def _max_cosine_conf(
    queries: torch.Tensor,
    refs: torch.Tensor,
    *,
    batch: int,
) -> List[float]:
    out: List[float] = []
    refs = refs / refs.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    qn = queries / queries.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    for start in range(
        0,
        qn.size(0),
        batch,
    ):
        q = qn[start : start + batch]
        sims = q @ refs.T
        (
            max_sim,
            _,
        ) = sims.max(dim=1)
        conf = (max_sim + 1.0) * 0.5
        out.extend([
                float(x)
                for x in conf.clamp(
                    0.0,
                    1.0,
                ).tolist()
            ])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='impute vad')
    ap.add_argument(
        '--config',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--out',
        type=str,
        default='data/derived/token_vad_imputed.json',
    )
    ap.add_argument(
        '--term_weights_path',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--extra_terms',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--embed_model',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--embed_backend',
        type=str,
        default=None,
        choices=('encoder', 'input_embeddings'),
    )
    ap.add_argument(
        '--embed_pooling',
        type=str,
        default=None,
        choices=('cls', 'mean'),
    )
    ap.add_argument(
        '--embed_batch_size',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--embed_max_length',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--embed_prompt_mode',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--embed_prompt_text',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--max_train',
        type=int,
        default=20000,
    )
    ap.add_argument(
        '--max_candidates',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--min_conf',
        type=float,
        default=0.2,
    )
    ap.add_argument(
        '--ridge_alpha',
        type=float,
        default=1.0,
    )
    ap.add_argument(
        '--sim_batch',
        type=int,
        default=512,
    )
    opts = ap.parse_args()
    setup = load_config(opts.config)
    device = get_device(getattr(
            setup,
            'device',
            'cpu',
        ))
    embed_model = str(opts.embed_model
        or getattr(
            setup,
            'token_ot_embed_model',
            None,
        )
        or getattr(
            setup,
            'embed_model_name',
            None,
        )
        or getattr(
            setup,
            'model_name',
            None,
        )
        or setup.model_name)
    embed_backend = str(opts.embed_backend
        or getattr(
            setup,
            'token_ot_embed_backend',
            None,
        )
        or 'encoder')
    embed_pooling = str(opts.embed_pooling
        or getattr(
            setup,
            'token_ot_embed_pooling',
            None,
        )
        or 'cls')
    embed_bs = int(opts.embed_batch_size
        if opts.embed_batch_size is not None
        else getattr(
            setup,
            'token_ot_embed_batch_size',
            64,
        ))
    embed_max_len = int(opts.embed_max_length
        if opts.embed_max_length is not None
        else getattr(
            setup,
            'token_ot_embed_max_length',
            32,
        ))
    embed_prompt_mode = str(opts.embed_prompt_mode
        or getattr(
            setup,
            'token_ot_embed_prompt_mode',
            None,
        )
        or 'none')
    embed_prompt_text = (
        opts.embed_prompt_text
        if opts.embed_prompt_text is not None
        else getattr(
            setup,
            'token_ot_embed_prompt_text',
            None,
        )
    )
    word_vad_path = getattr(
        setup,
        'vad_lexicon_path',
        None,
    )
    if not word_vad_path:
        raise SystemExit('error: SystemExit')
    word_vad_raw = load_word_vad(
        word_vad_path,
        vad_scale=getattr(
            setup,
            'word_vad_scale',
            None,
        ),
    )
    word_vad: Dict[str, Tuple[float, float, float]] = {}
    for term, vec in word_vad_raw.items():
        key = _canonical(term)
        if not key:
            continue
        word_vad[key] = (
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
        )
    train_terms = sorted(word_vad.keys())
    if not train_terms:
        raise SystemExit('error: SystemExit')
    if opts.max_train:
        train_terms = train_terms[
            : max(
                1,
                int(opts.max_train),
            )
        ]
    lex = load_lexicon(
        getattr(
            setup,
            'lexicon_path',
            None,
        ),
        getattr(
            setup,
            'vad_lexicon_path',
            None,
        ),
        lexicon_vad_scale=getattr(
            setup,
            'lexicon_vad_scale',
            None,
        ),
        word_vad_scale=getattr(
            setup,
            'word_vad_scale',
            None,
        ),
        stopwords_path=getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        ),
        extra_path=getattr(
            setup,
            'lexicon_extra_path',
            None,
        ),
        intensity_path=getattr(
            setup,
            'lexicon_intensity_path',
            None,
        ),
        intensity_min=getattr(
            setup,
            'lexicon_intensity_min',
            0.0,
        ),
        min_vad_salience=getattr(
            setup,
            'lexicon_min_vad_salience',
            0.0,
        ),
        min_vad_arousal=getattr(
            setup,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        require_word_vad=bool(getattr(
                setup,
                'lexicon_require_word_vad',
                False,
            )),
    )
    lex_terms = _unique_terms((
            t
            for entries in lex.values()
            for t in entries.keys()
        ))
    weight_terms = _unique_terms(_load_term_weights_terms(opts.term_weights_path
            or getattr(
                setup,
                'token_term_weights_path',
                None,
            )))
    extra_terms: List[str] = []
    if opts.extra_terms:
        p = Path(opts.extra_terms)
        if p.exists():
            extra_terms = _unique_terms((
                    line.strip()
                    for line in p.read_text(
                        encoding='utf-8',
                        errors='ignore',
                    ).splitlines()
                ))
    candidate_terms = _unique_terms(lex_terms + weight_terms + extra_terms)
    known = set(train_terms)
    candidate_terms = [
        t for t in candidate_terms if t not in known
    ]
    if opts.max_candidates:
        candidate_terms = candidate_terms[
            : max(
                1,
                int(opts.max_candidates),
            )
        ]
    if not candidate_terms:
        raise SystemExit('error: SystemExit')
    term_to_emb = _embed_terms(
        train_terms + candidate_terms,
        model_name=embed_model,
        device=device,
        backend=embed_backend,
        pooling=embed_pooling,
        batch_size=int(embed_bs),
        max_length=int(embed_max_len),
        amp=str(getattr(
                setup,
                'amp',
                None,
            ) or 'none'),
        prompt_mode=embed_prompt_mode,
        prompt_text=embed_prompt_text,
    )
    X_train = np.stack(
        [term_to_emb[t].numpy() for t in train_terms],
        axis=0,
    )
    y_train = np.stack(
        [
            np.array(
                word_vad[t],
                dtype=np.float32,
            )
            for t in train_terms
        ],
        axis=0,
    )
    model = Ridge(alpha=float(opts.ridge_alpha))
    model.fit(
        X_train,
        y_train,
    )
    X_guess = np.stack(
        [term_to_emb[t].numpy() for t in candidate_terms],
        axis=0,
    )
    preds = model.predict(X_guess)
    preds = np.clip(
        preds,
        -1.0,
        1.0,
    )
    refs = torch.tensor(
        X_train,
        dtype=torch.float,
    )
    queries = torch.tensor(
        X_guess,
        dtype=torch.float,
    )
    if device.type != 'cpu':
        refs = refs.to(device=device)
        queries = queries.to(device=device)
    confs = _max_cosine_conf(
        queries,
        refs,
        batch=int(opts.sim_batch),
    )
    entries = []
    min_conf = float(opts.min_conf)
    for term, vec, conf in zip(
        candidate_terms,
        preds,
        confs,
    ):
        if conf < min_conf:
            continue
        entries.append({
                'term': term,
                'vad': [
                    float(vec[0]),
                    float(vec[1]),
                    float(vec[2]),
                ],
                'conf': float(conf),
            })
    out_path = Path(opts.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = {
        '__meta__': {
            'embed_model': embed_model,
            'embed_backend': embed_backend,
            'embed_pooling': embed_pooling,
            'embed_batch_size': int(embed_bs),
            'embed_max_length': int(embed_max_len),
            'embed_prompt_mode': embed_prompt_mode,
            'embed_prompt_text': (
                str(embed_prompt_text)
                if embed_prompt_text
                else None
            ),
            'n_train': int(len(train_terms)),
            'n_candidates': int(len(candidate_terms)),
            'n_written': int(len(entries)),
            'min_conf': float(min_conf),
            'ridge_alpha': float(opts.ridge_alpha),
        },
        'terms': entries,
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
                'n_written': int(len(entries)),
            },
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
