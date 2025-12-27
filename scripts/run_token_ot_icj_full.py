#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import (
    iter_text_paths,
    parse_icj_meta,
)
from legal_emotion.token_compare import compare_token_clouds


def _write_jsonl(
    path: Path,
    rows: Iterable[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        'w',
        encoding='utf-8',
    ) as f:
        for r in rows:
            f.write(json.dumps(
                    r,
                    ensure_ascii=False,
                ) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description='token-ot full run')
    ap.add_argument(
        '--data_dir',
        type=str,
        default='data/EN_TXT_BEST_FULL',
    )
    ap.add_argument(
        '--config',
        type=str,
        default='config.icj.gpu.yaml',
    )
    ap.add_argument(
        '--out_dir',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--topk',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--candidate_k',
        type=int,
        default=50,
    )
    ap.add_argument(
        '--mode',
        type=str,
        default='unbalanced_divergence',
    )
    ap.add_argument(
        '--focus',
        type=str,
        default='emotional',
        choices=('all', 'emotional'),
    )
    ap.add_argument(
        '--cost',
        type=str,
        default='embedding_vad',
        choices=('embedding', 'vad', 'embedding_vad'),
    )
    ap.add_argument(
        '--alpha_embed',
        type=float,
        default=0.8,
    )
    ap.add_argument(
        '--beta_vad',
        type=float,
        default=0.2,
    )
    ap.add_argument(
        '--vad_threshold',
        type=float,
        default=0.45,
    )
    ap.add_argument(
        '--emotional_vocab',
        type=str,
        default='lexicon',
        choices=('lexicon', 'vad', 'lexicon_or_vad'),
    )
    ap.add_argument(
        '--vad_min_arousal_vad_only',
        type=float,
        default=0.35,
    )
    ap.add_argument(
        '--max_ngram',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--weight',
        type=str,
        default='tf',
        choices=('tfidf', 'tf', 'uniform'),
    )
    ap.add_argument(
        '--term_weights_path',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--term_weight_power',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--term_weight_min',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--term_weight_max',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--term_weight_mix',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--term_weight_default',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--vad_imputed_path',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--max_terms',
        type=int,
        default=256,
    )
    ap.add_argument(
        '--drop_top_df',
        type=int,
        default=100,
    )
    ap.add_argument(
        '--no_stopwords',
        action='store_true',
    )
    ap.add_argument(
        '--stopwords_file',
        type=str,
        default='data/stopwords_legal_en_token_ot.txt',
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
        '--no_explain',
        action='store_true',
    )
    ap.add_argument(
        '--no_vis',
        action='store_true',
    )
    opts = ap.parse_args()
    out_dir = (
        Path(opts.out_dir)
        if opts.out_dir
        else Path('outputs')
        / time.strftime('token_ot_icj_full_%Y%m%d_%H%M%S')
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    data_dir = Path(opts.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    input_jsonl = out_dir / 'icj_full_docs.jsonl'
    rows = []
    for p in iter_text_paths(data_dir):
        meta = parse_icj_meta(p)
        meta.setdefault(
            'id',
            p.stem,
        )
        meta['path'] = str(p)
        rows.append({'meta': meta})
        if opts.limit is not None and len(rows) >= int(opts.limit):
            break
    if not rows:
        raise ValueError('error: ValueError')
    _write_jsonl(
        input_jsonl,
        rows,
    )
    out_jsonl = out_dir / 'neighbours.jsonl'
    out_html = out_dir / 'neighbours.html'
    stats = compare_token_clouds(
        input_jsonl=str(input_jsonl),
        output_path=str(out_jsonl),
        cfg_path=str(opts.config) if opts.config else None,
        fmt='neighbours',
        topk=int(opts.topk),
        candidate_k=int(opts.candidate_k),
        mode=str(opts.mode),
        focus=opts.focus,
        cost=opts.cost,
        embed_model=opts.embed_model,
        embed_backend=opts.embed_backend,
        embed_pooling=opts.embed_pooling,
        embed_batch_size=opts.embed_batch_size,
        embed_max_length=opts.embed_max_length,
        embed_prompt_mode=opts.embed_prompt_mode,
        embed_prompt_text=opts.embed_prompt_text,
        alpha_embed=float(opts.alpha_embed),
        beta_vad=float(opts.beta_vad),
        vad_threshold=float(opts.vad_threshold),
        emotional_vocab=str(opts.emotional_vocab),
        vad_min_arousal_vad_only=float(opts.vad_min_arousal_vad_only),
        max_ngram=int(opts.max_ngram),
        weight=str(opts.weight),
        term_weights_path=opts.term_weights_path,
        term_weight_power=opts.term_weight_power,
        term_weight_min=opts.term_weight_min,
        term_weight_max=opts.term_weight_max,
        term_weight_mix=opts.term_weight_mix,
        term_weight_default=opts.term_weight_default,
        vad_imputed_path=opts.vad_imputed_path,
        max_terms=int(opts.max_terms),
        stopwords=not bool(opts.no_stopwords),
        stopwords_file=(
            str(opts.stopwords_file)
            if opts.stopwords_file
            else None
        ),
        drop_top_df=int(opts.drop_top_df),
        include_explain=not bool(opts.no_explain),
        vis=not bool(opts.no_vis),
        vis_path=str(out_html),
    )
    (out_dir / 'run_stats.json').write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    print(json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
