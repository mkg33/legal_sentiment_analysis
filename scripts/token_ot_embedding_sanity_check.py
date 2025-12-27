#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = (
    ROOT / 'scripts' / 'token_ot_meaningfulness_check.py'
)


def _maybe_add(
    cmd: List[str],
    flag: str,
    value: Optional[str | int | float],
) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _run_variant(
    *,
    name: str,
    out_dir: Path,
    base_args: argparse.Namespace,
    model: str,
    pooling: Optional[str],
    prompt_mode: Optional[str],
    prompt_text: Optional[str],
    embed_batch_size: Optional[int],
    embed_max_length: Optional[int],
) -> Dict[str, Any]:
    variant_dir = out_dir / name
    cmd = [
        sys.executable,
        str(CHECK_SCRIPT),
        '--input_dir',
        base_args.input_dir,
        '--config',
        base_args.config,
        '--out_dir',
        str(variant_dir),
        '--n_per_category',
        str(base_args.n_per_category),
        '--random_n',
        str(base_args.random_n),
        '--seed',
        str(base_args.seed),
        '--mode',
        base_args.mode,
        '--focus',
        base_args.focus,
        '--cost',
        base_args.cost,
        '--alpha_embed',
        str(base_args.alpha_embed),
        '--beta_vad',
        str(base_args.beta_vad),
        '--vad_threshold',
        str(base_args.vad_threshold),
        '--emotional_vocab',
        base_args.emotional_vocab,
        '--vad_min_arousal_vad_only',
        str(base_args.vad_min_arousal_vad_only),
        '--max_ngram',
        str(base_args.max_ngram),
        '--weight',
        base_args.weight,
        '--max_terms',
        str(base_args.max_terms),
        '--min_token_len',
        str(base_args.min_token_len),
        '--drop_top_df',
        str(base_args.drop_top_df),
        '--top_flows',
        str(base_args.top_flows),
        '--explain_pairs',
        str(base_args.explain_pairs),
    ]
    if base_args.no_stopwords:
        cmd.append('--no_stopwords')
    if base_args.stopwords_file:
        cmd.extend(['--stopwords_file', base_args.stopwords_file])
    _maybe_add(
        cmd,
        '--embed_model',
        model,
    )
    _maybe_add(
        cmd,
        '--embed_pooling',
        pooling,
    )
    _maybe_add(
        cmd,
        '--embed_prompt_mode',
        prompt_mode,
    )
    _maybe_add(
        cmd,
        '--embed_prompt_text',
        prompt_text,
    )
    _maybe_add(
        cmd,
        '--embed_batch_size',
        embed_batch_size,
    )
    _maybe_add(
        cmd,
        '--embed_max_length',
        embed_max_length,
    )
    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError('error: RuntimeError')
    summary_path = variant_dir / 'summary.json'
    report_path = variant_dir / 'report.md'
    if not summary_path.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    return {
        'name': name,
        'model': model,
        'pooling': pooling,
        'prompt_mode': prompt_mode,
        'prompt_text': prompt_text,
        'embed_batch_size': embed_batch_size,
        'embed_max_length': embed_max_length,
        'summary': summary,
        'summary_path': str(summary_path),
        'report_path': str(report_path),
        'stdout': out.stdout,
        'stderr': out.stderr,
    }


def _format_score(val: Optional[float]) -> str:
    if val is None:
        return 'n/a'
    return f'{val:.1f}'


def main() -> int:
    ap = argparse.ArgumentParser(description='embedding check')
    ap.add_argument(
        '--input_dir',
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
        '--n_per_category',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--random_n',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=13,
    )
    ap.add_argument(
        '--mode',
        type=str,
        default='sinkhorn_divergence',
    )
    ap.add_argument(
        '--focus',
        type=str,
        default='emotional',
    )
    ap.add_argument(
        '--cost',
        type=str,
        default='embedding_vad',
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
        default=0.55,
    )
    ap.add_argument(
        '--emotional_vocab',
        type=str,
        default='lexicon_or_vad',
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
        default='tfidf',
    )
    ap.add_argument(
        '--max_terms',
        type=int,
        default=128,
    )
    ap.add_argument(
        '--min_token_len',
        type=int,
        default=2,
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
        '--drop_top_df',
        type=int,
        default=100,
    )
    ap.add_argument(
        '--top_flows',
        type=int,
        default=12,
    )
    ap.add_argument(
        '--explain_pairs',
        type=int,
        default=6,
    )
    ap.add_argument(
        '--model_a',
        type=str,
        default='BAAI/bge-m3',
    )
    ap.add_argument(
        '--pooling_a',
        type=str,
        default='cls',
    )
    ap.add_argument(
        '--prompt_mode_a',
        type=str,
        default='none',
    )
    ap.add_argument(
        '--prompt_text_a',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--embed_batch_size_a',
        type=int,
        default=32,
    )
    ap.add_argument(
        '--embed_max_length_a',
        type=int,
        default=32,
    )
    ap.add_argument(
        '--model_b',
        type=str,
        default='intfloat/e5-mistral-7b-instruct',
    )
    ap.add_argument(
        '--pooling_b',
        type=str,
        default='mean',
    )
    ap.add_argument(
        '--prompt_mode_b',
        type=str,
        default='e5-mistral',
    )
    ap.add_argument(
        '--prompt_text_b',
        type=str,
        default='Represent the query for retrieval',
    )
    ap.add_argument(
        '--embed_batch_size_b',
        type=int,
        default=8,
    )
    ap.add_argument(
        '--embed_max_length_b',
        type=int,
        default=64,
    )
    opts = ap.parse_args()
    out_dir = (
        Path(opts.out_dir)
        if opts.out_dir
        else Path('outputs')
        / time.strftime('token_ot_embed_sanity_%Y%m%d_%H%M%S')
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    variants = [
        {
            'name': 'model_a',
            'model': opts.model_a,
            'pooling': opts.pooling_a,
            'prompt_mode': opts.prompt_mode_a,
            'prompt_text': opts.prompt_text_a,
            'embed_batch_size': opts.embed_batch_size_a,
            'embed_max_length': opts.embed_max_length_a,
        },
        {
            'name': 'model_b',
            'model': opts.model_b,
            'pooling': opts.pooling_b,
            'prompt_mode': opts.prompt_mode_b,
            'prompt_text': opts.prompt_text_b,
            'embed_batch_size': opts.embed_batch_size_b,
            'embed_max_length': opts.embed_max_length_b,
        },
    ]
    outs: List[Dict[str, Any]] = []
    for v in variants:
        outs.append(_run_variant(
                name=v['name'],
                out_dir=out_dir,
                base_args=opts,
                model=v['model'],
                pooling=v['pooling'],
                prompt_mode=v['prompt_mode'],
                prompt_text=v['prompt_text'],
                embed_batch_size=v['embed_batch_size'],
                embed_max_length=v['embed_max_length'],
            ))
    scores = []
    for r in outs:
        metrics = r.get(
            'summary',
            {},
        ).get(
            'metrics',
            {},
        )
        scores.append((r['name'], float(metrics.get(
                    'score',
                    0.0,
                ))))
    best = (
        max(
            scores,
            key=lambda x: x[1],
        )[0]
        if scores
        else None
    )
    comparison = {
        'out_dir': str(out_dir),
        'best_variant': best,
        'variants': outs,
    }
    (out_dir / 'comparison.json').write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    md_path = out_dir / 'comparison.md'
    with md_path.open(
        'w',
        encoding='utf-8',
    ) as f:
        f.write('# Token-OT embedding sanity check\n\n')
        f.write(f'- out_dir: `{out_dir}`\n')
        if best:
            f.write(f'- best_variant: `{best}`\n')
        f.write('\n| variant | model | pooling | prompt_mode | score | verdict | coverage_total | top1_rate |\n')
        f.write('|---|---|---|---|---:|---|---:|---:|\n')
        for r in outs:
            metrics = r.get(
                'summary',
                {},
            ).get(
                'metrics',
                {},
            )
            f.write(f"| {r['name']} | {r['model']} | {r.get('pooling')} | {r.get('prompt_mode')} | {_format_score(metrics.get('score'))} | {metrics.get('verdict')} | {metrics.get(
                    'coverage_total',
                    0.0,
                ):.4f} | {metrics.get('top1_same_category_rate') or 0.0:.3f} |\n")
        f.write('\nPer-variant reports:\n\n')
        for r in outs:
            f.write(f"- {r['name']}: `{r['report_path']}`\n")
    print(json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
