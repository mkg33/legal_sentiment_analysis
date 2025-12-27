#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

ROOT = Path(__file__).resolve().parents[1]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        return json.load(f)


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _eot_doc_type_coherence(neighbours_jsonl: Path) -> Dict[str, Any]:
    orig_to_doc_type: Dict[int, str] = {}
    top1_pairs: List[Tuple[int, int, float]] = []
    docs = 0
    for row in _iter_jsonl(neighbours_jsonl):
        docs += 1
        oi = row.get('orig_index')
        if not isinstance(
            oi,
            int,
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
        doc_type = str(meta.get('doc_type') or 'unknown')
        orig_to_doc_type[oi] = doc_type
        neigh = row.get('neighbours')
        if isinstance(
            neigh,
            list,
        ) and neigh:
            first = (
                neigh[0]
                if isinstance(
                    neigh[0],
                    dict,
                )
                else None
            )
            if first and isinstance(
                first.get('orig_index'),
                int,
            ):
                top1_pairs.append((
                        oi,
                        int(first['orig_index']),
                        float(first.get('distance') or 0.0),
                    ))
    eligible = 0
    matches = 0
    dist_sum = 0.0
    for oi, oj, d in top1_pairs:
        a = orig_to_doc_type.get(
            oi,
            'unknown',
        )
        b = orig_to_doc_type.get(
            oj,
            'unknown',
        )
        if 'unknown' in {a, b}:
            continue
        eligible += 1
        dist_sum += float(d)
        if a == b:
            matches += 1
    return {
        'docs': int(docs),
        'top1_denom': int(eligible),
        'top1_same_doc_type_rate': (
            float(matches / eligible) if eligible else None
        ),
        'top1_distance_mean': (
            float(dist_sum / eligible) if eligible else None
        ),
    }


def _score_signal_stats(scores_jsonl: Path) -> Dict[str, Any]:
    total = 0
    low = 0
    sig_sum = 0.0
    sig_n = 0
    for row in _iter_jsonl(scores_jsonl):
        total += 1
        if row.get('low_emotion_signal') is True:
            low += 1
        sig = row.get('emotion_signal_per_1k_words')
        if isinstance(
            sig,
            (int, float),
        ):
            sig_sum += float(sig)
            sig_n += 1
    return {
        'docs': int(total),
        'low_emotion_signal': int(low),
        'low_emotion_signal_rate': (
            float(low / total) if total else None
        ),
        'emotion_signal_per_1k_words_mean': (
            float(sig_sum / sig_n) if sig_n else None
        ),
    }


def _run_cmd(
    cmd: List[str],
    *,
    env: Dict[str, str],
    log,
) -> None:
    log('cmd')
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT),
        env=env,
    )


def _collect_run_metrics(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {'run_dir': str(run_dir)}
    token_check = (
        run_dir / 'token_ot_check' / 'summary.json'
    )
    if token_check.exists():
        out['token_ot_check'] = _read_json(token_check)
    best_settings = (
        run_dir / 'token_ot_tuning' / 'best_settings.json'
    )
    if best_settings.exists():
        out['token_ot_best_settings'] = _read_json(best_settings)
    scores = run_dir / 'icj_scores.jsonl'
    if scores.exists():
        out['icj_score_signal'] = _score_signal_stats(scores)
    neighbours = run_dir / 'icj_neighbours.jsonl'
    if neighbours.exists():
        out['icj_eot_doc_type_coherence'] = (
            _eot_doc_type_coherence(neighbours)
        )
    cfg_resolved = run_dir / 'emotion_config.resolved.yaml'
    if cfg_resolved.exists():
        out['emotion_config_resolved'] = str(cfg_resolved)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='warm-start ablation')
    ap.add_argument(
        '--emotion_config',
        type=str,
        default='config.icj.gpu.yaml',
    )
    ap.add_argument(
        '--sentiment_config',
        type=str,
        default='config.sentiment.sample.yaml',
    )
    ap.add_argument(
        '--emotion_num_epochs',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--sentiment_num_epochs',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--icj_dir',
        type=str,
        default='data/EN_TXT_BEST_FULL',
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=13,
    )
    ap.add_argument(
        '--stride',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--force',
        action='store_true',
    )
    ap.add_argument(
        '--purge_existing',
        action='store_true',
    )
    ap.add_argument(
        '--safe_dataloader',
        action='store_true',
    )
    ap.add_argument(
        '--gold_jsonl',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--gold_dev_ratio',
        type=float,
        default=0.2,
    )
    ap.add_argument(
        '--sigmalaw_csv',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--warm_start',
        type=str,
        default='both',
        choices=('on', 'off', 'both'),
    )
    ap.add_argument(
        '--max_iters',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--token_guard_min_coverage',
        type=float,
        default=0.03,
    )
    ap.add_argument(
        '--token_guard_min_selected_ratio',
        type=float,
        default=0.02,
    )
    ap.add_argument(
        '--token_guard_max_zero_ratio',
        type=float,
        default=0.02,
    )
    ap.add_argument(
        '--token_guard_min_score',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--token_guard_min_separation_effect',
        type=float,
        default=0.8,
    )
    ap.add_argument(
        '--token_guard_allow_warn',
        action='store_true',
    )
    ap.add_argument(
        '--progress_every',
        type=int,
        default=500,
    )
    ap.add_argument(
        '--check_n_per_category',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--check_random_n',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--check_seed',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--check_selected_jsonl',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--check_format',
        type=str,
        default='neighbours',
        choices=('matrix', 'neighbours'),
    )
    ap.add_argument(
        '--check_topk',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--check_candidate_k',
        type=int,
        default=200,
    )
    ap.add_argument(
        '--check_top_flows',
        type=int,
        default=12,
    )
    ap.add_argument(
        '--check_explain_pairs',
        type=int,
        default=6,
    )
    ap.add_argument(
        '--vad_threshold_start',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--vad_threshold_min',
        type=float,
        default=0.0,
    )
    ap.add_argument(
        '--vad_threshold_max',
        type=float,
        default=0.9,
    )
    ap.add_argument(
        '--vad_threshold_step',
        type=float,
        default=0.02,
    )
    ap.add_argument(
        '--drop_top_df_start',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--drop_top_df_min',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--drop_top_df_max',
        type=int,
        default=500,
    )
    ap.add_argument(
        '--drop_top_df_step',
        type=int,
        default=5,
    )
    ap.add_argument(
        '--vad_min_arousal_start',
        type=float,
        default=None,
    )
    ap.add_argument(
        '--vad_min_arousal_min',
        type=float,
        default=0.0,
    )
    ap.add_argument(
        '--vad_min_arousal_max',
        type=float,
        default=1.0,
    )
    ap.add_argument(
        '--vad_min_arousal_step',
        type=float,
        default=0.02,
    )
    ap.add_argument(
        '--max_terms_start',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--max_terms_min',
        type=int,
        default=128,
    )
    ap.add_argument(
        '--max_terms_max',
        type=int,
        default=2048,
    )
    ap.add_argument(
        '--max_terms_step',
        type=int,
        default=64,
    )
    ap.add_argument(
        '--restart_on_stuck',
        action='store_true',
    )
    ap.add_argument(
        '--restart_seed',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--restart_attempts',
        type=int,
        default=500,
    )
    ap.add_argument(
        '--restart_emotional_vocab',
        type=str,
        default='lexicon_or_vad,lexicon',
    )
    ap.add_argument(
        '--out_dir',
        type=str,
        default='outputs',
    )
    ap.add_argument(
        '--tag',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--quiet',
        action='store_true',
    )
    opts = ap.parse_args()

    def log(msg: str) -> None:
        if bool(opts.quiet):
            return
        print(
            msg,
            file=sys.stderr,
            flush=True,
        )

    out_base = Path(opts.out_dir)
    out_base.mkdir(
        parents=True,
        exist_ok=True,
    )
    stamp = time.strftime('%Y%m%d_%H%M%S')
    tag = f'_{opts.tag}' if opts.tag else ''
    variants: List[Tuple[str, List[str]]] = []
    if opts.warm_start in {'on', 'both'}:
        variants.append(('warm', []))
    if opts.warm_start in {'off', 'both'}:
        variants.append(('cold', ['--no_emotion_warm_start']))
    env = dict(os.environ)
    env.setdefault(
        'PYTHONUNBUFFERED',
        '1',
    )
    env.setdefault(
        'TOKENIZERS_PARALLELISM',
        'false',
    )
    outs: Dict[str, Any] = {
        'stamp': stamp,
        'variants': {},
    }
    for name, extra_flags in variants:
        run_name = f'gold_to_icj_ot_{stamp}_{name}{tag}'
        run_dir = (out_base / run_name).resolve()
        if run_dir.exists():
            if bool(opts.purge_existing):
                log(f'purge {run_dir}')
                _safe_rmtree(run_dir)
            else:
                raise SystemExit('error: SystemExit')
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        log(f'pipeline {name}')
        pipeline_cmd = [
            sys.executable,
            '-u',
            'scripts/run_gold_to_icj_ot_pipeline.py',
            '--run_dir',
            str(run_dir),
            '--seed',
            str(int(opts.seed)),
            '--sentiment_config',
            str(opts.sentiment_config),
            '--emotion_config',
            str(opts.emotion_config),
            *(
                [
                    '--sentiment_num_epochs',
                    str(int(opts.sentiment_num_epochs)),
                ]
                if opts.sentiment_num_epochs is not None
                else []
            ),
            *(
                [
                    '--emotion_num_epochs',
                    str(int(opts.emotion_num_epochs)),
                ]
                if opts.emotion_num_epochs is not None
                else []
            ),
            '--icj_dir',
            str(opts.icj_dir),
            '--stride',
            str(int(opts.stride)),
            '--no_token_ot',
            '--no_token_check',
            '--no_group_shift',
            '--no_token_guard',
        ]
        if opts.gold_jsonl:
            pipeline_cmd += [
                '--gold_jsonl',
                str(opts.gold_jsonl),
                '--gold_dev_ratio',
                str(float(opts.gold_dev_ratio)),
            ]
        if opts.sigmalaw_csv:
            pipeline_cmd += [
                '--sigmalaw_csv',
                str(opts.sigmalaw_csv),
            ]
        if opts.limit is not None:
            pipeline_cmd += [
                '--limit',
                str(int(opts.limit)),
            ]
        if bool(opts.force):
            pipeline_cmd.append('--force')
        if bool(opts.safe_dataloader):
            pipeline_cmd.append('--safe_dataloader')
        pipeline_cmd += list(extra_flags)
        _run_cmd(
            pipeline_cmd,
            env=env,
            log=log,
        )
        log(f'auto-tune {name}')
        check_seed = (
            int(opts.check_seed)
            if opts.check_seed is not None
            else int(opts.seed)
        )
        tune_cmd = [
            sys.executable,
            '-u',
            'scripts/auto_tune_token_ot_guardrails.py',
            '--run_dir',
            str(run_dir),
            '--icj_dir',
            str(opts.icj_dir),
            '--sentiment_config',
            str(opts.sentiment_config),
            '--max_iters',
            str(int(opts.max_iters)),
            '--token_guard_min_coverage',
            str(float(opts.token_guard_min_coverage)),
            '--token_guard_min_selected_ratio',
            str(float(opts.token_guard_min_selected_ratio)),
            '--token_guard_max_zero_ratio',
            str(float(opts.token_guard_max_zero_ratio)),
            '--token_guard_min_separation_effect',
            str(float(opts.token_guard_min_separation_effect)),
            '--progress_every',
            str(int(opts.progress_every)),
            '--check_format',
            str(opts.check_format),
            '--check_topk',
            str(int(opts.check_topk)),
            '--check_candidate_k',
            str(int(opts.check_candidate_k)),
            '--check_top_flows',
            str(int(opts.check_top_flows)),
            '--check_explain_pairs',
            str(int(opts.check_explain_pairs)),
            '--vad_threshold_min',
            str(float(opts.vad_threshold_min)),
            '--vad_threshold_max',
            str(float(opts.vad_threshold_max)),
            '--vad_threshold_step',
            str(float(opts.vad_threshold_step)),
            '--drop_top_df_min',
            str(int(opts.drop_top_df_min)),
            '--drop_top_df_max',
            str(int(opts.drop_top_df_max)),
            '--drop_top_df_step',
            str(int(opts.drop_top_df_step)),
            '--vad_min_arousal_min',
            str(float(opts.vad_min_arousal_min)),
            '--vad_min_arousal_max',
            str(float(opts.vad_min_arousal_max)),
            '--vad_min_arousal_step',
            str(float(opts.vad_min_arousal_step)),
            '--max_terms_min',
            str(int(opts.max_terms_min)),
            '--max_terms_max',
            str(int(opts.max_terms_max)),
            '--max_terms_step',
            str(int(opts.max_terms_step)),
        ]
        if opts.check_selected_jsonl:
            tune_cmd += [
                '--check_selected_jsonl',
                str(opts.check_selected_jsonl),
            ]
        else:
            tune_cmd += [
                '--check_n_per_category',
                str(int(opts.check_n_per_category)),
                '--check_random_n',
                str(int(opts.check_random_n)),
                '--check_seed',
                str(int(check_seed)),
            ]
        if opts.token_guard_min_score is not None:
            tune_cmd += [
                '--token_guard_min_score',
                str(float(opts.token_guard_min_score)),
            ]
        if bool(opts.token_guard_allow_warn):
            tune_cmd.append('--token_guard_allow_warn')
        if opts.vad_threshold_start is not None:
            tune_cmd += [
                '--vad_threshold_start',
                str(float(opts.vad_threshold_start)),
            ]
        if opts.drop_top_df_start is not None:
            tune_cmd += [
                '--drop_top_df_start',
                str(int(opts.drop_top_df_start)),
            ]
        if opts.vad_min_arousal_start is not None:
            tune_cmd += [
                '--vad_min_arousal_start',
                str(float(opts.vad_min_arousal_start)),
            ]
        if opts.max_terms_start is not None:
            tune_cmd += [
                '--max_terms_start',
                str(int(opts.max_terms_start)),
            ]
        if bool(opts.restart_on_stuck):
            tune_cmd.append('--restart_on_stuck')
            restart_seed = (
                int(opts.restart_seed)
                if opts.restart_seed is not None
                else int(opts.seed)
            )
            tune_cmd += [
                '--restart_seed',
                str(int(restart_seed)),
                '--restart_attempts',
                str(int(opts.restart_attempts)),
                '--restart_emotional_vocab',
                str(opts.restart_emotional_vocab),
            ]
        if opts.limit is not None:
            tune_cmd += ['--limit', str(int(opts.limit))]
        _run_cmd(
            tune_cmd,
            env=env,
            log=log,
        )
        token_neighbours = (
            run_dir / 'icj_token_neighbours.jsonl'
        )
        best_cmd_path = (
            run_dir
            / 'token_ot_tuning'
            / 'best_pipeline_command.txt'
        )
        if (
            not token_neighbours.exists()
            and best_cmd_path.exists()
        ):
            cmd = best_cmd_path.read_text(encoding='utf-8').strip()
            if cmd:
                log('tuner fallback')
                _run_cmd(
                    shlex.split(cmd),
                    env=env,
                    log=log,
                )
        outs['variants'][name] = _collect_run_metrics(run_dir)
    out_path = (
        out_base / f'warm_start_ablation_{stamp}{tag}.json'
    )
    out_path.write_text(
        json.dumps(
            outs,
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    print(json.dumps(
            {
                'output': str(out_path),
                'runs': outs.get(
                    'variants',
                    {},
                ),
            },
            ensure_ascii=False,
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
