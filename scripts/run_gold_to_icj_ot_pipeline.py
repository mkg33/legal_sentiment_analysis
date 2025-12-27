#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.compare import compare_scores
from legal_emotion.combined_viz import (
    write_combined_html_report,
)
from legal_emotion.gold_eval import (
    evaluate_checkpoint as evaluate_emotion_checkpoint,
)
from legal_emotion.prepare import prepare_txt_dir
from legal_emotion.score import score_txt_dir
from legal_emotion.sentiment_config import (
    SentimentConfig,
    load_sentiment_config,
)
from legal_emotion.sentiment_prepare import (
    prepare_sigmalaw_absa,
)
from legal_emotion.sentiment_report import (
    evaluate_sentiment_checkpoint,
)
from legal_emotion.sentiment_train import (
    run_sentiment_training,
)
from legal_emotion.summary_viz import (
    write_summary_html_report,
)
from legal_emotion.token_compare import compare_token_clouds
from legal_emotion.train import run_training
from legal_emotion.silver_teacher import make_teacher_silver
from legal_emotion.utils import load_config


def _write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        return json.load(f)


def _write_yaml(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding='utf-8',
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _jsonl_prepared_for_dir(
    jsonl_path: Path,
    input_dir: Path,
) -> bool:
    try:
        with jsonl_path.open(
            'r',
            encoding='utf-8',
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                meta = row.get('meta')
                if not isinstance(
                    meta,
                    dict,
                ):
                    continue
                path_val = meta.get('path')
                if (
                    not isinstance(
                        path_val,
                        str,
                    )
                    or not path_val.strip()
                ):
                    continue
                p = Path(path_val)
                if not p.is_absolute():
                    p = (_ROOT / p).resolve()
                else:
                    p = p.resolve()
                base = input_dir
                if not base.is_absolute():
                    base = (_ROOT / base).resolve()
                else:
                    base = base.resolve()
                try:
                    return p.is_relative_to(base)
                except AttributeError:
                    return str(p).startswith(str(base))
    except Exception:
        return False
    return False


def _write_jsonl(
    path: Path,
    rows: Iterable[Dict[str, Any]],
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b'',
        ):
            h.update(chunk)
    return h.hexdigest()


def _sig_matches(
    path: Path,
    sig: Dict[str, Any],
) -> bool:
    try:
        if not path.exists():
            return False
        existing = _read_json(path)
    except Exception:
        return False
    return existing == sig


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return


def _guard_token_coverage(
    *,
    token_stats: Optional[Dict[str, Any]],
    token_check: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if args.no_token_guard:
        return
    if not token_stats:
        return
    coverage_total = float(token_stats.get('coverage_total') or 0.0)
    docs = int(token_stats.get('docs') or 0)
    zero_docs = int(token_stats.get('docs_with_zero_selected') or 0)
    zero_ratio = float(zero_docs) / float(max(
            1,
            docs,
        ))
    reasons: List[str] = []
    if coverage_total < float(args.token_guard_min_coverage):
        reasons.append(f'coverage_total {coverage_total:.4f} below {float(args.token_guard_min_coverage):.4f}')
    if zero_ratio > float(args.token_guard_max_zero_ratio):
        reasons.append(f'docs_with_zero_selected ratio {zero_ratio:.3f} above {float(args.token_guard_max_zero_ratio):.3f}')
    min_score = getattr(args, 'token_guard_min_score', None)
    if token_check:
        check_payload = token_check
        if isinstance(
            check_payload.get('evaluation'),
            dict,
        ):
            check_payload = check_payload['evaluation']
        metrics = (
            check_payload.get('metrics')
            if isinstance(
                check_payload,
                dict,
            )
            else None
        )
        scores_dict: Dict[str, Any] = (
            metrics if isinstance(
                metrics,
                dict,
            ) else {}
        )
        median_ratio = float(scores_dict.get('selected_ratio_median')
            or check_payload.get('selected_ratio_median')
            or 0.0)
        if median_ratio < float(args.token_guard_min_selected_ratio):
            reasons.append(f'selected_ratio_median {median_ratio:.4f} below {float(args.token_guard_min_selected_ratio):.4f}')
        if min_score is not None:
            score = float(scores_dict.get('score')
                or check_payload.get('score')
                or 0.0)
            if score < float(min_score):
                reasons.append(f'meaningfulness score {score:.2f} below {float(min_score):.2f}')
        verdict = (
            str(check_payload.get('verdict')
                or scores_dict.get('verdict')
                or '')
            .strip()
            .lower()
        )
        if (
            verdict
            and verdict != 'pass'
            and (not args.token_guard_allow_warn)
        ):
            reasons.append(f'token_ot_check verdict is {verdict}')
    elif min_score is not None:
        reasons.append('missing token_ot_check summary for score guard')
    if reasons:
        detail = ' | '.join(reasons)
        raise RuntimeError('Token OT coverage guard failed '
            + detail
            + '. Adjust token_vad_threshold, token_emotional_vocab, token_drop_top_df, or lexicon_min_vad_salience then rerun.')


def split_labelled_jsonl(
    *,
    input_path: str | Path,
    train_out: str | Path,
    dev_out: str | Path,
    dev_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    if not 0.0 < float(dev_ratio) < 1.0:
        raise ValueError('error: ValueError')
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    rows = _read_jsonl(in_p)
    if not rows:
        raise ValueError('error: ValueError')
    for r in rows[:10]:
        if (
            not isinstance(
                r.get('text'),
                str,
            )
            or r.get('label') is None
        ):
            raise ValueError('error: ValueError')
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    n = len(rows)
    dev_n = int(round(n * float(dev_ratio)))
    dev_n = max(
        1,
        dev_n,
    )
    dev_n = min(
        dev_n,
        n - 1,
    )
    dev = rows[:dev_n]
    train = rows[dev_n:]
    _write_jsonl(
        Path(train_out),
        train,
    )
    _write_jsonl(
        Path(dev_out),
        dev,
    )
    return {
        'input': str(in_p),
        'n_total': int(n),
        'n_train': int(len(train)),
        'n_dev': int(len(dev)),
    }


def _resolve_sentiment_cfg(
    base_path: Optional[str],
    *,
    train_path: Path,
    dev_path: Path,
    save_dir: Path,
    seed: int,
) -> SentimentConfig:
    setup = load_sentiment_config(base_path)
    setup.data_path = str(train_path)
    setup.eval_path = str(dev_path)
    setup.save_dir = str(save_dir)
    setup.seed = int(seed)
    return setup


def _make_logger():
    def log(msg: str) -> None:
        print(
            msg,
            file=sys.stderr,
            flush=True,
        )

    return log


def _prepare_gold_split(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    log,
) -> Tuple[Path, Path]:
    gold_dir = run_dir / 'gold'
    gold_train = gold_dir / 'train.jsonl'
    gold_dev = gold_dir / 'dev.jsonl'
    gold_stats_path = run_dir / 'gold_split_stats.json'
    stats: Dict[str, Any]
    if (
        args.force
        or not gold_train.exists()
        or (not gold_dev.exists())
    ):
        log('gold split')
        if args.gold_jsonl:
            log(f'gold {args.gold_jsonl}')
            stats = split_labelled_jsonl(
                input_path=args.gold_jsonl,
                train_out=gold_train,
                dev_out=gold_dev,
                dev_ratio=float(args.gold_dev_ratio),
                seed=int(args.seed),
            )
        else:
            log('use sigmalaw')
            csv_path = args.sigmalaw_csv
            if csv_path is None:
                repo_csv = Path('data/sigmalaw_absa/SigmaLaw-ABSA.csv')
                if repo_csv.exists():
                    csv_path = str(repo_csv)
            stats = prepare_sigmalaw_absa(
                out_dir=gold_dir,
                csv_path=csv_path,
                dev_ratio=float(args.gold_dev_ratio),
                seed=int(args.seed),
                download=True,
            )
        _write_json(
            gold_stats_path,
            stats,
        )
    elif gold_stats_path.exists():
        stats = _read_json(gold_stats_path)
    else:
        stats = {
            'skipped': True,
            'gold_train': str(gold_train),
            'gold_dev': str(gold_dev),
            'seed': int(args.seed),
            'dev_ratio': float(args.gold_dev_ratio),
        }
        _write_json(
            gold_stats_path,
            stats,
        )
    return (gold_train, gold_dev)


def _sentiment_stage(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    gold_train: Path,
    gold_dev: Path,
    log,
) -> Path:
    log('sentiment config')
    sent_ckpt_dir = run_dir / 'checkpoints_sentiment'
    sent_ckpt = sent_ckpt_dir / 'sentiment.pt'
    sent_cfg = _resolve_sentiment_cfg(
        args.sentiment_config,
        train_path=gold_train,
        dev_path=gold_dev,
        save_dir=sent_ckpt_dir,
        seed=int(args.seed),
    )
    if args.sentiment_num_epochs is not None:
        sent_cfg.num_epochs = int(args.sentiment_num_epochs)
    if bool(args.safe_dataloader):
        sent_cfg.num_workers = 0
        sent_cfg.persistent_workers = False
        sent_cfg.pin_memory = False
    sent_cfg_path = (
        run_dir / 'sentiment_config.resolved.yaml'
    )
    _write_yaml(
        sent_cfg_path,
        dict(sent_cfg.__dict__),
    )
    if not args.skip_sentiment_train and (
        args.force or not sent_ckpt.exists()
    ):
        log('sentiment train')
        run_sentiment_training(sent_cfg)
    else:
        log('sentiment train skip')
    if not sent_ckpt.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    log('sentiment eval')
    sent_metrics = evaluate_sentiment_checkpoint(
        checkpoint=str(sent_ckpt),
        data_path=str(gold_dev),
        cfg_path=str(sent_cfg_path),
        batch_size=None,
        limit=None,
    )
    _write_json(
        run_dir / 'sentiment_eval_metrics.json',
        sent_metrics,
    )
    return sent_ckpt


def _emotion_stage(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    sent_ckpt: Path,
    log,
) -> Tuple[Any, Path]:
    log('emotion config')
    emo_ckpt_dir = run_dir / 'checkpoints_emotion'
    emo_ckpt = emo_ckpt_dir / 'model.pt'
    emo_cfg = load_config(args.emotion_config)
    emo_cfg.save_dir = str(emo_ckpt_dir)
    emo_cfg.seed = int(args.seed)
    if args.emotion_num_epochs is not None:
        emo_cfg.num_epochs = int(args.emotion_num_epochs)
    if bool(args.safe_dataloader):
        setattr(
            emo_cfg,
            'num_workers',
            0,
        )
        setattr(
            emo_cfg,
            'persistent_workers',
            False,
        )
        setattr(
            emo_cfg,
            'pin_memory',
            False,
        )
    warm_start_cfg = bool(getattr(
            emo_cfg,
            'emotion_init_from_sentiment',
            True,
        ))
    warm_start = bool(warm_start_cfg and (not args.no_emotion_warm_start))
    emo_cfg.emotion_init_from_sentiment = warm_start
    emo_cfg.init_from_sentiment_checkpoint = (
        str(sent_ckpt) if warm_start else None
    )
    if warm_start:
        log('emotion warm start on')
    elif args.no_emotion_warm_start and warm_start_cfg:
        log('emotion warm start off')
    else:
        log('emotion warm start off')
    data_p = Path(emo_cfg.data_path)
    dev_p = Path(emo_cfg.eval_path)
    need_prep = not data_p.exists() or not dev_p.exists()
    if not need_prep:
        corpus_dir = Path(args.icj_dir)
        if not _jsonl_prepared_for_dir(
            data_p,
            corpus_dir,
        ) or not _jsonl_prepared_for_dir(
            dev_p,
            corpus_dir,
        ):
            log('emotion prep refresh')
            data_p = run_dir / 'corpus_train.jsonl'
            dev_p = run_dir / 'corpus_dev.jsonl'
            emo_cfg.data_path = str(data_p)
            emo_cfg.eval_path = str(dev_p)
            need_prep = True
    if need_prep:
        log('emotion prep')
        prep_stats = prepare_txt_dir(
            str(args.icj_dir),
            str(data_p),
            str(dev_p),
            tokenizer_name=str(emo_cfg.model_name),
            max_length=int(emo_cfg.max_length),
            dev_ratio=0.1,
            seed=int(args.seed),
            recursive=True,
            limit=None,
            stride=int(args.stride),
        )
        _write_json(
            run_dir / 'icj_prepare_stats.json',
            prep_stats,
        )
    teacher_model = getattr(
        emo_cfg,
        'silver_teacher_model',
        None,
    )
    if teacher_model:
        train_silver = run_dir / 'icj_train_silver.jsonl'
        dev_silver = run_dir / 'icj_dev_silver.jsonl'
        if (
            args.force
            or not train_silver.exists()
            or (not dev_silver.exists())
        ):
            log(f'silver {teacher_model}')
            teacher_stats = {
                'train': make_teacher_silver(
                    input_path=str(data_p),
                    output_path=str(train_silver),
                    cfg_path=str(args.emotion_config),
                    model_name=str(teacher_model),
                    batch_size=int(getattr(
                            emo_cfg,
                            'silver_teacher_batch_size',
                            16,
                        )),
                    max_length=int(getattr(
                            emo_cfg,
                            'silver_teacher_max_length',
                            None,
                        )
                        or emo_cfg.max_length),
                    truncate_to_max_length=True,
                ),
                'dev': make_teacher_silver(
                    input_path=str(dev_p),
                    output_path=str(dev_silver),
                    cfg_path=str(args.emotion_config),
                    model_name=str(teacher_model),
                    batch_size=int(getattr(
                            emo_cfg,
                            'silver_teacher_batch_size',
                            16,
                        )),
                    max_length=int(getattr(
                            emo_cfg,
                            'silver_teacher_max_length',
                            None,
                        )
                        or emo_cfg.max_length),
                    truncate_to_max_length=True,
                ),
            }
            _write_json(
                run_dir / 'icj_teacher_silver_stats.json',
                teacher_stats,
            )
        emo_cfg.data_path = str(train_silver)
        emo_cfg.eval_path = str(dev_silver)
        emo_cfg.use_silver = True
        emo_cfg.silver_force_has_lex = True
    emo_cfg_path = run_dir / 'emotion_config.resolved.yaml'
    _write_yaml(
        emo_cfg_path,
        dict(emo_cfg.__dict__),
    )
    if not args.skip_emotion_train and (
        args.force or not emo_ckpt.exists()
    ):
        log('emotion train')
        run_training(emo_cfg)
    else:
        log('emotion train skip')
    if not emo_ckpt.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    if args.emotion_gold_jsonl:
        log('emotion eval gold')
        emo_metrics = evaluate_emotion_checkpoint(
            checkpoint=str(emo_ckpt),
            data_path=str(args.emotion_gold_jsonl),
            cfg_path=str(emo_cfg_path),
            batch_size=None,
            limit=None,
        )
        _write_json(
            run_dir / 'emotion_gold_eval_metrics.json',
            emo_metrics,
        )
    return (emo_cfg, emo_ckpt)


def _score_icj_docs(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    emo_ckpt: Path,
    log,
) -> Path:
    scores_jsonl = run_dir / 'icj_scores.jsonl'
    if args.force or not scores_jsonl.exists():
        log('icj score')
        score_stats = score_txt_dir(
            checkpoint=str(emo_ckpt),
            input_dir=str(args.icj_dir),
            output_jsonl=str(scores_jsonl),
            cfg_path=str(run_dir / 'emotion_config.resolved.yaml'),
            recursive=True,
            limit=(
                int(args.limit)
                if args.limit is not None
                else None
            ),
            stride=int(args.stride),
            batch_size=None,
        )
        _write_json(
            run_dir / 'icj_score_stats.json',
            score_stats,
        )
    else:
        log('icj score skip')
    return scores_jsonl


def _doc_level_compare(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scores_jsonl: Path,
    log,
) -> Tuple[Path, Path]:
    compare_out_jsonl = run_dir / 'icj_neighbours.jsonl'
    compare_vis_html = run_dir / 'icj_neighbours.html'
    if args.force or not compare_out_jsonl.exists():
        log('eot neighbours')
        compare_stats = compare_scores(
            input_jsonl=str(scores_jsonl),
            output_path=str(compare_out_jsonl),
            cfg_path=str(run_dir / 'emotion_config.resolved.yaml'),
            fmt='neighbours',
            topk=int(args.topk),
            top_flows=int(args.top_flows),
            limit=(
                int(args.limit)
                if args.limit is not None
                else None
            ),
            vis=not bool(args.no_vis),
            vis_path=str(compare_vis_html),
        )
        _write_json(
            run_dir / 'icj_compare_stats.json',
            compare_stats,
        )
    else:
        log('eot neighbours skip')
        if (
            not args.no_vis
            and compare_out_jsonl.exists()
            and (not compare_vis_html.exists())
        ):
            log('eot html rebuild')
            from legal_emotion.compare import (
                render_compare_neighbours_html,
            )

            render_compare_neighbours_html(
                scores_jsonl=str(scores_jsonl),
                neighbours_jsonl=str(compare_out_jsonl),
                output_html=str(compare_vis_html),
                cfg_path=str(run_dir / 'emotion_config.resolved.yaml'),
            )
    return (compare_out_jsonl, compare_vis_html)


def _resolve_token_check_selected_jsonl(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    return str(p)


def _token_neighbours_sig(
    *,
    args: argparse.Namespace,
    scores_jsonl: Path,
    token_cfg_sha: Optional[str],
    stopwords_file: str,
    stopwords_sha: Optional[str],
) -> Dict[str, Any]:
    return {
        'kind': 'token_neighbours',
        'cfg_sha256': token_cfg_sha,
        'input_jsonl': str(scores_jsonl),
        'icj_dir': str(args.icj_dir),
        'limit': (
            int(args.limit)
            if args.limit is not None
            else None
        ),
        'no_vis': bool(args.no_vis),
        'stopwords_file': stopwords_file,
        'stopwords_sha256': stopwords_sha,
        'token_topk': int(args.token_topk),
        'token_candidate_k': int(args.token_candidate_k),
        'token_mode': str(args.token_mode),
        'token_weight': str(args.token_weight),
        'token_cost': str(args.token_cost),
        'token_alpha_embed': float(args.token_alpha_embed),
        'token_beta_vad': float(args.token_beta_vad),
        'token_vad_threshold': float(args.token_vad_threshold),
        'token_emotional_vocab': str(args.token_emotional_vocab),
        'token_vad_min_arousal_vad_only': float(args.token_vad_min_arousal_vad_only),
        'token_max_ngram': int(args.token_max_ngram),
        'token_max_terms': int(args.token_max_terms),
        'token_drop_top_df': int(args.token_drop_top_df),
    }


def _token_check_sig(
    *,
    args: argparse.Namespace,
    token_cfg_sha: Optional[str],
    stopwords_file: str,
    stopwords_sha: Optional[str],
    token_check_selected_jsonl: Optional[str],
    token_check_seed: int,
) -> Dict[str, Any]:
    return {
        'kind': 'token_check',
        'cfg_sha256': token_cfg_sha,
        'icj_dir': str(args.icj_dir),
        'selected_jsonl': token_check_selected_jsonl,
        'n_per_category': (
            int(args.token_check_n_per_category)
            if not token_check_selected_jsonl
            else None
        ),
        'random_n': (
            int(args.token_check_random_n)
            if not token_check_selected_jsonl
            else None
        ),
        'seed': (
            int(token_check_seed)
            if not token_check_selected_jsonl
            else None
        ),
        'format': str(args.token_check_format),
        'topk': int(args.token_check_topk),
        'candidate_k': int(args.token_check_candidate_k),
        'top_flows': int(args.token_check_top_flows),
        'explain_pairs': int(args.token_check_explain_pairs),
        'stopwords_file': stopwords_file,
        'stopwords_sha256': stopwords_sha,
        'token_mode': str(args.token_mode),
        'token_weight': str(args.token_weight),
        'token_cost': str(args.token_cost),
        'token_alpha_embed': float(args.token_alpha_embed),
        'token_beta_vad': float(args.token_beta_vad),
        'token_vad_threshold': float(args.token_vad_threshold),
        'token_emotional_vocab': str(args.token_emotional_vocab),
        'token_vad_min_arousal_vad_only': float(args.token_vad_min_arousal_vad_only),
        'token_max_ngram': int(args.token_max_ngram),
        'token_max_terms': int(args.token_max_terms),
        'token_drop_top_df': int(args.token_drop_top_df),
    }


def _group_shift_sig(
    *,
    args: argparse.Namespace,
    token_cfg_sha: Optional[str],
    stopwords_file: str,
    stopwords_sha: Optional[str],
) -> Dict[str, Any]:
    return {
        'kind': 'group_shift',
        'cfg_sha256': token_cfg_sha,
        'data_dir': str(args.icj_dir),
        'group_by': str(args.group_shift_by),
        'group_a': (
            str(args.group_shift_a)
            if args.group_shift_a
            else None
        ),
        'group_b': (
            str(args.group_shift_b)
            if args.group_shift_b
            else None
        ),
        'exclude_unknown': bool(args.group_shift_exclude_unknown),
        'limit': (
            int(args.group_shift_limit)
            if args.group_shift_limit is not None
            else None
        ),
        'no_vis': bool(args.no_vis),
        'stopwords_file': stopwords_file,
        'stopwords_sha256': stopwords_sha,
        'top_flows': int(args.group_shift_top_flows),
        'token_mode': str(args.token_mode),
        'token_weight': str(args.token_weight),
        'token_cost': str(args.token_cost),
        'token_alpha_embed': float(args.token_alpha_embed),
        'token_beta_vad': float(args.token_beta_vad),
        'token_vad_threshold': float(args.token_vad_threshold),
        'token_emotional_vocab': str(args.token_emotional_vocab),
        'token_vad_min_arousal_vad_only': float(args.token_vad_min_arousal_vad_only),
        'token_max_ngram': int(args.token_max_ngram),
        'token_max_terms': int(args.token_max_terms),
        'token_drop_top_df': int(args.token_drop_top_df),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='gold to icj pipeline')
    ap.add_argument(
        '--run_dir',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=13,
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
        '--sentiment_config',
        type=str,
        default='config.sentiment.sample.yaml',
    )
    ap.add_argument(
        '--emotion_config',
        type=str,
        default='config.icj.gpu.yaml',
    )
    ap.add_argument(
        '--sentiment_num_epochs',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--emotion_num_epochs',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--icj_dir',
        type=str,
        default='data/EN_TXT_BEST_FULL',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--stride',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--topk',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--top_flows',
        type=int,
        default=8,
    )
    ap.add_argument(
        '--token_topk',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--token_candidate_k',
        type=int,
        default=50,
    )
    ap.add_argument(
        '--no_token_ot',
        action='store_true',
    )
    ap.add_argument(
        '--token_mode',
        type=str,
        default='sinkhorn_divergence',
    )
    ap.add_argument(
        '--token_weight',
        type=str,
        default='tfidf',
    )
    ap.add_argument(
        '--token_alpha_embed',
        type=float,
        default=0.8,
    )
    ap.add_argument(
        '--token_beta_vad',
        type=float,
        default=0.2,
    )
    ap.add_argument(
        '--token_cost',
        type=str,
        default='embedding_vad',
        choices=('embedding', 'vad', 'embedding_vad'),
    )
    ap.add_argument(
        '--token_vad_threshold',
        type=float,
        default=0.35,
    )
    ap.add_argument(
        '--token_emotional_vocab',
        type=str,
        default='lexicon_or_vad',
        choices=('lexicon', 'vad', 'lexicon_or_vad'),
    )
    ap.add_argument(
        '--token_vad_min_arousal_vad_only',
        type=float,
        default=0.4,
    )
    ap.add_argument(
        '--token_max_ngram',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--token_max_terms',
        type=int,
        default=512,
    )
    ap.add_argument(
        '--token_drop_top_df',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--token_check_n_per_category',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--token_check_random_n',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--token_check_seed',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--token_check_selected_jsonl',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--token_check_format',
        type=str,
        default='neighbours',
        choices=('matrix', 'neighbours'),
    )
    ap.add_argument(
        '--token_check_topk',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--token_check_candidate_k',
        type=int,
        default=200,
    )
    ap.add_argument(
        '--token_check_top_flows',
        type=int,
        default=12,
    )
    ap.add_argument(
        '--token_check_explain_pairs',
        type=int,
        default=6,
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
        '--token_guard_allow_warn',
        action='store_true',
    )
    ap.add_argument(
        '--no_token_guard',
        action='store_true',
    )
    ap.add_argument(
        '--no_group_shift',
        action='store_true',
    )
    ap.add_argument(
        '--no_token_check',
        action='store_true',
    )
    ap.add_argument(
        '--group_shift_by',
        type=str,
        default='doc_type',
    )
    ap.add_argument(
        '--group_shift_a',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--group_shift_b',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--group_shift_exclude_unknown',
        action='store_true',
    )
    ap.add_argument(
        '--group_shift_limit',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--group_shift_top_flows',
        type=int,
        default=16,
    )
    ap.add_argument(
        '--emotion_gold_jsonl',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--no_vis',
        action='store_true',
    )
    ap.add_argument(
        '--keep_standalone_html',
        action='store_true',
    )
    ap.add_argument(
        '--force',
        action='store_true',
    )
    ap.add_argument(
        '--safe_dataloader',
        action='store_true',
    )
    ap.add_argument(
        '--skip_sentiment_train',
        action='store_true',
    )
    ap.add_argument(
        '--skip_emotion_train',
        action='store_true',
    )
    ap.add_argument(
        '--no_emotion_warm_start',
        action='store_true',
    )
    args = ap.parse_args()
    log = _make_logger()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path('outputs')
        / time.strftime('gold_to_icj_ot_%Y%m%d_%H%M%S')
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    log(f'run {run_dir}')
    (
        gold_train,
        gold_dev,
    ) = _prepare_gold_split(
        args=args,
        run_dir=run_dir,
        log=log,
    )
    sent_ckpt = _sentiment_stage(
        args=args,
        run_dir=run_dir,
        gold_train=gold_train,
        gold_dev=gold_dev,
        log=log,
    )
    (
        emo_cfg,
        emo_ckpt,
    ) = _emotion_stage(
        args=args,
        run_dir=run_dir,
        sent_ckpt=sent_ckpt,
        log=log,
    )
    scores_jsonl = _score_icj_docs(
        args=args,
        run_dir=run_dir,
        emo_ckpt=emo_ckpt,
        log=log,
    )
    (
        compare_out_jsonl,
        compare_vis_html,
    ) = (
        _doc_level_compare(
            args=args,
            run_dir=run_dir,
            scores_jsonl=scores_jsonl,
            log=log,
        )
    )
    token_out_jsonl = run_dir / 'icj_token_neighbours.jsonl'
    token_vis_html = run_dir / 'icj_token_neighbours.html'
    token_stats_path = run_dir / 'icj_token_ot_stats.json'
    token_cfg_path = (
        run_dir / 'emotion_config.resolved.yaml'
    )
    token_cfg_sha = (
        _sha256_file(token_cfg_path)
        if token_cfg_path.exists()
        else None
    )
    stopwords_path = Path('data/stopwords_legal_en_token_ot.txt')
    stopwords_file = str(stopwords_path)
    stopwords_sha = (
        _sha256_file(stopwords_path)
        if stopwords_path.exists()
        else None
    )
    token_neighbours_sig_path = (
        run_dir / 'icj_token_neighbours.signature.json'
    )
    desired_token_neighbours_sig = _token_neighbours_sig(
        args=args,
        scores_jsonl=scores_jsonl,
        token_cfg_sha=token_cfg_sha,
        stopwords_file=stopwords_file,
        stopwords_sha=stopwords_sha,
    )
    need_token_neighbours = not args.no_token_ot and (
        bool(args.force)
        or not token_out_jsonl.exists()
        or (
            not _sig_matches(
                token_neighbours_sig_path,
                desired_token_neighbours_sig,
            )
        )
    )
    if need_token_neighbours:
        if token_out_jsonl.exists():
            log('token-ot neighbours rerun')
        else:
            log('token-ot neighbours')
        _safe_unlink(token_out_jsonl)
        _safe_unlink(token_vis_html)
        _safe_unlink(token_stats_path)
        token_stats = compare_token_clouds(
            input_jsonl=str(scores_jsonl),
            output_path=str(token_out_jsonl),
            cfg_path=str(run_dir / 'emotion_config.resolved.yaml'),
            fmt='neighbours',
            topk=int(args.token_topk),
            candidate_k=int(args.token_candidate_k),
            mode=str(args.token_mode),
            focus='emotional',
            cost=str(args.token_cost),
            alpha_embed=float(args.token_alpha_embed),
            beta_vad=float(args.token_beta_vad),
            vad_threshold=float(args.token_vad_threshold),
            emotional_vocab=str(args.token_emotional_vocab),
            vad_min_arousal_vad_only=float(args.token_vad_min_arousal_vad_only),
            max_ngram=int(args.token_max_ngram),
            weight=str(args.token_weight),
            max_terms=int(args.token_max_terms),
            stopwords=True,
            stopwords_file=stopwords_file,
            drop_top_df=int(args.token_drop_top_df),
            include_explain=True,
            top_flows=8,
            limit=(
                int(args.limit)
                if args.limit is not None
                else None
            ),
            vis=not bool(args.no_vis),
            vis_path=str(token_vis_html),
        )
        _write_json(
            token_stats_path,
            token_stats,
        )
        _write_json(
            token_neighbours_sig_path,
            desired_token_neighbours_sig,
        )
    elif args.no_token_ot:
        log('token-ot skip')
    else:
        log('token-ot neighbours skip')
    token_check_dir = run_dir / 'token_ot_check'
    token_check_summary = token_check_dir / 'summary.json'
    token_check_sig_path = (
        token_check_dir / 'signature.json'
    )
    token_check_seed = (
        int(args.token_check_seed)
        if args.token_check_seed is not None
        else int(args.seed)
    )
    token_check_selected_jsonl = (
        _resolve_token_check_selected_jsonl(args.token_check_selected_jsonl)
    )
    desired_token_check_sig = _token_check_sig(
        args=args,
        token_cfg_sha=token_cfg_sha,
        stopwords_file=stopwords_file,
        stopwords_sha=stopwords_sha,
        token_check_selected_jsonl=token_check_selected_jsonl,
        token_check_seed=token_check_seed,
    )
    need_token_check = not args.no_token_check and (
        bool(args.force)
        or not token_check_summary.exists()
        or (
            not _sig_matches(
                token_check_sig_path,
                desired_token_check_sig,
            )
        )
    )
    if need_token_check:
        if token_check_dir.exists():
            log('token-ot check rerun')
        else:
            log('token-ot check')
        _safe_rmtree(token_check_dir)
        token_check_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        embed_model = getattr(
            emo_cfg,
            'token_ot_embed_model',
            None,
        )
        embed_backend = getattr(
            emo_cfg,
            'token_ot_embed_backend',
            None,
        )
        embed_pooling = getattr(
            emo_cfg,
            'token_ot_embed_pooling',
            None,
        )
        embed_batch_size = getattr(
            emo_cfg,
            'token_ot_embed_batch_size',
            None,
        )
        embed_max_length = getattr(
            emo_cfg,
            'token_ot_embed_max_length',
            None,
        )
        embed_prompt_mode = getattr(
            emo_cfg,
            'token_ot_embed_prompt_mode',
            None,
        )
        embed_prompt_text = getattr(
            emo_cfg,
            'token_ot_embed_prompt_text',
            None,
        )
        cmd = [
            sys.executable,
            str(_ROOT
                / 'scripts'
                / 'token_ot_meaningfulness_check.py'),
            '--config',
            str(run_dir / 'emotion_config.resolved.yaml'),
            '--out_dir',
            str(token_check_dir),
        ]
        if token_check_selected_jsonl:
            cmd += [
                '--selected_jsonl',
                str(token_check_selected_jsonl),
            ]
        else:
            cmd += [
                '--input_dir',
                str(args.icj_dir),
                '--n_per_category',
                str(int(args.token_check_n_per_category)),
                '--random_n',
                str(int(args.token_check_random_n)),
                '--seed',
                str(int(token_check_seed)),
            ]
        cmd += [
            '--stopwords_file',
            str(stopwords_file),
            '--format',
            str(args.token_check_format),
            '--topk',
            str(int(args.token_check_topk)),
            '--candidate_k',
            str(int(args.token_check_candidate_k)),
            '--top_flows',
            str(int(args.token_check_top_flows)),
            '--explain_pairs',
            str(int(args.token_check_explain_pairs)),
            '--mode',
            str(args.token_mode),
            '--cost',
            str(args.token_cost),
            '--alpha_embed',
            str(args.token_alpha_embed),
            '--beta_vad',
            str(args.token_beta_vad),
            '--vad_threshold',
            str(args.token_vad_threshold),
            '--emotional_vocab',
            str(args.token_emotional_vocab),
            '--vad_min_arousal_vad_only',
            str(args.token_vad_min_arousal_vad_only),
            '--max_ngram',
            str(args.token_max_ngram),
            '--weight',
            str(args.token_weight),
            '--max_terms',
            str(args.token_max_terms),
            '--drop_top_df',
            str(args.token_drop_top_df),
        ]
        if embed_model:
            cmd += ['--embed_model', str(embed_model)]
        if embed_backend:
            cmd += ['--embed_backend', str(embed_backend)]
        if embed_pooling:
            cmd += ['--embed_pooling', str(embed_pooling)]
        if embed_batch_size:
            cmd += [
                '--embed_batch_size',
                str(embed_batch_size),
            ]
        if embed_max_length:
            cmd += [
                '--embed_max_length',
                str(embed_max_length),
            ]
        if embed_prompt_mode:
            cmd += [
                '--embed_prompt_mode',
                str(embed_prompt_mode),
            ]
        if embed_prompt_text:
            cmd += [
                '--embed_prompt_text',
                str(embed_prompt_text),
            ]
        subprocess.run(
            cmd,
            check=True,
        )
        _write_json(
            token_check_sig_path,
            desired_token_check_sig,
        )
    elif args.no_token_check:
        log('token-ot check skip')
    else:
        log('token-ot check skip')
    token_stats_payload: Optional[Dict[str, Any]] = None
    if not args.no_token_ot and token_stats_path.exists():
        if 'token_stats' in locals() and isinstance(
            token_stats,
            dict,
        ):
            token_stats_payload = token_stats
        else:
            token_stats_payload = _read_json(token_stats_path)
    token_check_payload: Optional[Dict[str, Any]] = None
    if (
        not args.no_token_check
        and token_check_summary.exists()
    ):
        token_check_payload = _read_json(token_check_summary)
    if not args.no_token_ot:
        log('token-ot guard')
        _guard_token_coverage(
            token_stats=token_stats_payload,
            token_check=token_check_payload,
            args=args,
        )
    group_shift_dir = run_dir / 'icj_group_shift'
    group_shift_summary = group_shift_dir / 'summary.json'
    group_shift_sig_path = (
        group_shift_dir / 'signature.json'
    )
    target_group_shift_sig = _group_shift_sig(
        args=args,
        token_cfg_sha=token_cfg_sha,
        stopwords_file=stopwords_file,
        stopwords_sha=stopwords_sha,
    )
    need_group_shift = not args.no_group_shift and (
        bool(args.force)
        or not group_shift_summary.exists()
        or (
            not _sig_matches(
                group_shift_sig_path,
                target_group_shift_sig,
            )
        )
    )
    if need_group_shift:
        if group_shift_dir.exists():
            log('group shift rerun')
        else:
            log('group shift run')
        _safe_rmtree(group_shift_dir)
        group_shift_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        cmd = [
            sys.executable,
            str(_ROOT
                / 'scripts'
                / 'run_token_ot_group_shift.py'),
            '--data_dir',
            str(args.icj_dir),
            '--config',
            str(run_dir / 'emotion_config.resolved.yaml'),
            '--out_dir',
            str(group_shift_dir),
            '--group_by',
            str(args.group_shift_by),
            '--mode',
            str(args.token_mode),
            '--focus',
            'emotional',
            '--cost',
            str(args.token_cost),
            '--alpha_embed',
            str(args.token_alpha_embed),
            '--beta_vad',
            str(args.token_beta_vad),
            '--vad_threshold',
            str(args.token_vad_threshold),
            '--emotional_vocab',
            str(args.token_emotional_vocab),
            '--vad_min_arousal_vad_only',
            str(args.token_vad_min_arousal_vad_only),
            '--max_ngram',
            str(args.token_max_ngram),
            '--weight',
            str(args.token_weight),
            '--max_terms',
            str(args.token_max_terms),
            '--stopwords_file',
            'data/stopwords_legal_en_token_ot.txt',
            '--drop_top_df',
            str(args.token_drop_top_df),
            '--top_flows',
            str(args.group_shift_top_flows),
        ]
        if args.group_shift_a:
            cmd += ['--group_a', str(args.group_shift_a)]
        if args.group_shift_b:
            cmd += ['--group_b', str(args.group_shift_b)]
        if bool(args.group_shift_exclude_unknown):
            cmd += ['--exclude_unknown']
        if args.group_shift_limit is not None:
            cmd += [
                '--limit',
                str(int(args.group_shift_limit)),
            ]
        if bool(args.no_vis):
            cmd += ['--no_vis']
        subprocess.run(
            cmd,
            check=True,
        )
        _write_json(
            group_shift_sig_path,
            target_group_shift_sig,
        )
    elif args.no_group_shift:
        log('group shift skip')
    else:
        log('group shift skip')
    most_emotional_html = None
    if not args.no_vis:
        from legal_emotion.emotional_docs_viz import (
            write_most_emotional_docs_html_report,
        )
        from legal_emotion.utils import load_config

        cfg = load_config(str(run_dir / 'emotion_config.resolved.yaml'))
        emotion_labels = getattr(
            cfg,
            'emotion_display_names',
            None,
        )
        if not isinstance(
            emotion_labels,
            dict,
        ):
            emotion_labels = None
        most_emotional_html = (
            write_most_emotional_docs_html_report(
                output_path=run_dir
                / 'icj_most_emotional_docs.html',
                scores_jsonl=scores_jsonl,
                top_n=10,
                excerpt_chars=800,
                title='Most Emotional ICJ Documents',
                emotion_labels=emotion_labels,
            )
        )
    summary = {
        'run_dir': str(run_dir),
        'gold_train': str(gold_train),
        'gold_dev': str(gold_dev),
        'teacher_silver_stats': (
            str(run_dir / 'icj_teacher_silver_stats.json')
            if (
                run_dir / 'icj_teacher_silver_stats.json'
            ).exists()
            else None
        ),
        'emotion_checkpoint': str(emo_ckpt),
        'icj_scores': str(scores_jsonl),
        'icj_compare_jsonl': str(compare_out_jsonl),
        'icj_compare_html': (
            str(compare_vis_html)
            if not args.no_vis
            else None
        ),
        'icj_most_emotional_html': most_emotional_html,
        'icj_token_compare_jsonl': (
            str(token_out_jsonl)
            if not args.no_token_ot
            else None
        ),
        'icj_token_compare_html': (
            str(token_vis_html)
            if not args.no_token_ot and (not args.no_vis)
            else None
        ),
        'icj_token_check_dir': (
            str(token_check_dir)
            if not args.no_token_check
            else None
        ),
        'icj_token_check_summary': (
            str(token_check_summary)
            if not args.no_token_check
            and token_check_summary.exists()
            else None
        ),
        'icj_group_shift_dir': (
            str(group_shift_dir)
            if not args.no_group_shift
            else None
        ),
        'icj_group_shift_summary': (
            str(group_shift_summary)
            if not args.no_group_shift
            and group_shift_summary.exists()
            else None
        ),
    }
    if (
        not args.no_group_shift
        and group_shift_summary.exists()
    ):
        try:
            shift = json.loads(group_shift_summary.read_text(encoding='utf-8'))
            if isinstance(
                shift,
                dict,
            ):
                summary['icj_group_shift_html'] = shift.get('report_html')
                summary['icj_group_shift_groups'] = [
                    shift.get('group_a'),
                    shift.get('group_b'),
                ]
        except Exception:
            pass
    summary_path = run_dir / 'summary.json'
    _write_json(
        summary_path,
        summary,
    )
    summary_sections: List[Tuple[str, Path]] = []
    summary_sections.append(('Run Summary', summary_path))
    teacher_stats_path = (
        run_dir / 'icj_teacher_silver_stats.json'
    )
    if teacher_stats_path.exists():
        summary_sections.append(('Teacher Silver Stats', teacher_stats_path))
    emotion_metrics = (
        run_dir / 'emotion_gold_eval_metrics.json'
    )
    if emotion_metrics.exists():
        summary_sections.append(('Emotion Gold Eval Metrics', emotion_metrics))
    compare_stats = run_dir / 'icj_compare_stats.json'
    if compare_stats.exists():
        summary_sections.append(('Doc-level OT Compare Stats', compare_stats))
    token_stats = run_dir / 'icj_token_ot_stats.json'
    if token_stats.exists():
        summary_sections.append((
                'Token-OT Compare Stats (experimental)',
                token_stats,
            ))
    if token_check_summary.exists():
        summary_sections.append((
                'Token-OT Check Summary (experimental)',
                token_check_summary,
            ))
    token_check_report = token_check_dir / 'report.md'
    if token_check_report.exists():
        summary_sections.append((
                'Token-OT Check Report (experimental)',
                token_check_report,
            ))
    summary_html = None
    if summary_sections and (not args.no_vis):
        try:
            summary_html = write_summary_html_report(
                output_path=run_dir / 'icj_summary.html',
                sections=[
                    (label, str(path))
                    for label, path in summary_sections
                ],
                title='Legal Emotion OT Summary',
            )
            summary['icj_summary_html'] = summary_html
        except Exception:
            summary_html = None
    if not args.no_vis:
        sections = []
        if summary_html:
            sections.append(('Summary', Path(summary_html)))
        if most_emotional_html:
            sections.append((
                    'Most Emotional Documents',
                    Path(most_emotional_html),
                ))
        if compare_vis_html.exists():
            sections.append((
                    'Doc-level OT Neighbours',
                    compare_vis_html,
                ))
        group_html = summary.get('icj_group_shift_html')
        if group_html:
            group_html_p = Path(str(group_html))
            if group_html_p.exists():
                sections.append((
                        'Group Shift (token-OT, experimental)',
                        group_html_p,
                    ))
        if not args.no_token_ot and token_vis_html.exists():
            sections.append((
                    'Token-OT Neighbours (experimental)',
                    token_vis_html,
                ))
        if sections:
            combined_html = write_combined_html_report(
                output_path=run_dir
                / 'icj_full_report.html',
                sections=[
                    (label, str(path))
                    for label, path in sections
                ],
                title='Legal Emotion OT Report',
            )
            summary['icj_combined_html'] = combined_html
            if not args.keep_standalone_html:
                to_remove = [
                    summary_html,
                    most_emotional_html,
                    (
                        str(compare_vis_html)
                        if compare_vis_html.exists()
                        else None
                    ),
                    (
                        str(token_vis_html)
                        if not args.no_token_ot
                        and token_vis_html.exists()
                        else None
                    ),
                    group_html,
                ]
                for item in to_remove:
                    if not item:
                        continue
                    p = Path(str(item))
                    if p.exists() and str(p) != str(combined_html):
                        p.unlink()
                summary['icj_summary_html'] = None
                summary['icj_compare_html'] = None
                summary['icj_most_emotional_html'] = None
                summary['icj_token_compare_html'] = None
                summary['icj_group_shift_html'] = None
    _write_json(
        summary_path,
        summary,
    )
    print(json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
