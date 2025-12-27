#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import iter_text_paths
from legal_emotion.token_compare import (
    ENGLISH_STOP_WORDS,
    _build_emotional_matcher,
    _build_term_vad_map,
    _extract_emotional_terms,
    _load_imputed_vad_map,
    _load_stopword_terms,
    _merge_imputed_vad,
    _normalise_negators,
    _term_vad_from_lexicon,
    _get_text,
)
from legal_emotion.lexicon import (
    load_lexicon,
    load_word_vad,
    resolve_vad_path,
    tokenize as lex_tokenize,
)
from legal_emotion.utils import load_config


@dataclass(frozen=True)
class TokenTuningSettings:
    mode: str = 'sinkhorn_divergence'
    weight: str = 'tfidf'
    cost: str = 'embedding_vad'
    alpha_embed: float = 0.8
    beta_vad: float = 0.2
    vad_threshold: float = 0.35
    emotional_vocab: str = 'lexicon_or_vad'
    vad_min_arousal_vad_only: float = 0.4
    max_ngram: int = 3
    max_terms: int = 512
    drop_top_df: int = 10


def _read_jsonl(
    path: Path,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
            if limit is not None and len(rows) >= max(
                0,
                int(limit),
            ):
                break
    return rows


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


def _sha256_file(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b'',
            ):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


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


def _build_check_params(
    *,
    opts: argparse.Namespace,
    cfg_sha: Optional[str],
    stopwords_sha: Optional[str],
    selected_jsonl: Optional[str],
) -> Dict[str, Any]:
    if selected_jsonl:
        return {
            'selection': 'jsonl',
            'selected_jsonl': selected_jsonl,
            'format': str(opts.check_format)
            .strip()
            .lower(),
            'topk': int(opts.check_topk),
            'candidate_k': int(opts.check_candidate_k),
            'top_flows': int(opts.check_top_flows),
            'explain_pairs': int(opts.check_explain_pairs),
            'cfg_sha256': cfg_sha,
            'stopwords_sha256': stopwords_sha,
        }
    return {
        'selection': 'sample',
        'n_per_category': int(opts.check_n_per_category),
        'random_n': int(opts.check_random_n),
        'seed': int(opts.check_seed),
        'format': str(opts.check_format).strip().lower(),
        'topk': int(opts.check_topk),
        'candidate_k': int(opts.check_candidate_k),
        'top_flows': int(opts.check_top_flows),
        'explain_pairs': int(opts.check_explain_pairs),
        'cfg_sha256': cfg_sha,
        'stopwords_sha256': stopwords_sha,
    }


def _build_check_sig(
    *,
    check_params: Dict[str, Any],
    selected_jsonl: Optional[str],
    cfg_sha: Optional[str],
    stopwords_sha: Optional[str],
) -> Tuple[Any, ...]:
    if selected_jsonl:
        return (
            'jsonl',
            str(selected_jsonl),
            str(check_params['format']),
            int(check_params['topk']),
            int(check_params['candidate_k']),
            int(check_params['top_flows']),
            int(check_params['explain_pairs']),
            cfg_sha,
            stopwords_sha,
        )
    return (
        'sample',
        int(check_params['n_per_category']),
        int(check_params['random_n']),
        int(check_params['seed']),
        str(check_params['format']),
        int(check_params['topk']),
        int(check_params['candidate_k']),
        int(check_params['top_flows']),
        int(check_params['explain_pairs']),
        cfg_sha,
        stopwords_sha,
    )


def _purge_token_outputs(
    run_dir: Path,
    *,
    purge_check: bool = True,
) -> None:
    _safe_unlink(run_dir / 'icj_token_neighbours.jsonl')
    _safe_unlink(run_dir / 'icj_token_neighbours.html')
    _safe_unlink(run_dir / 'icj_token_ot_stats.json')
    if purge_check:
        _safe_rmtree(run_dir / 'token_ot_check')
    _safe_rmtree(run_dir / 'icj_group_shift')


def _infer_icj_dir_from_jsonl(path: Path) -> Optional[Path]:
    try:
        with path.open(
            'r',
            encoding='utf-8',
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = row.get('meta')
                if isinstance(
                    meta,
                    dict,
                ) and isinstance(
                    meta.get('path'),
                    str,
                ):
                    p = Path(meta['path'])
                    return (
                        p.parent if p.exists() else p.parent
                    )
                break
    except Exception:
        return None
    return None


def _ensure_doc_index_jsonl(
    *,
    run_dir: Path,
    icj_dir: Path,
) -> Path:
    out = run_dir / 'token_ot_tuning' / 'icj_docs.jsonl'
    if out.exists():
        return out
    paths = list(iter_text_paths(
            icj_dir,
            recursive=True,
        ))
    rows = [
        {
            'meta': {
                'id': p.stem,
                'path': str(p),
                'source': 'ICJ',
            }
        }
        for p in paths
    ]
    _write_jsonl(
        out,
        rows,
    )
    return out


def _selection_stats(
    *,
    input_jsonl: Path,
    cfg_path: Path,
    settings: TokenTuningSettings,
    limit: Optional[int],
    progress_every: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    setup = load_config(str(cfg_path))
    rows = _read_jsonl(
        input_jsonl,
        limit=limit,
    )
    if not rows:
        raise ValueError('error: ValueError')
    focus = 'emotional'
    cost_mode = str(settings.cost).lower().strip()
    max_ngram_eff = max(
        1,
        int(settings.max_ngram),
    )
    min_len = 2
    include_english = bool(getattr(
            setup,
            'token_include_english_stopwords',
            False,
        ))
    sw: set[str] = set()
    stopwords_file = 'data/stopwords_legal_en_token_ot.txt'
    if include_english:
        sw |= set(ENGLISH_STOP_WORDS)
    (
        extra_sw,
        extra_phrase_sw,
    ) = (
        _load_stopword_terms(stopwords_file)
        if stopwords_file
        else (set(), set())
    )
    sw |= extra_sw
    neg_window = int(getattr(
            setup,
            'lexicon_negation_window',
            0,
        ) or 0)
    negator_list = getattr(
        setup,
        'lexicon_negators',
        None,
    )
    (
        neg_tokens,
        neg_phrases,
    ) = _normalise_negators(negator_list)
    allow_vad_stopwords = bool(getattr(
            setup,
            'token_allow_vad_stopwords',
            False,
        ))
    lex_stopwords_path = getattr(
        setup,
        'lexicon_stopwords_file',
        None,
    )
    word_vad_stopwords_path = (
        None if allow_vad_stopwords else lex_stopwords_path
    )
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
        stopwords_path=lex_stopwords_path,
        word_vad_stopwords_path=word_vad_stopwords_path,
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
        allow_seed_only=bool(getattr(
                setup,
                'lexicon_allow_seed_only',
                False,
            )),
        allow_missing_vad=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    )
    vad_path = resolve_vad_path(
        getattr(
            setup,
            'vad_lexicon_path',
            None,
        ),
        allow_missing=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    )
    word_vad = (
        load_word_vad(
            vad_path,
            vad_scale=getattr(
                setup,
                'word_vad_scale',
                None,
            ),
            stopwords_path=word_vad_stopwords_path,
        )
        if vad_path
        else {}
    )
    (
        term_vad,
        term_vad_conf,
    ) = _build_term_vad_map(
        lex,
        word_vad,
        imputed_weight=float(getattr(
                setup,
                'token_vad_imputed_weight',
                0.0,
            )
            or 0.0),
        return_conf=True,
    )
    lexicon_terms = set(_term_vad_from_lexicon(lex).keys())
    word_vad_terms = set((
            ' '.join(lex_tokenize(t))
            for t in (word_vad or {}).keys()
            if ' '.join(lex_tokenize(t))
        ))
    vad_imputed_path = getattr(
        setup,
        'token_vad_imputed_path',
        None,
    )
    if vad_imputed_path:
        (
            imputed_vad,
            imputed_conf,
        ) = _load_imputed_vad_map(vad_imputed_path)
        imputed_terms = _merge_imputed_vad(
            term_vad,
            term_vad_conf,
            imputed_vad,
            imputed_conf,
            default_conf=float(getattr(
                    setup,
                    'token_vad_imputed_weight',
                    0.0,
                )
                or 0.0),
        )
        if imputed_terms:
            word_vad_terms |= imputed_terms
    (
        allowed_terms,
        ngram_map,
    ) = _build_emotional_matcher(
        lexicon_terms=lexicon_terms,
        word_vad_terms=word_vad_terms,
        term_vad=term_vad,
        term_vad_conf=term_vad_conf,
        vad_threshold=float(settings.vad_threshold),
        vad_min_arousal_vad_only=float(settings.vad_min_arousal_vad_only),
        emotional_vocab=str(settings.emotional_vocab),
        max_ngram=int(max_ngram_eff),
        stopword_terms=extra_phrase_sw,
    )
    k_drop = int(settings.drop_top_df)
    df = None
    if k_drop > 0:
        from collections import Counter

        df = Counter()
    docs_terms: List[List[str]] = []
    raw_token_counts: List[int] = []
    t0 = time.time()
    total = int(len(rows))
    every = (
        int(progress_every)
        if progress_every is not None
        else 0
    )
    for idx, row in enumerate(
        rows,
        start=1,
    ):
        text = _get_text(row)
        toks_raw = lex_tokenize(text)
        raw_token_counts.append(int(len(toks_raw)))
        toks_all = [
            w
            for w in toks_raw
            if len(w) >= min_len and w not in sw
        ]
        if df is not None:
            for t in set(toks_all):
                df[t] += 1
        if focus == 'all':
            docs_terms.append(list(toks_all))
            continue
        docs_terms.append(_extract_emotional_terms(
                toks_raw,
                lexicon_terms=lexicon_terms,
                word_vad_terms=word_vad_terms,
                term_vad=term_vad,
                term_vad_conf=term_vad_conf,
                vad_threshold=float(settings.vad_threshold),
                vad_min_arousal_vad_only=float(settings.vad_min_arousal_vad_only),
                emotional_vocab=str(settings.emotional_vocab),
                max_ngram=int(max_ngram_eff),
                stopwords=sw,
                stopword_terms=extra_phrase_sw,
                min_token_len=min_len,
                negation_window=int(neg_window),
                negator_tokens=neg_tokens,
                negator_phrases=neg_phrases,
                allow_vad_terms_in_stopwords=bool(allow_vad_stopwords),
                allowed_terms=allowed_terms,
                ngram_to_term=ngram_map,
            ))
        if (
            every > 0
            and log is not None
            and (idx % every == 0)
        ):
            dt = max(
                1e-06,
                float(time.time() - t0),
            )
            rate = float(idx) / dt
            log(f'stats {idx}/{total}')
    dropped_top_df_terms: List[str] = []
    if k_drop > 0 and df:
        items = sorted(
            df.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
        if len(items) > k_drop:
            dropped_top_df_terms = [
                t for t, _ in items[:k_drop]
            ]
            drop_set = set(dropped_top_df_terms)
            if drop_set:
                docs_terms = [
                    [t for t in toks if t not in drop_set]
                    for toks in docs_terms
                ]
    if cost_mode == 'vad':
        docs_terms = [
            [
                t
                for t in toks
                if term_vad_conf.get(
                    t,
                    0.0,
                ) > 0.0
            ]
            for toks in docs_terms
        ]
    selected_token_counts = [
        int(len(toks)) for toks in docs_terms
    ]
    docs_with_zero_selected = int(sum((1 for c in selected_token_counts if c == 0)))
    total_raw = float(sum(raw_token_counts))
    total_selected = float(sum(selected_token_counts))
    coverage_total = float(total_selected / max(
            1.0,
            total_raw,
        ))
    ratios = [
        float(sel) / max(
            1.0,
            float(raw),
        )
        for sel, raw in zip(
            selected_token_counts,
            raw_token_counts,
        )
    ]
    ratios_sorted = sorted(ratios)
    median = (
        float(ratios_sorted[len(ratios_sorted) // 2])
        if ratios_sorted
        else 0.0
    )
    return {
        'docs': int(len(rows)),
        'coverage_total': float(coverage_total),
        'docs_with_zero_selected': int(docs_with_zero_selected),
        'selected_ratio_median_full': float(median),
        'selected_ratio_min_full': float(min(ratios) if ratios else 0.0),
        'selected_ratio_max_full': float(max(ratios) if ratios else 0.0),
        'drop_top_df': int(k_drop),
        'dropped_top_df_terms': dropped_top_df_terms,
    }


def _guard_reasons(
    *,
    full_stats: Dict[str, Any],
    check: Dict[str, Any],
    min_coverage: float,
    min_selected_ratio: float,
    max_zero_ratio: float,
    min_score: Optional[float],
    min_separation_effect: Optional[float],
    allow_warn: bool,
) -> List[str]:
    reasons: List[str] = []
    coverage_total = float(full_stats.get('coverage_total') or 0.0)
    docs = int(full_stats.get('docs') or 0)
    zero_docs = int(full_stats.get('docs_with_zero_selected') or 0)
    zero_ratio = float(zero_docs) / float(max(
            1,
            docs,
        ))
    if coverage_total < float(min_coverage):
        reasons.append(f'coverage_total {coverage_total:.4f} below {float(min_coverage):.4f}')
    if zero_ratio > float(max_zero_ratio):
        reasons.append(f'docs_with_zero_selected ratio {zero_ratio:.3f} above {float(max_zero_ratio):.3f}')
    metrics = (
        check.get('metrics')
        if isinstance(
            check.get('metrics'),
            dict,
        )
        else {}
    )
    median_ratio = float(metrics.get('selected_ratio_median')
        or check.get('selected_ratio_median')
        or 0.0)
    if median_ratio < float(min_selected_ratio):
        reasons.append(f'selected_ratio_median {median_ratio:.4f} below {float(min_selected_ratio):.4f}')
    score = float(check.get('score') or metrics.get('score') or 0.0)
    if min_score is not None and score < float(min_score):
        reasons.append(f'meaningfulness score {score:.2f} below {float(min_score):.2f}')
    sep_effect_raw = (
        metrics.get('separation_effect')
        if isinstance(
            metrics,
            dict,
        )
        else None
    )
    if sep_effect_raw is None:
        sep_effect_raw = check.get('separation_effect')
    sep_effect: Optional[float] = None
    if isinstance(
        sep_effect_raw,
        (int, float),
    ):
        sep_effect = float(sep_effect_raw)
    if (
        min_separation_effect is not None
        and float(min_separation_effect) > 0.0
    ):
        if sep_effect is None:
            reasons.append(f'missing separation_effect (need >= {float(min_separation_effect):.2f})')
        elif float(sep_effect) < float(min_separation_effect):
            reasons.append(f'separation_effect {float(sep_effect):.3f} below {float(min_separation_effect):.3f}')
    verdict = (
        str(check.get('verdict') or '').strip().lower()
    )
    if (
        verdict
        and verdict != 'pass'
        and (not bool(allow_warn))
    ):
        reasons.append(f'token_ot_check verdict is {verdict}')
    return reasons


def _settings_key(settings: TokenTuningSettings) -> Tuple[Any, ...]:
    return (
        str(settings.mode),
        str(settings.weight),
        str(settings.cost),
        round(
            float(settings.alpha_embed),
            4,
        ),
        round(
            float(settings.beta_vad),
            4,
        ),
        round(
            float(settings.vad_threshold),
            4,
        ),
        str(settings.emotional_vocab),
        round(
            float(settings.vad_min_arousal_vad_only),
            4,
        ),
        int(settings.max_ngram),
        int(settings.max_terms),
        int(settings.drop_top_df),
    )


def _step_count(
    min_val: float,
    max_val: float,
    step: float,
) -> int:
    if step <= 0.0:
        return 0
    return max(
        0,
        int(math.floor((max_val - min_val) / step + 1e-12)),
    )


def _sample_float_grid(
    rng: random.Random,
    *,
    min_val: float,
    max_val: float,
    step: float,
) -> float:
    lo = float(min_val)
    hi = float(max_val)
    if hi < lo:
        (
            lo,
            hi,
        ) = (hi, lo)
    s = float(step)
    if s <= 0.0:
        return float(rng.uniform(
                lo,
                hi,
            ))
    n = _step_count(
        lo,
        hi,
        s,
    )
    idx = int(rng.randint(
            0,
            n,
        ))
    return float(min(
            hi,
            max(
                lo,
                round(
                    lo + idx * s,
                    10,
                ),
            ),
        ))


def _sample_int_grid(
    rng: random.Random,
    *,
    min_val: int,
    max_val: int,
    step: int,
) -> int:
    lo = int(min_val)
    hi = int(max_val)
    if hi < lo:
        (
            lo,
            hi,
        ) = (hi, lo)
    s = int(step) if int(step) > 0 else 1
    n = max(
        0,
        int((hi - lo) // s),
    )
    idx = int(rng.randint(
            0,
            n,
        ))
    return int(min(
            hi,
            max(
                lo,
                lo + idx * s,
            ),
        ))


def _random_restart(
    *,
    current: TokenTuningSettings,
    tried: set[Tuple[Any, ...]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Optional[TokenTuningSettings]:
    vocab_candidates = [
        v.strip()
        for v in str(getattr(
                args,
                'restart_emotional_vocab',
                '',
            )).split(',')
        if v.strip()
    ]
    vocab_candidates = [
        v
        for v in vocab_candidates
        if v in {'lexicon', 'vad', 'lexicon_or_vad'}
    ]
    if not vocab_candidates:
        vocab_candidates = ['lexicon_or_vad', 'lexicon']
    min_terms = int(getattr(
            args,
            'max_terms_min',
            0,
        ) or 0)
    if min_terms <= 0:
        min_terms = 128
    max_terms = int(getattr(
            args,
            'max_terms_max',
            0,
        ) or 0)
    if max_terms <= 0:
        max_terms = 2048
    if max_terms < min_terms:
        (
            min_terms,
            max_terms,
        ) = (max_terms, min_terms)
    terms_step = int(getattr(
            args,
            'max_terms_step',
            0,
        ) or 0)
    ar_min = float(getattr(
            args,
            'vad_min_arousal_min',
            0.0,
        ) or 0.0)
    ar_max = float(getattr(
            args,
            'vad_min_arousal_max',
            1.0,
        ) or 1.0)
    ar_min = max(
        0.0,
        min(
            1.0,
            ar_min,
        ),
    )
    ar_max = max(
        0.0,
        min(
            1.0,
            ar_max,
        ),
    )
    if ar_max < ar_min:
        (
            ar_min,
            ar_max,
        ) = (ar_max, ar_min)
    ar_step = float(getattr(
            args,
            'vad_min_arousal_step',
            0.0,
        ) or 0.0)
    thr_min = float(getattr(
            args,
            'vad_threshold_min',
            0.0,
        ) or 0.0)
    thr_max = float(getattr(
            args,
            'vad_threshold_max',
            0.9,
        ) or 0.9)
    thr_min = max(
        0.0,
        min(
            1.0,
            thr_min,
        ),
    )
    thr_max = max(
        0.0,
        min(
            1.0,
            thr_max,
        ),
    )
    if thr_max < thr_min:
        (
            thr_min,
            thr_max,
        ) = (thr_max, thr_min)
    thr_step = float(getattr(
            args,
            'vad_threshold_step',
            0.0,
        ) or 0.0)
    drop_min = int(getattr(
            args,
            'drop_top_df_min',
            0,
        ) or 0)
    drop_max = int(getattr(
            args,
            'drop_top_df_max',
            0,
        ) or 0)
    if drop_max <= 0:
        drop_max = 500
    if drop_max < drop_min:
        (
            drop_min,
            drop_max,
        ) = (drop_max, drop_min)
    drop_step = int(getattr(
            args,
            'drop_top_df_step',
            0,
        ) or 0)
    attempts = max(
        1,
        int(getattr(
                args,
                'restart_attempts',
                250,
            ) or 250),
    )
    for _ in range(attempts):
        candidate = TokenTuningSettings(**{
                **asdict(current),
                'vad_threshold': _sample_float_grid(
                    rng,
                    min_val=thr_min,
                    max_val=thr_max,
                    step=thr_step,
                ),
                'drop_top_df': _sample_int_grid(
                    rng,
                    min_val=drop_min,
                    max_val=drop_max,
                    step=drop_step,
                ),
                'vad_min_arousal_vad_only': _sample_float_grid(
                    rng,
                    min_val=ar_min,
                    max_val=ar_max,
                    step=ar_step,
                ),
                'max_terms': _sample_int_grid(
                    rng,
                    min_val=min_terms,
                    max_val=max_terms,
                    step=terms_step,
                ),
                'emotional_vocab': str(rng.choice(vocab_candidates)),
            })
        if _settings_key(candidate) not in tried:
            return candidate
    return None


def _propose_next(
    *,
    current: TokenTuningSettings,
    full_stats: Dict[str, Any],
    check: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[TokenTuningSettings]:
    metrics = (
        check.get('metrics')
        if isinstance(
            check.get('metrics'),
            dict,
        )
        else {}
    )
    verdict = (
        str(check.get('verdict') or '').strip().lower()
    )
    score = float(check.get('score') or metrics.get('score') or 0.0)
    coverage_total = float(full_stats.get('coverage_total') or 0.0)
    zero_ratio = float(full_stats.get('docs_with_zero_selected') or 0) / float(max(
            1,
            int(full_stats.get('docs') or 0),
        ))
    median_ratio = float(metrics.get('selected_ratio_median') or 0.0)
    boilerplate_rate = metrics.get('boilerplate_rate')
    anchor_rate = metrics.get('anchor_rate')
    top1 = metrics.get('top1_same_category_rate')
    sep_effect_raw = metrics.get('separation_effect')
    sep_effect = (
        float(sep_effect_raw)
        if isinstance(
            sep_effect_raw,
            (int, float),
        )
        else None
    )
    drop_step = int(getattr(
            args,
            'drop_top_df_step',
            0,
        ) or 0)
    drop_max = int(getattr(
            args,
            'drop_top_df_max',
            0,
        ) or 0)

    def bump_drop_df(cur: int) -> Optional[int]:
        if drop_step <= 0:
            return None
        nxt = int(cur) + int(drop_step)
        if drop_max > 0:
            nxt = min(
                int(drop_max),
                nxt,
            )
        return None if nxt == int(cur) else int(nxt)

    need_more_coverage = (
        coverage_total
        < float(args.token_guard_min_coverage)
        or median_ratio
        < float(args.token_guard_min_selected_ratio)
        or zero_ratio
        > float(args.token_guard_max_zero_ratio)
    )
    if need_more_coverage:
        thr = float(current.vad_threshold)
        if thr - float(args.vad_threshold_step) >= float(args.vad_threshold_min):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'vad_threshold': float(thr - float(args.vad_threshold_step)),
                })
        if int(current.drop_top_df) > int(args.drop_top_df_min):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'drop_top_df': int(args.drop_top_df_min),
                })
        ar = float(current.vad_min_arousal_vad_only)
        if ar - float(args.vad_min_arousal_step) >= float(args.vad_min_arousal_min):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'vad_min_arousal_vad_only': float(ar
                        - float(args.vad_min_arousal_step)),
                })
        if str(current.emotional_vocab) != 'lexicon_or_vad':
            return TokenTuningSettings(**{
                    **asdict(current),
                    'emotional_vocab': 'lexicon_or_vad',
                })
        return None
    min_sep = getattr(
        args,
        'token_guard_min_separation_effect',
        None,
    )
    need_more_separation = (
        min_sep is not None
        and float(min_sep) > 0.0
        and (
            sep_effect is None
            or float(sep_effect) < float(min_sep)
        )
    )
    min_score = getattr(
        args,
        'token_guard_min_score',
        None,
    )
    need_better_score = (
        min_score is not None and score < float(min_score)
    )
    if need_better_score or need_more_separation:
        if (
            isinstance(
                anchor_rate,
                (int, float),
            )
            and float(anchor_rate) < 0.15
            and (
                str(current.emotional_vocab)
                == 'lexicon_or_vad'
            )
        ):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'emotional_vocab': 'lexicon',
                })
        max_terms_max = int(getattr(
                args,
                'max_terms_max',
                0,
            ) or 0)
        max_terms_step = int(getattr(
                args,
                'max_terms_step',
                0,
            ) or 0)
        if (
            max_terms_max > 0
            and max_terms_step > 0
            and (int(current.max_terms) < max_terms_max)
        ):
            nxt_terms = min(
                max_terms_max,
                int(current.max_terms) + max_terms_step,
            )
            return TokenTuningSettings(**{
                    **asdict(current),
                    'max_terms': int(nxt_terms),
                })
        if (
            isinstance(
                boilerplate_rate,
                (int, float),
            )
            and float(boilerplate_rate) > 0.35
            and (str(current.emotional_vocab) != 'lexicon')
        ):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'emotional_vocab': 'lexicon',
                })
        if (
            isinstance(
                boilerplate_rate,
                (int, float),
            )
            and float(boilerplate_rate) > 0.35
        ):
            nxt_drop = bump_drop_df(int(current.drop_top_df))
            if nxt_drop is not None:
                return TokenTuningSettings(**{
                        **asdict(current),
                        'drop_top_df': int(nxt_drop),
                    })
        if (
            isinstance(
                top1,
                (int, float),
            )
            and float(top1) < 0.5
        ):
            nxt_drop = bump_drop_df(int(current.drop_top_df))
            if nxt_drop is not None:
                return TokenTuningSettings(**{
                        **asdict(current),
                        'drop_top_df': int(nxt_drop),
                    })
        thr = float(current.vad_threshold)
        if thr + float(args.vad_threshold_step) <= float(args.vad_threshold_max):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'vad_threshold': float(thr + float(args.vad_threshold_step)),
                })
        return None
    if verdict and verdict != 'pass':
        if (
            isinstance(
                boilerplate_rate,
                (int, float),
            )
            and float(boilerplate_rate) > 0.45
        ):
            nxt_drop = bump_drop_df(int(current.drop_top_df))
            if nxt_drop is not None:
                return TokenTuningSettings(**{
                        **asdict(current),
                        'drop_top_df': int(nxt_drop),
                    })
        if (
            isinstance(
                top1,
                (int, float),
            )
            and float(top1) < 0.4
        ):
            nxt_drop = bump_drop_df(int(current.drop_top_df))
            if nxt_drop is not None:
                return TokenTuningSettings(**{
                        **asdict(current),
                        'drop_top_df': int(nxt_drop),
                    })
        thr = float(current.vad_threshold)
        if thr + float(args.vad_threshold_step) <= float(args.vad_threshold_max):
            return TokenTuningSettings(**{
                    **asdict(current),
                    'vad_threshold': float(thr + float(args.vad_threshold_step)),
                })
    return None


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
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )


def _append_jsonl(
    path: Path,
    row: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        'a',
        encoding='utf-8',
    ) as f:
        f.write(json.dumps(
                row,
                ensure_ascii=False,
            ) + '\n')


def _latest_run_dir(outputs_dir: Path) -> Optional[Path]:
    if not outputs_dir.exists():
        return None
    candidates = [
        p
        for p in outputs_dir.iterdir()
        if p.is_dir()
        and p.name.startswith('gold_to_icj_ot_')
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description='auto-tune token-ot')
    ap.add_argument(
        '--run_dir',
        type=str,
        default='latest',
    )
    ap.add_argument(
        '--icj_dir',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--emotion_config',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--sentiment_config',
        type=str,
        default='config.sentiment.sample.yaml',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
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
        default=13,
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
        '--resume',
        type=str,
        default='auto',
        choices=('auto', 'last', 'best', 'none'),
    )
    ap.add_argument(
        '--resume_from',
        type=str,
        default=None,
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
        default=13,
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
        '--no_finalize_pipeline',
        action='store_true',
    )
    ap.add_argument(
        '--strict',
        action='store_true',
    )
    opts = ap.parse_args()
    selected_jsonl: Optional[str] = None
    if opts.check_selected_jsonl:
        p = (
            Path(opts.check_selected_jsonl)
            .expanduser()
            .resolve()
        )
        if not p.exists():
            raise SystemExit('error: SystemExit')
        if p.is_dir():
            candidates = sorted([
                    c
                    for c in p.glob('*.jsonl')
                    if c.is_file()
                ])
            preview = (
                ', '.join((str(c.name) for c in candidates[:8]))
                if candidates
                else '(no .jsonl files found)'
            )
            raise SystemExit('error: SystemExit')
        try:
            rows = _read_jsonl(
                p,
                limit=None,
            )
        except Exception as e:
            raise SystemExit('error: SystemExit') from e
        bad: List[int] = []
        for i, row in enumerate(
            rows,
            start=1,
        ):
            meta = (
                row.get('meta')
                if isinstance(
                    row.get('meta'),
                    dict,
                )
                else {}
            )
            cat = (
                meta.get('category')
                if isinstance(
                    meta.get('category'),
                    str,
                )
                else row.get('category')
            )
            if (
                not isinstance(
                    cat,
                    str,
                )
                or not cat.strip()
                or cat.strip().lower()
                in {'unknown', 'unlabelled', 'none', 'null'}
            ):
                bad.append(int(i))
        if bad:
            preview = ', '.join((str(x) for x in bad[:8]))
            raise SystemExit('error: SystemExit')
        selected_jsonl = str(p)
    run_dir = None
    if (
        opts.run_dir
        and str(opts.run_dir).strip().lower() != 'latest'
    ):
        run_dir = Path(opts.run_dir)
    else:
        run_dir = _latest_run_dir(Path('outputs'))
        if run_dir is None:
            raise SystemExit('error: SystemExit')
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit('error: SystemExit')
    resolved_cfg = run_dir / 'emotion_config.resolved.yaml'
    cfg_path: Path | None = None
    if opts.emotion_config:
        cfg_path = (
            Path(opts.emotion_config).expanduser().resolve()
        )
    else:
        cfg_path = (
            resolved_cfg if resolved_cfg.exists() else None
        )
    if cfg_path is None or not cfg_path.exists():
        raise SystemExit('error: SystemExit')
    scores_jsonl = run_dir / 'icj_scores.jsonl'
    icj_dir = Path(opts.icj_dir) if opts.icj_dir else None
    if icj_dir is None and scores_jsonl.exists():
        icj_dir = _infer_icj_dir_from_jsonl(scores_jsonl)
    if icj_dir is None:
        icj_dir = Path('data/EN_TXT_BEST_FULL')
    if not icj_dir.exists():
        raise SystemExit('error: SystemExit')
    input_jsonl = (
        scores_jsonl
        if scores_jsonl.exists()
        else _ensure_doc_index_jsonl(
            run_dir=run_dir,
            icj_dir=icj_dir,
        )
    )
    cfg_sha = _sha256_file(cfg_path)
    stopwords_sha = _sha256_file(Path('data/stopwords_legal_en_token_ot.txt'))
    check_params = _build_check_params(
        opts=opts,
        cfg_sha=cfg_sha,
        stopwords_sha=stopwords_sha,
        selected_jsonl=selected_jsonl,
    )
    check_sig: Tuple[Any, ...] = _build_check_sig(
        check_params=check_params,
        selected_jsonl=selected_jsonl,
        cfg_sha=cfg_sha,
        stopwords_sha=stopwords_sha,
    )

    def attempt_check_sig(attempt: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        cp = attempt.get('check_params')
        if not isinstance(
            cp,
            dict,
        ):
            return None
        try:
            cfg_sha_ = cp.get('cfg_sha256')
            cfg_sha_ = (
                str(cfg_sha_)
                if isinstance(
                    cfg_sha_,
                    str,
                )
                and cfg_sha_.strip()
                else None
            )
            sw_sha = cp.get('stopwords_sha256')
            sw_sha = (
                str(sw_sha)
                if isinstance(
                    sw_sha,
                    str,
                )
                and sw_sha.strip()
                else None
            )
            sel = cp.get('selected_jsonl')
            if isinstance(
                sel,
                str,
            ) and sel.strip():
                return (
                    'jsonl',
                    str(sel).strip(),
                    str(cp.get('format') or '')
                    .strip()
                    .lower(),
                    int(cp.get('topk')),
                    int(cp.get('candidate_k')),
                    int(cp.get('top_flows')),
                    int(cp.get('explain_pairs')),
                    cfg_sha_,
                    sw_sha,
                )
            return (
                'sample',
                int(cp.get('n_per_category')),
                int(cp.get('random_n')),
                int(cp.get('seed')),
                str(cp.get('format') or '').strip().lower(),
                int(cp.get('topk')),
                int(cp.get('candidate_k')),
                int(cp.get('top_flows')),
                int(cp.get('explain_pairs')),
                cfg_sha_,
                sw_sha,
            )
        except Exception:
            return None

    def objective(attempt: Dict[str, Any]) -> tuple[float, float, float]:
        full_stats = (
            attempt.get('full_stats')
            if isinstance(
                attempt.get('full_stats'),
                dict,
            )
            else {}
        )
        check = (
            attempt.get('check')
            if isinstance(
                attempt.get('check'),
                dict,
            )
            else {}
        )
        metrics = (
            check.get('metrics')
            if isinstance(
                check.get('metrics'),
                dict,
            )
            else {}
        )
        coverage_total = float(full_stats.get('coverage_total') or 0.0)
        docs = int(full_stats.get('docs') or 0)
        zero_docs = int(full_stats.get('docs_with_zero_selected') or 0)
        zero_ratio = float(zero_docs) / float(max(
                1,
                docs,
            ))
        median_ratio = float(metrics.get('selected_ratio_median')
            or check.get('selected_ratio_median')
            or 0.0)
        score = float(metrics.get('score')
            or check.get('score')
            or 0.0)
        sep_effect_raw = (
            metrics.get('separation_effect')
            if isinstance(
                metrics,
                dict,
            )
            else None
        )
        if sep_effect_raw is None:
            sep_effect_raw = check.get('separation_effect')
        sep_effect = (
            float(sep_effect_raw)
            if isinstance(
                sep_effect_raw,
                (int, float),
            )
            else None
        )
        cov_def = max(
            0.0,
            float(opts.token_guard_min_coverage)
            - coverage_total,
        ) / max(
            1e-09,
            float(opts.token_guard_min_coverage),
        )
        sel_def = max(
            0.0,
            float(opts.token_guard_min_selected_ratio)
            - median_ratio,
        ) / max(
            1e-09,
            float(opts.token_guard_min_selected_ratio),
        )
        zero_def = max(
            0.0,
            zero_ratio
            - float(opts.token_guard_max_zero_ratio),
        ) / max(
            1e-09,
            float(opts.token_guard_max_zero_ratio),
        )
        score_def = 0.0
        if opts.token_guard_min_score is not None:
            score_def = max(
                0.0,
                float(opts.token_guard_min_score) - score,
            ) / max(
                1e-09,
                float(opts.token_guard_min_score),
            )
        sep_def = 0.0
        min_sep = getattr(
            opts,
            'token_guard_min_separation_effect',
            None,
        )
        if min_sep is not None and float(min_sep) > 0.0:
            if sep_effect is None:
                sep_def = 1.0
            else:
                sep_def = max(
                    0.0,
                    float(min_sep) - float(sep_effect),
                ) / max(
                    1e-09,
                    float(min_sep),
                )
        deficit = (
            2.0 * cov_def
            + 2.0 * score_def
            + 2.0 * sep_def
            + sel_def
            + zero_def
        )
        return (
            float(deficit),
            -float(score),
            -float(coverage_total),
        )

    def coerce_settings(
        raw: Dict[str, Any],
        fallback: TokenTuningSettings,
    ) -> TokenTuningSettings:
        base = dict(asdict(fallback))
        for k in list(base.keys()):
            if k in raw:
                base[k] = raw[k]
        return TokenTuningSettings(
            mode=str(base['mode']),
            weight=str(base['weight']),
            cost=str(base['cost']),
            alpha_embed=float(base['alpha_embed']),
            beta_vad=float(base['beta_vad']),
            vad_threshold=float(base['vad_threshold']),
            emotional_vocab=str(base['emotional_vocab']),
            vad_min_arousal_vad_only=float(base['vad_min_arousal_vad_only']),
            max_ngram=int(base['max_ngram']),
            max_terms=int(base['max_terms']),
            drop_top_df=int(base['drop_top_df']),
        )

    def load_attempt(path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None
        return (
            payload if isinstance(
                payload,
                dict,
            ) else None
        )

    setup = load_config(str(cfg_path))
    default_settings = TokenTuningSettings(
        mode='sinkhorn_divergence',
        weight='tfidf',
        cost='embedding_vad',
        alpha_embed=0.8,
        beta_vad=0.2,
        vad_threshold=(
            float(opts.vad_threshold_start)
            if opts.vad_threshold_start is not None
            else 0.35
        ),
        emotional_vocab='lexicon_or_vad',
        vad_min_arousal_vad_only=(
            float(opts.vad_min_arousal_start)
            if opts.vad_min_arousal_start is not None
            else 0.4
        ),
        max_ngram=3,
        max_terms=(
            int(opts.max_terms_start)
            if opts.max_terms_start is not None
            else 512
        ),
        drop_top_df=(
            int(opts.drop_top_df_start)
            if opts.drop_top_df_start is not None
            else 10
        ),
    )
    tuning_dir = run_dir / 'token_ot_tuning'
    tuning_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    history_path = tuning_dir / 'history.jsonl'
    best_path = tuning_dir / 'best_settings.json'
    best_cmd_path = tuning_dir / 'best_pipeline_command.txt'
    best_attempt: Optional[Dict[str, Any]] = None
    best_obj: Optional[tuple[float, float, float]] = None
    best_attempt_any: Optional[Dict[str, Any]] = None
    best_obj_any: Optional[tuple[float, float, float]] = (
        None
    )
    tried: set[Tuple[Any, ...]] = set()
    attempt_by_key: Dict[
        Tuple[Any, ...], Dict[str, Any]
    ] = {}
    last_attempt_any: Optional[Dict[str, Any]] = None
    max_iter_seen = 0
    if (
        history_path.exists()
        and str(opts.resume).lower() != 'none'
    ):
        with history_path.open(
            'r',
            encoding='utf-8',
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    a = json.loads(line)
                except Exception:
                    continue
                if not isinstance(
                    a,
                    dict,
                ):
                    continue
                last_attempt_any = a
                it = a.get('iter')
                if isinstance(
                    it,
                    int,
                ):
                    max_iter_seen = max(
                        max_iter_seen,
                        int(it),
                    )
                try:
                    obj = objective(a)
                except Exception:
                    obj = None
                if obj is not None and (
                    best_obj_any is None
                    or obj < best_obj_any
                ):
                    best_obj_any = obj
                    best_attempt_any = a
                if attempt_check_sig(a) != check_sig:
                    continue
                s = a.get('settings')
                if isinstance(
                    s,
                    dict,
                ):
                    try:
                        k = _settings_key(coerce_settings(
                                s,
                                default_settings,
                            ))
                        tried.add(k)
                        attempt_by_key[k] = a
                    except Exception:
                        pass
                if obj is not None and (
                    best_obj is None or obj < best_obj
                ):
                    best_obj = obj
                    best_attempt = a
    resume_choice = str(opts.resume).lower().strip()
    resume_from = (
        Path(opts.resume_from) if opts.resume_from else None
    )
    base_attempt: Optional[Dict[str, Any]] = None
    if resume_from is not None:
        base_attempt = load_attempt(resume_from)
    elif resume_choice in {'auto', 'last'}:
        base_attempt = last_attempt_any
    if base_attempt is None and resume_choice in {
        'auto',
        'best',
    }:
        base_attempt = best_attempt_any or best_attempt
    if base_attempt is None and resume_choice != 'none':
        if (
            resume_choice in {'auto', 'best'}
            and best_path.exists()
        ):
            base_attempt = load_attempt(best_path)
        if base_attempt is None and resume_choice in {
            'auto',
            'last',
        }:
            cand = tuning_dir / 'last_attempt.json'
            if cand.exists():
                base_attempt = load_attempt(cand)
    settings = default_settings
    if base_attempt and isinstance(
        base_attempt.get('settings'),
        dict,
    ):
        try:
            settings = coerce_settings(
                base_attempt['settings'],
                default_settings,
            )
        except Exception:
            settings = default_settings
    if opts.vad_threshold_start is not None:
        settings = TokenTuningSettings(**{
                **asdict(settings),
                'vad_threshold': float(opts.vad_threshold_start),
            })
    if opts.drop_top_df_start is not None:
        settings = TokenTuningSettings(**{
                **asdict(settings),
                'drop_top_df': int(opts.drop_top_df_start),
            })
    if opts.vad_min_arousal_start is not None:
        settings = TokenTuningSettings(**{
                **asdict(settings),
                'vad_min_arousal_vad_only': float(opts.vad_min_arousal_start),
            })
    if opts.max_terms_start is not None:
        settings = TokenTuningSettings(**{
                **asdict(settings),
                'max_terms': int(opts.max_terms_start),
            })
    if (
        base_attempt
        and isinstance(
            base_attempt.get('settings'),
            dict,
        )
        and (_settings_key(settings) in tried)
    ):
        prev_full = (
            base_attempt.get('full_stats')
            if isinstance(
                base_attempt.get('full_stats'),
                dict,
            )
            else None
        )
        prev_check = (
            base_attempt.get('check')
            if isinstance(
                base_attempt.get('check'),
                dict,
            )
            else None
        )
        if prev_full is None or prev_check is None:
            prev = attempt_by_key.get(_settings_key(settings))
            if isinstance(
                prev,
                dict,
            ):
                if prev_full is None and isinstance(
                    prev.get('full_stats'),
                    dict,
                ):
                    prev_full = prev.get('full_stats')
                if prev_check is None and isinstance(
                    prev.get('check'),
                    dict,
                ):
                    prev_check = prev.get('check')
        if prev_full is not None and prev_check is not None:
            nxt = _propose_next(
                current=settings,
                full_stats=prev_full,
                check=prev_check,
                args=opts,
            )
            if nxt is not None:
                settings = nxt
    token_check_dir = run_dir / 'token_ot_check'
    token_check_summary = token_check_dir / 'summary.json'
    restart_rng = random.Random(int(getattr(
                opts,
                'restart_seed',
                13,
            )))
    max_iters = int(getattr(
            opts,
            'max_iters',
            0,
        ) or 0)
    i = (
        int(max_iter_seen) + 1
        if max_iter_seen > 0
        and str(opts.resume).lower() != 'none'
        else 1
    )
    while True:
        if max_iters > 0 and i > max_iters:
            break
        if _settings_key(settings) in tried:
            if bool(getattr(
                    opts,
                    'restart_on_stuck',
                    False,
                )):
                nxt_restart = _random_restart(
                    current=settings,
                    tried=tried,
                    args=opts,
                    rng=restart_rng,
                )
                if nxt_restart is not None:
                    print(
                        f'iter {i} restart',
                        flush=True,
                    )
                    settings = nxt_restart
                    continue
            best = best_attempt
            if best is not None:
                _write_json(
                    best_path,
                    best,
                )
                best_settings = (
                    best.get('settings')
                    if isinstance(
                        best.get('settings'),
                        dict,
                    )
                    else {}
                )
            else:
                best_settings = asdict(settings)
            best_cmd_parts = [
                str(sys.executable),
                'scripts/run_gold_to_icj_ot_pipeline.py',
                '--run_dir',
                str(run_dir),
                '--sentiment_config',
                str(opts.sentiment_config),
                '--emotion_config',
                str(cfg_path),
                '--icj_dir',
                str(icj_dir),
                '--token_check_format',
                str(opts.check_format),
                '--token_check_topk',
                str(int(opts.check_topk)),
                '--token_check_candidate_k',
                str(int(opts.check_candidate_k)),
                '--token_check_top_flows',
                str(int(opts.check_top_flows)),
                '--token_check_explain_pairs',
                str(int(opts.check_explain_pairs)),
                '--skip_sentiment_train',
                '--skip_emotion_train',
                '--no_token_guard',
                '--token_mode',
                str(best_settings.get(
                        'mode',
                        default_settings.mode,
                    )),
                '--token_weight',
                str(best_settings.get(
                        'weight',
                        default_settings.weight,
                    )),
                '--token_cost',
                str(best_settings.get(
                        'cost',
                        default_settings.cost,
                    )),
                '--token_alpha_embed',
                str(best_settings.get(
                        'alpha_embed',
                        default_settings.alpha_embed,
                    )),
                '--token_beta_vad',
                str(best_settings.get(
                        'beta_vad',
                        default_settings.beta_vad,
                    )),
                '--token_vad_threshold',
                str(best_settings.get(
                        'vad_threshold',
                        default_settings.vad_threshold,
                    )),
                '--token_emotional_vocab',
                str(best_settings.get(
                        'emotional_vocab',
                        default_settings.emotional_vocab,
                    )),
                '--token_vad_min_arousal_vad_only',
                str(best_settings.get(
                        'vad_min_arousal_vad_only',
                        default_settings.vad_min_arousal_vad_only,
                    )),
                '--token_max_ngram',
                str(best_settings.get(
                        'max_ngram',
                        default_settings.max_ngram,
                    )),
                '--token_max_terms',
                str(best_settings.get(
                        'max_terms',
                        default_settings.max_terms,
                    )),
                '--token_drop_top_df',
                str(best_settings.get(
                        'drop_top_df',
                        default_settings.drop_top_df,
                    )),
            ]
            if selected_jsonl:
                best_cmd_parts += [
                    '--token_check_selected_jsonl',
                    str(selected_jsonl),
                ]
            else:
                best_cmd_parts += [
                    '--token_check_n_per_category',
                    str(int(opts.check_n_per_category)),
                    '--token_check_random_n',
                    str(int(opts.check_random_n)),
                    '--token_check_seed',
                    str(int(opts.check_seed)),
                ]
            best_cmd_path.write_text(
                ' '.join(best_cmd_parts) + '\n',
                encoding='utf-8',
            )
            payload = {
                'status': 'stuck_repeat',
                'run_dir': str(run_dir),
                'best_settings': best_settings,
                'best_pipeline_command': str(best_cmd_path),
                'history': str(history_path),
            }
            print(json.dumps(
                    payload,
                    indent=2,
                ))
            if bool(opts.strict):
                raise SystemExit('error: SystemExit')
            return 0
        tried.add(_settings_key(settings))
        iter_total = (
            str(max_iters) if max_iters > 0 else 'inf'
        )
        print(
            f'iter {i}/{iter_total} try',
            flush=True,
        )
        _purge_token_outputs(run_dir)
        print(
            f'iter {i} scan',
            flush=True,
        )
        full_stats = _selection_stats(
            input_jsonl=input_jsonl,
            cfg_path=cfg_path,
            settings=settings,
            limit=(
                int(opts.limit)
                if opts.limit is not None
                else None
            ),
            progress_every=(
                int(opts.progress_every)
                if int(opts.progress_every) > 0
                else None
            ),
            log=(
                (lambda m: print(
                    m,
                    flush=True,
                ))
                if int(opts.progress_every) > 0
                else None
            ),
        )
        print(
            f'iter {i} cov {full_stats.get('coverage_total'):.4f}',
            flush=True,
        )
        cmd = [
            sys.executable,
            str(_ROOT
                / 'scripts'
                / 'token_ot_meaningfulness_check.py'),
            '--config',
            str(cfg_path),
            '--out_dir',
            str(token_check_dir),
        ]
        if selected_jsonl:
            cmd += ['--selected_jsonl', str(selected_jsonl)]
        else:
            cmd += [
                '--input_dir',
                str(icj_dir),
                '--n_per_category',
                str(int(opts.check_n_per_category)),
                '--random_n',
                str(int(opts.check_random_n)),
                '--seed',
                str(int(opts.check_seed)),
            ]
        cmd += [
            '--format',
            str(opts.check_format),
            '--topk',
            str(int(opts.check_topk)),
            '--candidate_k',
            str(int(opts.check_candidate_k)),
            '--top_flows',
            str(int(opts.check_top_flows)),
            '--explain_pairs',
            str(int(opts.check_explain_pairs)),
            '--mode',
            str(settings.mode),
            '--cost',
            str(settings.cost),
            '--alpha_embed',
            str(settings.alpha_embed),
            '--beta_vad',
            str(settings.beta_vad),
            '--vad_threshold',
            str(settings.vad_threshold),
            '--emotional_vocab',
            str(settings.emotional_vocab),
            '--vad_min_arousal_vad_only',
            str(settings.vad_min_arousal_vad_only),
            '--max_ngram',
            str(settings.max_ngram),
            '--weight',
            str(settings.weight),
            '--max_terms',
            str(settings.max_terms),
            '--drop_top_df',
            str(settings.drop_top_df),
        ]
        embed_model = getattr(
            setup,
            'token_ot_embed_model',
            None,
        )
        embed_backend = getattr(
            setup,
            'token_ot_embed_backend',
            None,
        )
        embed_pooling = getattr(
            setup,
            'token_ot_embed_pooling',
            None,
        )
        embed_batch_size = getattr(
            setup,
            'token_ot_embed_batch_size',
            None,
        )
        embed_max_length = getattr(
            setup,
            'token_ot_embed_max_length',
            None,
        )
        embed_prompt_mode = getattr(
            setup,
            'token_ot_embed_prompt_mode',
            None,
        )
        embed_prompt_text = getattr(
            setup,
            'token_ot_embed_prompt_text',
            None,
        )
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
        if not token_check_summary.exists():
            raise SystemExit('error: SystemExit')
        check = json.loads(token_check_summary.read_text(encoding='utf-8'))
        reasons = _guard_reasons(
            full_stats=full_stats,
            check=check,
            min_coverage=float(opts.token_guard_min_coverage),
            min_selected_ratio=float(opts.token_guard_min_selected_ratio),
            max_zero_ratio=float(opts.token_guard_max_zero_ratio),
            min_score=(
                float(opts.token_guard_min_score)
                if opts.token_guard_min_score is not None
                else None
            ),
            min_separation_effect=(
                float(opts.token_guard_min_separation_effect)
                if getattr(
                    opts,
                    'token_guard_min_separation_effect',
                    None,
                )
                is not None
                else None
            ),
            allow_warn=bool(opts.token_guard_allow_warn),
        )
        if reasons:
            print(
                f'iter {i} guard fail',
                flush=True,
            )
        else:
            print(
                f'iter {i} guard pass',
                flush=True,
            )
        attempt = {
            'iter': int(i),
            'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'check_params': dict(check_params),
            'settings': asdict(settings),
            'full_stats': full_stats,
            'check': check,
            'guard_reasons': reasons,
            'guard_pass': not reasons,
        }
        _append_jsonl(
            history_path,
            attempt,
        )
        _write_json(
            tuning_dir / f'iter_{i:03d}.json',
            attempt,
        )
        obj = objective(attempt)
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_attempt = attempt
        if not reasons:
            print(
                'success',
                flush=True,
            )
            _write_json(
                best_path,
                attempt,
            )
            best_cmd_parts = [
                str(sys.executable),
                'scripts/run_gold_to_icj_ot_pipeline.py',
                '--run_dir',
                str(run_dir),
                '--sentiment_config',
                str(opts.sentiment_config),
                '--emotion_config',
                str(cfg_path),
                '--icj_dir',
                str(icj_dir),
                '--token_check_format',
                str(opts.check_format),
                '--token_check_topk',
                str(int(opts.check_topk)),
                '--token_check_candidate_k',
                str(int(opts.check_candidate_k)),
                '--token_check_top_flows',
                str(int(opts.check_top_flows)),
                '--token_check_explain_pairs',
                str(int(opts.check_explain_pairs)),
                '--token_mode',
                str(settings.mode),
                '--token_weight',
                str(settings.weight),
                '--token_cost',
                str(settings.cost),
                '--token_alpha_embed',
                str(settings.alpha_embed),
                '--token_beta_vad',
                str(settings.beta_vad),
                '--token_vad_threshold',
                str(settings.vad_threshold),
                '--token_emotional_vocab',
                str(settings.emotional_vocab),
                '--token_vad_min_arousal_vad_only',
                str(settings.vad_min_arousal_vad_only),
                '--token_max_ngram',
                str(settings.max_ngram),
                '--token_max_terms',
                str(settings.max_terms),
                '--token_drop_top_df',
                str(settings.drop_top_df),
                '--token_guard_min_coverage',
                str(float(opts.token_guard_min_coverage)),
                '--token_guard_min_selected_ratio',
                str(float(opts.token_guard_min_selected_ratio)),
                '--token_guard_max_zero_ratio',
                str(float(opts.token_guard_max_zero_ratio)),
            ]
            if selected_jsonl:
                best_cmd_parts += [
                    '--token_check_selected_jsonl',
                    str(selected_jsonl),
                ]
            else:
                best_cmd_parts += [
                    '--token_check_n_per_category',
                    str(int(opts.check_n_per_category)),
                    '--token_check_random_n',
                    str(int(opts.check_random_n)),
                    '--token_check_seed',
                    str(int(opts.check_seed)),
                ]
            if opts.token_guard_min_score is not None:
                best_cmd_parts += [
                    '--token_guard_min_score',
                    str(float(opts.token_guard_min_score)),
                ]
            if opts.limit is not None:
                best_cmd_parts += [
                    '--limit',
                    str(int(opts.limit)),
                ]
            if bool(opts.token_guard_allow_warn):
                best_cmd_parts.append('--token_guard_allow_warn')
            best_cmd = ' '.join(best_cmd_parts)
            best_cmd_path.write_text(
                best_cmd + '\n',
                encoding='utf-8',
            )
            if opts.no_finalize_pipeline:
                print(json.dumps(
                        {
                            'status': 'pass',
                            'run_dir': str(run_dir),
                            'best_settings': asdict(settings),
                        },
                        indent=2,
                    ))
                return 0
            _purge_token_outputs(
                run_dir,
                purge_check=False,
            )
            print(
                f'finalize {run_dir}',
                flush=True,
            )
            pipeline_cmd = [
                sys.executable,
                str(_ROOT
                    / 'scripts'
                    / 'run_gold_to_icj_ot_pipeline.py'),
                '--run_dir',
                str(run_dir),
                '--sentiment_config',
                str(opts.sentiment_config),
                '--emotion_config',
                str(cfg_path),
                '--icj_dir',
                str(icj_dir),
                '--token_check_format',
                str(opts.check_format),
                '--token_check_topk',
                str(int(opts.check_topk)),
                '--token_check_candidate_k',
                str(int(opts.check_candidate_k)),
                '--token_check_top_flows',
                str(int(opts.check_top_flows)),
                '--token_check_explain_pairs',
                str(int(opts.check_explain_pairs)),
                '--token_mode',
                str(settings.mode),
                '--token_weight',
                str(settings.weight),
                '--token_cost',
                str(settings.cost),
                '--token_alpha_embed',
                str(settings.alpha_embed),
                '--token_beta_vad',
                str(settings.beta_vad),
                '--token_vad_threshold',
                str(settings.vad_threshold),
                '--token_emotional_vocab',
                str(settings.emotional_vocab),
                '--token_vad_min_arousal_vad_only',
                str(settings.vad_min_arousal_vad_only),
                '--token_max_ngram',
                str(settings.max_ngram),
                '--token_max_terms',
                str(settings.max_terms),
                '--token_drop_top_df',
                str(settings.drop_top_df),
                '--token_guard_min_coverage',
                str(float(opts.token_guard_min_coverage)),
                '--token_guard_min_selected_ratio',
                str(float(opts.token_guard_min_selected_ratio)),
                '--token_guard_max_zero_ratio',
                str(float(opts.token_guard_max_zero_ratio)),
            ]
            if selected_jsonl:
                pipeline_cmd += [
                    '--token_check_selected_jsonl',
                    str(selected_jsonl),
                ]
            else:
                pipeline_cmd += [
                    '--token_check_n_per_category',
                    str(int(opts.check_n_per_category)),
                    '--token_check_random_n',
                    str(int(opts.check_random_n)),
                    '--token_check_seed',
                    str(int(opts.check_seed)),
                ]
            if opts.token_guard_min_score is not None:
                pipeline_cmd += [
                    '--token_guard_min_score',
                    str(float(opts.token_guard_min_score)),
                ]
            if opts.limit is not None:
                pipeline_cmd += [
                    '--limit',
                    str(int(opts.limit)),
                ]
            if bool(opts.token_guard_allow_warn):
                pipeline_cmd += ['--token_guard_allow_warn']
            subprocess.run(
                pipeline_cmd,
                check=True,
            )
            print(json.dumps(
                    {
                        'status': 'pipeline_ok',
                        'run_dir': str(run_dir),
                        'best_settings': asdict(settings),
                    },
                    indent=2,
                ))
            return 0
        nxt = _propose_next(
            current=settings,
            full_stats=full_stats,
            check=check,
            args=opts,
        )
        if nxt is None:
            if bool(getattr(
                    opts,
                    'restart_on_stuck',
                    False,
                )):
                nxt_restart = _random_restart(
                    current=settings,
                    tried=tried,
                    args=opts,
                    rng=restart_rng,
                )
                if nxt_restart is not None:
                    print(
                        f'iter {i} restart',
                        flush=True,
                    )
                    settings = nxt_restart
                    i += 1
                    continue
            _write_json(
                tuning_dir / 'last_attempt.json',
                attempt,
            )
            best = best_attempt or attempt
            _write_json(
                best_path,
                best,
            )
            best_settings = (
                best.get('settings')
                if isinstance(
                    best.get('settings'),
                    dict,
                )
                else asdict(settings)
            )
            best_cmd_parts = [
                str(sys.executable),
                'scripts/run_gold_to_icj_ot_pipeline.py',
                '--run_dir',
                str(run_dir),
                '--sentiment_config',
                str(opts.sentiment_config),
                '--emotion_config',
                str(cfg_path),
                '--icj_dir',
                str(icj_dir),
                '--token_check_format',
                str(opts.check_format),
                '--token_check_topk',
                str(int(opts.check_topk)),
                '--token_check_candidate_k',
                str(int(opts.check_candidate_k)),
                '--token_check_top_flows',
                str(int(opts.check_top_flows)),
                '--token_check_explain_pairs',
                str(int(opts.check_explain_pairs)),
                '--skip_sentiment_train',
                '--skip_emotion_train',
                '--no_token_guard',
                '--token_mode',
                str(best_settings.get(
                        'mode',
                        settings.mode,
                    )),
                '--token_weight',
                str(best_settings.get(
                        'weight',
                        settings.weight,
                    )),
                '--token_cost',
                str(best_settings.get(
                        'cost',
                        settings.cost,
                    )),
                '--token_alpha_embed',
                str(best_settings.get(
                        'alpha_embed',
                        settings.alpha_embed,
                    )),
                '--token_beta_vad',
                str(best_settings.get(
                        'beta_vad',
                        settings.beta_vad,
                    )),
                '--token_vad_threshold',
                str(best_settings.get(
                        'vad_threshold',
                        settings.vad_threshold,
                    )),
                '--token_emotional_vocab',
                str(best_settings.get(
                        'emotional_vocab',
                        settings.emotional_vocab,
                    )),
                '--token_vad_min_arousal_vad_only',
                str(best_settings.get(
                        'vad_min_arousal_vad_only',
                        settings.vad_min_arousal_vad_only,
                    )),
                '--token_max_ngram',
                str(best_settings.get(
                        'max_ngram',
                        settings.max_ngram,
                    )),
                '--token_max_terms',
                str(best_settings.get(
                        'max_terms',
                        settings.max_terms,
                    )),
                '--token_drop_top_df',
                str(best_settings.get(
                        'drop_top_df',
                        settings.drop_top_df,
                    )),
            ]
            if selected_jsonl:
                best_cmd_parts += [
                    '--token_check_selected_jsonl',
                    str(selected_jsonl),
                ]
            else:
                best_cmd_parts += [
                    '--token_check_n_per_category',
                    str(int(opts.check_n_per_category)),
                    '--token_check_random_n',
                    str(int(opts.check_random_n)),
                    '--token_check_seed',
                    str(int(opts.check_seed)),
                ]
            best_cmd_path.write_text(
                ' '.join(best_cmd_parts) + '\n',
                encoding='utf-8',
            )
            payload = {
                'status': 'stuck',
                'run_dir': str(run_dir),
                'best_settings': best_settings,
                'best_guard_reasons': best.get('guard_reasons'),
                'best_full_stats': best.get('full_stats'),
                'best_check': best.get('check'),
                'best_pipeline_command': str(best_cmd_path),
                'history': str(history_path),
            }
            print(json.dumps(
                    payload,
                    indent=2,
                ))
            if bool(opts.strict):
                raise SystemExit('error: SystemExit')
            return 0
        print(
            f'iter {i} next',
            flush=True,
        )
        settings = nxt
        i += 1
    best = best_attempt
    if best is not None:
        _write_json(
            best_path,
            best,
        )
        best_settings = (
            best.get('settings')
            if isinstance(
                best.get('settings'),
                dict,
            )
            else {}
        )
        best_cmd_parts = [
            str(sys.executable),
            'scripts/run_gold_to_icj_ot_pipeline.py',
            '--run_dir',
            str(run_dir),
            '--sentiment_config',
            str(opts.sentiment_config),
            '--emotion_config',
            str(cfg_path),
            '--icj_dir',
            str(icj_dir),
            '--token_check_n_per_category',
            str(int(opts.check_n_per_category)),
            '--token_check_random_n',
            str(int(opts.check_random_n)),
            '--token_check_seed',
            str(int(opts.check_seed)),
            '--token_check_format',
            str(opts.check_format),
            '--token_check_topk',
            str(int(opts.check_topk)),
            '--token_check_candidate_k',
            str(int(opts.check_candidate_k)),
            '--token_check_top_flows',
            str(int(opts.check_top_flows)),
            '--token_check_explain_pairs',
            str(int(opts.check_explain_pairs)),
            '--skip_sentiment_train',
            '--skip_emotion_train',
            '--no_token_guard',
            '--token_mode',
            str(best_settings.get(
                    'mode',
                    default_settings.mode,
                )),
            '--token_weight',
            str(best_settings.get(
                    'weight',
                    default_settings.weight,
                )),
            '--token_cost',
            str(best_settings.get(
                    'cost',
                    default_settings.cost,
                )),
            '--token_alpha_embed',
            str(best_settings.get(
                    'alpha_embed',
                    default_settings.alpha_embed,
                )),
            '--token_beta_vad',
            str(best_settings.get(
                    'beta_vad',
                    default_settings.beta_vad,
                )),
            '--token_vad_threshold',
            str(best_settings.get(
                    'vad_threshold',
                    default_settings.vad_threshold,
                )),
            '--token_emotional_vocab',
            str(best_settings.get(
                    'emotional_vocab',
                    default_settings.emotional_vocab,
                )),
            '--token_vad_min_arousal_vad_only',
            str(best_settings.get(
                    'vad_min_arousal_vad_only',
                    default_settings.vad_min_arousal_vad_only,
                )),
            '--token_max_ngram',
            str(best_settings.get(
                    'max_ngram',
                    default_settings.max_ngram,
                )),
            '--token_max_terms',
            str(best_settings.get(
                    'max_terms',
                    default_settings.max_terms,
                )),
            '--token_drop_top_df',
            str(best_settings.get(
                    'drop_top_df',
                    default_settings.drop_top_df,
                )),
        ]
        best_cmd_path.write_text(
            ' '.join(best_cmd_parts) + '\n',
            encoding='utf-8',
        )
    payload = {
        'status': 'max_iters',
        'run_dir': str(run_dir),
        'best_settings': (
            best.get('settings')
            if isinstance(
                best,
                dict,
            )
            else None
        ),
        'best_guard_reasons': (
            best.get('guard_reasons')
            if isinstance(
                best,
                dict,
            )
            else None
        ),
        'best_full_stats': (
            best.get('full_stats')
            if isinstance(
                best,
                dict,
            )
            else None
        ),
        'best_check': (
            best.get('check')
            if isinstance(
                best,
                dict,
            )
            else None
        ),
        'history': str(history_path),
    }
    print(json.dumps(
            payload,
            indent=2,
        ))
    if bool(opts.strict):
        raise SystemExit('error: SystemExit')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
