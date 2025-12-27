from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)
import torch
from .lexicon import emotion_prototypes, load_lexicon
from .losses import (
    cost_matrix,
    ot_loss,
    sinkhorn_cost_parts,
)
from .utils import get_device, load_config

_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(
    path: str | Path,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(
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


def _infer_emotions(
    rows: List[Dict[str, Any]],
    fallback: List[str],
) -> List[str]:
    if not rows:
        return list(fallback)
    first = rows[0].get('emotions')
    if (
        isinstance(
            first,
            list,
        )
        and all((isinstance(
                x,
                str,
            ) for x in first))
        and first
    ):
        return list(first)
    return list(fallback)


def _align_emotion_vector(
    row: Dict[str, Any],
    vec: torch.Tensor,
    emotions: List[str],
    *,
    measure: str,
    row_index: Optional[int] = None,
) -> torch.Tensor:
    row_feelings = row.get('emotions')
    if (
        not isinstance(
            row_feelings,
            list,
        )
        or not row_feelings
    ):
        return vec
    if row_feelings == emotions:
        return vec
    prefix = (
        f'row {row_index} ' if row_index is not None else ''
    )
    if len(set(row_feelings)) != len(row_feelings):
        raise ValueError('error: ValueError')
    if len(row_feelings) != len(emotions):
        raise ValueError('error: ValueError')
    if set(row_feelings) != set(emotions):
        missing = sorted(set(emotions) - set(row_feelings))
        extra = sorted(set(row_feelings) - set(emotions))
        raise ValueError('error: ValueError')
    idx_map = {e: i for i, e in enumerate(row_feelings)}
    order = [idx_map[e] for e in emotions]
    order_t = torch.tensor(
        order,
        device=vec.device,
    )
    return vec.index_select(
        0,
        order_t,
    )


def _as_float(x: Any) -> Optional[float]:
    if isinstance(
        x,
        (int, float),
    ):
        return float(x)
    return None


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
        return int(x)
    if isinstance(
        x,
        str,
    ):
        try:
            return int(x.strip())
        except ValueError:
            try:
                return int(float(x.strip()))
            except ValueError:
                return None
    return None


def _sum_list(x: Any) -> Optional[float]:
    if not isinstance(
        x,
        list,
    ) or not x:
        return None
    try:
        return float(sum((float(v) for v in x)))
    except Exception:
        return None


def _pred_per_1k_words(row: Dict[str, Any]) -> Optional[float]:
    guess_per_1k = _as_float(row.get('pred_counts_calibrated_per_1k_words'))
    if guess_per_1k is None:
        guess_per_1k = _as_float(row.get('pred_per_1k_words'))
    if guess_per_1k is None:
        guess_counts = _sum_list(row.get('pred_counts_calibrated'))
        if guess_counts is None:
            guess_counts = _sum_list(row.get('pred_counts'))
        n_words = int(row.get('n_words') or 0)
        if guess_counts is not None and n_words > 0:
            guess_per_1k = float(guess_counts / float(n_words) * 1000.0)
    return guess_per_1k


def _lex_per_1k_words(row: Dict[str, Any]) -> Optional[float]:
    lex_per_1k = _as_float(row.get('lex_per_1k_words'))
    if lex_per_1k is None:
        lex_counts = _sum_list(row.get('lex_counts'))
        n_words = int(row.get('n_words') or 0)
        if lex_counts is not None and n_words > 0:
            lex_per_1k = float(lex_counts / float(n_words) * 1000.0)
    return lex_per_1k


def _infer_signal_per_1k_words(row: Dict[str, Any]) -> Optional[float]:
    signal = _as_float(row.get('emotion_signal_per_1k_words'))
    if signal is not None:
        return signal
    guess_per_1k = _pred_per_1k_words(row)
    lex_per_1k = _lex_per_1k_words(row)
    if guess_per_1k is None and lex_per_1k is None:
        return None
    if guess_per_1k is None:
        return float(lex_per_1k)
    if lex_per_1k is None:
        return float(guess_per_1k)
    return float(max(
            guess_per_1k,
            lex_per_1k,
        ))


def _mass_scale_candidates(
    *,
    measure: str,
    use_token_scale: bool,
) -> List[str]:
    m = (measure or '').lower().strip()
    if m == 'lex_prior':
        return [
            (
                'lex_counts_per_1k_tokens'
                if use_token_scale
                else 'lex_counts_per_1k_words'
            ),
            'lex_counts',
        ]
    if use_token_scale:
        return [
            'pred_counts_calibrated_per_1k_tokens',
            'pred_counts_per_1k_tokens',
            'pred_mixscaled_per_1k_tokens',
            'pred_counts_calibrated',
            'pred_counts',
            'lex_counts',
        ]
    return [
        'pred_counts_calibrated_per_1k_words',
        'pred_counts_per_1k_words',
        'pred_mixscaled_per_1k_words',
        'pred_counts_calibrated',
        'pred_counts',
        'lex_counts',
    ]


def _find_mass_scale_key(
    rows: List[Dict[str, Any]],
    candidates: List[str],
) -> Optional[str]:
    for key in candidates:
        ok = True
        for row in rows:
            try:
                _ = extract_measure(
                    row,
                    key,
                )
            except Exception:
                ok = False
                break
        if ok:
            return key
    return None


def _filter_low_signal_rows(
    rows: List[Dict[str, Any]],
    cfg,
) -> Tuple[
    List[Dict[str, Any]], List[int], List[Dict[str, Any]]
]:
    drop_low = bool(getattr(
            cfg,
            'compare_drop_low_signal',
            False,
        ))
    if not drop_low:
        return (rows, list(range(len(rows))), [])
    min_signal = float(getattr(
            cfg,
            'semantic_min_signal_per_1k_words',
            0.0,
        )
        or 0.0)
    min_words = int(getattr(
            cfg,
            'compare_min_words',
            0,
        ) or 0)
    kept: List[Dict[str, Any]] = []
    orig_indices: List[int] = []
    dropped: List[Dict[str, Any]] = []
    missing_signal = 0
    for idx, row in enumerate(rows):
        if min_words > 0:
            n_words = _as_int(row.get('n_words'))
            if (
                isinstance(
                    n_words,
                    int,
                )
                and n_words < min_words
            ):
                dropped.append({
                        'index': idx,
                        'reason': 'min_words',
                        'n_words': n_words,
                        'threshold': min_words,
                    })
                continue
        low_flag = row.get('low_emotion_signal')
        signal = _as_float(row.get('emotion_signal_per_1k_words'))
        if signal is None:
            signal = _infer_signal_per_1k_words(row)
        if low_flag is True:
            dropped.append({
                    'index': idx,
                    'reason': 'low_emotion_signal',
                    'signal': signal,
                })
            continue
        if signal is None:
            missing_signal += 1
        if (
            low_flag is None
            and isinstance(
                signal,
                (int, float),
            )
            and (min_signal > 0.0)
            and (float(signal) < min_signal)
        ):
            dropped.append({
                    'index': idx,
                    'reason': 'signal_below_threshold',
                    'signal': float(signal),
                    'threshold': min_signal,
                })
            continue
        kept.append(row)
        orig_indices.append(idx)
    if missing_signal and min_signal > 0.0:
        warnings.warn(
            'warn: signal',
            RuntimeWarning,
            stacklevel=2,
        )
    return (kept, orig_indices, dropped)


def _as_float_list(
    x: Any,
    *,
    field: str,
) -> List[float]:
    if not isinstance(
        x,
        list,
    ) or not x:
        raise ValueError('error: ValueError')
    try:
        return [float(v) for v in x]
    except Exception as e:
        raise ValueError('error: ValueError') from e


def _counts_per_1k(
    row: Dict[str, Any],
    *,
    counts_field: str,
    denom_field: str,
) -> torch.Tensor:
    counts = torch.tensor(
        _as_float_list(
            row.get(counts_field),
            field=counts_field,
        ),
        dtype=torch.float,
    )
    denom_val = int(row.get(denom_field) or 0)
    denom = max(
        denom_val,
        1,
    )
    return counts / float(denom) * 1000.0


def extract_measure(
    row: Dict[str, Any],
    measure: str,
) -> torch.Tensor:
    m = (measure or '').lower().strip()
    if m in {
        'pred_softmax',
        'pred_sigmoid',
        'pred_dist',
        'pred_dist_calibrated',
        'pred_counts',
        'pred_counts_calibrated',
        'lex_prior',
        'lex_counts',
        'pred_mixscaled_per_1k_words',
        'pred_mixscaled_per_1k_tokens',
    }:
        return torch.tensor(
            _as_float_list(
                row.get(m),
                field=m,
            ),
            dtype=torch.float,
        )
    if m == 'pred_counts_per_1k_words':
        return _counts_per_1k(
            row,
            counts_field='pred_counts',
            denom_field='n_words',
        )
    if m == 'pred_counts_per_1k_tokens':
        return _counts_per_1k(
            row,
            counts_field='pred_counts',
            denom_field='n_tokens',
        )
    if m == 'pred_counts_calibrated_per_1k_words':
        return _counts_per_1k(
            row,
            counts_field='pred_counts_calibrated',
            denom_field='n_words',
        )
    if m == 'pred_counts_calibrated_per_1k_tokens':
        return _counts_per_1k(
            row,
            counts_field='pred_counts_calibrated',
            denom_field='n_tokens',
        )
    if m == 'lex_counts_per_1k_words':
        return _counts_per_1k(
            row,
            counts_field='lex_counts',
            denom_field='n_words',
        )
    if m == 'lex_counts_per_1k_tokens':
        counts = torch.tensor(
            _as_float_list(
                row.get('lex_counts'),
                field='lex_counts',
            ),
            dtype=torch.float,
        )
        n_tokens = int(row.get('n_tokens') or 0)
        denom = max(
            n_tokens,
            1,
        )
        return counts / float(denom) * 1000.0
    raise ValueError('error: ValueError')


def build_cost(
    emotions: List[str],
    *,
    device: torch.device,
    cost: str,
    lexicon_path: Optional[str] = None,
    vad_lexicon_path: Optional[str] = None,
    lexicon_vad_scale: Optional[str] = None,
    word_vad_scale: Optional[str] = None,
    lexicon_stopwords_file: Optional[str] = None,
    lexicon_extra_path: Optional[str] = None,
    lexicon_intensity_path: Optional[str] = None,
    lexicon_intensity_min: float = 0.0,
    lexicon_min_vad_salience: float = 0.0,
    lexicon_min_vad_arousal: float = 0.0,
    lexicon_require_word_vad: bool = False,
    lexicon_allow_seed_only: bool = False,
    vad_allow_missing: bool = False,
) -> torch.Tensor:
    cost = (cost or 'uniform').lower()
    if cost == 'uniform':
        return cost_matrix(
            len(emotions),
            device,
        )
    if cost == 'vad':
        lex = load_lexicon(
            lexicon_path,
            vad_lexicon_path,
            lexicon_vad_scale=lexicon_vad_scale,
            word_vad_scale=word_vad_scale,
            stopwords_path=lexicon_stopwords_file,
            extra_path=lexicon_extra_path,
            intensity_path=lexicon_intensity_path,
            intensity_min=lexicon_intensity_min,
            min_vad_salience=lexicon_min_vad_salience,
            min_vad_arousal=lexicon_min_vad_arousal,
            require_word_vad=lexicon_require_word_vad,
            allow_seed_only=lexicon_allow_seed_only,
            allow_missing_vad=vad_allow_missing,
        )
        protos = emotion_prototypes(
            lex,
            emotions,
        ).to(device)
        return cost_matrix(
            len(emotions),
            device,
            emotion_vad=protos,
        )
    raise ValueError('error: ValueError')


def eot_distance_matrix(
    vectors: torch.Tensor,
    C: torch.Tensor,
    *,
    mode: str,
    epsilon: float,
    iters: int,
    reg_m: float,
) -> torch.Tensor:
    if vectors.dim() != 2:
        raise ValueError('error: ValueError')
    if C.shape != (vectors.size(1), vectors.size(1)):
        raise ValueError('error: ValueError')
    mode_str = (mode or 'sinkhorn_divergence').lower()
    n = int(vectors.size(0))
    out = torch.empty(
        (n, n),
        device=vectors.device,
        dtype=torch.float,
    )
    for i in range(n):
        prior = vectors[i].expand_as(vectors)
        out[i] = ot_loss(
            vectors,
            prior,
            C,
            float(epsilon),
            int(iters),
            mode_str,
            float(reg_m),
        )
    if mode_str.endswith('_divergence'):
        out = 0.5 * (out + out.t())
        out.fill_diagonal_(0.0)
    return out


def eot_neighbours(
    vectors: torch.Tensor,
    C: torch.Tensor,
    *,
    mode: str,
    epsilon: float,
    iters: int,
    reg_m: float,
    topk: int,
) -> List[List[Tuple[int, float]]]:
    if topk <= 0:
        raise ValueError('error: ValueError')
    if vectors.dim() != 2:
        raise ValueError('error: ValueError')
    n = int(vectors.size(0))
    if n <= 1:
        return [[] for _ in range(n)]
    mode_str = (mode or 'sinkhorn_divergence').lower()
    k = min(
        int(topk),
        n - 1,
    )
    out: List[List[Tuple[int, float]]] = []
    for i in range(n):
        prior = vectors[i].expand_as(vectors)
        d = ot_loss(
            vectors,
            prior,
            C,
            float(epsilon),
            int(iters),
            mode_str,
            float(reg_m),
        ).to(dtype=torch.float)
        if mode_str.endswith('_divergence'):
            d_rev = ot_loss(
                prior,
                vectors,
                C,
                float(epsilon),
                int(iters),
                mode_str,
                float(reg_m),
            ).to(dtype=torch.float)
            d = 0.5 * (d + d_rev)
        d = d.clone()
        d[i] = float('inf')
        (
            vals,
            idx,
        ) = torch.topk(
            d,
            k=k,
            largest=False,
        )
        out.append([
                (int(j), float(v))
                for j, v in zip(
                    idx.tolist(),
                    vals.tolist(),
                )
            ])
    return out


def compare_scores(
    *,
    input_jsonl: str,
    output_path: str,
    cfg_path: Optional[str] = None,
    mode: Optional[str] = None,
    cost: Optional[str] = None,
    measure: Optional[str] = None,
    style_mode: Optional[str] = None,
    style_measure: Optional[str] = None,
    include_style: bool = True,
    include_explain: bool = True,
    epsilon: Optional[float] = None,
    iters: Optional[int] = None,
    reg_m: Optional[float] = None,
    fmt: str = 'neighbours',
    topk: int = 10,
    top_flows: int = 8,
    limit: Optional[int] = None,
    vis: bool = False,
    vis_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup = load_config(cfg_path)
    rows_all = _read_jsonl(
        input_jsonl,
        limit=limit,
    )
    emotions = _infer_emotions(
        rows_all,
        setup.emotions,
    )
    if not rows_all:
        raise ValueError('error: ValueError')
    feeling_labels = getattr(
        setup,
        'emotion_display_names',
        None,
    )
    if not isinstance(
        feeling_labels,
        dict,
    ):
        feeling_labels = None
    (
        rows,
        orig_indices,
        filtered_out,
    ) = (
        _filter_low_signal_rows(
            rows_all,
            setup,
        )
    )
    if not rows:
        raise ValueError('error: ValueError')
    mode = (
        mode or setup.ot_mode or 'sinkhorn_divergence'
    ).lower()
    cost = (cost or setup.ot_cost or 'uniform').lower()
    epsilon = float(epsilon
        if epsilon is not None
        else setup.sinkhorn_epsilon)
    iters = int(iters if iters is not None else setup.sinkhorn_iters)
    reg_m = float(reg_m if reg_m is not None else setup.ot_reg_m)
    count_scale = (
        str(getattr(
                setup,
                'count_pred_scale',
                'counts',
            )
            or 'counts')
        .lower()
        .strip()
    )
    use_token_scale = count_scale in {
        'density',
        'per_token',
        'per_tok',
    }
    if measure is None:
        row0 = rows[0] if rows else {}

        def _pick_scale(
            words_key: str,
            tokens_key: str,
        ) -> str:
            if use_token_scale and tokens_key in row0:
                return tokens_key
            return words_key

        if mode.startswith('unbalanced'):
            if 'pred_counts_calibrated' in row0:
                measure = _pick_scale(
                    'pred_counts_calibrated_per_1k_words',
                    'pred_counts_calibrated_per_1k_tokens',
                )
            elif 'pred_counts' in row0:
                measure = _pick_scale(
                    'pred_counts_per_1k_words',
                    'pred_counts_per_1k_tokens',
                )
            elif (
                'pred_mixscaled_per_1k_words' in row0
                or 'pred_mixscaled_per_1k_tokens' in row0
            ):
                measure = _pick_scale(
                    'pred_mixscaled_per_1k_words',
                    'pred_mixscaled_per_1k_tokens',
                )
            else:
                measure = _pick_scale(
                    'pred_counts_per_1k_words',
                    'pred_counts_per_1k_tokens',
                )
        elif rows and 'pred_dist_calibrated' in rows[0]:
            measure = 'pred_dist_calibrated'
        elif rows and 'pred_dist' in rows[0]:
            measure = 'pred_dist'
        else:
            measure = 'pred_sigmoid'
    measure = (measure or '').lower()
    style_mode = (
        style_mode or 'sinkhorn_divergence'
    ).lower()
    if style_measure is None:
        if rows and 'pred_dist_calibrated' in rows[0]:
            style_measure = 'pred_dist_calibrated'
        elif rows and 'pred_dist' in rows[0]:
            style_measure = 'pred_dist'
        else:
            style_measure = 'pred_sigmoid'
    style_measure = (
        style_measure or 'pred_sigmoid'
    ).lower()
    simplex_like = {
        'pred_softmax',
        'pred_sigmoid',
        'pred_dist',
        'pred_dist_calibrated',
        'lex_prior',
    }
    mass_like = {
        'pred_counts',
        'pred_counts_calibrated',
        'lex_counts',
        'pred_counts_per_1k_words',
        'pred_counts_per_1k_tokens',
        'pred_counts_calibrated_per_1k_words',
        'pred_counts_calibrated_per_1k_tokens',
        'lex_counts_per_1k_words',
        'lex_counts_per_1k_tokens',
        'pred_mixscaled_per_1k_words',
        'pred_mixscaled_per_1k_tokens',
    }
    scale_simplex = False
    mass_scale_key: Optional[str] = None
    if (
        mode.startswith('unbalanced')
        and measure in simplex_like
        and bool(getattr(
                setup,
                'compare_unbalanced_mass_scale_simplex',
                True,
            ))
    ):
        candidates = _mass_scale_candidates(
            measure=measure,
            use_token_scale=use_token_scale,
        )
        mass_scale_key = _find_mass_scale_key(
            rows,
            candidates,
        )
        if mass_scale_key is not None:
            scale_simplex = True
    if (
        mode.startswith('unbalanced')
        and measure in simplex_like
        and (not scale_simplex)
    ):
        warnings.warn(
            'warn: saturation',
            RuntimeWarning,
            stacklevel=2,
        )
    if (
        not mode.startswith('unbalanced')
        and measure in mass_like
    ):
        warnings.warn(
            'warn: saturation',
            RuntimeWarning,
            stacklevel=2,
        )
    device = get_device(setup.device)
    vectors: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []
    sat_words: List[Optional[float]] = []
    sat_tokens: List[Optional[float]] = []
    signal_vals: List[Optional[float]] = []
    low_flags: List[Optional[bool]] = []
    mass_scales: List[Optional[float]] = []
    kept_rows: List[Dict[str, Any]] = []
    kept_indices: List[int] = []
    min_signal = float(getattr(
            setup,
            'semantic_min_signal_per_1k_words',
            0.0,
        )
        or 0.0)
    for pos, row in enumerate(rows):
        vec = extract_measure(
            row,
            measure,
        )
        vec = _align_emotion_vector(
            row,
            vec,
            emotions,
            measure=measure,
            row_index=orig_indices[pos],
        )
        if vec.numel() != len(emotions):
            raise ValueError('error: ValueError')
        pw = row.get(
            'pred_counts_calibrated_per_1k_words',
            row.get('pred_per_1k_words'),
        )
        pt = row.get(
            'pred_counts_calibrated_per_1k_tokens',
            row.get('pred_per_1k_tokens'),
        )
        signal = _as_float(row.get('emotion_signal_per_1k_words'))
        if signal is None:
            signal = _infer_signal_per_1k_words(row)
        low_flag = row.get('low_emotion_signal')
        if (
            low_flag is None
            and signal is not None
            and (min_signal > 0.0)
        ):
            low_flag = bool(float(signal) < min_signal)
        if measure in simplex_like:
            invalid_flag = False
            if (
                measure == 'pred_dist_calibrated'
                and row.get('pred_dist_calibrated_valid')
                is False
            ):
                invalid_flag = True
            if float(vec.sum().item()) <= 1e-08:
                invalid_flag = True
            if invalid_flag:
                filtered_out.append({
                        'index': orig_indices[pos],
                        'reason': 'zero_mass_simplex',
                        'signal': (
                            float(signal)
                            if isinstance(
                                signal,
                                (int, float),
                            )
                            else None
                        ),
                        'low_emotion_signal': (
                            bool(low_flag)
                            if isinstance(
                                low_flag,
                                bool,
                            )
                            else None
                        ),
                    })
                continue
        if scale_simplex and mass_scale_key is not None:
            try:
                mass_vec = extract_measure(
                    row,
                    mass_scale_key,
                )
                mass_scales.append(float(mass_vec.sum().item()))
            except Exception:
                scale_simplex = False
                mass_scale_key = None
                mass_scales = []
        kept_rows.append(row)
        kept_indices.append(orig_indices[pos])
        vectors.append(vec)
        meta = (
            row.get('meta')
            if isinstance(
                row.get('meta'),
                dict,
            )
            else {}
        )
        metas.append(meta)
        sat_words.append(float(pw)
            if isinstance(
                pw,
                (int, float),
            )
            else None)
        sat_tokens.append(float(pt)
            if isinstance(
                pt,
                (int, float),
            )
            else None)
        signal_vals.append(float(signal)
            if isinstance(
                signal,
                (int, float),
            )
            else None)
        low_flags.append(bool(low_flag)
            if isinstance(
                low_flag,
                bool,
            )
            else None)
    rows = kept_rows
    orig_indices = kept_indices
    if not rows:
        raise ValueError('error: ValueError')
    if scale_simplex and mass_scale_key is not None:
        for idx, vec in enumerate(vectors):
            denom = vec.sum().clamp_min(1e-08)
            vec = vec / denom
            vec = vec * float(mass_scales[idx])
            vectors[idx] = vec
    X = torch.stack(
        vectors,
        dim=0,
    ).to(device)
    masses = (
        X.sum(dim=1).to(dtype=torch.float).detach().cpu()
    )
    sat_words_tensor: Optional[torch.Tensor] = None
    if all((v is not None for v in sat_words)):
        sat_words_tensor = torch.tensor(
            [float(v) for v in sat_words],
            dtype=torch.float,
        )
    sat_tokens_tensor: Optional[torch.Tensor] = None
    if all((v is not None for v in sat_tokens)):
        sat_tokens_tensor = torch.tensor(
            [float(v) for v in sat_tokens],
            dtype=torch.float,
        )
    X_style: Optional[torch.Tensor] = None
    X_style_cpu: Optional[torch.Tensor] = None
    if include_style:
        try:
            style_vecs: List[torch.Tensor] = []
            for pos, row in enumerate(rows):
                style_vec = extract_measure(
                    row,
                    style_measure,
                )
                style_vec = _align_emotion_vector(
                    row,
                    style_vec,
                    emotions,
                    measure=style_measure,
                    row_index=orig_indices[pos],
                )
                style_vecs.append(style_vec)
        except ValueError as exc:
            if 'emotion order mismatch' in str(exc):
                raise
            include_style = False
        except Exception:
            include_style = False
        else:
            X_style = torch.stack(
                style_vecs,
                dim=0,
            ).to(device)
            X_style_cpu = X_style.detach().cpu()
    C = build_cost(
        emotions,
        device=device,
        cost=cost,
        lexicon_path=getattr(
            setup,
            'lexicon_path',
            None,
        ),
        vad_lexicon_path=getattr(
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
        lexicon_stopwords_file=getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        ),
        lexicon_extra_path=getattr(
            setup,
            'lexicon_extra_path',
            None,
        ),
        lexicon_intensity_path=getattr(
            setup,
            'lexicon_intensity_path',
            None,
        ),
        lexicon_intensity_min=getattr(
            setup,
            'lexicon_intensity_min',
            0.0,
        ),
        lexicon_min_vad_salience=getattr(
            setup,
            'lexicon_min_vad_salience',
            0.0,
        ),
        lexicon_min_vad_arousal=getattr(
            setup,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        lexicon_require_word_vad=bool(getattr(
                setup,
                'lexicon_require_word_vad',
                False,
            )),
        lexicon_allow_seed_only=bool(getattr(
                setup,
                'lexicon_allow_seed_only',
                False,
            )),
        vad_allow_missing=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    ).to(
        device=device,
        dtype=torch.float,
    )

    def _base_mode(m: str) -> str:
        m = (m or '').lower()
        return (
            m[: -len('_divergence')]
            if m.endswith('_divergence')
            else m
        )

    def _pair_explain(
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        mode: str,
        measure: str,
        distance: float,
    ) -> Dict[str, Any]:
        mode_str = (mode or 'sinkhorn_divergence').lower()
        base = _base_mode(mode_str)
        is_div = mode_str.endswith('_divergence')
        if base not in {'sinkhorn', 'unbalanced'}:
            return {
                'mode': mode_str,
                'measure': measure,
                'distance': float(distance),
                'error': 'wrong explain mode!',
            }
        unbalanced = base == 'unbalanced'
        parts_ab = sinkhorn_cost_parts(
            a,
            b,
            C,
            epsilon=epsilon,
            iters=iters,
            unbalanced=unbalanced,
            reg_m=reg_m,
            return_plan=True,
        )
        plan = parts_ab['plan']
        top = _top_transport_contrib(
            plan,
            C,
            emotions,
            topk=int(top_flows),
            include_diagonal=False,
        )
        out: Dict[str, Any] = {
            'mode': mode_str,
            'measure': measure,
            'distance': float(distance),
            'cost_ab': float(parts_ab['total'].detach().cpu().item()),
            'cost_aa': None,
            'cost_bb': None,
            'divergence_calc': None,
            'distance_minus_calc': None,
            'transport_ab': float(parts_ab['transport'].detach().cpu().item()),
            'transport': float(parts_ab['transport'].detach().cpu().item()),
            'kl_a': float(parts_ab['kl_a'].detach().cpu().item()),
            'kl_b': float(parts_ab['kl_b'].detach().cpu().item()),
            'mass_a': float(parts_ab['mass_a'].detach().cpu().item()),
            'mass_b': float(parts_ab['mass_b'].detach().cpu().item()),
            'mass_plan': float(parts_ab['mass_plan'].detach().cpu().item()),
            'top_transport_cost_contrib': top,
        }
        if is_div:
            parts_aa = sinkhorn_cost_parts(
                a,
                a,
                C,
                epsilon=epsilon,
                iters=iters,
                unbalanced=unbalanced,
                reg_m=reg_m,
                return_plan=False,
            )
            parts_bb = sinkhorn_cost_parts(
                b,
                b,
                C,
                epsilon=epsilon,
                iters=iters,
                unbalanced=unbalanced,
                reg_m=reg_m,
                return_plan=False,
            )
            div_calc = float((
                    parts_ab['total']
                    - 0.5 * parts_aa['total']
                    - 0.5 * parts_bb['total']
                )
                .detach()
                .cpu()
                .item())
            out['cost_aa'] = float(parts_aa['total'].detach().cpu().item())
            out['cost_bb'] = float(parts_bb['total'].detach().cpu().item())
            out['divergence_calc'] = div_calc
            out['distance_minus_calc'] = float(distance - div_calc)
        else:
            cost_calc = float(parts_ab['total'].detach().cpu().item())
            out['distance_minus_calc'] = float(distance - cost_calc)
        return out

    fmt = (fmt or 'neighbours').lower().strip()
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    vis_enabled = bool(vis)
    vis_out_p: Optional[Path] = None
    if vis_enabled:
        vis_out_p = (
            Path(vis_path)
            if vis_path
            else out_p.with_suffix('.html')
        )
    if fmt == 'matrix':
        dist = eot_distance_matrix(
            X,
            C,
            mode=mode,
            epsilon=epsilon,
            iters=iters,
            reg_m=reg_m,
        ).cpu()
        mass_diff = (
            masses[:, None] - masses[None, :]
        ).abs()
        sat_diff_words = (
            (
                sat_words_tensor[:, None]
                - sat_words_tensor[None, :]
            ).abs()
            if sat_words_tensor is not None
            else None
        )
        sat_diff_tokens = (
            (
                sat_tokens_tensor[:, None]
                - sat_tokens_tensor[None, :]
            ).abs()
            if sat_tokens_tensor is not None
            else None
        )
        style_dist = None
        if include_style and X_style is not None:
            style_dist = eot_distance_matrix(
                X_style,
                C,
                mode=style_mode,
                epsilon=epsilon,
                iters=iters,
                reg_m=reg_m,
            ).cpu()
        payload = {
            'mode': mode,
            'cost': cost,
            'measure': measure,
            'epsilon': epsilon,
            'iters': iters,
            'reg_m': reg_m,
            'emotions': emotions,
            'docs': [
                {
                    'index': i,
                    'orig_index': orig_indices[i],
                    'meta': metas[i],
                    'mass': float(masses[i].item()),
                    'pred_per_1k_words': sat_words[i],
                    'pred_per_1k_tokens': sat_tokens[i],
                    'emotion_signal_per_1k_words': signal_vals[
                        i
                    ],
                    'low_emotion_signal': low_flags[i],
                }
                for i in range(len(metas))
            ],
            'distance': dist.tolist(),
            'dist': dist.tolist(),
            'mass_diff': mass_diff.tolist(),
        }
        if scale_simplex and mass_scale_key is not None:
            payload['mass_scale_simplex'] = True
            payload['mass_scale_key'] = mass_scale_key
        if filtered_out:
            payload['filtered_out'] = filtered_out
        if sat_diff_words is not None:
            payload['saturation_diff_per_1k_words'] = (
                sat_diff_words.tolist()
            )
        if sat_diff_tokens is not None:
            payload['saturation_diff_per_1k_tokens'] = (
                sat_diff_tokens.tolist()
            )
        if include_style and style_dist is not None:
            payload['style'] = {
                'mode': style_mode,
                'measure': style_measure,
                'epsilon': epsilon,
                'iters': iters,
                'distance': style_dist.tolist(),
                'dist': style_dist.tolist(),
            }
        need_neighbours = include_explain or vis_enabled
        neighbours_payload: Optional[
            List[Dict[str, Any]]
        ] = None
        if need_neighbours:
            n = int(dist.size(0))
            k = min(
                int(topk),
                max(
                    0,
                    n - 1,
                ),
            )
            neighbours_payload = []
            for i in range(n):
                if k <= 0:
                    neighbours_payload.append({
                            'index': i,
                            'orig_index': orig_indices[i],
                            'neighbours': [],
                        })
                    continue
                drow = dist[i].clone()
                drow[i] = float('inf')
                (
                    vals,
                    idx,
                ) = torch.topk(
                    drow,
                    k=k,
                    largest=False,
                )
                pairs = list(zip(
                        idx.tolist(),
                        vals.tolist(),
                    ))
                neigh_out: List[Dict[str, Any]] = []
                for j, dval in pairs:
                    j = int(j)
                    dval_f = float(dval)
                    top_feeling_deltas = None
                    if X_style_cpu is not None:
                        diffs = (
                            X_style_cpu[j] - X_style_cpu[i]
                        )
                        top_idx = torch.topk(
                            diffs.abs(),
                            k=min(
                                3,
                                len(emotions),
                            ),
                            largest=True,
                        ).indices.tolist()
                        top_feeling_deltas = [
                            {
                                'emotion': emotions[int(k)],
                                'delta': float(diffs[int(k)].item()),
                            }
                            for k in top_idx
                        ]
                    neigh_out.append({
                            'index': j,
                            'orig_index': orig_indices[j],
                            'distance': dval_f,
                            'mass_diff': float(abs(masses[i].item()
                                    - masses[j].item())),
                            'saturation_diff_per_1k_words': (
                                float(abs(float(sat_words[i])
                                        - float(sat_words[j])))
                                if sat_words[i] is not None
                                and sat_words[j] is not None
                                else None
                            ),
                            'saturation_diff_per_1k_tokens': (
                                float(abs(float(sat_tokens[i])
                                        - float(sat_tokens[j])))
                                if sat_tokens[i] is not None
                                and sat_tokens[j]
                                is not None
                                else None
                            ),
                            'style_distance': (
                                float(style_dist[i, j].item())
                                if include_style
                                and style_dist is not None
                                else None
                            ),
                            'top_emotion_deltas': top_feeling_deltas,
                            'primary_explain': (
                                _pair_explain(
                                    X[i],
                                    X[j],
                                    mode=mode,
                                    measure=measure,
                                    distance=dval_f,
                                )
                                if include_explain
                                else None
                            ),
                            'style_explain': (
                                _pair_explain(
                                    X_style[i],
                                    X_style[j],
                                    mode=style_mode,
                                    measure=style_measure,
                                    distance=float(style_dist[
                                            i, j
                                        ].item()),
                                )
                                if include_explain
                                and include_style
                                and (X_style is not None)
                                and (style_dist is not None)
                                else None
                            ),
                        })
                neighbours_payload.append({
                        'index': i,
                        'orig_index': orig_indices[i],
                        'neighbours': neigh_out,
                    })
            if include_explain:
                payload['neighbours'] = neighbours_payload
        out_p.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        out_stats: Dict[str, Any] = {
            'docs': len(rows),
            'format': 'matrix',
            'output': str(out_p),
        }
        if scale_simplex and mass_scale_key is not None:
            out_stats['mass_scale_simplex'] = True
            out_stats['mass_scale_key'] = mass_scale_key
        if filtered_out:
            out_stats['filtered_out'] = filtered_out
        if (
            vis_enabled
            and vis_out_p is not None
            and (neighbours_payload is not None)
        ):
            from .viz import write_compare_html_report

            X_cpu = X.detach().cpu().to(dtype=torch.float)
            X_style_cpu2 = (
                X_style.detach().cpu().to(dtype=torch.float)
                if X_style is not None
                else None
            )
            docs_payload = []
            for i in range(int(X_cpu.size(0))):
                docs_payload.append({
                        'index': i,
                        'orig_index': orig_indices[i],
                        'meta': metas[i],
                        'mass': float(masses[i].item()),
                        'pred_per_1k_words': sat_words[i],
                        'pred_per_1k_tokens': sat_tokens[i],
                        'emotion_signal_per_1k_words': signal_vals[
                            i
                        ],
                        'low_emotion_signal': low_flags[i],
                        'vector': X_cpu[i].tolist(),
                        'style_vector': (
                            X_style_cpu2[i].tolist()
                            if X_style_cpu2 is not None
                            else None
                        ),
                    })
            report = {
                'format': 'matrix',
                'mode': mode,
                'cost': cost,
                'measure': measure,
                'epsilon': epsilon,
                'iters': iters,
                'reg_m': reg_m,
                'emotions': emotions,
                'emotion_labels': (
                    {
                        str(e): str(feeling_labels.get(
                                str(e),
                                str(e),
                            ))
                        for e in emotions
                    }
                    if feeling_labels is not None
                    else None
                ),
                'docs': docs_payload,
                'neighbours': neighbours_payload,
                'topk': int(topk),
                'top_flows': int(top_flows),
                'include_explain': bool(include_explain),
                'style': (
                    {
                        'mode': style_mode,
                        'measure': style_measure,
                    }
                    if include_style
                    and X_style_cpu2 is not None
                    else None
                ),
            }
            out_stats['visualization'] = (
                write_compare_html_report(
                    output_path=vis_out_p,
                    payload=report,
                )
            )
        return out_stats
    if fmt == 'neighbours':
        neigh = eot_neighbours(
            X,
            C,
            mode=mode,
            epsilon=epsilon,
            iters=iters,
            reg_m=reg_m,
            topk=int(topk),
        )
        style_all: Optional[List[torch.Tensor]] = None
        if include_style and X_style is not None:
            style_all = []
            for i in range(len(rows)):
                prior_s = X_style[i].expand_as(X_style)
                d_s = ot_loss(
                    X_style,
                    prior_s,
                    C,
                    float(epsilon),
                    int(iters),
                    style_mode,
                    float(reg_m),
                ).to(dtype=torch.float)
                if style_mode.endswith('_divergence'):
                    d_s_rev = ot_loss(
                        prior_s,
                        X_style,
                        C,
                        float(epsilon),
                        int(iters),
                        style_mode,
                        float(reg_m),
                    ).to(dtype=torch.float)
                    d_s = 0.5 * (d_s + d_s_rev)
                d_s = d_s.detach().cpu()
                style_all.append(d_s)
        neighbours_payload: Optional[
            List[Dict[str, Any]]
        ] = ([] if vis_enabled else None)
        with out_p.open(
            'w',
            encoding='utf-8',
        ) as f:
            for i, pairs in enumerate(neigh):
                neighbours_out: List[Dict[str, Any]] = []
                for j, d in pairs:
                    j = int(j)
                    d_f = float(d)
                    primary_explain = (
                        _pair_explain(
                            X[i],
                            X[j],
                            mode=mode,
                            measure=measure,
                            distance=d_f,
                        )
                        if include_explain
                        else None
                    )
                    style_explain = None
                    if (
                        include_explain
                        and style_all is not None
                        and (X_style is not None)
                    ):
                        style_explain = _pair_explain(
                            X_style[i],
                            X_style[j],
                            mode=style_mode,
                            measure=style_measure,
                            distance=float(style_all[i][j].item()),
                        )
                    neighbours_out.append({
                            'index': j,
                            'orig_index': orig_indices[j],
                            'distance': d_f,
                            'mass_diff': float(abs(masses[i].item()
                                    - masses[j].item())),
                            'saturation_diff_per_1k_words': (
                                float(abs(float(sat_words[i])
                                        - float(sat_words[j])))
                                if sat_words[i] is not None
                                and sat_words[j] is not None
                                else None
                            ),
                            'saturation_diff_per_1k_tokens': (
                                float(abs(float(sat_tokens[i])
                                        - float(sat_tokens[j])))
                                if sat_tokens[i] is not None
                                and sat_tokens[j]
                                is not None
                                else None
                            ),
                            'style_distance': (
                                float(style_all[i][j].item())
                                if style_all is not None
                                else None
                            ),
                            'top_emotion_deltas': (
                                [
                                    {
                                        'emotion': emotions[
                                            int(k)
                                        ],
                                        'delta': float((
                                                X_style_cpu[
                                                    j,
                                                    int(k),
                                                ]
                                                - X_style_cpu[
                                                    i,
                                                    int(k),
                                                ]
                                            ).item()),
                                    }
                                    for k in (
                                        torch.topk(
                                            (
                                                X_style_cpu[
                                                    j
                                                ]
                                                - X_style_cpu[
                                                    i
                                                ]
                                            ).abs(),
                                            k=min(
                                                3,
                                                len(emotions),
                                            ),
                                            largest=True,
                                        ).indices.tolist()
                                        if X_style_cpu
                                        is not None
                                        else []
                                    )
                                ]
                                if X_style_cpu is not None
                                else None
                            ),
                            'primary_explain': primary_explain,
                            'style_explain': style_explain,
                        })
                if neighbours_payload is not None:
                    neighbours_payload.append({
                            'index': i,
                            'orig_index': orig_indices[i],
                            'neighbours': neighbours_out,
                        })
                f.write(json.dumps(
                        {
                            'index': i,
                            'orig_index': orig_indices[i],
                            'meta': metas[i],
                            'mass': float(masses[i].item()),
                            'pred_per_1k_words': sat_words[
                                i
                            ],
                            'pred_per_1k_tokens': sat_tokens[
                                i
                            ],
                            'emotion_signal_per_1k_words': signal_vals[
                                i
                            ],
                            'low_emotion_signal': low_flags[
                                i
                            ],
                            'neighbours': neighbours_out,
                        },
                        ensure_ascii=False,
                    )
                    + '\n')
        out_stats: Dict[str, Any] = {
            'docs': len(rows),
            'format': 'neighbours',
            'topk': int(topk),
            'output': str(out_p),
        }
        if scale_simplex and mass_scale_key is not None:
            out_stats['mass_scale_simplex'] = True
            out_stats['mass_scale_key'] = mass_scale_key
        if filtered_out:
            out_stats['filtered_out'] = filtered_out
        if (
            vis_enabled
            and vis_out_p is not None
            and (neighbours_payload is not None)
        ):
            from .viz import write_compare_html_report

            X_cpu = X.detach().cpu().to(dtype=torch.float)
            X_style_cpu2 = (
                X_style.detach().cpu().to(dtype=torch.float)
                if X_style is not None
                else None
            )
            docs_payload = []
            for i in range(int(X_cpu.size(0))):
                docs_payload.append({
                        'index': i,
                        'orig_index': orig_indices[i],
                        'meta': metas[i],
                        'mass': float(masses[i].item()),
                        'pred_per_1k_words': sat_words[i],
                        'pred_per_1k_tokens': sat_tokens[i],
                        'emotion_signal_per_1k_words': signal_vals[
                            i
                        ],
                        'low_emotion_signal': low_flags[i],
                        'vector': X_cpu[i].tolist(),
                        'style_vector': (
                            X_style_cpu2[i].tolist()
                            if X_style_cpu2 is not None
                            else None
                        ),
                    })
            report = {
                'format': 'neighbours',
                'mode': mode,
                'cost': cost,
                'measure': measure,
                'epsilon': epsilon,
                'iters': iters,
                'reg_m': reg_m,
                'emotions': emotions,
                'emotion_labels': (
                    {
                        str(e): str(feeling_labels.get(
                                str(e),
                                str(e),
                            ))
                        for e in emotions
                    }
                    if feeling_labels is not None
                    else None
                ),
                'docs': docs_payload,
                'neighbours': neighbours_payload,
                'topk': int(topk),
                'top_flows': int(top_flows),
                'include_explain': bool(include_explain),
                'style': (
                    {
                        'mode': style_mode,
                        'measure': style_measure,
                    }
                    if include_style
                    and X_style_cpu2 is not None
                    else None
                ),
            }
            out_stats['visualization'] = (
                write_compare_html_report(
                    output_path=vis_out_p,
                    payload=report,
                )
            )
        return out_stats
    raise ValueError('error: ValueError')


def render_compare_neighbours_html(
    *,
    scores_jsonl: str | Path,
    neighbours_jsonl: str | Path,
    output_html: str | Path,
    cfg_path: str | None = None,
    embed_text: bool = False,
    embed_text_max_chars: int = 0,
    embed_text_collapse_whitespace: bool = False,
) -> str:
    scores_p = Path(scores_jsonl)
    neigh_p = Path(neighbours_jsonl)
    out_p = Path(output_html)
    if not scores_p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    if not neigh_p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    setup = load_config(cfg_path)
    rows_all = _read_jsonl(scores_p)
    if not rows_all:
        raise ValueError('error: ValueError')
    emotions = _infer_emotions(
        rows_all,
        setup.emotions,
    )
    feeling_labels = getattr(
        setup,
        'emotion_display_names',
        None,
    )
    if not isinstance(
        feeling_labels,
        dict,
    ):
        feeling_labels = None
    mode = (setup.ot_mode or 'sinkhorn_divergence').lower()
    cost = (setup.ot_cost or 'uniform').lower()
    epsilon = float(getattr(
            setup,
            'sinkhorn_epsilon',
            0.1,
        ))
    iters = int(getattr(
            setup,
            'sinkhorn_iters',
            30,
        ))
    reg_m = float(getattr(
            setup,
            'ot_reg_m',
            0.1,
        ))
    count_scale = (
        str(getattr(
                setup,
                'count_pred_scale',
                'counts',
            )
            or 'counts')
        .lower()
        .strip()
    )
    use_token_scale = count_scale in {
        'density',
        'per_token',
        'per_tok',
    }
    row0 = rows_all[0]

    def _pick_scale(
        words_key: str,
        tokens_key: str,
    ) -> str:
        if use_token_scale and tokens_key in row0:
            return tokens_key
        return words_key

    if mode.startswith('unbalanced'):
        if 'pred_counts_calibrated' in row0:
            measure = _pick_scale(
                'pred_counts_calibrated_per_1k_words',
                'pred_counts_calibrated_per_1k_tokens',
            )
        elif 'pred_counts' in row0:
            measure = _pick_scale(
                'pred_counts_per_1k_words',
                'pred_counts_per_1k_tokens',
            )
        elif (
            'pred_mixscaled_per_1k_words' in row0
            or 'pred_mixscaled_per_1k_tokens' in row0
        ):
            measure = _pick_scale(
                'pred_mixscaled_per_1k_words',
                'pred_mixscaled_per_1k_tokens',
            )
        else:
            measure = _pick_scale(
                'pred_counts_per_1k_words',
                'pred_counts_per_1k_tokens',
            )
    elif 'pred_dist_calibrated' in row0:
        measure = 'pred_dist_calibrated'
    elif 'pred_dist' in row0:
        measure = 'pred_dist'
    else:
        measure = 'pred_sigmoid'
    measure = str(measure).lower()
    style_mode = 'sinkhorn_divergence'
    if 'pred_dist_calibrated' in row0:
        style_measure = 'pred_dist_calibrated'
    elif 'pred_dist' in row0:
        style_measure = 'pred_dist'
    else:
        style_measure = 'pred_sigmoid'
    style_measure = str(style_measure).lower()
    neighbours_payload: List[Dict[str, Any]] = []
    docs_payload: List[Dict[str, Any]] = []
    topk = 0
    top_flows = 0
    include_explain = False
    docs_tmp: List[Tuple[int, Dict[str, Any]]] = []
    with neigh_p.open(
        'r',
        encoding='utf-8',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs_tmp.append((len(docs_tmp), json.loads(line)))
    docs_tmp.sort(key=lambda x: int(x[1].get(
                'index',
                x[0],
            )))
    for _, doc_row in docs_tmp:
        i = int(doc_row.get(
                'index',
                len(docs_payload),
            ))
        orig = int(doc_row.get(
                'orig_index',
                i,
            ))
        if orig < 0 or orig >= len(rows_all):
            raise IndexError('error: IndexError')
        row = rows_all[orig]
        vec = extract_measure(
            row,
            measure,
        )
        vec = _align_emotion_vector(
            row,
            vec,
            list(emotions),
            measure=measure,
            row_index=orig,
        )
        style_vec = extract_measure(
            row,
            style_measure,
        )
        style_vec = _align_emotion_vector(
            row,
            style_vec,
            list(emotions),
            measure=style_measure,
            row_index=orig,
        )
        meta = (
            doc_row.get('meta')
            if isinstance(
                doc_row.get('meta'),
                dict,
            )
            else row.get(
                'meta',
                {},
            )
        )
        neigh = (
            doc_row.get('neighbours')
            if isinstance(
                doc_row.get('neighbours'),
                list,
            )
            else []
        )
        topk = max(
            topk,
            len(neigh),
        )
        for n in neigh:
            if (
                isinstance(
                    n,
                    dict,
                )
                and n.get('primary_explain') is not None
            ):
                include_explain = True
            flows = None
            if isinstance(
                n,
                dict,
            ):
                pe = n.get('primary_explain')
                if isinstance(
                    pe,
                    dict,
                ):
                    flows = pe.get('top_transport_cost_contrib')
            if isinstance(
                flows,
                list,
            ):
                top_flows = max(
                    top_flows,
                    len(flows),
                )
        docs_payload.append({
                'index': i,
                'orig_index': orig,
                'meta': meta,
                'mass': float(doc_row.get(
                        'mass',
                        float(vec.sum().detach().cpu().item()),
                    )),
                'pred_per_1k_words': doc_row.get('pred_per_1k_words'),
                'pred_per_1k_tokens': doc_row.get('pred_per_1k_tokens'),
                'emotion_signal_per_1k_words': doc_row.get('emotion_signal_per_1k_words'),
                'low_emotion_signal': doc_row.get('low_emotion_signal'),
                'text': None,
                'vector': vec.detach()
                .cpu()
                .to(dtype=torch.float)
                .tolist(),
                'style_vector': style_vec.detach()
                .cpu()
                .to(dtype=torch.float)
                .tolist(),
            })
        if embed_text:
            meta_path = (
                meta.get('path')
                if isinstance(
                    meta,
                    dict,
                )
                else None
            )
            if (
                isinstance(
                    meta_path,
                    str,
                )
                and meta_path.strip()
            ):
                p = Path(meta_path.strip())
                if not p.is_absolute():
                    p = (_ROOT / p).resolve()
                if p.exists():
                    try:
                        t = p.read_text(
                            encoding='utf-8',
                            errors='ignore',
                        )
                        if embed_text_collapse_whitespace:
                            t = ' '.join(t.split())
                        if int(embed_text_max_chars) > 0 and len(t) > int(embed_text_max_chars):
                            t = (
                                t[
                                    : int(embed_text_max_chars)
                                ].rstrip()
                                + '...'
                            )
                        docs_payload[-1]['text'] = t
                    except Exception:
                        pass
        neighbours_payload.append({
                'index': i,
                'orig_index': orig,
                'neighbours': neigh,
            })
    from .viz import write_compare_html_report

    report = {
        'format': 'neighbours',
        'mode': mode,
        'cost': cost,
        'measure': measure,
        'epsilon': epsilon,
        'iters': iters,
        'reg_m': reg_m,
        'emotions': list(emotions),
        'emotion_labels': (
            {
                str(e): str(feeling_labels.get(
                        str(e),
                        str(e),
                    ))
                for e in emotions
            }
            if feeling_labels is not None
            else None
        ),
        'docs': docs_payload,
        'neighbours': neighbours_payload,
        'topk': int(topk),
        'top_flows': int(top_flows or 8),
        'include_explain': bool(include_explain),
        'style': {
            'mode': style_mode,
            'measure': style_measure,
        },
    }
    return write_compare_html_report(
        output_path=out_p,
        payload=report,
    )


def _top_transport_contrib(
    plan: torch.Tensor,
    C: torch.Tensor,
    emotions: List[str],
    *,
    topk: int = 8,
    include_diagonal: bool = False,
) -> List[Dict[str, Any]]:
    if plan.dim() != 2 or C.dim() != 2:
        raise ValueError('error: ValueError')
    if plan.shape != C.shape:
        raise ValueError('error: ValueError')
    n = int(plan.size(0))
    if n != len(emotions):
        raise ValueError('error: ValueError')
    contrib = (
        (plan * C).detach().cpu().to(dtype=torch.float)
    )
    mass = plan.detach().cpu().to(dtype=torch.float)
    if not include_diagonal:
        contrib.fill_diagonal_(0.0)
        mass.fill_diagonal_(0.0)
    flat = contrib.view(-1)
    k = min(
        int(topk),
        flat.numel(),
    )
    if k <= 0:
        return []
    (
        vals,
        idx,
    ) = torch.topk(
        flat,
        k=k,
        largest=True,
    )
    out: List[Dict[str, Any]] = []
    for v, linear in zip(
        vals.tolist(),
        idx.tolist(),
    ):
        if v <= 0:
            continue
        i = int(linear // n)
        j = int(linear % n)
        out.append({
                'from': emotions[i],
                'to': emotions[j],
                'mass': float(mass[i, j].item()),
                'cost': float(C[i, j].detach().cpu().item()),
                'contribution': float(v),
            })
    return out


def explain_pair(
    *,
    input_jsonl: str,
    i: int,
    j: int,
    output_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cost: Optional[str] = None,
    mode: Optional[str] = None,
    measure: Optional[str] = None,
    style_mode: Optional[str] = None,
    style_measure: Optional[str] = None,
    epsilon: Optional[float] = None,
    iters: Optional[int] = None,
    reg_m: Optional[float] = None,
    top_flows: int = 8,
) -> Dict[str, Any]:
    setup = load_config(cfg_path)
    rows = _read_jsonl(
        input_jsonl,
        limit=None,
    )
    emotions = _infer_emotions(
        rows,
        setup.emotions,
    )
    if not rows:
        raise ValueError('error: ValueError')
    n = len(rows)
    i = int(i)
    j = int(j)
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError('error: ValueError')
    if i == j:
        raise ValueError('error: ValueError')
    mode = (
        mode or setup.ot_mode or 'unbalanced_divergence'
    ).lower()
    cost = (cost or setup.ot_cost or 'uniform').lower()
    epsilon = float(epsilon
        if epsilon is not None
        else setup.sinkhorn_epsilon)
    iters = int(iters if iters is not None else setup.sinkhorn_iters)
    reg_m = float(reg_m if reg_m is not None else setup.ot_reg_m)
    count_scale = (
        str(getattr(
                setup,
                'count_pred_scale',
                'counts',
            )
            or 'counts')
        .lower()
        .strip()
    )
    use_token_scale = count_scale in {
        'density',
        'per_token',
        'per_tok',
    }
    if measure is None:
        row0 = rows[0] if rows else {}

        def _pick_scale(
            words_key: str,
            tokens_key: str,
        ) -> str:
            if use_token_scale and tokens_key in row0:
                return tokens_key
            return words_key

        if mode.startswith('unbalanced'):
            if 'pred_counts_calibrated' in row0:
                measure = _pick_scale(
                    'pred_counts_calibrated_per_1k_words',
                    'pred_counts_calibrated_per_1k_tokens',
                )
            elif 'pred_counts' in row0:
                measure = _pick_scale(
                    'pred_counts_per_1k_words',
                    'pred_counts_per_1k_tokens',
                )
            elif (
                'pred_mixscaled_per_1k_words' in row0
                or 'pred_mixscaled_per_1k_tokens' in row0
            ):
                measure = _pick_scale(
                    'pred_mixscaled_per_1k_words',
                    'pred_mixscaled_per_1k_tokens',
                )
            else:
                measure = _pick_scale(
                    'pred_counts_per_1k_words',
                    'pred_counts_per_1k_tokens',
                )
        elif rows and 'pred_dist_calibrated' in rows[0]:
            measure = 'pred_dist_calibrated'
        elif rows and 'pred_dist' in rows[0]:
            measure = 'pred_dist'
        else:
            measure = 'pred_sigmoid'
    measure = (measure or '').lower()
    style_mode = (
        style_mode or 'sinkhorn_divergence'
    ).lower()
    if style_measure is None:
        if rows and 'pred_dist_calibrated' in rows[0]:
            style_measure = 'pred_dist_calibrated'
        elif rows and 'pred_dist' in rows[0]:
            style_measure = 'pred_dist'
        else:
            style_measure = 'pred_sigmoid'
    style_measure = (
        style_measure or 'pred_sigmoid'
    ).lower()
    device = get_device(setup.device)
    C = build_cost(
        emotions,
        device=device,
        cost=cost,
        lexicon_path=getattr(
            setup,
            'lexicon_path',
            None,
        ),
        vad_lexicon_path=getattr(
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
        lexicon_stopwords_file=getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        ),
        lexicon_extra_path=getattr(
            setup,
            'lexicon_extra_path',
            None,
        ),
        lexicon_intensity_path=getattr(
            setup,
            'lexicon_intensity_path',
            None,
        ),
        lexicon_intensity_min=getattr(
            setup,
            'lexicon_intensity_min',
            0.0,
        ),
        lexicon_min_vad_salience=getattr(
            setup,
            'lexicon_min_vad_salience',
            0.0,
        ),
        lexicon_min_vad_arousal=getattr(
            setup,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        lexicon_require_word_vad=bool(getattr(
                setup,
                'lexicon_require_word_vad',
                False,
            )),
        lexicon_allow_seed_only=bool(getattr(
                setup,
                'lexicon_allow_seed_only',
                False,
            )),
        vad_allow_missing=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    ).to(
        device=device,
        dtype=torch.float,
    )
    vec_i = extract_measure(
        rows[i],
        measure,
    )
    vec_i = _align_emotion_vector(
        rows[i],
        vec_i,
        emotions,
        measure=measure,
        row_index=i,
    )
    if vec_i.numel() != len(emotions):
        raise ValueError('error: ValueError')
    vec_i = vec_i.to(
        device=device,
        dtype=torch.float,
    )
    vec_j = extract_measure(
        rows[j],
        measure,
    )
    vec_j = _align_emotion_vector(
        rows[j],
        vec_j,
        emotions,
        measure=measure,
        row_index=j,
    )
    if vec_j.numel() != len(emotions):
        raise ValueError('error: ValueError')
    vec_j = vec_j.to(
        device=device,
        dtype=torch.float,
    )
    simplex_like = {
        'pred_softmax',
        'pred_sigmoid',
        'pred_dist',
        'pred_dist_calibrated',
        'lex_prior',
    }
    if measure in simplex_like:
        for idx, row, vec in (
            (i, rows[i], vec_i),
            (j, rows[j], vec_j),
        ):
            invalid_flag = False
            if (
                measure == 'pred_dist_calibrated'
                and row.get('pred_dist_calibrated_valid')
                is False
            ):
                invalid_flag = True
            if float(vec.sum().item()) <= 1e-08:
                invalid_flag = True
            if invalid_flag:
                raise ValueError('error: ValueError')
    style_i = extract_measure(
        rows[i],
        style_measure,
    )
    style_i = _align_emotion_vector(
        rows[i],
        style_i,
        emotions,
        measure=style_measure,
        row_index=i,
    )
    if style_i.numel() != len(emotions):
        raise ValueError('error: ValueError')
    style_i = style_i.to(
        device=device,
        dtype=torch.float,
    )
    style_j = extract_measure(
        rows[j],
        style_measure,
    )
    style_j = _align_emotion_vector(
        rows[j],
        style_j,
        emotions,
        measure=style_measure,
        row_index=j,
    )
    if style_j.numel() != len(emotions):
        raise ValueError('error: ValueError')
    style_j = style_j.to(
        device=device,
        dtype=torch.float,
    )
    dist_primary = float(ot_loss(
            vec_i,
            vec_j,
            C,
            epsilon,
            iters,
            mode,
            reg_m,
        )
        .detach()
        .cpu()
        .item())
    dist_style = float(ot_loss(
            style_i,
            style_j,
            C,
            epsilon,
            iters,
            style_mode,
            reg_m,
        )
        .detach()
        .cpu()
        .item())

    def _base_mode(m: str) -> str:
        m = (m or '').lower()
        return (
            m[: -len('_divergence')]
            if m.endswith('_divergence')
            else m
        )

    unbalanced_primary = _base_mode(mode) == 'unbalanced'
    unbalanced_style = (
        _base_mode(style_mode) == 'unbalanced'
    )
    if _base_mode(mode) not in {'sinkhorn', 'unbalanced'}:
        raise ValueError('error: ValueError')
    if _base_mode(style_mode) not in {
        'sinkhorn',
        'unbalanced',
    }:
        raise ValueError('error: ValueError')
    primary_is_div = mode.endswith('_divergence')
    style_is_div = style_mode.endswith('_divergence')
    parts_primary = sinkhorn_cost_parts(
        vec_i,
        vec_j,
        C,
        epsilon=epsilon,
        iters=iters,
        unbalanced=unbalanced_primary,
        reg_m=reg_m,
        return_plan=True,
    )
    parts_primary_aa = None
    parts_primary_bb = None
    if primary_is_div:
        parts_primary_aa = sinkhorn_cost_parts(
            vec_i,
            vec_i,
            C,
            epsilon=epsilon,
            iters=iters,
            unbalanced=unbalanced_primary,
            reg_m=reg_m,
            return_plan=False,
        )
        parts_primary_bb = sinkhorn_cost_parts(
            vec_j,
            vec_j,
            C,
            epsilon=epsilon,
            iters=iters,
            unbalanced=unbalanced_primary,
            reg_m=reg_m,
            return_plan=False,
        )
    parts_style = sinkhorn_cost_parts(
        style_i,
        style_j,
        C,
        epsilon=epsilon,
        iters=iters,
        unbalanced=unbalanced_style,
        reg_m=reg_m,
        return_plan=True,
    )
    parts_style_aa = None
    parts_style_bb = None
    if style_is_div:
        parts_style_aa = sinkhorn_cost_parts(
            style_i,
            style_i,
            C,
            epsilon=epsilon,
            iters=iters,
            unbalanced=unbalanced_style,
            reg_m=reg_m,
            return_plan=False,
        )
        parts_style_bb = sinkhorn_cost_parts(
            style_j,
            style_j,
            C,
            epsilon=epsilon,
            iters=iters,
            unbalanced=unbalanced_style,
            reg_m=reg_m,
            return_plan=False,
        )
    plan_primary = parts_primary.get('plan')
    plan_style = parts_style.get('plan')
    top_primary = _top_transport_contrib(
        plan_primary,
        C,
        emotions,
        topk=int(top_flows),
        include_diagonal=False,
    )
    top_style = _top_transport_contrib(
        plan_style,
        C,
        emotions,
        topk=int(top_flows),
        include_diagonal=False,
    )
    meta_i = (
        rows[i].get('meta')
        if isinstance(
            rows[i].get('meta'),
            dict,
        )
        else {}
    )
    meta_j = (
        rows[j].get('meta')
        if isinstance(
            rows[j].get('meta'),
            dict,
        )
        else {}
    )
    style_div_calc = None
    style_div_error = None
    if (
        style_is_div
        and parts_style_aa is not None
        and (parts_style_bb is not None)
    ):
        style_div_calc = float((
                parts_style['total']
                - 0.5 * parts_style_aa['total']
                - 0.5 * parts_style_bb['total']
            )
            .detach()
            .cpu()
            .item())
        style_div_error = float(dist_style - style_div_calc)
    primary_div_calc = None
    primary_div_error = None
    if (
        primary_is_div
        and parts_primary_aa is not None
        and (parts_primary_bb is not None)
    ):
        primary_div_calc = float((
                parts_primary['total']
                - 0.5 * parts_primary_aa['total']
                - 0.5 * parts_primary_bb['total']
            )
            .detach()
            .cpu()
            .item())
        primary_div_error = float(dist_primary - primary_div_calc)
    signal_i = _infer_signal_per_1k_words(rows[i])
    signal_j = _infer_signal_per_1k_words(rows[j])
    min_signal = float(getattr(
            setup,
            'semantic_min_signal_per_1k_words',
            0.0,
        )
        or 0.0)
    low_i = rows[i].get('low_emotion_signal')
    if (
        low_i is None
        and signal_i is not None
        and (min_signal > 0.0)
    ):
        low_i = bool(float(signal_i) < min_signal)
    low_j = rows[j].get('low_emotion_signal')
    if (
        low_j is None
        and signal_j is not None
        and (min_signal > 0.0)
    ):
        low_j = bool(float(signal_j) < min_signal)
    out: Dict[str, Any] = {
        'i': {
            'index': i,
            'meta': meta_i,
            'pred_per_1k_words': rows[i].get(
                'pred_counts_calibrated_per_1k_words',
                rows[i].get('pred_per_1k_words'),
            ),
            'pred_per_1k_tokens': rows[i].get(
                'pred_counts_calibrated_per_1k_tokens',
                rows[i].get('pred_per_1k_tokens'),
            ),
            'emotion_signal_per_1k_words': signal_i,
            'low_emotion_signal': (
                bool(low_i)
                if isinstance(
                    low_i,
                    bool,
                )
                else None
            ),
        },
        'j': {
            'index': j,
            'meta': meta_j,
            'pred_per_1k_words': rows[j].get(
                'pred_counts_calibrated_per_1k_words',
                rows[j].get('pred_per_1k_words'),
            ),
            'pred_per_1k_tokens': rows[j].get(
                'pred_counts_calibrated_per_1k_tokens',
                rows[j].get('pred_per_1k_tokens'),
            ),
            'emotion_signal_per_1k_words': signal_j,
            'low_emotion_signal': (
                bool(low_j)
                if isinstance(
                    low_j,
                    bool,
                )
                else None
            ),
        },
        'emotions': emotions,
        'cost': cost,
        'epsilon': epsilon,
        'iters': iters,
        'reg_m': reg_m,
        'style': {
            'mode': style_mode,
            'measure': style_measure,
            'distance': dist_style,
            'cost_ab': float(parts_style['total'].detach().cpu().item()),
            'cost_aa': (
                float(parts_style_aa['total']
                    .detach()
                    .cpu()
                    .item())
                if parts_style_aa is not None
                else None
            ),
            'cost_bb': (
                float(parts_style_bb['total']
                    .detach()
                    .cpu()
                    .item())
                if parts_style_bb is not None
                else None
            ),
            'divergence_calc': style_div_calc,
            'distance_minus_calc': style_div_error,
            'transport_ab': float(parts_style['transport']
                .detach()
                .cpu()
                .item()),
            'transport': float(parts_style['transport']
                .detach()
                .cpu()
                .item()),
            'top_transport_cost_contrib': top_style,
        },
        'primary': {
            'mode': mode,
            'measure': measure,
            'distance': dist_primary,
            'cost_ab': float(parts_primary['total'].detach().cpu().item()),
            'cost_aa': (
                float(parts_primary_aa['total']
                    .detach()
                    .cpu()
                    .item())
                if parts_primary_aa is not None
                else None
            ),
            'cost_bb': (
                float(parts_primary_bb['total']
                    .detach()
                    .cpu()
                    .item())
                if parts_primary_bb is not None
                else None
            ),
            'divergence_calc': primary_div_calc,
            'distance_minus_calc': primary_div_error,
            'transport_ab': float(parts_primary['transport']
                .detach()
                .cpu()
                .item()),
            'transport': float(parts_primary['transport']
                .detach()
                .cpu()
                .item()),
            'kl_a': float(parts_primary['kl_a'].detach().cpu().item()),
            'kl_b': float(parts_primary['kl_b'].detach().cpu().item()),
            'mass_a': float(parts_primary['mass_a']
                .detach()
                .cpu()
                .item()),
            'mass_b': float(parts_primary['mass_b']
                .detach()
                .cpu()
                .item()),
            'mass_plan': float(parts_primary['mass_plan']
                .detach()
                .cpu()
                .item()),
            'top_transport_cost_contrib': top_primary,
        },
    }
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        out_p.write_text(
            json.dumps(
                out,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
    return out
