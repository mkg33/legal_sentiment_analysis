#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Any,
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
import torch
from legal_emotion.corpus import (
    iter_text_paths,
    parse_icj_meta,
    read_text,
)
from legal_emotion.lexicon import (
    load_lexicon,
    load_word_vad,
    tokenize as lex_tokenize,
)
from legal_emotion.token_compare import (
    TokenCloud,
    _apply_term_weights,
    _build_emotional_matcher,
    _build_term_vad_map,
    _embed_terms,
    _load_imputed_vad_map,
    _load_stopword_terms,
    _load_term_weights,
    _pair_distance_and_explain,
    _reverse_primary_explain,
    _term_vad_from_lexicon,
    _merge_imputed_vad,
    _extract_emotional_terms,
)
from legal_emotion.utils import get_device, load_config
from legal_emotion.token_viz import (
    write_token_ot_html_report,
)

try:
    from sklearn.feature_extraction.text import (
        ENGLISH_STOP_WORDS,
    )
except Exception:
    ENGLISH_STOP_WORDS = frozenset()


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


def _group_value(
    meta: Dict[str, Any],
    group_by: str,
) -> str:
    key = (group_by or '').strip()
    if not key:
        return 'unknown'
    if key == 'year':
        date = meta.get('date')
        if isinstance(
            date,
            str,
        ) and len(date) >= 4:
            return date[:4]
        return 'unknown'
    val = meta.get(key)
    if isinstance(
        val,
        str,
    ) and val:
        return val
    return 'unknown'


def _term_salience(vad: torch.Tensor) -> float:
    v = vad.to(dtype=torch.float).view(-1)
    if v.numel() < 2:
        return float(torch.abs(v).max().item())
    valence = float(torch.abs(v[0]).clamp(
            0.0,
            1.0,
        ).item())
    arousal_pos = float(v[1].clamp(
            0.0,
            1.0,
        ).item())
    return max(
        valence,
        arousal_pos,
    )


def _select_terms_and_weights(
    counts: Counter[str],
    *,
    weight: str,
    max_terms: int,
    idf: Optional[Dict[str, float]] = None,
    unk_token: str = '[UNK]',
) -> Tuple[List[str], List[float]]:
    max_k = max(
        1,
        int(max_terms),
    )
    weight_mode = (weight or 'tf').lower().strip()
    items: List[Tuple[str, float]] = []
    if weight_mode == 'tfidf':
        idf = idf or {}
        for t, c in counts.items():
            items.append((t, float(c) * float(idf.get(
                        t,
                        1.0,
                    ))))
    elif weight_mode in {'tf', 'count', 'counts'}:
        for t, c in counts.items():
            items.append((t, float(c)))
    elif weight_mode in {'uniform', 'binary'}:
        for t in counts.keys():
            items.append((t, 1.0))
    else:
        raise ValueError('error: ValueError')
    items = [(t, w) for t, w in items if w > 0]
    items.sort(
        key=lambda x: (x[1], x[0]),
        reverse=True,
    )
    items = items[:max_k]
    if not items:
        return ([unk_token], [1.0])
    return ([t for t, _ in items], [w for _, w in items])


def main() -> int:
    ap = argparse.ArgumentParser(description='token-ot group shift')
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
        '--group_by',
        type=str,
        default='doc_type',
    )
    ap.add_argument(
        '--group_a',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--group_b',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--exclude_unknown',
        action='store_true',
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
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
        default=16,
    )
    ap.add_argument(
        '--no_vis',
        action='store_true',
    )
    opts = ap.parse_args()
    setup = load_config(opts.config)
    device = get_device(getattr(
            setup,
            'device',
            'cpu',
        ))
    term_weights_path = opts.term_weights_path or getattr(
        setup,
        'token_term_weights_path',
        None,
    )
    term_weight_power = float(opts.term_weight_power
        if opts.term_weight_power is not None
        else getattr(
            setup,
            'token_term_weight_power',
            1.0,
        ))
    term_weight_min = float(opts.term_weight_min
        if opts.term_weight_min is not None
        else getattr(
            setup,
            'token_term_weight_min',
            0.0,
        ))
    term_weight_max = float(opts.term_weight_max
        if opts.term_weight_max is not None
        else getattr(
            setup,
            'token_term_weight_max',
            1.0,
        ))
    term_weight_mix = float(opts.term_weight_mix
        if opts.term_weight_mix is not None
        else getattr(
            setup,
            'token_term_weight_mix',
            1.0,
        ))
    term_weight_default = float(opts.term_weight_default
        if opts.term_weight_default is not None
        else getattr(
            setup,
            'token_term_weight_default',
            1.0,
        ))
    (
        term_weights,
        term_weights_meta,
    ) = (
        _load_term_weights(term_weights_path)
        if term_weights_path
        else ({}, None)
    )
    term_weights_active = bool(term_weights)
    vad_imputed_path = opts.vad_imputed_path or getattr(
        setup,
        'token_vad_imputed_path',
        None,
    )
    vad_imputed_weight = float(getattr(
            setup,
            'token_vad_imputed_weight',
            0.0,
        )
        or 0.0)
    out_dir = (
        Path(opts.out_dir)
        if opts.out_dir
        else Path('outputs')
        / time.strftime('token_ot_group_shift_%Y%m%d_%H%M%S')
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    data_dir = Path(opts.data_dir)
    paths = list(iter_text_paths(
            data_dir,
            recursive=True,
        ))
    if not paths:
        raise SystemExit('error: SystemExit')
    group_to_paths: Dict[str, List[Path]] = defaultdict(list)
    rows_meta: List[Dict[str, Any]] = []
    for p in paths:
        meta = parse_icj_meta(p)
        meta.setdefault(
            'id',
            p.stem,
        )
        meta['path'] = str(p)
        g = _group_value(
            meta,
            str(opts.group_by),
        )
        if opts.exclude_unknown and g == 'unknown':
            continue
        group_to_paths[g].append(p)
        rows_meta.append({
                'meta': {
                    **meta,
                    'group': g,
                }
            })
        if opts.limit is not None and len(rows_meta) >= int(opts.limit):
            break
    if not group_to_paths:
        raise SystemExit('error: SystemExit')
    group_sizes = {
        g: len(ps) for g, ps in group_to_paths.items()
    }
    groups_sorted = sorted(
        group_sizes.items(),
        key=lambda x: (-x[1], x[0]),
    )
    _write_json(
        out_dir / 'groups.json',
        {
            'group_by': opts.group_by,
            'groups': groups_sorted,
        },
    )
    _write_jsonl(
        out_dir / 'docs.jsonl',
        rows_meta,
    )
    if opts.group_a and opts.group_b:
        ga = str(opts.group_a)
        gb = str(opts.group_b)
    else:
        if len(groups_sorted) < 2:
            raise SystemExit('error: SystemExit')
        ga = groups_sorted[0][0]
        gb = groups_sorted[1][0]
    if ga not in group_to_paths:
        raise SystemExit('error: SystemExit')
    if gb not in group_to_paths:
        raise SystemExit('error: SystemExit')
    selected_groups = [ga, gb]
    selected_paths = [
        (ga, p) for p in group_to_paths[ga]
    ] + [(gb, p) for p in group_to_paths[gb]]
    sw = (
        set(ENGLISH_STOP_WORDS)
        if not bool(opts.no_stopwords)
        else set()
    )
    (
        extra_sw,
        extra_phrase_sw,
    ) = _load_stopword_terms(str(opts.stopwords_file)
        if opts.stopwords_file
        else None)
    sw |= extra_sw
    min_len = int(opts.min_token_len)
    df: Counter[str] = Counter()
    for _, p in selected_paths:
        toks_raw = lex_tokenize(read_text(p))
        toks_all = [
            w
            for w in toks_raw
            if len(w) >= min_len and w not in sw
        ]
        for t in set(toks_all):
            df[t] += 1
    drop_set: set[str] = set()
    dropped_top_df_terms: List[str] = []
    k_drop = int(opts.drop_top_df)
    if k_drop > 0 and df:
        items = sorted(
            df.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
        dropped_top_df_terms = [
            t for t, _ in items[: min(
                k_drop,
                len(items),
            )]
        ]
        drop_set = set(dropped_top_df_terms)
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
    word_vad = (
        load_word_vad(
            getattr(
                setup,
                'vad_lexicon_path',
                None,
            ),
            vad_scale=getattr(
                setup,
                'word_vad_scale',
                None,
            ),
        )
        if getattr(
            setup,
            'vad_lexicon_path',
            None,
        )
        else {}
    )
    term_vad_conf: Dict[str, float] = {}
    (
        term_vad,
        term_vad_conf,
    ) = _build_term_vad_map(
        lex,
        word_vad,
        imputed_weight=vad_imputed_weight,
        return_conf=True,
    )
    vad_support = any((v > 0.0 for v in term_vad_conf.values()))
    cost_mode = (
        str(opts.cost or 'embedding').lower().strip()
    )
    if cost_mode == 'vad' and (not vad_support):
        warnings.warn(
            'warn: vad',
            RuntimeWarning,
            stacklevel=2,
        )
        cost_mode = 'embedding'
        opts.cost = 'embedding'
    if (
        cost_mode == 'embedding_vad'
        and (not vad_support)
        and (float(getattr(
                opts,
                'beta_vad',
                0.0,
            )) > 0.0)
    ):
        warnings.warn(
            'warn: beta_vad',
            RuntimeWarning,
            stacklevel=2,
        )
        opts.beta_vad = 0.0
    lexicon_terms = set(_term_vad_from_lexicon(lex).keys())
    word_vad_terms = set((
            ' '.join(lex_tokenize(t))
            for t in (word_vad or {}).keys()
            if ' '.join(lex_tokenize(t))
        ))
    imputed_terms: set[str] = set()
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
            default_conf=vad_imputed_weight,
        )
        if imputed_terms:
            word_vad_terms |= imputed_terms
    allowed_terms: Optional[set[str]] = None
    ngram_map: Optional[Dict[Tuple[str, ...], str]] = None
    focus = (opts.focus or 'emotional').lower().strip()
    if focus != 'all':
        (
            allowed_terms,
            ngram_map,
        ) = _build_emotional_matcher(
            lexicon_terms=lexicon_terms,
            word_vad_terms=word_vad_terms,
            term_vad=term_vad,
            vad_threshold=float(opts.vad_threshold),
            vad_min_arousal_vad_only=float(opts.vad_min_arousal_vad_only),
            emotional_vocab=str(opts.emotional_vocab),
            max_ngram=int(opts.max_ngram),
            stopword_terms=extra_phrase_sw,
        )
    group_term_counts: Dict[str, Counter[str]] = {
        ga: Counter(),
        gb: Counter(),
    }
    group_raw_tokens: Dict[str, int] = {
        ga: 0,
        gb: 0,
    }
    group_selected_tokens: Dict[str, int] = {
        ga: 0,
        gb: 0,
    }
    for g, p in selected_paths:
        text = read_text(p)
        toks_raw = lex_tokenize(text)
        group_raw_tokens[g] += int(len(toks_raw))
        if focus == 'all':
            terms = [
                w
                for w in toks_raw
                if len(w) >= min_len and w not in sw
            ]
        else:
            terms = _extract_emotional_terms(
                toks_raw,
                lexicon_terms=lexicon_terms,
                word_vad_terms=word_vad_terms,
                term_vad=term_vad,
                vad_threshold=float(opts.vad_threshold),
                vad_min_arousal_vad_only=float(opts.vad_min_arousal_vad_only),
                emotional_vocab=str(opts.emotional_vocab),
                max_ngram=int(opts.max_ngram),
                stopwords=sw,
                stopword_terms=extra_phrase_sw,
                min_token_len=min_len,
                allowed_terms=allowed_terms,
                ngram_to_term=ngram_map,
            )
        if drop_set:
            terms = [t for t in terms if t not in drop_set]
        group_selected_tokens[g] += int(len(terms))
        group_term_counts[g].update(terms)
    df_groups: Counter[str] = Counter()
    for g in selected_groups:
        for t in group_term_counts[g].keys():
            df_groups[t] += 1
    idf: Dict[str, float] = {}
    n_groups = len(selected_groups)
    for t, dfi in df_groups.items():
        idf[t] = (
            math.log((1.0 + float(n_groups)) / (1.0 + float(dfi)))
            + 1.0
        )
    embed_model_name = str(getattr(
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
    embed_backend = str(getattr(
            setup,
            'token_ot_embed_backend',
            None,
        )
        or 'encoder')
    embed_pooling = str(getattr(
            setup,
            'token_ot_embed_pooling',
            None,
        )
        or 'cls')
    embed_bs = int(getattr(
            setup,
            'token_ot_embed_batch_size',
            64,
        ))
    embed_max_len = int(getattr(
            setup,
            'token_ot_embed_max_length',
            32,
        ))
    unk = str(getattr(
            _load_tokenizer(embed_model_name),
            'unk_token',
            None,
        )
        or '[UNK]')
    group_terms: Dict[str, List[str]] = {}
    group_weights: Dict[str, List[float]] = {}
    group_term_weights: Optional[Dict[str, List[float]]] = (
        {} if term_weights_active else None
    )
    for g in selected_groups:
        (
            terms_g,
            weights_g,
        ) = _select_terms_and_weights(
            group_term_counts[g],
            weight=str(opts.weight),
            max_terms=int(opts.max_terms),
            idf=idf,
            unk_token=unk,
        )
        term_weight_g = None
        if term_weights_active:
            (
                weights_g,
                term_weight_g,
            ) = _apply_term_weights(
                terms_g,
                weights_g,
                term_weights,
                weight_default=term_weight_default,
                weight_min=term_weight_min,
                weight_max=term_weight_max,
                weight_power=term_weight_power,
                weight_mix=term_weight_mix,
            )
        if cost_mode == 'vad' and term_vad_conf:
            keep_idx = [
                idx
                for idx, t in enumerate(terms_g)
                if term_vad_conf.get(
                    t,
                    0.0,
                ) > 0.0
            ]
            if keep_idx:
                terms_g = [terms_g[idx] for idx in keep_idx]
                weights_g = [
                    weights_g[idx] for idx in keep_idx
                ]
                if term_weight_g is not None:
                    term_weight_g = [
                        term_weight_g[idx]
                        for idx in keep_idx
                    ]
            else:
                (
                    terms_g,
                    weights_g,
                ) = ([unk], [1.0])
                if term_weight_g is not None:
                    term_weight_g = [
                        float(term_weight_default)
                    ]
        group_terms[g] = terms_g
        group_weights[g] = weights_g
        if group_term_weights is not None:
            group_term_weights[g] = term_weight_g or [
                float(term_weight_default)
            ]
    vocab = sorted({t for g in selected_groups for t in group_terms[g]})
    need_embed = cost_mode in {'embedding', 'embedding_vad'}
    term_to_emb: Dict[str, torch.Tensor] = {}
    if need_embed:
        term_to_emb = _embed_terms(
            vocab,
            model_name=embed_model_name,
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
        )
    clouds: Dict[str, TokenCloud] = {}
    masses: Dict[str, float] = {}
    for g in selected_groups:
        terms_g = group_terms[g]
        weights_g = torch.tensor(
            group_weights[g],
            dtype=torch.float,
        ).clamp_min(1e-08)
        if need_embed:
            X = torch.stack(
                [term_to_emb[t] for t in terms_g],
                dim=0,
            ).to(dtype=torch.float)
        else:
            X = torch.zeros(
                (len(terms_g), 1),
                dtype=torch.float,
            )
        V = torch.stack(
            [
                term_vad.get(
                    t,
                    torch.zeros(
                        3,
                        dtype=torch.float,
                    ),
                )
                for t in terms_g
            ],
            dim=0,
        ).to(dtype=torch.float)
        V_conf = None
        if term_vad_conf:
            V_conf = torch.tensor(
                [
                    term_vad_conf.get(
                        t,
                        0.0,
                    )
                    for t in terms_g
                ],
                dtype=torch.float,
            )
        clouds[g] = TokenCloud(
            terms=terms_g,
            weights=weights_g,
            X=X,
            vad=V,
            vad_conf=V_conf,
        )
        masses[g] = float(weights_g.sum().item())
    (
        dist,
        explain_ab,
    ) = _pair_distance_and_explain(
        clouds[ga],
        clouds[gb],
        mode=str(opts.mode),
        cost=str(opts.cost),
        alpha_embed=float(opts.alpha_embed),
        beta_vad=float(opts.beta_vad),
        epsilon=float(getattr(
                setup,
                'sinkhorn_epsilon',
                0.1,
            )),
        iters=int(getattr(
                setup,
                'sinkhorn_iters',
                30,
            )),
        reg_m=float(getattr(
                setup,
                'ot_reg_m',
                0.1,
            )),
        top_flows=int(opts.top_flows),
        include_plan=True,
        device=device,
    )
    explain_ba = _reverse_primary_explain(explain_ab)
    docs_payload: List[Dict[str, Any]] = []
    for idx, g in enumerate(selected_groups):
        terms_g = group_terms[g]
        weights_g = group_weights[g]
        V = (
            clouds[g]
            .vad.detach()
            .cpu()
            .to(dtype=torch.float)
        )
        sources_out: List[str] = []
        sal: List[float] = []
        vad_out: List[List[float]] = []
        for t, v in zip(
            terms_g,
            V,
        ):
            if t == unk:
                sources_out.append('unk')
            elif t in lexicon_terms and t in word_vad_terms:
                sources_out.append('lexicon+vad_imputed'
                    if t in imputed_terms
                    else 'lexicon+vad')
            elif t in lexicon_terms:
                sources_out.append('lexicon')
            elif t in word_vad_terms:
                sources_out.append('vad_imputed'
                    if t in imputed_terms
                    else 'vad')
            else:
                sources_out.append('other')
            sal.append(float(_term_salience(v)))
            vad_out.append([float(x) for x in v.tolist()])
        raw_n = int(group_raw_tokens[g])
        sel_n = int(group_selected_tokens[g])
        docs_payload.append({
                'index': idx,
                'meta': {
                    'id': g,
                    'group_by': str(opts.group_by),
                    'group': g,
                    'n_docs': int(len(group_to_paths[g])),
                },
                'mass': float(masses[g]),
                'n_terms': int(len(terms_g)),
                'n_raw_tokens': raw_n,
                'n_selected_tokens': sel_n,
                'n_selected_unique': int(len(group_term_counts[g])),
                'selected_ratio': float(sel_n / max(
                        1.0,
                        float(raw_n),
                    )),
                'used_unk_fallback': bool(len(terms_g) == 1 and terms_g[0] == unk),
                'terms': list(terms_g),
                'weights': [float(x) for x in weights_g],
                'term_weight': (
                    [
                        float(x)
                        for x in (
                            group_term_weights or {}
                        ).get(
                            g,
                            [],
                        )
                    ]
                    if group_term_weights is not None
                    else None
                ),
                'term_vad': vad_out,
                'term_salience': sal,
                'term_source': sources_out,
                'term_vad_conf': (
                    [
                        float(term_vad_conf.get(
                                t,
                                0.0,
                            ))
                        for t in terms_g
                    ]
                    if term_vad_conf
                    else None
                ),
            })
    neighbours_payload = [
        {
            'index': 0,
            'neighbours': [
                {
                    'index': 1,
                    'distance': float(dist),
                    'mass_diff': float(abs(masses[ga] - masses[gb])),
                    'primary_explain': explain_ab,
                },
            ],
        },
        {
            'index': 1,
            'neighbours': [
                {
                    'index': 0,
                    'distance': float(dist),
                    'mass_diff': float(abs(masses[ga] - masses[gb])),
                    'primary_explain': explain_ba,
                },
            ],
        },
    ]
    report = {
        'format': 'neighbours',
        'mode': str(opts.mode),
        'focus': focus,
        'cost': str(opts.cost),
        'embed_model': (
            embed_model_name if need_embed else None
        ),
        'embed_backend': (
            embed_backend if need_embed else None
        ),
        'embed_pooling': (
            embed_pooling if need_embed else None
        ),
        'embed_batch_size': (
            int(embed_bs) if need_embed else None
        ),
        'embed_max_length': (
            int(embed_max_len) if need_embed else None
        ),
        'alpha_embed': float(opts.alpha_embed),
        'beta_vad': float(opts.beta_vad),
        'vad_threshold': float(opts.vad_threshold),
        'emotional_vocab': (
            str(opts.emotional_vocab)
            if focus != 'all'
            else None
        ),
        'vad_min_arousal_vad_only': (
            float(opts.vad_min_arousal_vad_only)
            if focus != 'all'
            else None
        ),
        'max_ngram': int(opts.max_ngram),
        'weight': str(opts.weight),
        'term_weight_path': (
            str(term_weights_path)
            if term_weights_path
            else None
        ),
        'term_weight_meta': term_weights_meta,
        'term_weight_power': float(term_weight_power),
        'term_weight_min': float(term_weight_min),
        'term_weight_max': float(term_weight_max),
        'term_weight_mix': float(term_weight_mix),
        'term_weight_default': float(term_weight_default),
        'vad_imputed_path': (
            str(vad_imputed_path)
            if vad_imputed_path
            else None
        ),
        'epsilon': float(getattr(
                setup,
                'sinkhorn_epsilon',
                0.1,
            )),
        'iters': int(getattr(
                setup,
                'sinkhorn_iters',
                30,
            )),
        'reg_m': float(getattr(
                setup,
                'ot_reg_m',
                0.1,
            )),
        'max_terms': int(opts.max_terms),
        'stopwords': not bool(opts.no_stopwords),
        'stopwords_file': (
            str(opts.stopwords_file)
            if opts.stopwords_file
            else None
        ),
        'drop_top_df': int(k_drop),
        'dropped_top_df_terms': list(dropped_top_df_terms),
        'docs': docs_payload,
        'neighbours': neighbours_payload,
        'topk': 1,
        'top_flows': int(opts.top_flows),
    }
    _write_json(
        out_dir / 'group_shift_report.json',
        report,
    )
    html_path = None
    if not opts.no_vis:
        html_path = write_token_ot_html_report(
            output_path=out_dir / 'group_shift.html',
            payload=report,
        )
    summary = {
        'out_dir': str(out_dir),
        'group_by': str(opts.group_by),
        'group_a': ga,
        'group_b': gb,
        'report_json': str(out_dir / 'group_shift_report.json'),
        'report_html': html_path,
    }
    _write_json(
        out_dir / 'summary.json',
        summary,
    )
    print(json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ))
    return 0


@torch.inference_mode()
def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )


if __name__ == '__main__':
    raise SystemExit(main())
