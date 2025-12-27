from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
from typing import Optional
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from .corpus import (
    chunk_paragraphs_with_token_counts,
    iter_text_paths,
    parse_icj_meta,
    read_text,
    split_paragraphs,
)
from .lexicon import (
    LexiconFeaturizer,
    load_lexicon,
    load_word_vad,
    resolve_vad_path,
    tokenize as lex_tokenize,
)
from .predict import load_model
from .utils import get_device, load_config


def _normalise_dist(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    total = x.sum()
    if total <= 0:
        return torch.full_like(
            x,
            1.0 / float(x.numel()),
        )
    return x / total


def _entropy(dist: torch.Tensor) -> float:
    n = int(dist.numel())
    if n <= 1:
        return 0.0
    p = dist.clamp_min(1e-08)
    return float((-(p * p.log()).sum()).item()
        / max(
            math.log(n),
            1e-08,
        ))


def _blend_counts_with_lexicon(
    pred_counts: torch.Tensor,
    lex_counts: torch.Tensor,
    lex_prior: torch.Tensor,
    *,
    n_words: int,
    strength_per_1k: float,
) -> tuple[torch.Tensor, float]:
    lex_mass = float(lex_counts.sum().item())
    guess_mass = float(pred_counts.sum().item())
    lex_per_1k = float(lex_mass / max(
            n_words,
            1,
        ) * 1000.0)
    k = max(
        float(strength_per_1k),
        1e-06,
    )
    lex_strength = (
        float(lex_per_1k / (lex_per_1k + k))
        if lex_per_1k > 0
        else 0.0
    )
    if lex_strength <= 0.0:
        return (pred_counts, 0.0)
    guess_dist = _normalise_dist(pred_counts)
    lex_dist = _normalise_dist(lex_prior
        if lex_prior.numel() == pred_counts.numel()
        else lex_counts)
    mix = _normalise_dist((1.0 - lex_strength) * guess_dist
        + lex_strength * lex_dist)
    mass_mix = (
        1.0 - lex_strength
    ) * guess_mass + lex_strength * lex_mass
    return (mix * float(mass_mix), lex_strength)


def _scale_lex_features(
    counts: torch.Tensor,
    prior: torch.Tensor,
    vad: torch.Tensor,
    *,
    new_ratio: float,
    has_lex: bool,
    has_lex_vad: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, float, float
]:
    ratio = float(new_ratio)
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    lex_mask = 1.0 if has_lex else 0.0
    vad_mask = 1.0 if has_lex_vad else 0.0
    counts = counts * (ratio * lex_mask)
    prior = prior * (ratio * lex_mask)
    vad = vad * (ratio * vad_mask)
    return (counts, prior, vad, lex_mask, vad_mask)


def score_txt_dir(
    *,
    checkpoint: str,
    input_dir: str,
    output_jsonl: str,
    cfg_path: Optional[str] = None,
    recursive: bool = True,
    limit: Optional[int] = None,
    stride: int = 0,
    batch_size: Optional[int] = None,
) -> dict:
    setup = load_config(cfg_path)
    (
        model,
        setup,
    ) = load_model(
        checkpoint,
        setup,
    )
    device = get_device(setup.device)
    model = model.to(device)
    amp_mode = str(getattr(
            setup,
            'amp',
            'none',
        ) or 'none').lower()
    autocast_device = (
        'cuda' if device.type == 'cuda' else 'cpu'
    )
    use_amp = device.type == 'cuda' and amp_mode in {
        'fp16',
        'bf16',
    }
    amp_dtype = (
        torch.float16
        if amp_mode == 'fp16'
        else torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        setup.model_name,
        use_fast=True,
    )
    lexicon = load_lexicon(
        setup.lexicon_path,
        setup.vad_lexicon_path,
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
    vad_terms = (
        load_word_vad(
            vad_path,
            vad_scale=getattr(
                setup,
                'word_vad_scale',
                None,
            ),
            stopwords_path=getattr(
                setup,
                'lexicon_stopwords_file',
                None,
            ),
        )
        if vad_path
        else {}
    )
    featurizer = LexiconFeaturizer(
        lexicon,
        setup.emotions,
        vad_lexicon=vad_terms,
        negation_window=getattr(
            setup,
            'lexicon_negation_window',
            0,
        ),
        negators=getattr(
            setup,
            'lexicon_negators',
            None,
        ),
        shared_term_weighting=getattr(
            setup,
            'lexicon_shared_term_weighting',
            'split',
        ),
    )
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
    count_is_density = count_scale in {
        'density',
        'per_token',
        'per_tok',
    }
    semantic_calibration = bool(getattr(
            setup,
            'semantic_calibration',
            True,
        ))
    if semantic_calibration and (
        not getattr(
            setup,
            'lexicon_path',
            None,
        )
    ):
        if not bool(getattr(
                setup,
                'semantic_calibration_allow_seed',
                False,
            )):
            warnings.warn(
                'warn: semantic_calibration',
                RuntimeWarning,
                stacklevel=2,
            )
            semantic_calibration = False
        else:
            warnings.warn(
                'warn: semantic_calibration seed',
                RuntimeWarning,
                stacklevel=2,
            )
    paths = list(iter_text_paths(
            input_dir,
            recursive=recursive,
            suffix='.txt',
        ))
    if limit is not None:
        paths = paths[: max(
            0,
            limit,
        )]
    bs = int(batch_size or setup.batch_size or 4)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    stats = {
        'docs': len(paths),
        'written': 0,
        'skipped_empty': 0,
        'chunks': 0,
    }
    with out_path.open(
        'w',
        encoding='utf-8',
    ) as out_f:
        for p in tqdm(
            paths,
            desc='score',
        ):
            text = read_text(p)
            n_words = len(lex_tokenize(text))
            paras = split_paragraphs(text)
            chunks = chunk_paragraphs_with_token_counts(
                paras,
                tokenizer,
                max_length=setup.max_length,
                stride=stride,
            )
            if not chunks:
                stats['skipped_empty'] += 1
                continue
            (
                lex_counts_full,
                lex_prior_full,
                lex_vad_full,
                lex_stats_full,
            ) = featurizer.vectors_with_stats(text)
            lex_hits_full = int(lex_stats_full.get(
                    'lex_hits',
                    0,
                ))
            lex_vad_hits_full = int(lex_stats_full.get(
                    'vad_hits',
                    0,
                ))
            chunk_texts = [c.text for c in chunks]
            lex_counts_list = []
            lex_prior_list = []
            lex_vad_list = []
            lex_mask_list = []
            lex_vad_mask_list = []
            lex_vad_hits_list = []
            for c in chunks:
                (
                    counts_raw,
                    prior_raw,
                    vad_raw,
                    lex_stats,
                ) = featurizer.vectors_with_stats(c.text)
                lex_hits = float(lex_stats.get(
                        'lex_hits',
                        0,
                    ))
                vad_hits = float(lex_stats.get(
                        'vad_hits',
                        0,
                    ))
                has_lex = lex_hits > 0
                has_lex_vad = vad_hits > 0
                lex_counts_list.append(counts_raw)
                lex_prior_list.append(prior_raw)
                lex_vad_list.append(vad_raw)
                lex_mask_list.append(1.0 if has_lex else 0.0)
                lex_vad_mask_list.append(1.0 if has_lex_vad else 0.0)
                lex_vad_hits_list.append(float(vad_hits))
            all_logits = []
            all_vad = []
            all_counts = []
            all_weights = []
            all_scales = []
            all_body_tokens = []
            special = tokenizer.num_special_tokens_to_add(pair=False)
            for start in range(
                0,
                len(chunk_texts),
                bs,
            ):
                end = min(
                    len(chunk_texts),
                    start + bs,
                )
                batch_texts = chunk_texts[start:end]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=setup.max_length,
                    return_tensors='pt',
                )
                attn_tokens = (
                    enc['attention_mask']
                    .sum(dim=1)
                    .to(dtype=torch.float)
                )
                body_tokens = (
                    attn_tokens - float(special)
                ).clamp_min(1.0)
                new_tokens = torch.tensor(
                    [
                        c.new_body_tokens
                        for c in chunks[start:end]
                    ],
                    dtype=torch.float,
                ).clamp_min(1.0)
                new_tokens = torch.minimum(
                    new_tokens,
                    body_tokens,
                )
                all_weights.append(new_tokens)
                all_scales.append(new_tokens / body_tokens)
                all_body_tokens.append(body_tokens)
                batch = {
                    k: v.to(device)
                    for k, v in enc.items()
                    if torch.is_tensor(v)
                }
                batch['lex_counts'] = torch.stack(lex_counts_list[start:end]).to(device)
                batch['lex_prior'] = torch.stack(lex_prior_list[start:end]).to(device)
                batch['lex_vad'] = torch.stack(lex_vad_list[start:end]).to(device)
                batch['lex_mask'] = torch.tensor(
                    lex_mask_list[start:end],
                    dtype=torch.float,
                ).to(device)
                batch['lex_vad_mask'] = torch.tensor(
                    lex_vad_mask_list[start:end],
                    dtype=torch.float,
                ).to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type=autocast_device,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    (
                        logits,
                        vad_guess,
                        count_guess,
                    ) = model(**batch)
                all_logits.append(logits.cpu())
                all_vad.append(vad_guess.cpu())
                all_counts.append(count_guess.cpu())
            logits = torch.cat(
                all_logits,
                dim=0,
            )
            vad_guess = torch.cat(
                all_vad,
                dim=0,
            )
            count_guess = torch.cat(
                all_counts,
                dim=0,
            )
            weights = torch.cat(
                all_weights,
                dim=0,
            ).clamp_min(1.0)
            scales = (
                torch.cat(
                    all_scales,
                    dim=0,
                )
                .clamp_min(0.0)
                .clamp_max(1.0)
            )
            body_tokens_all = torch.cat(
                all_body_tokens,
                dim=0,
            ).clamp_min(1.0)
            wsum = weights.sum().clamp_min(1.0)
            guess_sigmoid = torch.sigmoid(logits)
            guess_softmax = torch.softmax(
                logits,
                dim=-1,
            )
            guess_sigmoid_doc = (
                guess_sigmoid * weights[:, None]
            ).sum(dim=0) / wsum
            guess_softmax_doc = (
                guess_softmax * weights[:, None]
            ).sum(dim=0) / wsum
            guess_vad_doc = (
                vad_guess * weights[:, None]
            ).sum(dim=0) / wsum
            use_count_dist = (
                str(getattr(
                        setup,
                        'ot_mode',
                        '',
                    ) or '')
                .lower()
                .startswith('unbalanced')
            )
            alpha_mass = float(getattr(
                    setup,
                    'alpha_mass',
                    0.0,
                ) or 0.0)
            use_count_mass = alpha_mass > 0.0
            guess_counts_source = 'count_head'
            if use_count_dist:
                if count_is_density:
                    guess_counts_chunk = (
                        count_guess
                        * body_tokens_all[:, None]
                    )
                else:
                    guess_counts_chunk = count_guess
                guess_counts_chunk = (
                    guess_counts_chunk.clamp_min(0.0)
                )
            else:
                dist_denom = guess_sigmoid.sum(
                    dim=-1,
                    keepdim=True,
                )
                guess_dist = (
                    guess_sigmoid
                    / dist_denom.clamp_min(1e-08)
                )
                if torch.any(dist_denom <= 1e-08):
                    uniform = torch.full_like(
                        guess_sigmoid,
                        1.0 / guess_sigmoid.size(-1),
                    )
                    guess_dist = torch.where(
                        dist_denom <= 1e-08,
                        uniform,
                        guess_dist,
                    )
                if use_count_mass:
                    if count_is_density:
                        guess_mass_chunk = (
                            (
                                count_guess
                                * body_tokens_all[:, None]
                            )
                            .sum(
                                dim=-1,
                                keepdim=True,
                            )
                            .clamp_min(1e-08)
                        )
                    else:
                        guess_mass_chunk = count_guess.sum(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(1e-08)
                    guess_counts_chunk = (
                        guess_dist * guess_mass_chunk
                    ).clamp_min(0.0)
                else:
                    guess_counts_source = (
                        'sigmoid_token_scaled'
                    )
                    guess_counts_chunk = (
                        guess_sigmoid
                        * body_tokens_all[:, None]
                    ).clamp_min(0.0)
            guess_counts_doc = (
                guess_counts_chunk * scales[:, None]
            ).sum(dim=0)
            guess_mass = guess_counts_doc.sum().item()
            n_tokens = int(wsum.item())
            guess_density = float(guess_mass / max(
                    n_tokens,
                    1,
                ))
            guess_density_words = float(guess_mass / max(
                    n_words,
                    1,
                ))
            guess_mixscaled_per_1k_words = (
                guess_counts_doc / max(
                    n_words,
                    1,
                ) * 1000.0
            ).tolist()
            guess_mixscaled_per_1k_tokens = (
                guess_counts_doc / max(
                    n_tokens,
                    1,
                ) * 1000.0
            ).tolist()
            lex_mask_tensor = torch.tensor(
                lex_mask_list,
                dtype=torch.float,
            )
            lex_chunk_weighted_sum = float((lex_mask_tensor * scales).sum().item())
            lex_chunk_total_weight = float(scales.sum().item())
            lex_chunk_ratio = float(lex_chunk_weighted_sum
                / max(
                    lex_chunk_total_weight,
                    1e-08,
                ))
            lex_hit_chunks = int((lex_mask_tensor > 0).sum().item())
            lex_counts_chunk = (
                torch.stack(
                    lex_counts_list,
                    dim=0,
                )
                * scales[:, None]
            ).sum(dim=0)
            lex_prior_chunk = _normalise_dist(lex_counts_chunk)
            vad_hits_tensor = torch.tensor(
                lex_vad_hits_list,
                dtype=torch.float,
            ).clamp_min(0.0)
            vad_hits_weighted = vad_hits_tensor * scales
            vad_hits_weighted_sum = float(vad_hits_weighted.sum().item())
            if vad_hits_weighted_sum > 0.0:
                lex_vad_chunk = (
                    torch.stack(
                        lex_vad_list,
                        dim=0,
                    )
                    * vad_hits_weighted[:, None]
                ).sum(dim=0) / vad_hits_weighted_sum
            else:
                lex_vad_chunk = torch.zeros(
                    3,
                    dtype=torch.float,
                )
            lex_basis = (
                str(getattr(
                        setup,
                        'semantic_lexicon_basis',
                        'chunks',
                    )
                    or 'chunks')
                .lower()
                .strip()
            )
            if lex_basis not in {'chunks', 'full'}:
                warnings.warn(
                    'warn: lexicon_basis',
                    RuntimeWarning,
                    stacklevel=2,
                )
                lex_basis = 'chunks'
            if lex_basis == 'chunks':
                lex_counts_doc = lex_counts_chunk
                lex_prior_doc = lex_prior_chunk
                lex_vad_doc = lex_vad_chunk
                lex_hits_doc_weighted = float(lex_counts_chunk.sum().item())
                lex_vad_hits_doc_weighted = float(vad_hits_weighted_sum)
            else:
                lex_counts_doc = lex_counts_full
                lex_prior_doc = lex_prior_full
                lex_vad_doc = lex_vad_full
                lex_hits_doc_weighted = float(lex_counts_full.sum().item())
                lex_vad_hits_doc_weighted = float(lex_vad_hits_full)
            lex_hits_doc = int(lex_hits_full)
            lex_vad_hits_doc = int(lex_vad_hits_full)
            lex_mass = float(lex_counts_doc.sum().item())
            lex_density = float(lex_mass / max(
                    n_tokens,
                    1,
                ))
            lex_density_words = float(lex_mass / max(
                    n_words,
                    1,
                ))
            lex_per_1k_words = float(lex_density_words * 1000.0)
            lex_per_1k_tokens = float(lex_density * 1000.0)
            guess_dist_doc = _normalise_dist(guess_counts_doc)
            guess_entropy = _entropy(guess_dist_doc)
            guess_counts_calibrated = guess_counts_doc
            lex_strength = 0.0
            if semantic_calibration:
                (
                    guess_counts_calibrated,
                    lex_strength,
                ) = (
                    _blend_counts_with_lexicon(
                        guess_counts_doc,
                        lex_counts_doc,
                        lex_prior_doc,
                        n_words=int(n_words),
                        strength_per_1k=float(getattr(
                                setup,
                                'semantic_lexicon_strength_per_1k',
                                3.0,
                            )),
                    )
                )
            guess_dist_calibrated = _normalise_dist(guess_counts_calibrated)
            guess_entropy_calibrated = _entropy(guess_dist_calibrated)
            guess_calib_mass = float(guess_counts_calibrated.sum().item())
            guess_calib_per_1k_words = float(guess_calib_mass / max(
                    n_words,
                    1,
                ) * 1000.0)
            guess_calib_per_1k_tokens = float(guess_calib_mass / max(
                    n_tokens,
                    1,
                ) * 1000.0)
            guess_dist_calibrated_valid = True
            min_signal = float(getattr(
                    setup,
                    'semantic_min_signal_per_1k_words',
                    0.0,
                )
                or 0.0)
            entropy_cutoff = float(getattr(
                    setup,
                    'semantic_low_signal_entropy_ratio',
                    0.9,
                )
                or 0.9)
            entropy_cutoff = min(
                max(
                    entropy_cutoff,
                    0.0,
                ),
                1.0,
            )
            strict_lex_gate = bool(getattr(
                    setup,
                    'semantic_strict_lex_gate',
                    False,
                ))
            min_lex_hits = int(getattr(
                    setup,
                    'semantic_min_lex_hits',
                    0,
                )
                or 0)
            min_lex_chunk_ratio = float(getattr(
                    setup,
                    'semantic_min_lex_chunk_ratio',
                    0.0,
                )
                or 0.0)
            lex_hits_ok = (
                True
                if min_lex_hits <= 0
                else lex_hits_doc >= min_lex_hits
            )
            lex_chunk_ok = (
                True
                if min_lex_chunk_ratio <= 0.0
                else lex_chunk_ratio >= min_lex_chunk_ratio
            )
            lex_density_ok = (
                True
                if min_signal <= 0.0
                else lex_per_1k_words >= min_signal
            )
            strict_low_signal = bool(strict_lex_gate
                and (
                    not (
                        lex_hits_ok
                        and lex_chunk_ok
                        and lex_density_ok
                    )
                ))
            low_feeling_signal = bool(min_signal > 0.0
                and lex_per_1k_words < min_signal
                and (
                    guess_entropy_calibrated
                    >= entropy_cutoff
                ))
            if strict_low_signal:
                low_feeling_signal = True
            if low_feeling_signal and bool(getattr(
                    setup,
                    'semantic_zero_low_signal',
                    True,
                )):
                guess_counts_calibrated = torch.zeros_like(guess_counts_doc)
                guess_dist_calibrated = torch.zeros_like(guess_dist_doc)
                guess_entropy_calibrated = 0.0
                guess_calib_mass = 0.0
                guess_calib_per_1k_words = 0.0
                guess_calib_per_1k_tokens = 0.0
                guess_dist_calibrated_valid = False
            feeling_signal_per_1k_words = float(max(
                    guess_calib_per_1k_words,
                    lex_per_1k_words,
                ))
            feeling_signal_source = (
                'pred'
                if guess_calib_per_1k_words
                >= lex_per_1k_words
                else 'lex'
            )
            row = {
                'meta': {
                    'path': str(p),
                    **parse_icj_meta(p),
                },
                'n_chunks': int(logits.size(0)),
                'emotions': list(setup.emotions),
                'pred_sigmoid': guess_sigmoid_doc.tolist(),
                'pred_softmax': guess_softmax_doc.tolist(),
                'pred_dist': guess_dist_doc.tolist(),
                'pred_entropy': float(guess_entropy),
                'pred_vad': guess_vad_doc.tolist(),
                'pred_counts': guess_counts_doc.tolist(),
                'pred_counts_source': guess_counts_source,
                'pred_mass': float(guess_mass),
                'pred_density': float(guess_density),
                'pred_per_1k_tokens': float(guess_density * 1000.0),
                'pred_density_words': float(guess_density_words),
                'pred_per_1k_words': float(guess_density_words * 1000.0),
                'pred_mixscaled_per_1k_words': guess_mixscaled_per_1k_words,
                'pred_mixscaled_per_1k_tokens': guess_mixscaled_per_1k_tokens,
                'pred_counts_calibrated': guess_counts_calibrated.tolist(),
                'pred_dist_calibrated': guess_dist_calibrated.tolist(),
                'pred_entropy_calibrated': float(guess_entropy_calibrated),
                'pred_dist_calibrated_valid': bool(guess_dist_calibrated_valid),
                'pred_counts_calibrated_mass': float(guess_calib_mass),
                'pred_counts_calibrated_per_1k_words': float(guess_calib_per_1k_words),
                'pred_counts_calibrated_per_1k_tokens': float(guess_calib_per_1k_tokens),
                'lex_strength': float(lex_strength),
                'emotion_signal_per_1k_words': float(feeling_signal_per_1k_words),
                'emotion_signal_source': feeling_signal_source,
                'low_emotion_signal': bool(low_feeling_signal),
                'strict_low_signal': bool(strict_low_signal),
                'n_tokens': n_tokens,
                'n_words': int(n_words),
                'lex_basis': lex_basis,
                'lex_counts': lex_counts_doc.tolist(),
                'lex_counts_chunk': lex_counts_chunk.tolist(),
                'lex_counts_full': lex_counts_full.tolist(),
                'lex_mass': float(lex_mass),
                'lex_density': float(lex_density),
                'lex_per_1k_tokens': float(lex_per_1k_tokens),
                'lex_density_words': float(lex_density_words),
                'lex_per_1k_words': float(lex_per_1k_words),
                'lex_chunk_ratio': float(lex_chunk_ratio),
                'lex_hit_chunks': int(lex_hit_chunks),
                'lex_prior': lex_prior_doc.tolist(),
                'lex_prior_chunk': lex_prior_chunk.tolist(),
                'lex_prior_full': lex_prior_full.tolist(),
                'lex_vad': lex_vad_doc.tolist(),
                'lex_vad_chunk': lex_vad_chunk.tolist(),
                'lex_vad_full': lex_vad_full.tolist(),
                'lex_hits': int(lex_hits_doc),
                'lex_hits_weighted': float(lex_hits_doc_weighted),
                'lex_hits_weighted_chunk': float(lex_counts_chunk.sum().item()),
                'lex_hits_full': int(lex_hits_full),
                'lex_vad_hits': int(lex_vad_hits_doc),
                'lex_vad_hits_weighted': float(lex_vad_hits_doc_weighted),
                'lex_vad_hits_weighted_chunk': float(vad_hits_weighted_sum),
                'lex_vad_hits_full': int(lex_vad_hits_full),
            }
            out_f.write(json.dumps(
                    row,
                    ensure_ascii=False,
                ) + '\n')
            stats['written'] += 1
            stats['chunks'] += int(logits.size(0))
    return stats
