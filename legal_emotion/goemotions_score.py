from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from .corpus import (
    chunk_paragraphs_with_token_counts,
    iter_text_paths,
    parse_icj_meta,
    read_text,
    split_paragraphs,
)
from .lexicon import (
    load_stopwords,
    tokenize as lex_tokenize,
)
from .utils import get_device, load_config

_LABEL_SPLIT_RE = re.compile('[\\s\\-]+')
GOEMOTIONS_MAP: Dict[str, Sequence[str]] = {
    'anger': ('anger', 'annoyance', 'disapproval'),
    'fear': ('fear', 'nervousness'),
    'joy': ('joy', 'amusement', 'excitement', 'relief'),
    'sadness': (
        'sadness',
        'grief',
        'disappointment',
        'remorse',
    ),
    'trust': (
        'admiration',
        'approval',
        'caring',
        'gratitude',
        'love',
        'pride',
    ),
    'disgust': ('disgust', 'embarrassment'),
    'surprise': ('surprise', 'realization', 'confusion'),
    'anticipation': ('desire', 'curiosity', 'optimism'),
}


def _normalise_label(label: str) -> str:
    return _LABEL_SPLIT_RE.sub(
        '_',
        str(label).strip().lower(),
    )


def _label_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    idx_map: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        key = _normalise_label(label)
        idx_map.setdefault(
            key,
            [],
        ).append(int(i))
    return idx_map


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


def _normalise_dist(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    total = x.sum().clamp_min(0.0)
    if float(total.item()) <= 1e-08:
        return torch.full_like(
            x,
            1.0 / float(x.numel()),
        )
    return x / total


def _combine_probs_1d(
    probs: torch.Tensor,
    idxs: Sequence[int],
) -> torch.Tensor:
    if not idxs:
        return torch.tensor(
            0.0,
            device=probs.device,
            dtype=probs.dtype,
        )
    vals = probs[list(idxs)]
    return 1.0 - torch.prod(1.0 - vals)


def _compile_stopphrase_patterns(stopwords_file: Optional[str]) -> List[re.Pattern[str]]:
    if not stopwords_file:
        return []
    try:
        stop = load_stopwords(stopwords_file)
    except Exception:
        return []
    phrases = [p for p in stop if ' ' in p]
    if not phrases:
        return []
    patterns: List[re.Pattern[str]] = []
    for phrase in sorted(
        phrases,
        key=len,
        reverse=True,
    ):
        toks = phrase.split()
        if len(toks) < 2:
            continue
        pat = (
            '\\b'
            + '\\s+'.join((re.escape(tok) for tok in toks))
            + '\\b'
        )
        patterns.append(re.compile(
                pat,
                flags=re.IGNORECASE,
            ))
    return patterns


def score_txt_dir_goemotions(
    *,
    input_dir: str,
    output_jsonl: str,
    cfg_path: Optional[str] = None,
    model_name: Optional[str] = None,
    recursive: bool = True,
    limit: Optional[int] = None,
    stride: int = 0,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    intensity_scale_per_1k_words: float = 20.0,
) -> dict:
    setup = load_config(cfg_path)
    teacher = (
        model_name
        or getattr(
            setup,
            'silver_teacher_model',
            None,
        )
        or 'SamLowe/roberta-base-go_emotions'
    )
    bs = int(batch_size
        or getattr(
            setup,
            'silver_teacher_batch_size',
            None,
        )
        or setup.batch_size
        or 16)
    max_len = int(max_length
        or getattr(
            setup,
            'silver_teacher_max_length',
            None,
        )
        or setup.max_length
        or 512)
    stopwords_file = getattr(
        setup,
        'lexicon_stopwords_file',
        None,
    )
    stopphrase_patterns = _compile_stopphrase_patterns(str(stopwords_file) if stopwords_file else None)
    device = get_device(getattr(
            setup,
            'device',
            'cpu',
        ))
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
        teacher,
        use_fast=True,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(teacher)
    )
    model.eval()
    model.to(device)
    labels = [
        str(model.config.id2label[i])
        for i in range(int(model.config.num_labels))
    ]
    idx_map = _label_indices(labels)
    emotions: List[str] = list(getattr(
            setup,
            'emotions',
            None,
        )
        or list(GOEMOTIONS_MAP.keys()))
    out_feelings = [
        e for e in emotions if str(e) in GOEMOTIONS_MAP
    ]
    if not out_feelings:
        out_feelings = list(GOEMOTIONS_MAP.keys())
    paths = list(iter_text_paths(
            input_dir,
            recursive=recursive,
            suffix='.txt',
        ))
    if limit is not None:
        paths = paths[: max(
            0,
            int(limit),
        )]
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
        'teacher_model': str(teacher),
    }
    if stopphrase_patterns:
        stats['stopphrases_file'] = str(stopwords_file)
        stats['stopphrase_patterns'] = int(len(stopphrase_patterns))
    with out_path.open(
        'w',
        encoding='utf-8',
    ) as out_f:
        for path in tqdm(
            paths,
            desc='goemotions_score',
        ):
            text = read_text(path)
            stopphrase_hits = 0
            if stopphrase_patterns:
                for pat in stopphrase_patterns:
                    (
                        text,
                        n,
                    ) = pat.subn(
                        ' ',
                        text,
                    )
                    stopphrase_hits += int(n)
            paras = split_paragraphs(text)
            chunks = chunk_paragraphs_with_token_counts(
                paras,
                tokenizer,
                max_length=int(max_len),
                stride=int(stride),
            )
            if not chunks:
                stats['skipped_empty'] += 1
                continue
            chunk_texts = [c.text for c in chunks]
            all_odds: List[torch.Tensor] = []
            all_weights: List[torch.Tensor] = []
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
                    max_length=int(max_len),
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
                batch = {
                    k: v.to(device)
                    for k, v in enc.items()
                    if torch.is_tensor(v)
                }
                with torch.inference_mode(), torch.autocast(
                    device_type=autocast_device,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits = model(
                        **batch,
                        return_dict=True,
                    ).logits
                    probs = torch.sigmoid(logits)
                all_odds.append(probs.detach().cpu())
            odds_chunks = torch.cat(
                all_odds,
                dim=0,
            )
            weights = torch.cat(
                all_weights,
                dim=0,
            ).clamp_min(1.0)
            wsum = weights.sum().clamp_min(1.0)
            odds_doc = (odds_chunks * weights[:, None]).sum(dim=0) / wsum
            odds_doc = odds_doc.clamp(
                0.0,
                1.0,
            )
            target = torch.zeros(
                len(out_feelings),
                dtype=torch.float,
            )
            for emo_idx, emo in enumerate(out_feelings):
                src_labels = GOEMOTIONS_MAP.get(
                    str(emo),
                    (),
                )
                idxs: List[int] = []
                for lbl in src_labels:
                    idxs.extend(idx_map.get(
                            _normalise_label(lbl),
                            [],
                        ))
                target[emo_idx] = _combine_probs_1d(
                    odds_doc,
                    idxs,
                )
            dist = _normalise_dist(target)
            entropy = _entropy(dist)
            n_tokens = int(wsum.item())
            n_words = int(len(lex_tokenize(text)))
            scale_words = float(intensity_scale_per_1k_words)
            scale_tokens = float(scale_words) * (
                float(n_words) / float(max(
                        n_tokens,
                        1,
                    ))
            )
            per_1k_words_vec = (
                target * scale_words
            ).tolist()
            per_1k_tokens_vec = (
                target * scale_tokens
            ).tolist()
            per_1k_words_total = float(sum(per_1k_words_vec))
            per_1k_tokens_total = float(sum(per_1k_tokens_vec))
            row = {
                'meta': {
                    'path': str(path),
                    **parse_icj_meta(path),
                    'goemotions_teacher_model': str(teacher),
                    'goemotions_stopphrase_hits': int(stopphrase_hits),
                },
                'n_chunks': int(odds_chunks.size(0)),
                'emotions': list(out_feelings),
                'pred_sigmoid': target.tolist(),
                'pred_dist': dist.tolist(),
                'pred_entropy': float(entropy),
                'pred_mixscaled_per_1k_words': per_1k_words_vec,
                'pred_mixscaled_per_1k_tokens': per_1k_tokens_vec,
                'pred_per_1k_words': float(per_1k_words_total),
                'pred_per_1k_tokens': float(per_1k_tokens_total),
                'emotion_signal_per_1k_words': float(per_1k_words_total),
                'low_emotion_signal': False,
                'n_tokens': int(n_tokens),
                'n_words': int(n_words),
            }
            out_f.write(json.dumps(
                    row,
                    ensure_ascii=False,
                ) + '\n')
            stats['written'] += 1
            stats['chunks'] += int(odds_chunks.size(0))
    return stats
