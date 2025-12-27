from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from .corpus import (
    chunk_paragraphs_with_token_counts,
    iter_text_paths,
    read_text,
    split_paragraphs,
)
from .sentiment_config import load_sentiment_config
from .sentiment_predict import load_sentiment_model
from .utils import get_device

_INV_LABEL = {
    0: -1,
    1: 0,
    2: 1,
}


def score_txt_dir_sentiment(
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
    setup = load_sentiment_config(cfg_path)
    (
        model,
        setup,
    ) = load_sentiment_model(
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
    bs = int(batch_size or setup.batch_size or 8)
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
            desc='score_sentiment',
        ):
            text = read_text(p)
            paras = split_paragraphs(text)
            chunks = chunk_paragraphs_with_token_counts(
                paras,
                tokenizer,
                max_length=int(setup.max_length),
                stride=int(stride),
            )
            if not chunks:
                stats['skipped_empty'] += 1
                continue
            chunk_texts = [c.text for c in chunks]
            all_odds = []
            all_weights = []
            all_scales = []
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
                    max_length=int(setup.max_length),
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
                    logits = model(**batch).logits
                    probs = torch.softmax(
                        logits,
                        dim=-1,
                    )
                all_odds.append(probs.cpu())
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
            odds_doc = odds_doc / odds_doc.sum().clamp_min(1e-08)
            guess_idx = int(torch.argmax(
                    odds_doc,
                    dim=-1,
                ).item())
            guess_sent = int(_INV_LABEL.get(
                    guess_idx,
                    0,
                ))
            score = float(odds_doc[2].item() - odds_doc[0].item())
            row = {
                'meta': {'path': str(p)},
                'n_chunks': int(odds_chunks.size(0)),
                'probs': odds_doc.tolist(),
                'pred_label': guess_idx,
                'pred_sentiment': guess_sent,
                'sentiment_score': score,
            }
            out_f.write(json.dumps(
                    row,
                    ensure_ascii=False,
                ) + '\n')
            stats['written'] += 1
            stats['chunks'] += int(odds_chunks.size(0))
    return stats
