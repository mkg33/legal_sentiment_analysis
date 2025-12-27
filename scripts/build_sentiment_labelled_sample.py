#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import (
    chunk_paragraphs_with_token_counts,
    iter_text_paths,
    read_text,
    split_paragraphs,
)
from legal_emotion.sentiment_config import (
    load_sentiment_config,
)
from legal_emotion.sentiment_predict import (
    load_sentiment_model,
)
from legal_emotion.utils import get_device

_SENT_TO_CATEGORY = {
    -1: 'neg',
    0: 'neu',
    1: 'pos',
}
_TERTILE_CATEGORIES = ('neg', 'neu', 'pos')


def _write_jsonl(
    path: Path,
    rows: List[Dict[str, Any]],
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


def _infer_paths_from_run_dir(run_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    ckpt = (
        run_dir / 'checkpoints_sentiment' / 'sentiment.pt'
    )
    setup = run_dir / 'sentiment_config.resolved.yaml'
    return (
        ckpt if ckpt.exists() else None,
        setup if setup.exists() else None,
    )


def _score_doc(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    max_length: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    batch_size: int,
) -> Optional[Dict[str, Any]]:
    paras = split_paragraphs(text)
    chunks = chunk_paragraphs_with_token_counts(
        paras,
        tokenizer,
        max_length=int(max_length),
        stride=int(stride),
    )
    if not chunks:
        return None
    chunk_texts = [c.text for c in chunks]
    all_odds: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []
    special = tokenizer.num_special_tokens_to_add(pair=False)
    for start in range(
        0,
        len(chunk_texts),
        int(batch_size),
    ):
        end = min(
            len(chunk_texts),
            start + int(batch_size),
        )
        batch_texts = chunk_texts[start:end]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
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
            [c.new_body_tokens for c in chunks[start:end]],
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
            device_type=(
                'cuda' if device.type == 'cuda' else 'cpu'
            ),
            dtype=amp_dtype,
            enabled=bool(use_amp),
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
    inv_label = {
        0: -1,
        1: 0,
        2: 1,
    }
    guess_sent = int(inv_label.get(
            guess_idx,
            0,
        ))
    score = float(odds_doc[2].item() - odds_doc[0].item())
    return {
        'n_chunks': int(odds_chunks.size(0)),
        'probs': odds_doc.tolist(),
        'pred_label': guess_idx,
        'pred_sentiment': guess_sent,
        'sentiment_score': score,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='labelled sample')
    ap.add_argument(
        '--input_dir',
        type=str,
        required=True,
    )
    ap.add_argument(
        '--output_jsonl',
        type=str,
        required=True,
    )
    ap.add_argument(
        '--n_docs',
        type=int,
        default=500,
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=13,
    )
    ap.add_argument(
        '--category_mode',
        type=str,
        default='pred',
        choices=('pred', 'score_tertiles'),
    )
    ap.add_argument(
        '--run_dir',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--checkpoint',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--cfg_path',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--stride',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--batch_size',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--limit_paths',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--recursive',
        dest='recursive',
        action='store_true',
        default=True,
    )
    ap.add_argument(
        '--no_recursive',
        dest='recursive',
        action='store_false',
    )
    ap.add_argument(
        '--balance',
        dest='balance',
        action='store_true',
        default=True,
    )
    ap.add_argument(
        '--no_balance',
        dest='balance',
        action='store_false',
    )
    ap.add_argument(
        '--strict_balance',
        action='store_true',
    )
    opts = ap.parse_args()

    def log(msg: str) -> None:
        print(
            msg,
            file=sys.stderr,
            flush=True,
        )

    run_dir = (
        Path(opts.run_dir).resolve()
        if opts.run_dir
        else None
    )
    inferred_ckpt = None
    inferred_cfg = None
    if run_dir is not None:
        (
            inferred_ckpt,
            inferred_cfg,
        ) = (
            _infer_paths_from_run_dir(run_dir)
        )
    ckpt_path = (
        Path(opts.checkpoint).resolve()
        if opts.checkpoint
        else inferred_ckpt
    )
    cfg_path = (
        Path(opts.cfg_path).resolve()
        if opts.cfg_path
        else inferred_cfg
    )
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit('error: SystemExit')
    setup = load_sentiment_config(str(cfg_path)
        if cfg_path is not None and cfg_path.exists()
        else None)
    (
        model,
        setup,
    ) = load_sentiment_model(
        str(ckpt_path),
        setup,
    )
    device = get_device(setup.device)
    model = model.to(device)
    amp_mode = str(getattr(
            setup,
            'amp',
            'none',
        ) or 'none').lower()
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
    bs = int(opts.batch_size or setup.batch_size or 8)
    max_length = int(getattr(
            setup,
            'max_length',
            256,
        ) or 256)
    paths = list(iter_text_paths(
            opts.input_dir,
            recursive=bool(opts.recursive),
            suffix='.txt',
        ))
    if opts.limit_paths is not None:
        paths = paths[: max(
            0,
            int(opts.limit_paths),
        )]
    if not paths:
        raise SystemExit('error: SystemExit')
    rng = random.Random(int(opts.seed))
    rng.shuffle(paths)
    labels = ['neg', 'neu', 'pos']
    n_docs = max(
        1,
        int(opts.n_docs),
    )
    base = n_docs // len(labels)
    rem = n_docs % len(labels)
    targets = {
        lab: int(base + (1 if i < rem else 0))
        for i, lab in enumerate(labels)
    }
    buckets: Dict[str, List[Dict[str, Any]]] = {
        lab: [] for lab in labels
    }
    extras: List[Dict[str, Any]] = []
    log(f'ckpt {ckpt_path}')
    if cfg_path is not None and cfg_path.exists():
        log(f'cfg {cfg_path}')
    log(f'input {opts.input_dir} n={n_docs} mode={opts.category_mode}')
    scored = 0
    skipped_empty = 0
    scored_rows: List[Dict[str, Any]] = []
    mode = str(opts.category_mode).strip().lower()
    for p in paths:
        have = (
            sum((len(v) for v in buckets.values()))
            if mode == 'pred'
            else len(scored_rows)
        )
        if have >= n_docs:
            break
        text = read_text(p)
        out_2 = _score_doc(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_length=max_length,
            stride=int(opts.stride),
            device=device,
            use_amp=bool(use_amp),
            amp_dtype=amp_dtype,
            batch_size=bs,
        )
        scored += 1
        if out_2 is None:
            skipped_empty += 1
            continue
        if mode == 'pred':
            cat = _SENT_TO_CATEGORY.get(
                int(out_2['pred_sentiment']),
                'neu',
            )
        else:
            cat = ''
        row = {
            'meta': {
                'id': p.stem,
                'category': cat,
                'path': str(p),
                'source': 'ICJ',
                'label_source': (
                    'sentiment_model'
                    if mode == 'pred'
                    else 'sentiment_model_score_tertiles'
                ),
            },
            **out_2,
        }
        if mode != 'pred':
            scored_rows.append(row)
            continue
        if not bool(opts.balance):
            buckets.setdefault(
                cat,
                [],
            ).append(row)
            continue
        if cat in buckets and len(buckets[cat]) < targets.get(
            cat,
            0,
        ):
            buckets[cat].append(row)
        else:
            extras.append(row)
    if mode != 'pred':
        if len(scored_rows) < n_docs:
            raise SystemExit('error: SystemExit')
        scored_rows = scored_rows[:n_docs]
        order = sorted(
            range(len(scored_rows)),
            key=lambda i: float(scored_rows[i].get('sentiment_score') or 0.0),
        )
        cut1 = len(order) // 3
        cut2 = 2 * len(order) // 3
        for rank, idx in enumerate(order):
            if rank < cut1:
                cat = _TERTILE_CATEGORIES[0]
            elif rank < cut2:
                cat = _TERTILE_CATEGORIES[1]
            else:
                cat = _TERTILE_CATEGORIES[2]
            meta = (
                scored_rows[idx].get('meta')
                if isinstance(
                    scored_rows[idx].get('meta'),
                    dict,
                )
                else {}
            )
            meta = dict(meta)
            meta['category'] = cat
            scored_rows[idx]['meta'] = meta
        selected = scored_rows
    else:
        selected = []
        for lab in labels:
            selected.extend(buckets.get(
                    lab,
                    [],
                ))
    if (
        mode == 'pred'
        and bool(opts.balance)
        and (len(selected) < n_docs)
    ):
        missing = n_docs - len(selected)
        if bool(opts.strict_balance):
            counts = {k: len(v) for k, v in buckets.items()}
            raise SystemExit('error: SystemExit')
        if missing > 0 and extras:
            selected.extend(extras[:missing])
    if len(selected) < n_docs:
        raise SystemExit('error: SystemExit')
    selected = selected[:n_docs]
    out = Path(opts.output_jsonl).resolve()
    _write_jsonl(
        out,
        selected,
    )
    counts = Counter((
            (r.get('meta') or {}).get('category')
            or 'unknown'
            for r in selected
        ))
    payload = {
        'output_jsonl': str(out),
        'n_docs': int(n_docs),
        'seed': int(opts.seed),
        'balance': (
            bool(opts.balance) if mode == 'pred' else None
        ),
        'targets': targets if bool(opts.balance) else None,
        'counts': dict(counts),
        'scored': int(scored),
        'skipped_empty': int(skipped_empty),
        'category_mode': str(opts.category_mode),
        'checkpoint': str(ckpt_path),
        'cfg_path': (
            str(cfg_path)
            if cfg_path is not None and cfg_path.exists()
            else None
        ),
    }
    print(json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
