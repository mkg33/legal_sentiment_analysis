from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .sentiment_config import (
    SentimentConfig,
    load_sentiment_config,
)
from .sentiment_data import SentimentDataset
from .sentiment_predict import load_sentiment_model
from .utils import get_device


def _collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


def _per_class_metrics(cm: torch.Tensor) -> Dict[str, Any]:
    num = int(cm.size(0))
    per = []
    for c in range(num):
        tp = float(cm[c, c].item())
        fp = float(cm[:, c].sum().item() - tp)
        fn = float(cm[c, :].sum().item() - tp)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec + rec > 0
            else 0.0
        )
        support = int(cm[c, :].sum().item())
        per.append({
                'label': c,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'support': support,
            })
    macro_f1 = float(sum((m['f1'] for m in per)) / max(
            1,
            len(per),
        ))
    macro_precision = float(sum((m['precision'] for m in per))
        / max(
            1,
            len(per),
        ))
    macro_recall = float(sum((m['recall'] for m in per)) / max(
            1,
            len(per),
        ))
    return {
        'per_class': per,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
    }


def evaluate_sentiment_checkpoint(
    *,
    checkpoint: str,
    data_path: str,
    cfg_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    setup: SentimentConfig = load_sentiment_config(cfg_path)
    (
        model,
        setup,
    ) = load_sentiment_model(
        checkpoint,
        setup,
    )
    device = get_device(setup.device)
    model = model.to(device)
    ds = SentimentDataset(
        data_path,
        setup.model_name,
        max_length=setup.max_length,
    )
    if limit is not None:
        ds.samples = ds.samples[: max(
            0,
            int(limit),
        )]
    bs = int(batch_size
        or setup.eval_batch_size
        or setup.batch_size
        or 16)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=_collate,
    )
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
    total = 0
    total_loss = 0.0
    correct = 0
    guesses_all: list[int] = []
    gold_all: list[int] = []
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc='sentiment_eval',
            leave=False,
        ):
            inputs = {
                k: batch[k].to(device)
                for k in (
                    'input_ids',
                    'attention_mask',
                    'token_type_ids',
                )
                if k in batch
            }
            labels = batch['label'].to(device)
            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
                loss = F.cross_entropy(
                    logits,
                    labels,
                )
            pred = torch.argmax(
                logits,
                dim=-1,
            )
            bs = int(labels.size(0))
            total += bs
            total_loss += float(loss.item()) * bs
            correct += int((pred == labels).sum().item())
            guesses_all.extend(pred.detach().cpu().tolist())
            gold_all.extend(labels.detach().cpu().tolist())
    if total <= 0:
        return {'n_total': 0}
    num_labels = int(getattr(
            setup,
            'num_labels',
            3,
        ) or 3)
    cm = torch.zeros(
        (num_labels, num_labels),
        dtype=torch.long,
    )
    for y, p in zip(
        gold_all,
        guesses_all,
    ):
        if (
            0 <= int(y) < num_labels
            and 0 <= int(p) < num_labels
        ):
            cm[int(y), int(p)] += 1
    per = _per_class_metrics(cm.to(dtype=torch.float))
    metrics = {
        'checkpoint': str(checkpoint),
        'data_path': str(data_path),
        'eval_loss': total_loss / total,
        'accuracy': float(correct) / float(total),
        'macro_f1': float(per['macro_f1']),
        'macro_precision': float(per['macro_precision']),
        'macro_recall': float(per['macro_recall']),
        'n_total': int(total),
        'confusion_matrix': cm.tolist(),
        'per_class': per['per_class'],
    }
    return metrics


def write_json(
    path: str | Path,
    payload: Dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    p.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
