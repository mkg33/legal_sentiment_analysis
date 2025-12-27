from __future__ import annotations
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from .sentiment_config import (
    SentimentConfig,
    default_sentiment_config,
)
from .utils import get_device

_INV_LABEL = {
    0: -1,
    1: 0,
    2: 1,
}


def load_sentiment_model(
    checkpoint_path: str,
    cfg: SentimentConfig | None = None,
):
    try:
        data = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )
    except TypeError:
        data = torch.load(
            checkpoint_path,
            map_location='cpu',
        )
    cfg = cfg or default_sentiment_config()
    if isinstance(
        data,
        dict,
    ) and 'cfg' in data:
        saved_cfg = data['cfg']
        for k, v in saved_cfg.items():
            if hasattr(
                cfg,
                k,
            ):
                setattr(
                    cfg,
                    k,
                    v,
                )
        state = data['model']
    else:
        state = data
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=int(cfg.num_labels),
            ignore_mismatched_sizes=True,
        )
    )
    (
        missing,
        unexpected,
    ) = model.load_state_dict(
        state,
        strict=False,
    )
    if missing or unexpected:
        print('warn: checkpoint mismatch')
    model.eval()
    return (model, cfg)


def predict_sentiment(
    texts,
    checkpoint_path: str,
    cfg: SentimentConfig | None = None,
):
    (
        model,
        cfg,
    ) = load_sentiment_model(
        checkpoint_path,
        cfg,
    )
    device = get_device(cfg.device)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
    )
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(cfg.max_length),
        return_tensors='pt',
    )
    batch = {
        k: v.to(device)
        for k, v in enc.items()
        if torch.is_tensor(v)
    }
    amp_mode = str(getattr(
            cfg,
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
        pred = torch.argmax(
            probs,
            dim=-1,
        )
    guess_sent = torch.tensor(
        [
            _INV_LABEL.get(
                int(i),
                0,
            )
            for i in pred.cpu().tolist()
        ],
        dtype=torch.long,
    )
    return (probs.cpu(), guess_sent)
