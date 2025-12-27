from __future__ import annotations
from dataclasses import dataclass
import yaml
from .utils import _coerce_to_default_type


@dataclass
class SentimentConfig:
    model_name: str = 'nlpaueb/legal-bert-base-uncased'
    max_length: int = 256
    batch_size: int = 16
    eval_batch_size: int | None = None
    lr: float = 2e-05
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 0
    device: str = 'auto'
    amp: str = 'none'
    tf32: bool = True
    grad_accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    compile: bool = False
    gradient_checkpointing: bool = False
    log_every: int = 20
    grad_clip: float = 1.0
    save_dir: str = 'checkpoints_sentiment'
    data_path: str = 'data/sigmalaw_absa/train.jsonl'
    eval_path: str = 'data/sigmalaw_absa/dev.jsonl'
    seed: int = 13
    num_labels: int = 3


def default_sentiment_config() -> SentimentConfig:
    return SentimentConfig()


def load_sentiment_config(path: str | None = None) -> SentimentConfig:
    setup = SentimentConfig()
    if not path:
        return setup
    with open(
        path,
        'r',
        encoding='utf-8',
    ) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(
        data,
        dict,
    ):
        raise ValueError('error: ValueError')
    for k, v in data.items():
        if hasattr(
            setup,
            k,
        ):
            default = getattr(
                setup,
                k,
            )
            setattr(
                setup,
                k,
                _coerce_to_default_type(
                    default,
                    v,
                ),
            )
    return setup
