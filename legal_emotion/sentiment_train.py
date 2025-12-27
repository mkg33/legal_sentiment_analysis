from __future__ import annotations
import os
import sys
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from .sentiment_config import (
    SentimentConfig,
    default_sentiment_config,
)
from .sentiment_data import SentimentDataset
from .utils import get_device, set_seed


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    cfg,
    device: torch.device,
):
    num_workers = int(getattr(
            cfg,
            'num_workers',
            0,
        ) or 0)
    pin_memory = (
        bool(getattr(
                cfg,
                'pin_memory',
                False,
            ))
        and device.type == 'cuda'
    )
    persistent_workers = (
        bool(getattr(
                cfg,
                'persistent_workers',
                False,
            ))
        and num_workers > 0
    )
    prefetch_factor = getattr(
        cfg,
        'prefetch_factor',
        None,
    )
    kw_opts = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kw_opts['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            kw_opts['prefetch_factor'] = int(prefetch_factor)
    return DataLoader(
        dataset,
        **kw_opts,
    )


def _f1_macro(
    pred: torch.Tensor,
    gold: torch.Tensor,
    *,
    num_labels: int,
) -> float:
    num = int(num_labels)
    scores = []
    for c in range(num):
        tp = int(((pred == c) & (gold == c)).sum().item())
        fp = int(((pred == c) & (gold != c)).sum().item())
        fn = int(((pred != c) & (gold == c)).sum().item())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return float(sum(scores) / max(
            1,
            len(scores),
        ))


def run_sentiment_training(cfg: SentimentConfig = None):
    cfg = cfg or default_sentiment_config()

    def log(msg: str) -> None:
        print(
            msg,
            file=sys.stderr,
            flush=True,
        )

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    if int(getattr(
            cfg,
            'num_workers',
            0,
        ) or 0) > 0:
        os.environ.setdefault(
            'TOKENIZERS_PARALLELISM',
            'false',
        )
        log('tokenizers_parallelism=0')
    log(f'model={cfg.model_name} device={device.type}')
    if device.type == 'cuda' and bool(getattr(
            cfg,
            'tf32',
            True,
        )):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    log('load data')
    train_data = SentimentDataset(
        cfg.data_path,
        cfg.model_name,
        max_length=cfg.max_length,
    )
    dev_data = SentimentDataset(
        cfg.eval_path,
        cfg.model_name,
        max_length=cfg.max_length,
    )
    log(f'data train={len(train_data)} dev={len(dev_data)}')
    log('load model')
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=int(cfg.num_labels),
            ignore_mismatched_sizes=True,
        ).to(device)
    )
    if bool(getattr(
            cfg,
            'gradient_checkpointing',
            False,
        )) and hasattr(
        model,
        'gradient_checkpointing_enable',
    ):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    log('build loaders')
    train_loader = _make_loader(
        train_data,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        cfg=cfg,
        device=device,
    )
    eval_bs = int(getattr(
            cfg,
            'eval_batch_size',
            None,
        )
        or cfg.batch_size)
    dev_loader = _make_loader(
        dev_data,
        batch_size=eval_bs,
        shuffle=False,
        cfg=cfg,
        device=device,
    )
    optimiser = AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    grad_accum = max(
        1,
        int(getattr(
                cfg,
                'grad_accum_steps',
                1,
            ) or 1),
    )
    steps_per_epoch = int(math.ceil(len(train_loader) / grad_accum))
    total_steps = steps_per_epoch * int(cfg.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        int(cfg.warmup_steps),
        total_steps,
    )
    save_path = Path(cfg.save_dir)
    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    best_value: float | None = None
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
    scaler_enabled = (
        device.type == 'cuda' and amp_mode == 'fp16'
    )
    try:
        scaler = torch.amp.GradScaler(
            'cuda',
            enabled=scaler_enabled,
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if bool(getattr(
            cfg,
            'compile',
            False,
        )) and hasattr(
        torch,
        'compile',
    ):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    log(f'train epochs={int(cfg.num_epochs)} steps={len(train_loader)}')
    log_every = int(getattr(
            cfg,
            'log_every',
            20,
        ) or 20)
    global_step = 0
    for epoch in range(int(cfg.num_epochs)):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        log(f'epoch {epoch} start')
        for step, batch in enumerate(tqdm(
                train_loader,
                desc=f'train_sentiment {epoch}',
            )):
            inputs = {
                k: batch[k].to(
                    device,
                    non_blocking=True,
                )
                for k in (
                    'input_ids',
                    'attention_mask',
                    'token_type_ids',
                )
                if k in batch
            }
            labels = batch['label'].to(
                device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
                raw_loss = F.cross_entropy(
                    logits,
                    labels,
                )
            loss = raw_loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            is_accum_boundary = (
                step + 1
            ) % grad_accum == 0 or step + 1 == len(train_loader)
            if is_accum_boundary:
                if scaler.is_enabled():
                    scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg.grad_clip),
                )
                if scaler.is_enabled():
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                global_step += 1
                if (
                    log_every > 0
                    and global_step % log_every == 0
                ):
                    print(f'step {global_step} loss {raw_loss.item():.4f}')
        log(f'epoch {epoch} eval')
        metrics = evaluate(
            model,
            dev_loader,
            cfg,
            device,
        )
        f1 = float(metrics.get(
                'f1_macro',
                0.0,
            ))
        better = best_value is None or f1 > float(best_value)
        if better:
            best_value = f1
            torch.save(
                {
                    'model': model.state_dict(),
                    'cfg': cfg.__dict__,
                },
                save_path / 'sentiment.pt',
            )
            print(f'saved f1 {best_value:.4f}')


def evaluate(
    model,
    loader,
    cfg: SentimentConfig,
    device: torch.device,
):
    model.eval()
    total = 0
    total_loss = 0.0
    correct = 0
    guesses_all: list[int] = []
    gold_all: list[int] = []
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
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                k: batch[k].to(
                    device,
                    non_blocking=True,
                )
                for k in (
                    'input_ids',
                    'attention_mask',
                    'token_type_ids',
                )
                if k in batch
            }
            labels = batch['label'].to(
                device,
                non_blocking=True,
            )
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
        return {
            'eval_loss': 0.0,
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'n_total': 0,
        }
    guess_t = torch.tensor(
        guesses_all,
        dtype=torch.long,
    )
    gold_t = torch.tensor(
        gold_all,
        dtype=torch.long,
    )
    f1 = _f1_macro(
        guess_t,
        gold_t,
        num_labels=int(cfg.num_labels),
    )
    metrics = {
        'eval_loss': total_loss / total,
        'accuracy': float(correct) / float(total),
        'f1_macro': float(f1),
        'n_total': int(total),
    }
    print(f"eval loss {metrics['eval_loss']:.4f}")
    return metrics
