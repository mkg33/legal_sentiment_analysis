import os
import sys
from pathlib import Path
from typing import Optional
import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from .config import default_config
from .data import LegalEmotionDataset
from .lexicon import emotion_prototypes
from .losses import cost_matrix, emotion_loss
from .model import LegalEmotionModel
from .metrics import f1_macro, f1_micro
from .transfer import init_encoder_from_sentiment_checkpoint
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


def _pseudo_thresholds(cfg) -> Optional[torch.Tensor]:
    raw = getattr(
        cfg,
        'pseudo_class_thresholds',
        None,
    )
    if raw is None:
        return None
    if isinstance(
        raw,
        dict,
    ):
        return torch.tensor(
            [
                float(raw.get(
                        e,
                        cfg.threshold,
                    ))
                for e in cfg.emotions
            ],
            dtype=torch.float,
        )
    if isinstance(
        raw,
        (list, tuple),
    ):
        if len(raw) != len(cfg.emotions):
            raise ValueError('error: ValueError')
        return torch.tensor(
            [float(x) for x in raw],
            dtype=torch.float,
        )
    return None


def run_training(cfg=default_config()):
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
    train_data = LegalEmotionDataset(
        cfg.data_path,
        cfg.model_name,
        cfg.emotions,
        cfg.max_length,
        cfg.lexicon_path,
        cfg.vad_lexicon_path,
        lexicon_vad_scale=getattr(
            cfg,
            'lexicon_vad_scale',
            None,
        ),
        word_vad_scale=getattr(
            cfg,
            'word_vad_scale',
            None,
        ),
        label_vad_scale=getattr(
            cfg,
            'label_vad_scale',
            None,
        ),
        lexicon_stopwords_file=getattr(
            cfg,
            'lexicon_stopwords_file',
            None,
        ),
        lexicon_negation_window=getattr(
            cfg,
            'lexicon_negation_window',
            0,
        ),
        lexicon_negators=getattr(
            cfg,
            'lexicon_negators',
            None,
        ),
        lexicon_shared_term_weighting=getattr(
            cfg,
            'lexicon_shared_term_weighting',
            'split',
        ),
        lexicon_min_vad_salience=getattr(
            cfg,
            'lexicon_min_vad_salience',
            0.0,
        ),
        lexicon_min_vad_arousal=getattr(
            cfg,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        lexicon_require_word_vad=bool(getattr(
                cfg,
                'lexicon_require_word_vad',
                False,
            )),
        lexicon_allow_seed_only=bool(getattr(
                cfg,
                'lexicon_allow_seed_only',
                False,
            )),
        vad_allow_missing=bool(getattr(
                cfg,
                'vad_allow_missing',
                False,
            )),
        lexicon_extra_path=getattr(
            cfg,
            'lexicon_extra_path',
            None,
        ),
        lexicon_intensity_path=getattr(
            cfg,
            'lexicon_intensity_path',
            None,
        ),
        lexicon_intensity_min=getattr(
            cfg,
            'lexicon_intensity_min',
            0.0,
        ),
        silver_force_has_lex=bool(getattr(
                cfg,
                'silver_force_has_lex',
                False,
            )),
    )
    dev_data = LegalEmotionDataset(
        cfg.eval_path,
        cfg.model_name,
        cfg.emotions,
        cfg.max_length,
        cfg.lexicon_path,
        cfg.vad_lexicon_path,
        lexicon_vad_scale=getattr(
            cfg,
            'lexicon_vad_scale',
            None,
        ),
        word_vad_scale=getattr(
            cfg,
            'word_vad_scale',
            None,
        ),
        label_vad_scale=getattr(
            cfg,
            'label_vad_scale',
            None,
        ),
        lexicon_stopwords_file=getattr(
            cfg,
            'lexicon_stopwords_file',
            None,
        ),
        lexicon_negation_window=getattr(
            cfg,
            'lexicon_negation_window',
            0,
        ),
        lexicon_negators=getattr(
            cfg,
            'lexicon_negators',
            None,
        ),
        lexicon_shared_term_weighting=getattr(
            cfg,
            'lexicon_shared_term_weighting',
            'split',
        ),
        lexicon_min_vad_salience=getattr(
            cfg,
            'lexicon_min_vad_salience',
            0.0,
        ),
        lexicon_min_vad_arousal=getattr(
            cfg,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        lexicon_require_word_vad=bool(getattr(
                cfg,
                'lexicon_require_word_vad',
                False,
            )),
        lexicon_allow_seed_only=bool(getattr(
                cfg,
                'lexicon_allow_seed_only',
                False,
            )),
        vad_allow_missing=bool(getattr(
                cfg,
                'vad_allow_missing',
                False,
            )),
        lexicon_extra_path=getattr(
            cfg,
            'lexicon_extra_path',
            None,
        ),
        lexicon_intensity_path=getattr(
            cfg,
            'lexicon_intensity_path',
            None,
        ),
        lexicon_intensity_min=getattr(
            cfg,
            'lexicon_intensity_min',
            0.0,
        ),
        silver_force_has_lex=bool(getattr(
                cfg,
                'silver_force_has_lex',
                False,
            )),
    )
    unl_loader = None
    if (
        cfg.unlabelled_path
        and Path(cfg.unlabelled_path).exists()
    ):
        unl_data = LegalEmotionDataset(
            cfg.unlabelled_path,
            cfg.model_name,
            cfg.emotions,
            cfg.max_length,
            cfg.lexicon_path,
            cfg.vad_lexicon_path,
            lexicon_vad_scale=getattr(
                cfg,
                'lexicon_vad_scale',
                None,
            ),
            word_vad_scale=getattr(
                cfg,
                'word_vad_scale',
                None,
            ),
            label_vad_scale=getattr(
                cfg,
                'label_vad_scale',
                None,
            ),
            lexicon_stopwords_file=getattr(
                cfg,
                'lexicon_stopwords_file',
                None,
            ),
            lexicon_negation_window=getattr(
                cfg,
                'lexicon_negation_window',
                0,
            ),
            lexicon_negators=getattr(
                cfg,
                'lexicon_negators',
                None,
            ),
            lexicon_shared_term_weighting=getattr(
                cfg,
                'lexicon_shared_term_weighting',
                'split',
            ),
            lexicon_min_vad_salience=getattr(
                cfg,
                'lexicon_min_vad_salience',
                0.0,
            ),
            lexicon_min_vad_arousal=getattr(
                cfg,
                'lexicon_min_vad_arousal',
                0.0,
            ),
            lexicon_require_word_vad=bool(getattr(
                    cfg,
                    'lexicon_require_word_vad',
                    False,
                )),
            lexicon_allow_seed_only=bool(getattr(
                    cfg,
                    'lexicon_allow_seed_only',
                    False,
                )),
            vad_allow_missing=bool(getattr(
                    cfg,
                    'vad_allow_missing',
                    False,
                )),
            lexicon_extra_path=getattr(
                cfg,
                'lexicon_extra_path',
                None,
            ),
            lexicon_intensity_path=getattr(
                cfg,
                'lexicon_intensity_path',
                None,
            ),
            lexicon_intensity_min=getattr(
                cfg,
                'lexicon_intensity_min',
                0.0,
            ),
            silver_force_has_lex=bool(getattr(
                    cfg,
                    'silver_force_has_lex',
                    False,
                )),
        )
        unl_loader = _make_loader(
            unl_data,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            cfg=cfg,
            device=device,
        )
        log(f'unlabelled {len(unl_data)}')
    log('build model')
    model = LegalEmotionModel(
        cfg.model_name,
        len(cfg.emotions),
    ).to(device)
    init_ckpt = getattr(
        cfg,
        'init_from_sentiment_checkpoint',
        None,
    )
    if init_ckpt:
        log(f'init encoder {init_ckpt}')
        info = init_encoder_from_sentiment_checkpoint(
            model.encoder,
            init_ckpt,
            expected_model_name=cfg.model_name,
        )
        name_note = ''
        if info.get('model_name_match') is False:
            name_note = f" saved_model={info.get('saved_model_name')!r} expected_model={info.get('expected_model_name')!r} coverage={info.get('coverage'):.2f}"
        print(f"encoder init {info['checkpoint']}")
    if bool(getattr(
            cfg,
            'gradient_checkpointing',
            False,
        )) and hasattr(
        model.encoder,
        'gradient_checkpointing_enable',
    ):
        try:
            model.encoder.gradient_checkpointing_enable()
            if hasattr(
                model.encoder.config,
                'use_cache',
            ):
                model.encoder.config.use_cache = False
        except Exception:
            pass
    pseudo_class_thr = _pseudo_thresholds(cfg)
    C = None
    if cfg.alpha_sinkhorn > 0:
        if getattr(
            cfg,
            'ot_cost',
            'uniform',
        ) == 'vad':
            protos = emotion_prototypes(
                train_data.lexicon,
                cfg.emotions,
            ).to(device)
            C = cost_matrix(
                len(cfg.emotions),
                device,
                emotion_vad=protos,
            )
        else:
            C = cost_matrix(
                len(cfg.emotions),
                device,
            )
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
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    grad_accum = int(getattr(
            cfg,
            'grad_accum_steps',
            1,
        ) or 1)
    grad_accum = max(
        1,
        grad_accum,
    )
    steps_per_epoch = int(math.ceil(len(train_loader) / grad_accum))
    total_steps = steps_per_epoch * int(cfg.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        cfg.warmup_steps,
        total_steps,
    )
    best_value = None
    best_key = None
    save_path = Path(cfg.save_dir)
    save_path.mkdir(exist_ok=True)
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
    log_every = int(getattr(
            cfg,
            'log_every',
            20,
        ) or 20)
    global_step = 0
    log(f'train epochs={int(cfg.num_epochs)} steps={len(train_loader)}')
    for epoch in range(cfg.num_epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        log(f'epoch {epoch} start')
        for step, batch in enumerate(tqdm(
                train_loader,
                desc=f'train {epoch}',
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
            lex_counts = batch['lex_counts'].to(
                device,
                non_blocking=True,
            )
            lex_prior = batch['lex_prior'].to(
                device,
                non_blocking=True,
            )
            lex_vad = batch['lex_vad'].to(
                device,
                non_blocking=True,
            )
            lex_mask = batch.get('has_lex')
            lex_vad_mask = batch.get('has_lex_vad')
            if lex_mask is not None:
                lex_mask = lex_mask.to(
                    device,
                    non_blocking=True,
                )
            if lex_vad_mask is not None:
                lex_vad_mask = lex_vad_mask.to(
                    device,
                    non_blocking=True,
                )
            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                (
                    logits,
                    vad_guess,
                    count_guess,
                ) = model(
                    **inputs,
                    lex_counts=lex_counts,
                    lex_prior=lex_prior,
                    lex_vad=lex_vad,
                    lex_mask=lex_mask,
                    lex_vad_mask=lex_vad_mask,
                )
            (
                raw_loss,
                parts,
            ) = emotion_loss(
                logits.float(),
                vad_guess.float(),
                batch,
                cfg.alpha_cls,
                cfg.alpha_vad,
                cfg.alpha_sinkhorn,
                cfg.sinkhorn_epsilon,
                cfg.sinkhorn_iters,
                cfg.silver_weight,
                cfg.use_silver,
                cfg.ot_mode,
                cfg.ot_reg_m,
                C=C,
                count_pred=count_guess.float(),
                alpha_mass=getattr(
                    cfg,
                    'alpha_mass',
                    0.0,
                ),
                count_pred_scale=getattr(
                    cfg,
                    'count_pred_scale',
                    'counts',
                ),
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
                    cfg.grad_clip,
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
        if unl_loader:
            for ub in tqdm(
                unl_loader,
                desc=f'pseudo {epoch}',
            ):
                with torch.no_grad():
                    inputs = {
                        k: ub[k].to(
                            device,
                            non_blocking=True,
                        )
                        for k in (
                            'input_ids',
                            'attention_mask',
                            'token_type_ids',
                        )
                        if k in ub
                    }
                    lc = ub['lex_counts'].to(
                        device,
                        non_blocking=True,
                    )
                    lp = ub['lex_prior'].to(
                        device,
                        non_blocking=True,
                    )
                    lv = ub['lex_vad'].to(
                        device,
                        non_blocking=True,
                    )
                    lm = ub.get('has_lex')
                    lvm = ub.get('has_lex_vad')
                    if lm is not None:
                        lm = lm.to(
                            device,
                            non_blocking=True,
                        )
                    if lvm is not None:
                        lvm = lvm.to(
                            device,
                            non_blocking=True,
                        )
                    with torch.autocast(
                        device_type=autocast_device,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        (
                            logits_u,
                            vad_u,
                            _,
                        ) = model(
                            **inputs,
                            lex_counts=lc,
                            lex_prior=lp,
                            lex_vad=lv,
                            lex_mask=lm,
                            lex_vad_mask=lvm,
                        )
                    probs = torch.sigmoid(logits_u)
                    max_mask = probs.max(dim=1).values > float(cfg.pseudo_threshold)
                    if pseudo_class_thr is not None:
                        thr = pseudo_class_thr.to(device=probs.device)
                        mask = max_mask & (
                            probs >= thr
                        ).any(dim=1)
                    else:
                        mask = max_mask
                if mask.any():
                    pseudo = {}
                    for k, v in ub.items():
                        if torch.is_tensor(v):
                            pseudo[k] = v[mask]
                        else:
                            pseudo[k] = [
                                val
                                for i, val in enumerate(v)
                                if mask[i].item()
                            ]
                    if pseudo_class_thr is not None:
                        thr = pseudo_class_thr.to(device=probs.device)
                        pseudo['labels'] = (
                            probs[mask] > thr
                        ).float()
                    else:
                        pseudo['labels'] = (
                            probs[mask] > cfg.threshold
                        ).float()
                    pseudo['label_mask'] = torch.ones_like(pseudo['labels'])
                    if 'has_labels' in pseudo:
                        pseudo['has_labels'] = (
                            torch.ones_like(pseudo['has_labels'])
                        )
                    optimiser.zero_grad(set_to_none=True)
                    inputs = {
                        k: pseudo[k].to(
                            device,
                            non_blocking=True,
                        )
                        for k in (
                            'input_ids',
                            'attention_mask',
                            'token_type_ids',
                        )
                        if k in pseudo
                    }
                    lc = pseudo['lex_counts'].to(
                        device,
                        non_blocking=True,
                    )
                    lp = pseudo['lex_prior'].to(
                        device,
                        non_blocking=True,
                    )
                    lv = pseudo['lex_vad'].to(
                        device,
                        non_blocking=True,
                    )
                    lm = pseudo.get('has_lex')
                    lvm = pseudo.get('has_lex_vad')
                    if lm is not None:
                        lm = lm.to(
                            device,
                            non_blocking=True,
                        )
                    if lvm is not None:
                        lvm = lvm.to(
                            device,
                            non_blocking=True,
                        )
                    with torch.autocast(
                        device_type=autocast_device,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        (
                            logits_p,
                            vad_p,
                            count_p,
                        ) = model(
                            **inputs,
                            lex_counts=lc,
                            lex_prior=lp,
                            lex_vad=lv,
                            lex_mask=lm,
                            lex_vad_mask=lvm,
                        )
                    (
                        loss_p,
                        _,
                    ) = emotion_loss(
                        logits_p.float(),
                        vad_p.float(),
                        pseudo,
                        cfg.alpha_cls,
                        cfg.alpha_vad,
                        cfg.alpha_sinkhorn,
                        cfg.sinkhorn_epsilon,
                        cfg.sinkhorn_iters,
                        cfg.silver_weight,
                        cfg.use_silver,
                        cfg.ot_mode,
                        cfg.ot_reg_m,
                        C=C,
                        count_pred=count_p.float(),
                        alpha_mass=getattr(
                            cfg,
                            'alpha_mass',
                            0.0,
                        ),
                        count_pred_scale=getattr(
                            cfg,
                            'count_pred_scale',
                            'counts',
                        ),
                    )
                    if scaler.is_enabled():
                        scaler.scale(loss_p).backward()
                        scaler.unscale_(optimiser)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            cfg.grad_clip,
                        )
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        loss_p.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            cfg.grad_clip,
                        )
                        optimiser.step()
        log(f'epoch {epoch} eval')
        metrics = evaluate(
            model,
            dev_loader,
            cfg,
            device,
            C=C,
        )
        if metrics.get(
            'n_gold_labels',
            0,
        ) > 0:
            score_key = 'f1_macro_gold'
            score_value = float(metrics.get(
                    score_key,
                    0.0,
                ))
            better = (
                best_value is None
                or score_value > float(best_value)
            )
        else:
            score_key = 'eval_loss'
            score_value = float(metrics.get(
                    score_key,
                    0.0,
                ))
            better = (
                best_value is None
                or score_value < float(best_value)
            )
        if better:
            best_value = score_value
            best_key = score_key
            torch.save(
                {
                    'model': model.state_dict(),
                    'cfg': cfg.__dict__,
                },
                save_path / 'model.pt',
            )
            print(f'saved {best_key} {best_value:.4f}')


def evaluate(
    model,
    loader,
    cfg,
    device,
    *,
    C=None,
):
    model.eval()
    total = 0
    total_loss = 0.0
    parts_sum = {
        'cls': 0.0,
        'vad': 0.0,
        'ot': 0.0,
        'mass': 0.0,
    }
    gold_label_n = 0
    gold_label_positions = 0
    gold_vad_n = 0
    f1m_gold = 0.0
    f1M_gold = 0.0
    cls_gold = 0.0
    vad_gold = 0.0
    silver_label_n = 0
    silver_vad_n = 0
    ce_silver = 0.0
    vad_silver = 0.0
    mass_mse_silver = 0.0
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
            lex_counts = batch['lex_counts'].to(
                device,
                non_blocking=True,
            )
            lex_prior = batch['lex_prior'].to(
                device,
                non_blocking=True,
            )
            lex_vad = batch['lex_vad'].to(
                device,
                non_blocking=True,
            )
            lex_mask = batch.get('has_lex')
            lex_vad_mask = batch.get('has_lex_vad')
            if lex_mask is not None:
                lex_mask = lex_mask.to(
                    device,
                    non_blocking=True,
                )
            if lex_vad_mask is not None:
                lex_vad_mask = lex_vad_mask.to(
                    device,
                    non_blocking=True,
                )
            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                (
                    logits,
                    vad_guess,
                    count_guess,
                ) = model(
                    **inputs,
                    lex_counts=lex_counts,
                    lex_prior=lex_prior,
                    lex_vad=lex_vad,
                    lex_mask=lex_mask,
                    lex_vad_mask=lex_vad_mask,
                )
            (
                loss,
                parts,
            ) = emotion_loss(
                logits.float(),
                vad_guess.float(),
                batch,
                cfg.alpha_cls,
                cfg.alpha_vad,
                cfg.alpha_sinkhorn,
                cfg.sinkhorn_epsilon,
                cfg.sinkhorn_iters,
                cfg.silver_weight,
                cfg.use_silver,
                cfg.ot_mode,
                cfg.ot_reg_m,
                C=C,
                count_pred=count_guess.float(),
                alpha_mass=getattr(
                    cfg,
                    'alpha_mass',
                    0.0,
                ),
                count_pred_scale=getattr(
                    cfg,
                    'count_pred_scale',
                    'counts',
                ),
            )
            bs = int(logits.size(0))
            total += bs
            total_loss += loss.item() * bs
            for k in parts_sum:
                parts_sum[k] += (
                    float(parts.get(
                            k,
                            0.0,
                        )) * bs
                )
            labels = batch['labels'].to(device)
            vad_target = batch['vad'].to(device)
            label_mask = batch.get('label_mask')
            if label_mask is not None:
                label_mask = label_mask.to(device).float()
            has_labels = batch.get('has_labels')
            if has_labels is None:
                gold_mask = labels.sum(dim=1) > 0
            else:
                gold_mask = (
                    has_labels.to(device).view(-1) > 0.5
                )
            if gold_mask.any():
                gl_logits = logits[gold_mask]
                gl_labels = labels[gold_mask]
                gl_mask = (
                    label_mask[gold_mask]
                    if label_mask is not None
                    else None
                )
                n = int(gl_labels.size(0))
                gold_label_n += n
                if gl_mask is None:
                    cls_gold += torch.nn.functional.binary_cross_entropy_with_logits(
                        gl_logits,
                        gl_labels,
                        reduction='sum',
                    ).item()
                    gold_label_positions += n * max(
                        1,
                        len(cfg.emotions),
                    )
                else:
                    cls_loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
                        gl_logits,
                        gl_labels,
                        reduction='none',
                    )
                    cls_gold += (
                        (cls_loss_vec * gl_mask)
                        .sum()
                        .item()
                    )
                    gold_label_positions += int(gl_mask.sum().item())
                f1m_gold += (
                    f1_micro(
                        gl_logits,
                        gl_labels,
                        cfg.threshold,
                        label_mask=gl_mask,
                    )
                    * n
                )
                f1M_gold += (
                    f1_macro(
                        gl_logits,
                        gl_labels,
                        cfg.threshold,
                        label_mask=gl_mask,
                    )
                    * n
                )
            has_vad = batch.get('has_vad')
            if has_vad is not None:
                vad_gold_mask = (
                    has_vad.to(device).view(-1) > 0.5
                )
                if vad_gold_mask.any():
                    gv_guess = vad_guess[vad_gold_mask]
                    gv_target = vad_target[vad_gold_mask]
                    n = int(gv_target.size(0))
                    gold_vad_n += n
                    vad_gold += (
                        torch.mean(
                            (gv_guess - gv_target) ** 2,
                            dim=-1,
                        )
                        .sum()
                        .item()
                    )
            has_lex = batch.get('has_lex')
            if has_lex is not None and cfg.use_silver:
                lex_mask = has_lex.to(device).view(-1) > 0.5
                no_gold = ~gold_mask
                silver_mask = lex_mask & no_gold
                if silver_mask.any():
                    sl_logits = logits[silver_mask]
                    sl_prior = batch['silver_labels'].to(device)[silver_mask]
                    n = int(sl_prior.size(0))
                    silver_label_n += n
                    q = torch.sigmoid(sl_logits).clamp_min(1e-08)
                    q = q / q.sum(
                        dim=-1,
                        keepdim=True,
                    ).clamp_min(1e-08)
                    logq = q.log()
                    ce_silver += (
                        -(sl_prior * logq)
                        .sum(dim=-1)
                        .mean()
                        .item()
                        * n
                    )
                    if (
                        cfg.ot_mode or ''
                    ).lower().startswith('unbalanced') and 'lex_counts' in batch:
                        mass_guess = count_guess[
                            silver_mask
                        ].sum(dim=-1)
                        mass_lex = (
                            batch['lex_counts']
                            .to(device)[silver_mask]
                            .sum(dim=-1)
                        )
                        count_scale = (
                            str(getattr(
                                    cfg,
                                    'count_pred_scale',
                                    'counts',
                                )
                                or 'counts')
                            .lower()
                            .strip()
                        )
                        if (
                            count_scale
                            in {
                                'density',
                                'per_token',
                                'per_tok',
                            }
                            and 'n_tokens' in batch
                        ):
                            denom = (
                                batch['n_tokens']
                                .to(device)[silver_mask]
                                .clamp_min(1.0)
                            )
                            mass_lex = mass_lex / denom
                        mass_mse_silver += (
                            torch.mean((mass_guess - mass_lex) ** 2).item()
                            * n
                        )
            has_lex_vad = batch.get('has_lex_vad')
            if has_lex_vad is not None and cfg.use_silver:
                vad_mask = (
                    has_lex_vad.to(device).view(-1) > 0.5
                )
                no_gold_vad = has_vad is not None and ~(
                    has_vad.to(device).view(-1) > 0.5
                )
                if isinstance(
                    no_gold_vad,
                    torch.Tensor,
                ):
                    silver_vad_mask = vad_mask & no_gold_vad
                else:
                    silver_vad_mask = vad_mask
                if silver_vad_mask.any():
                    sv_guess = vad_guess[silver_vad_mask]
                    sv_target = batch['silver_vad'].to(device)[silver_vad_mask]
                    n = int(sv_target.size(0))
                    silver_vad_n += n
                    vad_silver += (
                        torch.mean(
                            (sv_guess - sv_target) ** 2,
                            dim=-1,
                        )
                        .sum()
                        .item()
                    )
    if total <= 0:
        return {
            'eval_loss': 0.0,
            'cls': 0.0,
            'vad': 0.0,
            'ot': 0.0,
            'n_total': 0,
            'n_gold_labels': 0,
            'n_gold_vad': 0,
        }
    metrics = {
        'eval_loss': total_loss / total,
        'cls': parts_sum['cls'] / total,
        'vad': parts_sum['vad'] / total,
        'ot': parts_sum['ot'] / total,
        'mass': parts_sum['mass'] / total,
        'n_total': total,
        'n_gold_labels': gold_label_n,
        'n_gold_vad': gold_vad_n,
    }
    if gold_label_n > 0:
        metrics['f1_micro_gold'] = f1m_gold / gold_label_n
        metrics['f1_macro_gold'] = f1M_gold / gold_label_n
        if gold_label_positions > 0:
            metrics['cls_gold'] = (
                cls_gold / gold_label_positions
            )
    if gold_vad_n > 0:
        metrics['vad_gold'] = vad_gold / gold_vad_n
    if silver_label_n > 0:
        metrics['ce_silver'] = ce_silver / silver_label_n
        metrics['n_silver_labels'] = silver_label_n
        if mass_mse_silver > 0:
            metrics['mass_mse_silver'] = (
                mass_mse_silver / silver_label_n
            )
    if silver_vad_n > 0:
        metrics['vad_silver'] = vad_silver / silver_vad_n
        metrics['n_silver_vad'] = silver_vad_n
    if metrics.get(
        'n_gold_labels',
        0,
    ) > 0:
        print(f"eval loss {metrics['eval_loss']:.4f}")
    else:
        extra = []
        if 'ce_silver' in metrics:
            extra.append(f"ce_silver {metrics['ce_silver']:.4f}")
        if 'mass_mse_silver' in metrics:
            extra.append(f"mass_mse_silver {metrics['mass_mse_silver']:.4f}")
        if 'vad_silver' in metrics:
            extra.append(f"vad_silver {metrics['vad_silver']:.4f}")
        extra_str = ' ' + ' '.join(extra) if extra else ''
        print(f"eval loss {metrics['eval_loss']:.4f}")
    return metrics


if __name__ == '__main__':
    run_training()
