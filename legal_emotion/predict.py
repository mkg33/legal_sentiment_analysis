import torch
from transformers import AutoTokenizer
from .config import default_config, Config
from .lexicon import (
    LexiconFeaturizer,
    load_lexicon,
    load_word_vad,
    resolve_vad_path,
)
from .model import LegalEmotionModel
from .utils import get_device


def build_inputs(
    texts,
    tokenizer,
    cfg,
    featurizer: LexiconFeaturizer,
):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors='pt',
    )
    counts = []
    priors = []
    vads = []
    lex_masks = []
    lex_vad_masks = []
    lex_texts = [
        tokenizer.decode(
            ids.tolist(),
            skip_special_tokens=True,
        )
        for ids in enc['input_ids']
    ]
    for t in lex_texts:
        (
            c,
            p,
            v,
            stats,
        ) = featurizer.vectors_with_stats(t)
        has_lex = float(stats.get(
                'lex_hits',
                0,
            ) > 0)
        has_lex_vad = float(stats.get(
                'vad_hits',
                0,
            ) > 0)
        counts.append(c)
        priors.append(p)
        vads.append(v)
        lex_masks.append(has_lex)
        lex_vad_masks.append(has_lex_vad)
    batch = {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'lex_counts': torch.stack(counts),
        'lex_prior': torch.stack(priors),
        'lex_vad': torch.stack(vads),
        'lex_mask': torch.tensor(
            lex_masks,
            dtype=torch.float,
        ),
        'lex_vad_mask': torch.tensor(
            lex_vad_masks,
            dtype=torch.float,
        ),
    }
    if 'token_type_ids' in enc:
        batch['token_type_ids'] = enc['token_type_ids']
    return batch


def load_model(
    checkpoint_path: str,
    cfg: Config = None,
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
    cfg = cfg or default_config()
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
        if 'lexicon_shared_term_weighting' not in saved_cfg:
            cfg.lexicon_shared_term_weighting = 'none'
        if 'semantic_lexicon_basis' not in saved_cfg:
            cfg.semantic_lexicon_basis = 'full'
        state = data['model']
    else:
        state = data
    model = LegalEmotionModel(
        cfg.model_name,
        len(cfg.emotions),
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


def predict(
    texts,
    checkpoint_path: str,
    cfg: Config = None,
):
    (
        model,
        cfg,
    ) = load_model(
        checkpoint_path,
        cfg,
    )
    device = get_device(cfg.device)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
    )
    lexicon = load_lexicon(
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
        stopwords_path=getattr(
            cfg,
            'lexicon_stopwords_file',
            None,
        ),
        extra_path=getattr(
            cfg,
            'lexicon_extra_path',
            None,
        ),
        intensity_path=getattr(
            cfg,
            'lexicon_intensity_path',
            None,
        ),
        intensity_min=getattr(
            cfg,
            'lexicon_intensity_min',
            0.0,
        ),
        min_vad_salience=getattr(
            cfg,
            'lexicon_min_vad_salience',
            0.0,
        ),
        min_vad_arousal=getattr(
            cfg,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        require_word_vad=bool(getattr(
                cfg,
                'lexicon_require_word_vad',
                False,
            )),
        allow_seed_only=bool(getattr(
                cfg,
                'lexicon_allow_seed_only',
                False,
            )),
        allow_missing_vad=bool(getattr(
                cfg,
                'vad_allow_missing',
                False,
            )),
    )
    vad_path = resolve_vad_path(
        getattr(
            cfg,
            'vad_lexicon_path',
            None,
        ),
        allow_missing=bool(getattr(
                cfg,
                'vad_allow_missing',
                False,
            )),
    )
    vad_terms = (
        load_word_vad(
            vad_path,
            vad_scale=getattr(
                cfg,
                'word_vad_scale',
                None,
            ),
            stopwords_path=getattr(
                cfg,
                'lexicon_stopwords_file',
                None,
            ),
        )
        if vad_path
        else {}
    )
    featurizer = LexiconFeaturizer(
        lexicon,
        cfg.emotions,
        vad_lexicon=vad_terms,
        negation_window=getattr(
            cfg,
            'lexicon_negation_window',
            0,
        ),
        negators=getattr(
            cfg,
            'lexicon_negators',
            None,
        ),
        shared_term_weighting=getattr(
            cfg,
            'lexicon_shared_term_weighting',
            'split',
        ),
    )
    batch = build_inputs(
        texts,
        tokenizer,
        cfg,
        featurizer,
    )
    batch = {
        k: v.to(device)
        for k, v in batch.items()
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
        (
            logits,
            vad,
            counts,
        ) = model(**batch)
        probs = torch.sigmoid(logits)
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
    use_count_dist = (
        str(getattr(
                cfg,
                'ot_mode',
                '',
            ) or '')
        .lower()
        .startswith('unbalanced')
    )
    if use_count_dist:
        if count_scale in {
            'density',
            'per_token',
            'per_tok',
        }:
            special = tokenizer.num_special_tokens_to_add(pair=False)
            body_tokens = (
                batch['attention_mask']
                .sum(dim=1)
                .to(dtype=torch.float)
                - float(special)
            ).clamp_min(1.0)
            counts = counts * body_tokens[:, None]
        counts = counts.clamp_min(0.0)
    else:
        if count_scale in {
            'density',
            'per_token',
            'per_tok',
        }:
            special = tokenizer.num_special_tokens_to_add(pair=False)
            body_tokens = (
                batch['attention_mask']
                .sum(dim=1)
                .to(dtype=torch.float)
                - float(special)
            ).clamp_min(1.0)
            mass = (
                (counts * body_tokens[:, None])
                .sum(
                    dim=-1,
                    keepdim=True,
                )
                .clamp_min(1e-08)
            )
        else:
            mass = counts.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-08)
        denom = probs.sum(
            dim=-1,
            keepdim=True,
        )
        dist = probs / denom.clamp_min(1e-08)
        if torch.any(denom <= 1e-08):
            uniform = torch.full_like(
                probs,
                1.0 / probs.size(-1),
            )
            dist = torch.where(
                denom <= 1e-08,
                uniform,
                dist,
            )
        counts = (dist * mass).clamp_min(0.0)
    return (probs.cpu(), vad.cpu(), counts.cpu())
