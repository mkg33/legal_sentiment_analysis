import argparse
import json
import torch
from .data import LegalEmotionDataset
from .metrics import f1_macro, f1_micro
from .lexicon import tokenize as lex_tokenize
from .predict import load_model
from .utils import get_device, load_config


def evaluate_path(
    checkpoint,
    data_path,
    cfg_path=None,
):
    setup = load_config(cfg_path)
    data = LegalEmotionDataset(
        data_path,
        setup.model_name,
        setup.emotions,
        setup.max_length,
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
        label_vad_scale=getattr(
            setup,
            'label_vad_scale',
            None,
        ),
        lexicon_stopwords_file=getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        ),
        lexicon_negation_window=getattr(
            setup,
            'lexicon_negation_window',
            0,
        ),
        lexicon_negators=getattr(
            setup,
            'lexicon_negators',
            None,
        ),
        lexicon_shared_term_weighting=getattr(
            setup,
            'lexicon_shared_term_weighting',
            'split',
        ),
        lexicon_min_vad_salience=getattr(
            setup,
            'lexicon_min_vad_salience',
            0.0,
        ),
        lexicon_min_vad_arousal=getattr(
            setup,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        lexicon_require_word_vad=bool(getattr(
                setup,
                'lexicon_require_word_vad',
                False,
            )),
        lexicon_allow_seed_only=bool(getattr(
                setup,
                'lexicon_allow_seed_only',
                False,
            )),
        vad_allow_missing=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
        lexicon_extra_path=getattr(
            setup,
            'lexicon_extra_path',
            None,
        ),
        lexicon_intensity_path=getattr(
            setup,
            'lexicon_intensity_path',
            None,
        ),
        lexicon_intensity_min=getattr(
            setup,
            'lexicon_intensity_min',
            0.0,
        ),
        silver_force_has_lex=bool(getattr(
                setup,
                'silver_force_has_lex',
                False,
            )),
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=setup.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    (
        model,
        setup,
    ) = load_model(
        checkpoint,
        setup,
    )
    device = get_device(setup.device)
    model = model.to(device)
    total = 0
    f1m = 0.0
    f1M = 0.0
    vad_loss = 0.0
    cls_loss = 0.0
    cls_positions = 0
    outs = []
    with torch.inference_mode():
        for batch in loader:
            b = default_collate(batch)
            inputs = {
                k: b[k].to(device)
                for k in (
                    'input_ids',
                    'attention_mask',
                    'token_type_ids',
                )
                if k in b
            }
            lex_counts = b['lex_counts'].to(device)
            lex_prior = b['lex_prior'].to(device)
            lex_vad = b['lex_vad'].to(device)
            (
                logits,
                vad,
                counts,
            ) = model(
                **inputs,
                lex_counts=lex_counts,
                lex_prior=lex_prior,
                lex_vad=lex_vad,
            )
            labels = b['labels'].to(device)
            label_mask = b.get('label_mask')
            if label_mask is None:
                cls_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction='sum',
                ).item()
                cls_positions += labels.numel()
            else:
                mask = label_mask.to(device).float()
                if mask.shape != labels.shape:
                    raise ValueError('error: ValueError')
                cls_loss_vec = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction='none',
                )
                cls_loss += (
                    (cls_loss_vec * mask).sum().item()
                )
                cls_positions += int(mask.sum().item())
            vad_loss += torch.nn.functional.mse_loss(
                vad,
                b['vad'].to(device),
                reduction='sum',
            ).item()
            total += labels.size(0)
            if label_mask is not None:
                label_mask = mask
            f1m += f1_micro(
                logits,
                labels,
                setup.threshold,
                label_mask=label_mask,
            ) * labels.size(0)
            f1M += f1_macro(
                logits,
                labels,
                setup.threshold,
                label_mask=label_mask,
            ) * labels.size(0)
            probs = torch.sigmoid(logits)
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
            use_count_dist = (
                str(getattr(
                        setup,
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
                    special = data.tokenizer.num_special_tokens_to_add(pair=False)
                    body_tokens = (
                        b['attention_mask']
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
                    special = data.tokenizer.num_special_tokens_to_add(pair=False)
                    body_tokens = (
                        b['attention_mask']
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
            counts = counts.cpu()
            probs = probs.cpu()
            vad = vad.cpu()
            labels_cpu = labels.cpu()
            for i, text in enumerate(b['text']):
                n_words = len(lex_tokenize(text))
                outs.append({
                        'text': text,
                        'probs': probs[i].tolist(),
                        'labels': labels_cpu[i].tolist(),
                        'vad': vad[i].tolist(),
                        'pred_counts': counts[i].tolist(),
                        'pred_mass': float(counts[i].sum().item()),
                        'pred_per_1k_words': float(counts[i].sum().item()
                            / max(
                                n_words,
                                1,
                            )
                            * 1000.0),
                    })
    cls_denom = (
        cls_positions
        if cls_positions > 0
        else max(
            1,
            total * len(setup.emotions),
        )
    )
    metrics = {
        'cls': cls_loss / cls_denom,
        'vad': vad_loss / total,
        'f1_micro': f1m / total,
        'f1_macro': f1M / total,
    }
    return (metrics, outs)


def default_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        required=True,
    )
    parser.add_argument(
        '--data',
        required=True,
    )
    parser.add_argument(
        '--config',
        default=None,
    )
    parser.add_argument(
        '--save_json',
        default=None,
    )
    opts = parser.parse_args()
    (
        metrics,
        rows,
    ) = evaluate_path(
        opts.checkpoint,
        opts.data,
        opts.config,
    )
    print(json.dumps(
            metrics,
            indent=2,
        ))
    if opts.save_json:
        with open(
            opts.save_json,
            'w',
        ) as f:
            json.dump(
                rows,
                f,
                indent=2,
            )


if __name__ == '__main__':
    main()
