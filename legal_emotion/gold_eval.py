from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import LegalEmotionDataset
from .lexicon import emotion_prototypes
from .losses import cost_matrix
from .predict import load_model
from .train import evaluate as _evaluate
from .utils import get_device, load_config


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


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )


def evaluate_checkpoint(
    *,
    checkpoint: str,
    data_path: str,
    cfg_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    setup: Config = load_config(cfg_path)
    (
        model,
        setup,
    ) = load_model(
        checkpoint,
        setup,
    )
    ds = LegalEmotionDataset(
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
    if limit is not None:
        ds.samples = ds.samples[: max(
            0,
            int(limit),
        )]
    device = get_device(setup.device)
    model = model.to(device)
    C = None
    if setup.alpha_sinkhorn > 0:
        if getattr(
            setup,
            'ot_cost',
            'uniform',
        ) == 'vad':
            protos = emotion_prototypes(
                ds.lexicon,
                setup.emotions,
            ).to(device)
            C = cost_matrix(
                len(setup.emotions),
                device,
                emotion_vad=protos,
            )
        else:
            C = cost_matrix(
                len(setup.emotions),
                device,
            )
    bs = int(batch_size
        or getattr(
            setup,
            'eval_batch_size',
            None,
        )
        or setup.batch_size
        or 4)
    loader = _make_loader(
        ds,
        batch_size=bs,
        shuffle=False,
    )
    metrics = _evaluate(
        model,
        loader,
        setup,
        device,
        C=C,
    )
    metrics['checkpoint'] = str(checkpoint)
    metrics['data_path'] = str(data_path)
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
