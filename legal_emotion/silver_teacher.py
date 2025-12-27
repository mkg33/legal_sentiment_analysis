from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from .lexicon import (
    LexiconFeaturizer,
    emotion_prototypes,
    load_lexicon,
    load_word_vad,
    resolve_vad_path,
)
from .utils import get_device, load_config

_LABEL_SPLIT_RE = re.compile('[\\s\\-]+')
GOEMOTIONS_MAP: Dict[str, Sequence[str]] = {
    'anger': ('anger', 'annoyance', 'disapproval'),
    'fear': ('fear', 'nervousness'),
    'joy': ('joy', 'amusement', 'excitement', 'relief'),
    'sadness': (
        'sadness',
        'grief',
        'disappointment',
        'remorse',
    ),
    'trust': (
        'admiration',
        'approval',
        'caring',
        'gratitude',
        'love',
        'pride',
    ),
    'disgust': ('disgust', 'embarrassment'),
    'surprise': ('surprise', 'realization', 'confusion'),
    'anticipation': ('desire', 'curiosity', 'optimism'),
}


def _normalise_label(label: str) -> str:
    return _LABEL_SPLIT_RE.sub(
        '_',
        str(label).strip().lower(),
    )


def _label_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    idx_map: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        key = _normalise_label(label)
        idx_map.setdefault(
            key,
            [],
        ).append(int(i))
    return idx_map


def _combine_probs(
    probs: torch.Tensor,
    idxs: Sequence[int],
) -> torch.Tensor:
    if not idxs:
        return torch.zeros(
            probs.size(0),
            device=probs.device,
            dtype=probs.dtype,
        )
    vals = probs[:, list(idxs)]
    return 1.0 - torch.prod(
        1.0 - vals,
        dim=1,
    )


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(
    path: Path,
    rows: Iterable[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        'w',
        encoding='utf-8',
    ) as f:
        for row in rows:
            f.write(json.dumps(
                    row,
                    ensure_ascii=False,
                ) + '\n')


def make_teacher_silver(
    *,
    input_path: str,
    output_path: str,
    cfg_path: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    truncate_to_max_length: bool = True,
) -> dict:
    setup = load_config(cfg_path)
    teacher = model_name or getattr(
        setup,
        'silver_teacher_model',
        None,
    )
    if not teacher:
        raise ValueError('error: ValueError')
    bs = int(batch_size
        or getattr(
            setup,
            'silver_teacher_batch_size',
            16,
        ))
    max_len = int(max_length
        or getattr(
            setup,
            'silver_teacher_max_length',
            None,
        )
        or setup.max_length)
    device = get_device(getattr(
            setup,
            'device',
            'cpu',
        ))
    tokenizer = AutoTokenizer.from_pretrained(
        teacher,
        use_fast=True,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(teacher)
    )
    model.eval()
    model.to(device)
    labels = [
        str(model.config.id2label[i])
        for i in range(int(model.config.num_labels))
    ]
    idx_map = _label_indices(labels)
    lex = load_lexicon(
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
        stopwords_path=getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        ),
        extra_path=getattr(
            setup,
            'lexicon_extra_path',
            None,
        ),
        intensity_path=getattr(
            setup,
            'lexicon_intensity_path',
            None,
        ),
        intensity_min=getattr(
            setup,
            'lexicon_intensity_min',
            0.0,
        ),
        min_vad_salience=getattr(
            setup,
            'lexicon_min_vad_salience',
            0.0,
        ),
        min_vad_arousal=getattr(
            setup,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        require_word_vad=bool(getattr(
                setup,
                'lexicon_require_word_vad',
                False,
            )),
        allow_seed_only=bool(getattr(
                setup,
                'lexicon_allow_seed_only',
                False,
            )),
        allow_missing_vad=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    )
    vad_path = resolve_vad_path(
        getattr(
            setup,
            'vad_lexicon_path',
            None,
        ),
        allow_missing=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    )
    vad_terms = (
        load_word_vad(
            vad_path,
            vad_scale=getattr(
                setup,
                'word_vad_scale',
                None,
            ),
            stopwords_path=getattr(
                setup,
                'lexicon_stopwords_file',
                None,
            ),
        )
        if vad_path
        else {}
    )
    featurizer = LexiconFeaturizer(
        lex,
        setup.emotions,
        vad_lexicon=vad_terms,
        negation_window=getattr(
            setup,
            'lexicon_negation_window',
            0,
        ),
        negators=getattr(
            setup,
            'lexicon_negators',
            None,
        ),
        shared_term_weighting=getattr(
            setup,
            'lexicon_shared_term_weighting',
            'split',
        ),
    )
    protos = emotion_prototypes(
        lex,
        setup.emotions,
    )
    in_path = Path(input_path)
    out_path = Path(output_path)
    rows_out: List[dict] = []
    buffer: List[dict] = []
    texts: List[str] = []

    def flush_batch() -> None:
        if not buffer:
            return
        enc = tokenizer(
            texts,
            padding=True,
            truncation=bool(truncate_to_max_length),
            max_length=max_len,
            return_tensors='pt',
        )
        enc = {
            k: v.to(
                device,
                non_blocking=True,
            )
            for k, v in enc.items()
            if isinstance(
                v,
                torch.Tensor,
            )
        }
        with torch.no_grad():
            logits = model(
                **enc,
                return_dict=True,
            ).logits
        probs = torch.sigmoid(logits).detach().cpu()
        target = torch.zeros(
            (probs.size(0), len(setup.emotions)),
            dtype=torch.float,
        )
        for emo_idx, emo in enumerate(setup.emotions):
            src_labels = GOEMOTIONS_MAP.get(
                emo,
                (),
            )
            idxs: List[int] = []
            for lbl in src_labels:
                idxs.extend(idx_map.get(
                        _normalise_label(lbl),
                        [],
                    ))
            target[:, emo_idx] = _combine_probs(
                probs,
                idxs,
            )
        target = target.clamp(
            0.0,
            1.0,
        )
        weights_sum = target.sum(
            dim=1,
            keepdim=True,
        )
        norm = torch.where(
            weights_sum > 1e-08,
            target / weights_sum,
            target,
        )
        for i, row in enumerate(buffer):
            text = texts[i]
            text_for_lex = text
            if truncate_to_max_length:
                ids = (
                    enc['input_ids'][i]
                    .detach()
                    .cpu()
                    .tolist()
                )
                text_for_lex = tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                )
            (
                _,
                _,
                vad_fallback,
            ) = featurizer.vectors(text_for_lex)
            if weights_sum[i].item() > 1e-08:
                vad_vec = (
                    norm[i].unsqueeze(0) @ protos
                ).squeeze(0)
            else:
                vad_vec = vad_fallback
            row['silver_labels'] = target[i].tolist()
            row['silver_vad'] = [
                float(vad_vec[0]),
                float(vad_vec[1]),
                float(vad_vec[2]),
            ]
            if truncate_to_max_length:
                row['silver_max_length'] = int(max_len)
            meta = row.get('meta')
            if isinstance(
                meta,
                dict,
            ):
                meta = dict(meta)
            else:
                meta = {}
            meta['silver_teacher_model'] = str(teacher)
            row['meta'] = meta
            rows_out.append(row)
        buffer.clear()
        texts.clear()

    for row in tqdm(
        _iter_jsonl(in_path),
        desc='teacher_silver',
    ):
        text = row.get(
            'text',
            '',
        )
        buffer.append(row)
        texts.append(text)
        if len(buffer) >= bs:
            flush_batch()
    flush_batch()
    _write_jsonl(
        out_path,
        rows_out,
    )
    return {
        'input': str(in_path),
        'output': str(out_path),
        'teacher_model': str(teacher),
        'batch_size': int(bs),
        'max_length': int(max_len),
        'rows': int(len(rows_out)),
    }


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='make silver labels')
    ap.add_argument(
        '--input',
        required=True,
    )
    ap.add_argument(
        '--output',
        required=True,
    )
    ap.add_argument(
        '--config',
        default=None,
    )
    ap.add_argument(
        '--model',
        default=None,
    )
    ap.add_argument(
        '--batch_size',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--max_length',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--no_truncate',
        action='store_true',
    )
    args = ap.parse_args()
    make_teacher_silver(
        input_path=args.input,
        output_path=args.output,
        cfg_path=args.config,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        truncate_to_max_length=not bool(args.no_truncate),
    )
