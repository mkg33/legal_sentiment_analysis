import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .lexicon import (
    LexiconFeaturizer,
    detect_vad_scale,
    load_lexicon,
    load_word_vad,
    normalise_vad_scale,
    resolve_vad_path,
    scale_vad,
)


class LegalEmotionDataset(Dataset):

    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        emotions: List[str],
        max_length: int,
        lexicon_path: Optional[str] = None,
        vad_lexicon_path: Optional[str] = None,
        lexicon_vad_scale: Optional[str] = None,
        word_vad_scale: Optional[str] = None,
        label_vad_scale: Optional[str] = None,
        lexicon_stopwords_file: Optional[str] = None,
        lexicon_negation_window: int = 0,
        lexicon_negators: Optional[List[str]] = None,
        lexicon_shared_term_weighting: str = 'split',
        lexicon_min_vad_salience: float = 0.0,
        lexicon_min_vad_arousal: float = 0.0,
        lexicon_require_word_vad: bool = False,
        lexicon_allow_seed_only: bool = False,
        vad_allow_missing: bool = False,
        lexicon_extra_path: Optional[str] = None,
        lexicon_intensity_path: Optional[str] = None,
        lexicon_intensity_min: float = 0.0,
        silver_force_has_lex: bool = False,
    ):
        self.path = Path(path)
        self.tokenizer_name = str(tokenizer_name)
        self._tokenizer = None
        self.emotions = emotions
        self.label_to_idx = {
            e: i for i, e in enumerate(self.emotions)
        }
        self.max_length = max_length
        self.lexicon = load_lexicon(
            lexicon_path,
            vad_lexicon_path,
            lexicon_vad_scale=lexicon_vad_scale,
            word_vad_scale=word_vad_scale,
            stopwords_path=lexicon_stopwords_file,
            extra_path=lexicon_extra_path,
            intensity_path=lexicon_intensity_path,
            intensity_min=lexicon_intensity_min,
            min_vad_salience=lexicon_min_vad_salience,
            min_vad_arousal=lexicon_min_vad_arousal,
            require_word_vad=lexicon_require_word_vad,
            allow_seed_only=lexicon_allow_seed_only,
            allow_missing_vad=vad_allow_missing,
        )
        vad_path = resolve_vad_path(
            vad_lexicon_path,
            allow_missing=vad_allow_missing,
        )
        vad_terms = (
            load_word_vad(
                vad_path,
                vad_scale=word_vad_scale,
                stopwords_path=lexicon_stopwords_file,
            )
            if vad_path
            else {}
        )
        self.featurizer = LexiconFeaturizer(
            self.lexicon,
            self.emotions,
            vad_lexicon=vad_terms,
            negation_window=lexicon_negation_window,
            negators=lexicon_negators,
            shared_term_weighting=lexicon_shared_term_weighting,
        )
        self.label_vad_scale = None
        self.silver_force_has_lex = bool(silver_force_has_lex)
        if label_vad_scale is not None:
            if (
                str(label_vad_scale).strip().lower()
                == 'auto'
            ):
                self.label_vad_scale = 'auto'
            else:
                self.label_vad_scale = normalise_vad_scale(label_vad_scale)
        self.samples = []
        with self.path.open() as f:
            for line in f:
                self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError('error: ValueError')
        vad_values: List[Tuple[float, float, float]] = []
        for s in self.samples:
            v = s.get('vad')
            if isinstance(
                v,
                (list, tuple),
            ) and len(v) >= 3:
                try:
                    vad_values.append((
                            float(v[0]),
                            float(v[1]),
                            float(v[2]),
                        ))
                except Exception:
                    continue

        def _maybe_warn_ambiguous(
            scale: Optional[str],
            values: List[Tuple[float, float, float]],
        ) -> None:
            if (
                scale not in {'zero_one', 'one_nine'}
                or not values
            ):
                return
            flat = [x for trio in values for x in trio]
            vmin = min(flat)
            vmax = max(flat)
            mean = sum(flat) / float(len(flat))
            span = vmax - vmin
            if scale == 'zero_one':
                skewed = (
                    mean < 0.2 or mean > 0.8 or span < 0.3
                )
                if skewed:
                    warnings.warn(
                        'warn: vad 0_1',
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if scale == 'one_nine':
                skewed = (
                    mean < 3.0 or mean > 7.0 or span < 2.5
                )
                if skewed:
                    warnings.warn(
                        'warn: vad 1_9',
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if self.label_vad_scale == 'auto':
            if vad_values:
                self.label_vad_scale = detect_vad_scale(vad_values)
                _maybe_warn_ambiguous(
                    self.label_vad_scale,
                    vad_values,
                )
            else:
                self.label_vad_scale = None
        elif self.label_vad_scale is None and vad_values:
            detected = detect_vad_scale(vad_values)
            if detected in {'zero_one', 'one_nine'}:
                warnings.warn(
                    'warn: label_vad_scale',
                    RuntimeWarning,
                    stacklevel=2,
                )
                _maybe_warn_ambiguous(
                    detected,
                    vad_values,
                )
        if any((
                'silver_labels' in s
                and 'silver_max_length' not in s
                for s in self.samples
            )):
            warnings.warn(
                'warn: silver_max_length',
                RuntimeWarning,
                stacklevel=2,
            )

    def __len__(self):
        return len(self.samples)

    def _get_tokenizer(self):
        tok = self._tokenizer
        if tok is None:
            tok = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
            )
            self._tokenizer = tok
        return tok

    @property
    def tokenizer(self):
        return self._get_tokenizer()

    def __getitem__(
        self,
        idx,
    ):
        row = self.samples[idx]
        text = row['text']
        labels = row.get(
            'labels',
            [],
        )
        labels_negative = row.get('labels_negative')
        vad = row.get('vad')
        silver = row.get('silver_labels')
        silver_vad = row.get('silver_vad')
        silver_max_len = row.get('silver_max_length')
        meta = row.get(
            'meta',
            {},
        )
        labels_exhaustive = row.get('labels_exhaustive')
        labels_list = (
            labels if isinstance(
                labels,
                list,
            ) else []
        )
        neg_labels = (
            labels_negative
            if isinstance(
                labels_negative,
                list,
            )
            else []
        )
        if labels_exhaustive is True:
            has_labels = 1.0
        elif labels_list or neg_labels:
            has_labels = 1.0
        elif 'labels' not in row or labels is None:
            has_labels = 0.0
        else:
            has_labels = 0.0
            if labels_exhaustive is None:
                warnings.warn(
                    'warn: empty',
                    RuntimeWarning,
                    stacklevel=2,
                )
        has_vad = float(isinstance(
                vad,
                (list, tuple),
            ) and len(vad) >= 3)
        label_vec = torch.zeros(
            len(self.emotions),
            dtype=torch.float,
        )
        for lbl in labels_list:
            idx = self.label_to_idx.get(lbl)
            if idx is not None:
                label_vec[idx] = 1.0
        tokenizer = self._get_tokenizer()
        enc = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        attn_len = int(enc['attention_mask'].sum().item())
        special = tokenizer.num_special_tokens_to_add(pair=False)
        body_tokens = max(
            1,
            attn_len - int(special),
        )
        text_for_lex = tokenizer.decode(
            enc['input_ids'][0].tolist(),
            skip_special_tokens=True,
        )
        (
            counts,
            prior,
            vad_avg,
            stats,
        ) = (
            self.featurizer.vectors_with_stats(text_for_lex)
        )
        has_lex = float(stats.get(
                'lex_hits',
                0,
            ) > 0)
        has_lex_vad = float(stats.get(
                'vad_hits',
                0,
            ) > 0)
        if has_vad:
            if self.label_vad_scale:
                vad_target = torch.tensor(
                    scale_vad(
                        vad,
                        scale=self.label_vad_scale,
                    ),
                    dtype=torch.float,
                )
            else:
                vad_target = torch.tensor(
                    vad,
                    dtype=torch.float,
                )
        else:
            vad_target = vad_avg
        if silver is not None and silver_max_len not in (
            None,
            self.max_length,
        ):
            silver = None
            silver_vad = None
        if silver is not None and silver_max_len is None:
            if attn_len >= int(self.max_length):
                silver = None
                silver_vad = None
        if self.silver_force_has_lex and silver is not None:
            has_lex = 1.0
            has_lex_vad = 1.0
        silver_vec = (
            torch.tensor(
                silver,
                dtype=torch.float,
            )
            if silver is not None
            else None
        )
        silver_vad_vec = (
            torch.tensor(
                silver_vad,
                dtype=torch.float,
            )
            if silver_vad is not None
            else None
        )
        label_mask = torch.zeros(
            len(self.emotions),
            dtype=torch.float,
        )
        if labels_exhaustive is True:
            label_mask[:] = 1.0
        else:
            for lbl in labels_list:
                idx = self.label_to_idx.get(lbl)
                if idx is not None:
                    label_mask[idx] = 1.0
            for lbl in neg_labels:
                idx = self.label_to_idx.get(lbl)
                if idx is not None:
                    label_mask[idx] = 1.0
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = label_vec
        item['label_mask'] = label_mask
        item['vad'] = vad_target
        item['lex_counts'] = counts
        item['lex_prior'] = prior
        item['lex_vad'] = vad_avg
        item['text'] = text
        item['meta'] = meta
        item['n_tokens'] = torch.tensor(
            float(body_tokens),
            dtype=torch.float,
        )
        item['has_labels'] = torch.tensor(
            has_labels,
            dtype=torch.float,
        )
        item['has_vad'] = torch.tensor(
            has_vad,
            dtype=torch.float,
        )
        item['has_lex'] = torch.tensor(
            has_lex,
            dtype=torch.float,
        )
        item['has_lex_vad'] = torch.tensor(
            has_lex_vad,
            dtype=torch.float,
        )
        if silver_vec is None:
            silver_vec = prior
        if silver_vad_vec is None:
            silver_vad_vec = vad_avg
        item['silver_labels'] = silver_vec
        item['silver_vad'] = silver_vad_vec
        return item
