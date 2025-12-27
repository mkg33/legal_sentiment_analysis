from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SentimentDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        tokenizer_name: str,
        *,
        max_length: int,
    ):
        self.path = Path(path)
        self.tokenizer_name = str(tokenizer_name)
        self._tokenizer = None
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError('error: ValueError')
        self.samples: list[Dict[str, Any]] = []
        with self.path.open(
            'r',
            encoding='utf-8',
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError('error: ValueError')

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        row = self.samples[idx]
        text = row.get('text')
        if not isinstance(
            text,
            str,
        ):
            raise ValueError('error: ValueError')
        label = row.get('label')
        if label is None:
            raise ValueError('error: ValueError')
        label_i = int(label)
        tokenizer = self._get_tokenizer()
        enc = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['label'] = torch.tensor(
            label_i,
            dtype=torch.long,
        )
        item['text'] = text
        item['meta'] = (
            row.get('meta')
            if isinstance(
                row.get('meta'),
                dict,
            )
            else {}
        )
        return item
