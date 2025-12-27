from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from .corpus import (
    chunk_paragraphs,
    iter_text_paths,
    parse_icj_meta,
    read_text,
    split_paragraphs,
)


def prepare_txt_dir(
    input_dir: str,
    train_out: str,
    dev_out: str,
    *,
    tokenizer_name: str,
    max_length: int,
    dev_ratio: float = 0.1,
    seed: int = 13,
    recursive: bool = True,
    limit: Optional[int] = None,
    stride: int = 0,
) -> dict:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError('error: ValueError')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
    )
    paths = list(iter_text_paths(
            input_dir,
            recursive=recursive,
            suffix='.txt',
        ))
    if limit is not None:
        paths = paths[: max(
            0,
            limit,
        )]
    metas = {p: parse_icj_meta(p) for p in paths}
    groups: dict[str, list[Path]] = {}
    for p in paths:
        meta = metas.get(
            p,
            {},
        )
        key = (
            meta.get('case_id')
            or meta.get('stem')
            or str(p)
        )
        groups.setdefault(
            key,
            [],
        ).append(p)
    rng = random.Random(seed)
    case_keys = list(groups.keys())
    rng.shuffle(case_keys)
    if len(case_keys) <= 1:
        dev_cases = []
    else:
        dev_n_cases = int(round(len(case_keys) * dev_ratio))
        dev_n_cases = max(
            1,
            dev_n_cases,
        )
        dev_n_cases = min(
            dev_n_cases,
            len(case_keys) - 1,
        )
        dev_cases = case_keys[:dev_n_cases]
    dev_case_set = set(dev_cases)
    train_path = Path(train_out)
    dev_path = Path(dev_out)
    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dev_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    stats = {
        'docs': len(paths),
        'cases': len(case_keys),
        'dev_cases': len(dev_case_set),
        'train_chunks': 0,
        'dev_chunks': 0,
    }
    with train_path.open(
        'w',
        encoding='utf-8',
    ) as train_f, dev_path.open(
        'w',
        encoding='utf-8',
    ) as dev_f:
        for p in tqdm(
            paths,
            desc='prepare',
        ):
            text = read_text(p)
            paras = split_paragraphs(text)
            chunks = chunk_paragraphs(
                paras,
                tokenizer,
                max_length=max_length,
                stride=stride,
            )
            parsed = metas.get(p) or parse_icj_meta(p)
            meta = {
                'path': str(p),
                **parsed,
            }
            case_key = (
                parsed.get('case_id')
                or parsed.get('stem')
                or str(p)
            )
            target_f = (
                dev_f
                if case_key in dev_case_set
                else train_f
            )
            for i, chunk in enumerate(chunks):
                row = {
                    'text': chunk,
                    'meta': {
                        **meta,
                        'chunk_index': i,
                        'num_chunks': len(chunks),
                    },
                }
                target_f.write(json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + '\n')
            if case_key in dev_case_set:
                stats['dev_chunks'] += len(chunks)
            else:
                stats['train_chunks'] += len(chunks)
    return stats
