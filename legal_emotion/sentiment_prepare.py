from __future__ import annotations
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen
import pandas as pd

SIGMALAW_ABSA_URL = 'https://osf.io/download/37gkh/'
_LABEL_MAP = {
    -1: 0,
    0: 1,
    1: 2,
}
_INV_LABEL_MAP = {v: k for k, v in _LABEL_MAP.items()}


def _download(
    url: str,
    dest: Path,
) -> None:
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    req = Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0'},
    )
    with urlopen(
        req,
        timeout=60,
    ) as r, dest.open('wb') as f:
        shutil.copyfileobj(
            r,
            f,
        )


def _row_to_example(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get('Sentence')
    if not isinstance(
        text,
        str,
    ) or not text.strip():
        return None
    s = row.get('Overall Sentiment')
    if s is None:
        return None
    try:
        sentiment = int(s)
    except Exception:
        return None
    if sentiment not in _LABEL_MAP:
        return None
    label = int(_LABEL_MAP[sentiment])
    meta = {
        'source': 'sigmalaw-absa',
        'party_raw': row.get('Party'),
        'party_sentiment_raw': row.get('Sentiment'),
    }
    return {
        'text': text,
        'label': label,
        'sentiment': sentiment,
        'meta': meta,
    }


def prepare_sigmalaw_absa(
    *,
    out_dir: str | Path,
    csv_path: str | Path | None = None,
    dev_ratio: float = 0.1,
    seed: int = 13,
    download: bool = True,
) -> Dict[str, Any]:
    if not 0.0 < float(dev_ratio) < 1.0:
        raise ValueError('error: ValueError')
    out_p = Path(out_dir)
    out_p.mkdir(
        parents=True,
        exist_ok=True,
    )
    if csv_path is None:
        csv_p = out_p / 'SigmaLaw-ABSA.csv'
    else:
        csv_p = Path(csv_path)
    if not csv_p.exists():
        if not download:
            raise FileNotFoundError('error: FileNotFoundError')
        _download(
            SIGMALAW_ABSA_URL,
            csv_p,
        )
    df = pd.read_csv(csv_p)
    expected = {'Sentence', 'Overall Sentiment'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError('error: ValueError')
    examples: list[Dict[str, Any]] = []
    for _, r in df.iterrows():
        ex = _row_to_example(r.to_dict())
        if ex is not None:
            examples.append(ex)
    if not examples:
        raise ValueError('error: ValueError')
    rng = random.Random(int(seed))
    rng.shuffle(examples)
    n = len(examples)
    dev_n = int(round(n * float(dev_ratio)))
    dev_n = max(
        1,
        dev_n,
    )
    dev_n = min(
        dev_n,
        n - 1,
    )
    dev = examples[:dev_n]
    train = examples[dev_n:]
    train_path = out_p / 'train.jsonl'
    dev_path = out_p / 'dev.jsonl'
    for p, rows in [(train_path, train), (dev_path, dev)]:
        with p.open(
            'w',
            encoding='utf-8',
        ) as f:
            for row in rows:
                f.write(json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + '\n')
    dist = {k: 0 for k in sorted(_LABEL_MAP.keys())}
    for ex in examples:
        dist[int(ex['sentiment'])] += 1
    return {
        'downloaded': str(csv_p),
        'out_dir': str(out_p),
        'n_total': int(n),
        'n_train': int(len(train)),
        'n_dev': int(len(dev)),
        'label_dist': {
            str(k): int(v) for k, v in dist.items()
        },
        'label_map': {
            str(k): int(v) for k, v in _LABEL_MAP.items()
        },
        'inv_label_map': {
            str(k): int(v)
            for k, v in _INV_LABEL_MAP.items()
        },
    }
