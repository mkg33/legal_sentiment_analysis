#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import iter_text_paths, read_text


def _write_jsonl(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        'w',
        encoding='utf-8',
    ) as f:
        for r in rows:
            f.write(json.dumps(
                    r,
                    ensure_ascii=False,
                ) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description='sample docs for labelling')
    ap.add_argument(
        '--input_dir',
        type=str,
        required=True,
    )
    ap.add_argument(
        '--output_jsonl',
        type=str,
        required=True,
    )
    ap.add_argument(
        '--n_docs',
        type=int,
        default=500,
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=13,
    )
    ap.add_argument(
        '--preview_chars',
        type=int,
        default=0,
    )
    ap.add_argument(
        '--limit_paths',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--recursive',
        dest='recursive',
        action='store_true',
        default=True,
    )
    ap.add_argument(
        '--no_recursive',
        dest='recursive',
        action='store_false',
    )
    opts = ap.parse_args()

    def log(msg: str) -> None:
        print(
            msg,
            file=sys.stderr,
            flush=True,
        )

    in_dir = Path(opts.input_dir)
    paths = list(iter_text_paths(
            in_dir,
            recursive=bool(opts.recursive),
            suffix='.txt',
        ))
    if opts.limit_paths is not None:
        paths = paths[: max(
            0,
            int(opts.limit_paths),
        )]
    if not paths:
        raise SystemExit('error: SystemExit')
    n_docs = max(
        1,
        int(opts.n_docs),
    )
    if len(paths) < n_docs:
        raise SystemExit('error: SystemExit')
    rng = random.Random(int(opts.seed))
    rng.shuffle(paths)
    chosen = paths[:n_docs]
    rows: List[Dict[str, Any]] = []
    preview_n = max(
        0,
        int(opts.preview_chars),
    )
    for p in chosen:
        row: Dict[str, Any] = {
            'meta': {
                'id': p.stem,
                'path': str(p),
                'category': '',
                'source': 'ICJ',
                'label_source': 'manual',
            }
        }
        if preview_n > 0:
            try:
                text = read_text(p)
                row['preview'] = text[:preview_n]
            except Exception:
                row['preview'] = None
        rows.append(row)
    out = Path(opts.output_jsonl).expanduser().resolve()
    _write_jsonl(
        out,
        rows,
    )
    log(f'wrote {len(rows)} {out}')
    print(json.dumps(
            {
                'output_jsonl': str(out),
                'n_docs': int(len(rows)),
                'seed': int(opts.seed),
                'input_dir': str(in_dir),
                'preview_chars': int(preview_n),
            },
            ensure_ascii=False,
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
