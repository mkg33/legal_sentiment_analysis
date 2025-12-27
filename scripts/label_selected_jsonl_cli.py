#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import read_text


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open(
        'w',
        encoding='utf-8',
    ) as f:
        for r in rows:
            f.write(json.dumps(
                    r,
                    ensure_ascii=False,
                ) + '\n')
    tmp.replace(path)


def _is_unlabelled(cat: Any) -> bool:
    if not isinstance(
        cat,
        str,
    ):
        return True
    c = cat.strip().lower()
    return not c or c in {
        'unknown',
        'unlabelled',
        'none',
        'null',
        'na',
        'n/a',
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='label selected docs')
    ap.add_argument(
        '--input_jsonl',
        type=str,
        required=True,
    )
    ap.add_argument(
        '--output_jsonl',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--labels',
        type=str,
        default='neg,neu,pos',
    )
    ap.add_argument(
        '--only_unlabelled',
        action='store_true',
        default=True,
    )
    ap.add_argument(
        '--label_all',
        dest='only_unlabelled',
        action='store_false',
    )
    ap.add_argument(
        '--show_chars',
        type=int,
        default=1200,
    )
    ap.add_argument(
        '--save_every',
        type=int,
        default=1,
    )
    opts = ap.parse_args()
    in_path = Path(opts.input_jsonl).expanduser().resolve()
    out_path = (
        Path(opts.output_jsonl).expanduser().resolve()
        if opts.output_jsonl
        else in_path
    )
    labels = [
        s.strip()
        for s in str(opts.labels).split(',')
        if s.strip()
    ]
    if not labels:
        raise SystemExit('error: SystemExit')
    allowed: Set[str] = set((s.lower() for s in labels))
    rows = _read_jsonl(in_path)
    if not rows:
        raise SystemExit('error: SystemExit')
    show_n = max(
        200,
        int(opts.show_chars),
    )
    save_every = max(
        1,
        int(opts.save_every),
    )
    labelled_since_save = 0
    for idx, row in enumerate(
        rows,
        start=1,
    ):
        meta = (
            row.get('meta')
            if isinstance(
                row.get('meta'),
                dict,
            )
            else {}
        )
        path_raw = meta.get('path')
        if (
            not isinstance(
                path_raw,
                str,
            )
            or not path_raw.strip()
        ):
            print(
                f'skip meta.path {idx}',
                file=sys.stderr,
            )
            continue
        p = Path(path_raw)
        cat = meta.get('category')
        if bool(opts.only_unlabelled) and (
            not _is_unlabelled(cat)
        ):
            continue
        print('\n' + '=' * 80)
        print(f'[{idx}/{len(rows)}] {p}')
        print(f'current category: {cat!r}')
        preview = row.get('preview')
        text: Optional[str] = None
        if isinstance(
            preview,
            str,
        ) and preview.strip():
            text = preview
        else:
            try:
                text = read_text(p)
            except Exception as e:
                print(
                    f'error read {type(e).__name__} {p}',
                    file=sys.stderr,
                )
                text = None
        if text:
            print('-' * 80)
            print(text[:show_n].rstrip())
            if len(text) > show_n:
                print('\n[... truncated ...]')
        else:
            print('[no text available]')
        prompt = f'label {labels} (s=skip, q=quit): '
        while True:
            ans = input(prompt).strip().lower()
            if ans in {'q', 'quit', 'exit'}:
                _atomic_write_jsonl(
                    out_path,
                    rows,
                )
                print(f'\nsaved and exited: {out_path}')
                return 0
            if ans in {'s', 'skip', ''}:
                break
            if ans in allowed:
                meta = dict(meta)
                meta['category'] = ans
                row['meta'] = meta
                labelled_since_save += 1
                break
            print(f'invalid label: {ans!r}')
        if labelled_since_save >= save_every:
            _atomic_write_jsonl(
                out_path,
                rows,
            )
            labelled_since_save = 0
            print(f'saved: {out_path}')
    _atomic_write_jsonl(
        out_path,
        rows,
    )
    print(f'done: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
