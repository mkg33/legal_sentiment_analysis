#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional


def _remove_path(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def _purge_dir(
    path: Path,
    *,
    keep_latest: bool,
    dry_run: bool,
) -> List[str]:
    removed: List[str] = []
    if not path.exists():
        return removed
    entries = sorted(
        path.iterdir(),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    keep: Optional[Path] = (
        entries[0] if keep_latest and entries else None
    )
    for entry in entries:
        if keep is not None and entry == keep:
            continue
        if dry_run:
            removed.append(str(entry))
            continue
        _remove_path(entry)
        removed.append(str(entry))
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description='purge outputs')
    ap.add_argument(
        '--outputs',
        type=str,
        default='outputs',
    )
    ap.add_argument(
        '--checkpoints',
        type=str,
        default='checkpoints',
    )
    ap.add_argument(
        '--sentiment_checkpoints',
        type=str,
        default='checkpoints_sentiment',
    )
    ap.add_argument(
        '--keep_latest',
        action='store_true',
    )
    ap.add_argument(
        '--dry_run',
        action='store_true',
    )
    opts = ap.parse_args()
    removed = {
        'outputs': _purge_dir(
            Path(opts.outputs),
            keep_latest=bool(opts.keep_latest),
            dry_run=bool(opts.dry_run),
        ),
        'checkpoints': _purge_dir(
            Path(opts.checkpoints),
            keep_latest=bool(opts.keep_latest),
            dry_run=bool(opts.dry_run),
        ),
        'checkpoints_sentiment': _purge_dir(
            Path(opts.sentiment_checkpoints),
            keep_latest=bool(opts.keep_latest),
            dry_run=bool(opts.dry_run),
        ),
    }
    print(json.dumps(
            {
                'removed': removed,
                'dry_run': bool(opts.dry_run),
            },
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
