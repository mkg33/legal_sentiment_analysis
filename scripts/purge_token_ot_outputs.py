#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _safe_rmtree(path: Path) -> bool:
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return False


def _find_run_dirs(outputs_dir: Path) -> List[Path]:
    if not outputs_dir.exists():
        return []
    out: List[Path] = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and p.name.startswith('gold_to_icj_ot_'):
            out.append(p)
    out.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return out


def _purge_one(
    run_dir: Path,
    *,
    apply: bool,
) -> List[str]:
    targets = [
        run_dir / 'token_ot_check',
        run_dir / 'icj_token_neighbours.jsonl',
        run_dir / 'icj_token_neighbours.html',
        run_dir / 'icj_token_ot_stats.json',
        run_dir / 'icj_token_neighbours.signature.json',
        run_dir / 'icj_group_shift',
    ]
    removed: List[str] = []
    for t in targets:
        if not t.exists():
            continue
        if not apply:
            removed.append(str(t))
            continue
        ok = (
            _safe_rmtree(t)
            if t.is_dir()
            else _safe_unlink(t)
        )
        if ok:
            removed.append(str(t))
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description='purge token-ot outputs')
    ap.add_argument(
        '--run_dir',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--outputs_dir',
        type=str,
        default='outputs',
    )
    ap.add_argument(
        '--apply',
        action='store_true',
    )
    opts = ap.parse_args()
    if opts.run_dir:
        run_dirs = [
            Path(opts.run_dir).expanduser().resolve()
        ]
    else:
        run_dirs = _find_run_dirs(Path(opts.outputs_dir).expanduser().resolve())
    removed_all = {}
    for rd in run_dirs:
        removed_all[str(rd)] = _purge_one(
            rd,
            apply=bool(opts.apply),
        )
    print(json.dumps(
            {
                'apply': bool(opts.apply),
                'removed': removed_all,
            },
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
