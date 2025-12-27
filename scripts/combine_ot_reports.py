#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.combined_viz import (
    write_combined_html_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description='combine ot reports')
    ap.add_argument(
        '--out',
        type=str,
        required=True,
    )
    ap.add_argument(
        '--summary_html',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--compare_html',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--token_html',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--group_html',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--title',
        type=str,
        default='Legal Emotion OT Report',
    )
    opts = ap.parse_args()
    sections = []
    if opts.summary_html:
        sections.append(('Summary', Path(opts.summary_html)))
    if opts.compare_html:
        sections.append(('EOT Neighbours', Path(opts.compare_html)))
    if opts.token_html:
        sections.append(('Token-OT Neighbours', Path(opts.token_html)))
    if opts.group_html:
        sections.append(('Group Shift', Path(opts.group_html)))
    if not sections:
        raise SystemExit('error: SystemExit')
    write_combined_html_report(
        output_path=Path(opts.out),
        sections=[
            (label, str(path)) for label, path in sections
        ],
        title=str(opts.title),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
