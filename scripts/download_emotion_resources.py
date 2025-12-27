#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
RESOURCES: Dict[str, Dict[str, str]] = {
    'nrc_emotion_intensity': {
        'urls': [
            'https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Intensity-Lexicon.zip',
            'http://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Intensity-Lexicon.zip',
        ],
        'folder': 'NRC-Emotion-Intensity-Lexicon-v1',
        'zip_name': 'NRC-Emotion-Intensity-Lexicon-v1.zip',
    }
}


def _download(
    urls: list[str],
    dest: Path,
    *,
    force: bool,
) -> None:
    if dest.exists() and (not force):
        return
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    if tmp.exists():
        tmp.unlink()
    last_err = None
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://saifmohammad.com/WebPages/NRC-Emotion-Intensity-Lexicon.htm',
    }
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers=headers,
            )
            with urllib.request.urlopen(req) as resp, tmp.open('wb') as f:
                f.write(resp.read())
            last_err = None
            break
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err
    tmp.replace(dest)


def _extract_zip(
    zip_path: Path,
    out_dir: Path,
    *,
    force: bool,
) -> None:
    if out_dir.exists() and (not force):
        return
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    with zipfile.ZipFile(
        zip_path,
        'r',
    ) as zf:
        zf.extractall(out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description='get emotion resources')
    ap.add_argument(
        '--out_dir',
        type=str,
        default='data/external',
    )
    ap.add_argument(
        '--dataset',
        type=str,
        default='nrc_emotion_intensity',
    )
    ap.add_argument(
        '--force',
        action='store_true',
    )
    opts = ap.parse_args()
    out_dir = Path(opts.out_dir)
    dataset = str(opts.dataset).strip().lower()
    selected = (
        RESOURCES.keys() if dataset == 'all' else [dataset]
    )
    outs = {}
    for key in selected:
        if key not in RESOURCES:
            raise SystemExit('error: SystemExit')
        spec = RESOURCES[key]
        zip_path = out_dir / spec['zip_name']
        extract_dir = out_dir / spec['folder']
        _download(
            spec['urls'],
            zip_path,
            force=bool(opts.force),
        )
        _extract_zip(
            zip_path,
            extract_dir,
            force=bool(opts.force),
        )
        outs[key] = {
            'zip': str(zip_path),
            'dir': str(extract_dir),
        }
    print(json.dumps(
            {
                'out_dir': str(out_dir),
                'resources': outs,
            },
            indent=2,
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
