#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _normalise_ws(text: str) -> str:
    s = unicodedata.normalize(
        'NFKC',
        str(text),
    )
    s = s.translate(str.maketrans({
                '"': '"',
                '"': '"',
                '"': '"',
                '"': '"',
                ''': "'",
                ''': "'",
                ''': "'",
                ''': "'",
                '-': '-',
                '-': '-',
                '-': '-',
                '-': '-',
                '\xad': '',
            }))
    return re.sub(
        '\\s+',
        ' ',
        s,
    ).strip()


def _iter_llm_outputs(out_dir: Path) -> Iterable[Path]:
    yield from sorted((
            p
            for p in out_dir.rglob('*_llm.txt')
            if p.is_file()
        ))


def _orig_path_for_llm(
    *,
    llm_path: Path,
    out_dir: Path,
    input_dir: Path,
) -> Optional[Path]:
    try:
        rel = llm_path.resolve().relative_to(out_dir.resolve())
    except Exception:
        rel = Path(llm_path.name)
    name = rel.name
    if not name.endswith('_llm.txt'):
        return None
    orig_name = name[: -len('_llm.txt')] + '.txt'
    return (input_dir / rel.parent / orig_name).resolve()


def _split_blocks(text: str) -> List[str]:
    blocks = [
        b.strip()
        for b in re.split(
            '\\n\\s*\\n+',
            text,
        )
        if b.strip()
    ]
    return blocks


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str
    missing_block_index: Optional[int] = None
    missing_block_preview: Optional[str] = None


def _make_anchors(block_norm: str) -> List[str]:
    s = str(block_norm).lstrip()
    anchors: List[str] = []
    anchors.extend([s[:80], s[:60], s[:40]])
    words = s.split()
    if len(words) >= 8:
        anchors.append(' '.join(words[:8]))
    if len(words) >= 12:
        anchors.append(' '.join(words[:12]))
    if len(s) > 160:
        mid = len(s) // 2
        anchors.append(s[mid : mid + 80])
    uniq: List[str] = []
    seen = set()
    for a in anchors:
        a = a.strip()
        if len(a) < 20:
            continue
        if a in seen:
            continue
        seen.add(a)
        uniq.append(a)
    return uniq


def _coverage_and_end(
    *,
    short: str,
    long: str,
) -> Tuple[float, int]:
    if not short:
        return (1.0, 0)
    sm = SequenceMatcher(
        None,
        short,
        long,
        autojunk=False,
    )
    blocks = sm.get_matching_blocks()
    matched = sum((b.size for b in blocks))
    cov = float(matched) / float(max(
            1,
            len(short),
        ))
    end = 0
    for b in blocks:
        if b.size <= 0:
            continue
        end = max(
            end,
            int(b.b + b.size),
        )
    return (cov, end)


def _best_partial_ratio_start(
    *,
    short: str,
    long: str,
) -> Tuple[float, int]:
    if not short or not long or len(short) > len(long):
        return (0.0, 0)
    m = SequenceMatcher(
        None,
        short,
        long,
        autojunk=False,
    )
    best = 0.0
    best_start = 0
    for b in m.get_matching_blocks():
        start = int(b.b - b.a)
        if start < 0:
            start = 0
        substr = long[start : start + len(short)]
        r = SequenceMatcher(
            None,
            short,
            substr,
            autojunk=False,
        ).ratio()
        if r > best:
            best = float(r)
            best_start = start
            if best >= 0.999:
                break
    return (best, best_start)


def _check_subset(
    *,
    orig_text: str,
    llm_text: str,
    min_ratio: float,
    window_margin_chars: int,
    max_anchor_occurrences: int,
) -> CheckResult:
    if not llm_text.strip():
        return CheckResult(
            ok=True,
            reason='empty (NONE)',
        )
    orig_norm = _normalise_ws(orig_text)
    blocks = _split_blocks(llm_text)
    if not blocks:
        return CheckResult(
            ok=True,
            reason='empty after split',
        )
    pos = 0
    for i, block in enumerate(
        blocks,
        1,
    ):
        block_norm = _normalise_ws(block)
        if not block_norm:
            continue
        rem = orig_norm[pos:]
        idx = rem.find(block_norm)
        if idx >= 0:
            pos = pos + idx + len(block_norm)
            continue
        anchors = _make_anchors(block_norm)
        starts: List[int] = []
        for anchor in anchors:
            search_from = 0
            while len(starts) < int(max_anchor_occurrences):
                j = rem.find(
                    anchor,
                    search_from,
                )
                if j < 0:
                    break
                starts.append(int(j))
                search_from = int(j) + 1
            if starts:
                break
        best_cov = 0.0
        best_start = 0
        best_end = 0
        found = False
        for j in sorted(set(starts))[
            : int(max_anchor_occurrences)
        ]:
            window = rem[
                j : j
                + len(block_norm)
                + int(window_margin_chars)
            ]
            (
                cov,
                end,
            ) = _coverage_and_end(
                short=block_norm,
                long=window,
            )
            if cov > best_cov:
                best_cov = cov
                best_start = j
                best_end = j + end
            if cov >= float(min_ratio):
                pos = pos + j + end
                found = True
                break
        if not found:
            (
                ratio,
                j,
            ) = _best_partial_ratio_start(
                short=block_norm,
                long=rem,
            )
            window = rem[
                j : j
                + len(block_norm)
                + int(window_margin_chars)
            ]
            (
                cov,
                end,
            ) = _coverage_and_end(
                short=block_norm,
                long=window,
            )
            if cov >= float(min_ratio):
                pos = pos + j + end
                found = True
            else:
                best_cov = max(
                    best_cov,
                    cov,
                )
                if best_cov == cov:
                    best_start = j
                    best_end = j + end
        if not found:
            preview = block_norm[:200] + (
                '...' if len(block_norm) > 200 else ''
            )
            return CheckResult(
                ok=False,
                reason=f'block match ratio {best_cov:.3f} < {float(min_ratio):.3f} (best_start~={best_start}, best_end~={best_end})',
                missing_block_index=i,
                missing_block_preview=preview,
            )
    return CheckResult(
        ok=True,
        reason='all blocks matched (approx, in order)',
    )


def main() -> int:
    ap = argparse.ArgumentParser(description='check llm outputs')
    ap.add_argument(
        '--input_dir',
        default='data/EN_TXT_BEST_FULL',
    )
    ap.add_argument(
        '--out_dir',
        default='dataset_llm',
    )
    ap.add_argument(
        '--name_regex',
        default=None,
    )
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
    )
    ap.add_argument(
        '--min_ratio',
        type=float,
        default=0.98,
    )
    ap.add_argument(
        '--window_margin_chars',
        type=int,
        default=2000,
    )
    ap.add_argument(
        '--max_anchor_occurrences',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--fail_fast',
        action='store_true',
    )
    opts = ap.parse_args()
    root = Path.cwd().resolve()
    input_dir = Path(opts.input_dir)
    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    out_dir = Path(opts.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    if not out_dir.exists():
        print(
            f'error: missing out_dir {out_dir}',
            file=sys.stderr,
        )
        return 2
    if not input_dir.exists():
        print(
            f'error: missing input_dir {input_dir}',
            file=sys.stderr,
        )
        return 2
    llm_paths = list(_iter_llm_outputs(out_dir))
    if opts.name_regex:
        pat = re.compile(str(opts.name_regex))
        llm_paths = [
            p for p in llm_paths if pat.search(str(p))
        ]
    if opts.limit is not None:
        llm_paths = llm_paths[: max(
            0,
            int(opts.limit),
        )]
    total = len(llm_paths)
    ok = 0
    empty = 0
    fail = 0
    missing_orig = 0
    for p in llm_paths:
        orig_path = _orig_path_for_llm(
            llm_path=p,
            out_dir=out_dir,
            input_dir=input_dir,
        )
        if orig_path is None or not orig_path.exists():
            missing_orig += 1
            fail += 1
            print(
                f'fail missing {p}',
                file=sys.stderr,
            )
            if opts.fail_fast:
                return 2
            continue
        llm_text = p.read_text(
            encoding='utf-8',
            errors='ignore',
        )
        orig_text = orig_path.read_text(
            encoding='utf-8',
            errors='ignore',
        )
        out = _check_subset(
            orig_text=orig_text,
            llm_text=llm_text,
            min_ratio=float(opts.min_ratio),
            window_margin_chars=int(opts.window_margin_chars),
            max_anchor_occurrences=int(opts.max_anchor_occurrences),
        )
        if out.ok:
            ok += 1
            if not llm_text.strip():
                empty += 1
        else:
            fail += 1
            msg = f'fail {p}'
            if out.missing_block_index is not None:
                msg += f' block {out.missing_block_index}'
            print(
                msg,
                file=sys.stderr,
            )
            if out.missing_block_preview:
                print(
                    f'preview_len {len(out.missing_block_preview)}',
                    file=sys.stderr,
                )
            if opts.fail_fast:
                return 2
    print(f'done {total} ok={ok} empty={empty} fail={fail} missing={missing_orig}')
    return 0 if fail == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
