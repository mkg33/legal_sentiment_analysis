#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _strip_sentiment_keys(payload: dict) -> dict:
    return {
        k: v
        for k, v in payload.items()
        if 'sentiment' not in str(k).lower()
    }


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def main() -> int:
    ap = argparse.ArgumentParser(description='rebuild icj report')
    ap.add_argument(
        '--run_dir',
        required=True,
        type=str,
    )
    ap.add_argument(
        '--out',
        default=None,
        type=str,
    )
    ap.add_argument(
        '--scores_jsonl',
        default=None,
        type=str,
    )
    ap.add_argument(
        '--neighbours_jsonl',
        default=None,
        type=str,
    )
    ap.add_argument(
        '--top_n',
        default=10,
        type=int,
    )
    ap.add_argument(
        '--excerpt_chars',
        default=800,
        type=int,
    )
    ap.add_argument(
        '--reasoning_scores_jsonl',
        default=None,
        type=str,
    )
    ap.add_argument(
        '--reasoning_top_n',
        default=None,
        type=int,
    )
    ap.add_argument(
        '--reasoning_pre_max_height_px',
        default=560,
        type=int,
    )
    ap.add_argument(
        '--reasoning_neighbours_jsonl',
        default=None,
        type=str,
    )
    ap.add_argument(
        '--reasoning_topk',
        default=10,
        type=int,
    )
    ap.add_argument(
        '--manual_top_docs',
        default=None,
        type=str,
    )
    ap.add_argument(
        '--cleanup',
        action='store_true',
    )
    opts = ap.parse_args()
    run_dir = Path(opts.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    scores_jsonl = (
        Path(opts.scores_jsonl)
        if opts.scores_jsonl
        else run_dir / 'icj_scores.jsonl'
    )
    neighbours_jsonl = (
        Path(opts.neighbours_jsonl)
        if opts.neighbours_jsonl
        else run_dir / 'icj_neighbours.jsonl'
    )
    cfg_path = run_dir / 'emotion_config.resolved.yaml'
    if not scores_jsonl.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    if not neighbours_jsonl.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    out_html = (
        Path(opts.out)
        if opts.out
        else run_dir / 'icj_full_report.html'
    )
    from legal_emotion.compare import (
        compare_scores,
        render_compare_neighbours_html,
    )
    from legal_emotion.combined_viz import (
        write_combined_html_report,
    )
    from legal_emotion.emotional_docs_viz import (
        write_most_emotional_docs_html_report,
    )
    from legal_emotion.summary_viz import (
        write_summary_html_report,
    )
    from legal_emotion.utils import load_config

    setup = (
        load_config(str(cfg_path))
        if cfg_path.exists()
        else None
    )
    feeling_labels = (
        getattr(
            setup,
            'emotion_display_names',
            None,
        )
        if setup is not None
        else None
    )
    if not isinstance(
        feeling_labels,
        dict,
    ):
        feeling_labels = None
    summary_json = run_dir / 'summary.json'
    summary_html: Optional[str] = None
    summary_sanitized: Optional[Path] = None
    if summary_json.exists():
        run_summary_path = summary_json
        try:
            payload = _read_json(summary_json)
            if isinstance(
                payload,
                dict,
            ):
                payload = _strip_sentiment_keys(payload)
                summary_sanitized = (
                    run_dir / '_summary_for_report.json'
                )
                summary_sanitized.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                run_summary_path = summary_sanitized
        except Exception:
            run_summary_path = summary_json
        summary_sections: List[Tuple[str, str]] = [
            ('Run Summary', str(run_summary_path))
        ]
        for label, p in [
            (
                'Teacher Silver Stats',
                run_dir / 'icj_teacher_silver_stats.json',
            ),
            (
                'Emotion Gold Eval Metrics',
                run_dir / 'emotion_gold_eval_metrics.json',
            ),
            (
                'Doc-level OT Compare Stats',
                run_dir / 'icj_compare_stats.json',
            ),
            (
                'Token-OT Compare Stats (experimental)',
                run_dir / 'icj_token_ot_stats.json',
            ),
            (
                'Token-OT Check Summary (experimental)',
                run_dir / 'token_ot_check' / 'summary.json',
            ),
            (
                'Token-OT Check Report (experimental)',
                run_dir / 'token_ot_check' / 'report.md',
            ),
        ]:
            if Path(p).exists():
                summary_sections.append((label, str(p)))
        summary_html = write_summary_html_report(
            output_path=run_dir / 'icj_summary.html',
            sections=summary_sections,
            title='Summary',
        )
        if summary_sanitized is not None:
            _safe_unlink(summary_sanitized)
    most_emotional_general_html = (
        write_most_emotional_docs_html_report(
            output_path=run_dir
            / 'icj_most_emotional_general.html',
            scores_jsonl=scores_jsonl,
            top_n=int(opts.top_n),
            excerpt_chars=int(opts.excerpt_chars),
            title='Most Emotional ICJ Documents (General)',
            manual_top_docs=(
                opts.manual_top_docs
                if opts.manual_top_docs
                else None
            ),
            emotion_labels=feeling_labels,
        )
    )
    reasoning_scores: Optional[Path] = None
    reasoning_html: Optional[str] = None
    if opts.reasoning_scores_jsonl:
        reasoning_scores = Path(opts.reasoning_scores_jsonl)
        if not reasoning_scores.exists():
            raise FileNotFoundError('error: FileNotFoundError')
        reasoning_html = write_most_emotional_docs_html_report(
            output_path=run_dir
            / 'icj_court_reasoning.html',
            scores_jsonl=reasoning_scores,
            top_n=int(opts.reasoning_top_n or opts.top_n),
            excerpt_chars=0,
            full_text=True,
            collapse_whitespace=False,
            pre_max_height_px=int(opts.reasoning_pre_max_height_px),
            title='ICJ Court Reasoning (Extracted) - Most Emotional Excerpts',
            emotion_labels=feeling_labels,
        )
    most_emotional_html: str
    if reasoning_html:
        most_emotional_html = write_combined_html_report(
            output_path=run_dir
            / 'icj_most_emotional_docs.html',
            sections=[
                ('General', most_emotional_general_html),
                ('Reasoning', reasoning_html),
            ],
            title='Most Emotional Documents',
        )
    else:
        most_emotional_html = most_emotional_general_html
    from legal_emotion.timeline_viz import (
        write_emotion_timeline_html_report,
    )

    timeline_series: List[Tuple[str, str]] = [
        ('General', str(scores_jsonl))
    ]
    if reasoning_scores is not None:
        timeline_series.append(('Reasoning', str(reasoning_scores)))
    timeline_html = write_emotion_timeline_html_report(
        output_path=run_dir / 'icj_timeline.html',
        series=timeline_series,
        title='ICJ Emotionality Timeline',
        emotion_labels=feeling_labels,
    )
    compare_html = render_compare_neighbours_html(
        scores_jsonl=scores_jsonl,
        neighbours_jsonl=neighbours_jsonl,
        output_html=run_dir / 'icj_neighbours.html',
        cfg_path=(
            str(cfg_path) if cfg_path.exists() else None
        ),
        embed_text=True,
        embed_text_max_chars=0,
        embed_text_collapse_whitespace=False,
    )
    reasoning_compare_html: Optional[str] = None
    if reasoning_scores is not None:
        reasoning_neighbours = (
            Path(opts.reasoning_neighbours_jsonl)
            if opts.reasoning_neighbours_jsonl
            else run_dir / 'icj_reasoning_neighbours.jsonl'
        )
        if not reasoning_neighbours.exists():
            compare_scores(
                input_jsonl=str(reasoning_scores),
                output_path=str(reasoning_neighbours),
                cfg_path=(
                    str(cfg_path)
                    if cfg_path.exists()
                    else None
                ),
                fmt='neighbours',
                topk=int(opts.reasoning_topk),
                vis=False,
            )
        reasoning_compare_html = (
            render_compare_neighbours_html(
                scores_jsonl=reasoning_scores,
                neighbours_jsonl=reasoning_neighbours,
                output_html=run_dir
                / 'icj_reasoning_neighbours.html',
                cfg_path=(
                    str(cfg_path)
                    if cfg_path.exists()
                    else None
                ),
                embed_text=True,
                embed_text_max_chars=0,
                embed_text_collapse_whitespace=False,
            )
        )
    group_shift_html: Optional[str] = None
    summary_payload = (
        _read_json(summary_json)
        if summary_json.exists()
        else {}
    )
    maybe_group = summary_payload.get('icj_group_shift_html')
    if (
        isinstance(
            maybe_group,
            str,
        )
        and Path(maybe_group).exists()
    ):
        group_shift_html = maybe_group
    token_neighbours_html: Optional[str] = None
    maybe_token = summary_payload.get('icj_token_compare_html')
    if (
        isinstance(
            maybe_token,
            str,
        )
        and Path(maybe_token).exists()
    ):
        token_neighbours_html = maybe_token
    else:
        fallback_token = (
            run_dir / 'icj_token_neighbours.html'
        )
        if fallback_token.exists():
            token_neighbours_html = str(fallback_token)
    sections: List[Tuple[str, str]] = []
    if summary_html:
        sections.append(('Summary', summary_html))
    sections.append(('Most Emotional Documents', most_emotional_html))
    sections.append(('Timeline', timeline_html))
    if reasoning_html:
        sections.append(('ICJ_court_reasoning', reasoning_html))
    sections.append(('Doc-level OT Neighbours', compare_html))
    if reasoning_compare_html:
        sections.append((
                'Doc-level OT Neighbours (Reasoning)',
                reasoning_compare_html,
            ))
    if group_shift_html:
        sections.append((
                'Group Shift (token-OT, experimental)',
                group_shift_html,
            ))
    if token_neighbours_html:
        sections.append((
                'Token-OT Neighbours (experimental)',
                token_neighbours_html,
            ))
    combined = write_combined_html_report(
        output_path=out_html,
        sections=sections,
        title='Legal Emotion OT Report',
    )
    if opts.cleanup:
        for _, p in sections:
            pp = Path(p)
            if (
                pp.exists()
                and pp.resolve() != Path(combined).resolve()
            ):
                _safe_unlink(pp)
    print(str(combined))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
