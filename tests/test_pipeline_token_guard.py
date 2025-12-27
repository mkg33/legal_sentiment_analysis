import argparse
from scripts.run_gold_to_icj_ot_pipeline import (
    _guard_token_coverage,
)


def _args(**overrides):
    base = {
        'no_token_guard': False,
        'token_guard_min_coverage': 0.03,
        'token_guard_min_selected_ratio': 0.02,
        'token_guard_max_zero_ratio': 0.02,
        'token_guard_allow_warn': False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_token_guard_reads_selected_ratio_from_metrics():
    token_stats = {
        'coverage_total': 0.05,
        'docs': 100,
        'docs_with_zero_selected': 0,
    }
    token_check = {
        'verdict': 'pass',
        'metrics': {'selected_ratio_median': 0.03},
    }
    _guard_token_coverage(
        token_stats=token_stats,
        token_check=token_check,
        args=_args(),
    )


def test_token_guard_accepts_evaluation_wrapper_schema():
    token_stats = {
        'coverage_total': 0.05,
        'docs': 100,
        'docs_with_zero_selected': 0,
    }
    token_check = {
        'evaluation': {
            'verdict': 'pass',
            'metrics': {'selected_ratio_median': 0.03},
        }
    }
    _guard_token_coverage(
        token_stats=token_stats,
        token_check=token_check,
        args=_args(),
    )
