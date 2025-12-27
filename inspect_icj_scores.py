import json
import random
import sys
from pathlib import Path


def inspect_scores(
    jsonl_path,
    num_samples=30,
):
    print(f'read {jsonl_path}')
    try:
        with open(
            jsonl_path,
            'r',
            encoding='utf-8',
        ) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'error: missing {jsonl_path}')
        return
    total_docs = len(lines)
    print(f'total {total_docs}')
    if total_docs == 0:
        print('empty')
        return
    sampled_lines = random.sample(
        lines,
        min(
            num_samples,
            total_docs,
        ),
    )
    print(f'\nsample {len(sampled_lines)}\n')
    for i, line in enumerate(sampled_lines):
        data = json.loads(line)
        meta = data.get(
            'meta',
            {},
        )
        path = meta.get(
            'path',
            'UNKNOWN_PATH',
        )
        print(f'doc {i + 1}')
        print(f'path {path}')
        text_content = '(Could not read file)'
        try:
            with open(
                path,
                'r',
                encoding='utf-8',
            ) as tf:
                text_content = tf.read(1500) + '...'
        except Exception as e:
            text_content = f'(Error reading text: {e})'
        emotions = data.get(
            'emotions',
            [],
        )
        guess_dist = data.get(
            'pred_dist',
            [],
        )
        guess_counts = data.get(
            'pred_counts',
            [],
        )
        lex_counts = data.get(
            'lex_counts',
            [],
        )
        scores_display = []
        for emp, score, count, lex in zip(
            emotions,
            guess_dist,
            guess_counts,
            lex_counts,
        ):
            scores_display.append(f'{emp}: {score:.3f} (cnt: {count:.1f}, lex: {lex:.1f})')
        print('scores')
        for s in scores_display:
            print(f'  {s}')
        print('\ntext')
        print(text_content)
        print('\n' + '-' * 20 + '\n')


if __name__ == '__main__':
    jsonl_path = 'outputs/gold_to_icj_ot_20251222_132451_warm/icj_scores.jsonl'
    inspect_scores(jsonl_path)
