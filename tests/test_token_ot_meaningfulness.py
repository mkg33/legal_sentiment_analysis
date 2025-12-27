import json
from pathlib import Path
from legal_emotion.token_compare import compare_token_clouds


def _write_jsonl(
    path: Path,
    rows,
):
    path.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    return path


def _write_cfg(
    tmp_path: Path,
    lex_path: Path,
    vad_path: Path,
) -> Path:
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 20',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{lex_path}"',
                f'vad_lexicon_path: "{vad_path}"',
                'token_emotional_vocab: lexicon',
                'token_vad_imputed_weight: 0.0',
            ])
        + '\n',
        encoding='utf-8',
    )
    return cfg_path


def test_token_ot_vad_meaningfulness(tmp_path):
    lex_path = tmp_path / 'emolex.txt'
    lex_path.write_text(
        '\n'.join([
                'happy\tjoy\t1',
                'sad\tsadness\t1',
                'angry\tanger\t1',
            ])
        + '\n',
        encoding='utf-8',
    )
    vad_path = tmp_path / 'vad.tsv'
    vad_path.write_text(
        '\n'.join([
                'word\tvalence\tarousal\tdominance',
                'happy\t0.9\t0.6\t0.6',
                'sad\t0.1\t0.4\t0.4',
                'angry\t0.1\t0.7\t0.4',
            ])
        + '\n',
        encoding='utf-8',
    )
    setup = _write_cfg(
        tmp_path,
        lex_path,
        vad_path,
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'happy happy',
        },
        {
            'meta': {'id': 'b'},
            'text': 'happy',
        },
        {
            'meta': {'id': 'c'},
            'text': 'sad sad',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = tmp_path / 'matrix.json'
    stats = compare_token_clouds(
        input_jsonl=str(inp),
        output_path=str(out),
        cfg_path=str(setup),
        fmt='matrix',
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        epsilon=0.1,
        iters=20,
        reg_m=0.2,
        weight='tf',
        max_terms=16,
        include_explain=False,
    )
    assert stats['docs'] == 3
    payload = json.loads(out.read_text(encoding='utf-8'))
    D = payload['distance']
    assert D[0][1] < D[0][2]
