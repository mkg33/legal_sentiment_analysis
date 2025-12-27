import json
import subprocess
import sys
from pathlib import Path


def test_group_shift_script_smoke(tmp_path):
    data_dir = tmp_path / 'icj'
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        data_dir / 'ICJ_001_2020-01-01_Judgment.txt'
    ).write_text(
        'We condemn the attack and mourn the suffering.',
        encoding='utf-8',
    )
    (data_dir / 'ICJ_002_2020-01-02_Order.txt').write_text(
        'They welcome the agreement and celebrate the decision.',
        encoding='utf-8',
    )
    setup = tmp_path / 'cfg.yaml'
    setup.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                'token_ot_embed_model: hf-internal-testing/tiny-random-bert',
                'token_ot_embed_backend: encoder',
                'token_ot_embed_pooling: cls',
                'token_ot_embed_batch_size: 16',
                'token_ot_embed_max_length: 16',
            ])
        + '\n',
        encoding='utf-8',
    )
    out_dir = tmp_path / 'out'
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1]
            / 'scripts'
            / 'run_token_ot_group_shift.py'),
        '--data_dir',
        str(data_dir),
        '--config',
        str(setup),
        '--out_dir',
        str(out_dir),
        '--group_by',
        'doc_type',
        '--focus',
        'all',
        '--cost',
        'embedding',
        '--drop_top_df',
        '0',
        '--weight',
        'tf',
        '--max_terms',
        '16',
        '--no_vis',
    ]
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((out_dir / 'summary.json').read_text(encoding='utf-8'))
    report_json = Path(summary['report_json'])
    assert report_json.exists()
    report = json.loads(report_json.read_text(encoding='utf-8'))
    assert len(report['docs']) == 2
    assert len(report['neighbours']) == 2
    ex = report['neighbours'][0]['neighbours'][0][
        'primary_explain'
    ]
    assert 'top_transport_cost_contrib' in ex
    assert 'top_transport_mass' in ex
