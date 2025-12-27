import torch
import pytest
from legal_emotion.compare import (
    eot_distance_matrix,
    eot_neighbours,
    extract_measure,
)
from legal_emotion.losses import cost_matrix


def test_extract_measure_per_1k_words():
    row = {
        'pred_counts': [10.0, 0.0],
        'n_words': 2000,
        'n_tokens': 100,
    }
    vec = extract_measure(
        row,
        'pred_counts_per_1k_words',
    )
    assert vec.tolist() == pytest.approx([5.0, 0.0])


def test_eot_distance_matrix_symmetry_and_diag_zero():
    X = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        dtype=torch.float,
    )
    C = cost_matrix(
        3,
        torch.device('cpu'),
    )
    D = eot_distance_matrix(
        X,
        C,
        mode='sinkhorn_divergence',
        epsilon=0.1,
        iters=50,
        reg_m=0.1,
    )
    assert D.shape == (3, 3)
    assert torch.allclose(
        D,
        D.t(),
        atol=1e-05,
    )
    assert torch.allclose(
        torch.diag(D),
        torch.zeros(3),
        atol=1e-05,
    )
    assert D[0, 1].item() > 0.0001


def test_eot_neighbours_prefers_mixture_over_opposite():
    X = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
        dtype=torch.float,
    )
    C = cost_matrix(
        3,
        torch.device('cpu'),
    )
    neigh = eot_neighbours(
        X,
        C,
        mode='sinkhorn_divergence',
        epsilon=0.1,
        iters=50,
        reg_m=0.1,
        topk=1,
    )
    assert neigh[0][0][0] == 2


def test_compare_scores_neighbours_include_ot_explanations(tmp_path):
    import json
    from legal_emotion.compare import compare_scores

    scores = tmp_path / 'scores.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.9, 0.1],
        },
        {
            'meta': {'id': 'b'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.1, 0.9],
        },
    ]
    scores.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    out = tmp_path / 'neighbours.jsonl'
    compare_scores(
        input_jsonl=str(scores),
        output_path=str(out),
        fmt='neighbours',
        topk=1,
        mode='sinkhorn_divergence',
        measure='pred_softmax',
        style_mode='sinkhorn_divergence',
        style_measure='pred_softmax',
        cost='uniform',
        epsilon=0.1,
        iters=50,
        reg_m=0.1,
        include_style=True,
        include_explain=True,
        top_flows=5,
    )
    lines = (
        out.read_text(encoding='utf-8').strip().splitlines()
    )
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0['index'] == 0
    assert len(row0['neighbours']) == 1
    n0 = row0['neighbours'][0]
    assert n0['primary_explain'] is not None
    assert (
        n0['primary_explain']['divergence_calc'] is not None
    )
    assert (
        abs(float(n0['primary_explain']['distance_minus_calc']))
        < 0.001
    )
    assert isinstance(
        n0['primary_explain']['top_transport_cost_contrib'],
        list,
    )
    assert (
        len(n0['primary_explain'][
                'top_transport_cost_contrib'
            ])
        <= 5
    )
    assert n0['style_explain'] is not None


def test_compare_scores_matrix_include_neighbours_and_explanations(tmp_path):
    import json
    from legal_emotion.compare import compare_scores

    scores = tmp_path / 'scores.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.9, 0.1],
        },
        {
            'meta': {'id': 'b'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.1, 0.9],
        },
    ]
    scores.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    out = tmp_path / 'matrix.json'
    compare_scores(
        input_jsonl=str(scores),
        output_path=str(out),
        fmt='matrix',
        topk=1,
        mode='sinkhorn_divergence',
        measure='pred_softmax',
        style_mode='sinkhorn_divergence',
        style_measure='pred_softmax',
        cost='uniform',
        epsilon=0.1,
        iters=50,
        reg_m=0.1,
        include_style=True,
        include_explain=True,
        top_flows=5,
    )
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['docs'][0]['index'] == 0
    assert 'neighbours' in payload
    assert payload['neighbours'][0]['index'] == 0
    assert len(payload['neighbours'][0]['neighbours']) == 1
    n0 = payload['neighbours'][0]['neighbours'][0]
    assert n0['primary_explain'] is not None
    assert (
        n0['primary_explain']['divergence_calc'] is not None
    )
    assert (
        abs(float(n0['primary_explain']['distance_minus_calc']))
        < 0.001
    )


def test_compare_scores_vis_writes_html_report(tmp_path):
    import json
    from pathlib import Path
    from legal_emotion.compare import compare_scores

    scores = tmp_path / 'scores.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.9, 0.1],
        },
        {
            'meta': {'id': 'b'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.1, 0.9],
        },
    ]
    scores.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    out = tmp_path / 'neighbours.jsonl'
    stats = compare_scores(
        input_jsonl=str(scores),
        output_path=str(out),
        fmt='neighbours',
        topk=1,
        mode='sinkhorn_divergence',
        measure='pred_softmax',
        style_mode='sinkhorn_divergence',
        style_measure='pred_softmax',
        cost='uniform',
        epsilon=0.1,
        iters=50,
        reg_m=0.1,
        include_style=True,
        include_explain=True,
        top_flows=3,
        vis=True,
    )
    assert 'visualization' in stats
    html_path = Path(stats['visualization'])
    assert html_path.exists()
    html = html_path.read_text(encoding='utf-8')
    assert 'Legal Emotion OT Report' in html
    assert 'top_transport_cost_contrib' in html
