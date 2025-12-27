import json


def test_explain_pair_smoke(tmp_path):
    from legal_emotion.compare import explain_pair

    scores = tmp_path / 'scores.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.8, 0.2],
            'pred_mixscaled_per_1k_words': [8.0, 2.0],
            'pred_per_1k_words': 10.0,
            'pred_per_1k_tokens': 5.0,
        },
        {
            'meta': {'id': 'b'},
            'emotions': ['anger', 'joy'],
            'pred_softmax': [0.3, 0.7],
            'pred_mixscaled_per_1k_words': [3.0, 7.0],
            'pred_per_1k_words': 10.0,
            'pred_per_1k_tokens': 5.0,
        },
    ]
    scores.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    out = explain_pair(
        input_jsonl=str(scores),
        i=0,
        j=1,
        cost='uniform',
        mode='unbalanced_divergence',
        measure='pred_mixscaled_per_1k_words',
        style_mode='sinkhorn_divergence',
        style_measure='pred_softmax',
        epsilon=0.1,
        iters=100,
        reg_m=0.5,
        top_flows=5,
    )
    assert out['i']['meta']['id'] == 'a'
    assert out['j']['meta']['id'] == 'b'
    assert out['primary']['mode'] == 'unbalanced_divergence'
    assert out['style']['measure'] == 'pred_softmax'
    assert out['primary']['distance'] >= 0.0
    assert out['style']['distance'] >= 0.0
    assert out['primary']['divergence_calc'] is not None
    assert out['style']['divergence_calc'] is not None
    assert (
        abs(float(out['primary']['distance_minus_calc']))
        < 0.001
    )
    assert (
        abs(float(out['style']['distance_minus_calc']))
        < 0.001
    )
    assert (
        len(out['primary']['top_transport_cost_contrib'])
        <= 5
    )
    assert (
        len(out['style']['top_transport_cost_contrib']) <= 5
    )
