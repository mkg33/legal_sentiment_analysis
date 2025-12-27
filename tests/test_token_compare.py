import json
from pathlib import Path
import pytest
from legal_emotion.token_compare import (
    compare_token_clouds,
    explain_token_pair,
)

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
NRC_LEXICON_PATH = DATA_DIR / 'NRC-Emotion-Lexicon'
NRC_VAD_PATH = DATA_DIR / 'NRC-VAD-Lexicon-v2.1'


def _write_jsonl(
    path,
    rows,
):
    path.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    return path


def _write_cfg(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                'vad_lexicon_path: null',
            ])
        + '\n',
        encoding='utf-8',
    )
    return cfg_path


def test_compare_token_clouds_neighbours_smoke(tmp_path):
    setup = _write_cfg(tmp_path)
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'The quick brown fox jumps over the lazy dog.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'A fast brown fox leaps above a sleepy dog.',
        },
        {
            'meta': {'id': 'c'},
            'text': 'Completely unrelated sentence about contracts and courts.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = tmp_path / 'neighbours.jsonl'
    stats = compare_token_clouds(
        input_jsonl=str(inp),
        output_path=str(out),
        cfg_path=str(setup),
        fmt='neighbours',
        topk=1,
        mode='sinkhorn_divergence',
        focus='all',
        cost='embedding',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=32,
        top_flows=3,
        include_explain=True,
    )
    assert stats['docs'] == 3
    lines = (
        out.read_text(encoding='utf-8').strip().splitlines()
    )
    assert len(lines) == 3
    row0 = json.loads(lines[0])
    assert row0['index'] == 0
    assert len(row0['neighbours']) == 1
    n0 = row0['neighbours'][0]
    assert n0['primary_explain'] is not None
    assert isinstance(
        n0['primary_explain']['top_transport_cost_contrib'],
        list,
    )
    assert (
        len(n0['primary_explain'][
                'top_transport_cost_contrib'
            ])
        <= 3
    )


def test_compare_token_clouds_neighbours_candidate_k_smoke(tmp_path):
    setup = _write_cfg(tmp_path)
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'The quick brown fox jumps over the lazy dog.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'A fast brown fox leaps above a sleepy dog.',
        },
        {
            'meta': {'id': 'c'},
            'text': 'Completely unrelated sentence about contracts and courts.',
        },
        {
            'meta': {'id': 'd'},
            'text': 'Another unrelated sentence about treaties and jurisdiction.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = tmp_path / 'neighbours.jsonl'
    stats = compare_token_clouds(
        input_jsonl=str(inp),
        output_path=str(out),
        cfg_path=str(setup),
        fmt='neighbours',
        topk=1,
        candidate_k=2,
        mode='sinkhorn_divergence',
        focus='all',
        cost='embedding',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=32,
        top_flows=2,
        include_explain=True,
    )
    assert stats['docs'] == 4
    assert stats['candidate_k'] == 2
    lines = (
        out.read_text(encoding='utf-8').strip().splitlines()
    )
    assert len(lines) == 4
    row0 = json.loads(lines[0])
    assert row0['index'] == 0
    assert len(row0['neighbours']) == 1
    n0 = row0['neighbours'][0]
    assert n0['primary_explain'] is not None
    assert isinstance(
        n0['primary_explain']['top_transport_cost_contrib'],
        list,
    )
    assert (
        len(n0['primary_explain'][
                'top_transport_cost_contrib'
            ])
        <= 2
    )


def test_compare_token_clouds_matrix_identical_docs_zero_distance(tmp_path):
    setup = _write_cfg(tmp_path)
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'hello world hello world',
        },
        {
            'meta': {'id': 'b'},
            'text': 'hello world hello world',
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
        focus='all',
        cost='embedding',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=16,
        include_explain=False,
    )
    assert stats['docs'] == 2
    payload = json.loads(out.read_text(encoding='utf-8'))
    D = payload['distance']
    assert len(D) == 2 and len(D[0]) == 2
    assert D[0][0] == pytest.approx(
        0.0,
        abs=1e-06,
    )
    assert D[1][1] == pytest.approx(
        0.0,
        abs=1e-06,
    )
    assert D[0][1] == pytest.approx(
        D[1][0],
        abs=1e-06,
    )
    assert D[0][1] == pytest.approx(
        0.0,
        abs=0.001,
    )


def test_explain_token_pair_smoke(tmp_path):
    setup = _write_cfg(tmp_path)
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'This agreement is binding.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'The contract shall be enforceable.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='all',
        cost='embedding',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=32,
        top_flows=5,
    )
    assert out['i']['meta']['id'] == 'a'
    assert out['j']['meta']['id'] == 'b'
    assert out['primary']['mode'] == 'sinkhorn_divergence'
    assert out['primary']['distance'] >= 0.0
    assert (
        abs(float(out['primary']['distance_minus_calc']))
        < 1e-06
    )
    assert isinstance(
        out['primary']['top_transport_cost_contrib'],
        list,
    )
    assert (
        len(out['primary']['top_transport_cost_contrib'])
        <= 5
    )


def test_explain_token_pair_includes_cost_breakdown_and_context(tmp_path):
    setup = tmp_path / 'cfg.yaml'
    setup.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{NRC_VAD_PATH}"',
                'word_vad_scale: zero_one',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'We condemn the attack and mourn the suffering.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'They condemn the threat and protest the violence.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='embedding_vad',
        alpha_embed=0.8,
        beta_vad=0.2,
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        max_terms=32,
        top_flows=5,
    )
    flows = out['primary']['top_transport_cost_contrib']
    assert flows
    f0 = flows[0]
    assert f0['cost_embed'] is not None
    assert f0['cost_vad'] is not None
    assert 'from_context' in f0 and 'to_context' in f0
    assert isinstance(
        f0['from_context'],
        (str, type(None)),
    )
    assert isinstance(
        f0['to_context'],
        (str, type(None)),
    )
    assert isinstance(
        f0['from_weight_norm'],
        float,
    )
    assert isinstance(
        f0['to_weight_norm'],
        float,
    )
    mass_flows = out['primary']['top_transport_mass']
    assert isinstance(
        mass_flows,
        list,
    )
    assert mass_flows
    m0 = mass_flows[0]
    assert m0['mass'] is not None
    assert m0['cost_embed'] is not None
    assert m0['cost_vad'] is not None


def test_emotional_focus_uses_seed_lexicon_terms(tmp_path):
    setup = tmp_path / 'cfg.yaml'
    setup.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{NRC_VAD_PATH}"',
                'word_vad_scale: zero_one',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'We condemn the threat and protest the attack.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'We welcome and celebrate the agreement.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        max_terms=32,
        top_flows=5,
    )
    flows = out['primary']['top_transport_cost_contrib']
    assert flows
    seen = {f['from'] for f in flows} | {
        f['to'] for f in flows
    }
    assert (
        'condemn' in seen
        or 'threat' in seen
        or 'protest' in seen
        or ('welcome' in seen)
        or ('celebrate' in seen)
    )


def test_emotional_vocab_controls_vad_only_terms(tmp_path):
    vad = tmp_path / 'vad.tsv'
    vad.write_text(
        '\n'.join([
                'Word\tValence\tArousal\tDominance',
                'obligation\t0.9\t0.9\t0.5',
            ])
        + '\n',
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
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{vad}"',
                'word_vad_scale: zero_one',
                'token_allow_vad_stopwords: true',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'obligation obligation obligation',
        },
        {
            'meta': {'id': 'b'},
            'text': 'obligation',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out_vad = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        vad_threshold=0.55,
        emotional_vocab='vad',
        max_terms=32,
        top_flows=5,
    )
    assert 'obligation' in set(out_vad['i']['terms']) | set(out_vad['j']['terms'])
    out_lex = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        vad_threshold=0.55,
        emotional_vocab='lexicon',
        max_terms=32,
        top_flows=5,
    )
    assert 'obligation' not in set(out_lex['i']['terms']) | set(out_lex['j']['terms'])


def test_compare_token_clouds_vis_writes_html(tmp_path):
    setup = tmp_path / 'cfg.yaml'
    setup.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{NRC_VAD_PATH}"',
                'word_vad_scale: zero_one',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'We condemn the threat.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'We condemn the attack.',
        },
        {
            'meta': {'id': 'c'},
            'text': 'We welcome and celebrate the agreement.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = tmp_path / 'neighbours.jsonl'
    stats = compare_token_clouds(
        input_jsonl=str(inp),
        output_path=str(out),
        cfg_path=str(setup),
        fmt='neighbours',
        topk=1,
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=32,
        top_flows=3,
        include_explain=True,
        vis=True,
    )
    assert 'visualization' in stats
    html_path = stats['visualization']
    assert html_path
    html = Path(html_path).read_text(encoding='utf-8')
    assert 'Token-cloud OT report' in html


def test_stopwords_file_removes_unigram_emotion_terms(tmp_path):
    setup = tmp_path / 'cfg.yaml'
    setup.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{NRC_VAD_PATH}"',
                'word_vad_scale: zero_one',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'We condemn the attack.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'We condemn the threat.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    sw = tmp_path / 'stop.txt'
    sw.write_text(
        'condemn\n',
        encoding='utf-8',
    )
    out = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        max_terms=32,
        top_flows=5,
        stopwords_file=str(sw),
    )
    assert 'condemn' not in set(out['i']['terms'])
    assert 'condemn' not in set(out['j']['terms'])


def test_stopwords_file_removes_phrase_emotion_terms(tmp_path):
    lex = tmp_path / 'lex.json'
    lex.write_text(
        json.dumps({'joy': [['not happy', [0.8, 0.4, 0.5]]]}),
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
                f'lexicon_path: "{lex}"',
                f'vad_lexicon_path: "{NRC_VAD_PATH}"',
                'word_vad_scale: zero_one',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'I am not happy.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'They are not happy.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    baseline = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        max_terms=32,
        top_flows=5,
    )
    assert 'not happy' in set(baseline['i']['terms']) | set(baseline['j']['terms'])
    sw = tmp_path / 'stop.txt'
    sw.write_text(
        'not happy\n',
        encoding='utf-8',
    )
    out = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        max_terms=32,
        top_flows=5,
        stopwords_file=str(sw),
    )
    assert 'not happy' not in set(out['i']['terms'])
    assert 'not happy' not in set(out['j']['terms'])


def test_drop_top_df_removes_corpus_boilerplate(tmp_path):
    setup = _write_cfg(tmp_path)
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'alpha common common common',
        },
        {
            'meta': {'id': 'b'},
            'text': 'beta common common common',
        },
        {
            'meta': {'id': 'c'},
            'text': 'gamma common common common',
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
        focus='all',
        cost='embedding',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=16,
        include_explain=False,
        drop_top_df=1,
    )
    assert stats['docs'] == 3
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['drop_top_df'] == 1
    assert 'common' in set(payload['dropped_top_df_terms'])
    for d in payload['docs']:
        assert 'common' not in set(d['terms'])


def test_drop_top_df_is_computed_on_corpus_tokens_not_emotional_terms(tmp_path):
    setup = tmp_path / 'cfg.yaml'
    setup.write_text(
        '\n'.join([
                'model_name: hf-internal-testing/tiny-random-bert',
                'device: cpu',
                'sinkhorn_epsilon: 0.1',
                'sinkhorn_iters: 25',
                'ot_reg_m: 0.2',
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{NRC_VAD_PATH}"',
                'word_vad_scale: zero_one',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'zzzz zzzz condemn condemn condemn',
        },
        {
            'meta': {'id': 'b'},
            'text': 'zzzz condemn condemn',
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
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=16,
        include_explain=False,
        drop_top_df=1,
        vad_threshold=0.55,
    )
    assert stats['docs'] == 2
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['drop_top_df'] == 1
    assert 'zzzz' in set(payload['dropped_top_df_terms'])
    assert 'condemn' not in set(payload['dropped_top_df_terms'])
    for d in payload['docs']:
        assert 'condemn' in set(d['terms'])


def test_focus_all_cost_vad_requires_word_vad_lexicon(tmp_path):
    setup = _write_cfg(tmp_path)
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'alpha beta gamma',
        },
        {
            'meta': {'id': 'b'},
            'text': 'alpha beta gamma',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = tmp_path / 'matrix.json'
    with pytest.warns(
        RuntimeWarning,
        match='fallback',
    ):
        compare_token_clouds(
            input_jsonl=str(inp),
            output_path=str(out),
            cfg_path=str(setup),
            fmt='matrix',
            mode='sinkhorn_divergence',
            focus='all',
            cost='vad',
            epsilon=0.1,
            iters=25,
            reg_m=0.2,
            weight='tfidf',
            max_terms=16,
            include_explain=False,
        )
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['cost'] == 'embedding'


def test_focus_all_cost_vad_filters_terms_without_vad(tmp_path):
    vad = tmp_path / 'vad.tsv'
    vad.write_text(
        '\n'.join([
                'Word\tValence\tArousal\tDominance',
                'alpha\t0.9\t0.5\t0.5',
                'beta\t0.1\t0.6\t0.5',
            ])
        + '\n',
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
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{vad}"',
                'word_vad_scale: zero_one',
                'token_allow_vad_stopwords: true',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'alpha alpha alpha zzz zzz',
        },
        {
            'meta': {'id': 'b'},
            'text': 'alpha alpha',
        },
        {
            'meta': {'id': 'c'},
            'text': 'beta beta',
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
        focus='all',
        cost='vad',
        epsilon=0.1,
        iters=25,
        reg_m=0.2,
        weight='tfidf',
        max_terms=16,
        include_explain=False,
    )
    assert stats['docs'] == 3
    payload = json.loads(out.read_text(encoding='utf-8'))
    for d in payload['docs']:
        assert 'zzz' not in set(d['terms'])


def test_emotional_focus_vad_only_terms_require_arousal_by_default(tmp_path):
    vad = tmp_path / 'vad.tsv'
    vad.write_text(
        '\n'.join([
                'Word\tValence\tArousal\tDominance',
                'obligation\t-0.8\t0.10\t0.40',
                'genocide\t-0.95\t0.80\t0.20',
            ])
        + '\n',
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
                f'lexicon_path: "{NRC_LEXICON_PATH}"',
                f'vad_lexicon_path: "{vad}"',
                'word_vad_scale: signed',
                'token_allow_vad_stopwords: true',
            ])
        + '\n',
        encoding='utf-8',
    )
    inp = tmp_path / 'docs.jsonl'
    rows = [
        {
            'meta': {'id': 'a'},
            'text': 'This is an obligation.',
        },
        {
            'meta': {'id': 'b'},
            'text': 'This is genocide.',
        },
    ]
    _write_jsonl(
        inp,
        rows,
    )
    out = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        emotional_vocab='lexicon_or_vad',
        max_terms=32,
        top_flows=3,
    )
    assert 'obligation' not in set(out['i']['terms'])
    assert 'genocide' in set(out['j']['terms'])
    relaxed = explain_token_pair(
        input_jsonl=str(inp),
        i=0,
        j=1,
        cfg_path=str(setup),
        mode='sinkhorn_divergence',
        focus='emotional',
        cost='vad',
        emotional_vocab='lexicon_or_vad',
        max_terms=32,
        top_flows=3,
        vad_min_arousal_vad_only=0.0,
    )
    assert 'obligation' in set(relaxed['i']['terms'])
