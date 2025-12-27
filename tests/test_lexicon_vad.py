import pytest
from legal_emotion.lexicon import (
    LexiconFeaturizer,
    load_lexicon,
    load_word_vad,
)


def test_load_lexicon_applies_word_vad(tmp_path):
    emolex_path = tmp_path / 'emolex.txt'
    emolex_path.write_text(
        'abandon\tfear\t1\nwelcome\tjoy\t1\n',
        encoding='utf-8',
    )
    vad_path = tmp_path / 'vad.tsv'
    vad_path.write_text(
        'Word\tValence\tArousal\tDominance\nabandon\t0.25\t0.75\t0.5\nwelcome\t0.9\t0.4\t0.6\n',
        encoding='utf-8',
    )
    lex = load_lexicon(
        str(emolex_path),
        str(vad_path),
    )
    assert lex['fear']['abandon'] == pytest.approx((-0.5, 0.5, 0.0))
    assert lex['joy']['welcome'] == pytest.approx((0.8, -0.2, 0.2))


def test_featurizer_uses_multiword_vad_terms(tmp_path):
    emolex_path = tmp_path / 'emolex.txt'
    emolex_path.write_text(
        'abandon\tfear\t1\n',
        encoding='utf-8',
    )
    vad_path = tmp_path / 'vad.tsv'
    vad_path.write_text(
        'term\tvalence\tarousal\tdominance\na battery\t-0.2\t0.3\t0.4\n',
        encoding='utf-8',
    )
    lex = load_lexicon(
        str(emolex_path),
        str(vad_path),
    )
    vad_terms = load_word_vad(str(vad_path))
    feat = LexiconFeaturizer(
        lex,
        ['fear'],
        vad_lexicon=vad_terms,
    )
    (
        _,
        _,
        vad,
    ) = feat.vectors('This is a battery test.')
    assert vad.tolist() == pytest.approx([-0.2, 0.3, 0.4])


def test_load_word_vad_does_not_double_scale_signed_files(tmp_path):
    vad_path = tmp_path / 'vad.tsv'
    vad_path.write_text(
        'term\tvalence\tarousal\tdominance\nneutralpos\t0.000\t0.556\t0.519\nneg\t-0.200\t0.300\t0.400\n',
        encoding='utf-8',
    )
    vad_terms = load_word_vad(str(vad_path))
    assert vad_terms['neutralpos'] == pytest.approx((0.0, 0.556, 0.519))
    assert vad_terms['neg'] == pytest.approx((-0.2, 0.3, 0.4))
