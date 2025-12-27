import json
from pathlib import Path
import pytest
from legal_emotion.data import LegalEmotionDataset

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DEFAULT_LEXICON_PATH = DATA_DIR / 'NRC-Emotion-Lexicon'
DEFAULT_VAD_LEXICON_PATH = DATA_DIR / 'NRC-VAD-Lexicon-v2.1'


def make_tmp_jsonl(
    tmp_path,
    rows,
):
    path = tmp_path / 'data.jsonl'
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return path


def test_dataset_reads_minimal(tmp_path):
    rows = [
        {
            'text': 'example text',
            'labels': ['anger'],
            'vad': [0.1, 0.2, 0.3],
        }
    ]
    path = make_tmp_jsonl(
        tmp_path,
        rows,
    )
    ds = LegalEmotionDataset(
        str(path),
        'hf-internal-testing/tiny-random-bert',
        ['anger', 'joy'],
        16,
        lexicon_path=str(DEFAULT_LEXICON_PATH),
        vad_lexicon_path=str(DEFAULT_VAD_LEXICON_PATH),
    )
    item = ds[0]
    assert item['labels'].shape[0] == 2
    assert item['vad'].shape[0] == 3
    assert item['lex_prior'].shape[0] == 2


def test_dataset_raises_on_empty(tmp_path):
    path = tmp_path / 'empty.jsonl'
    path.touch()
    with pytest.raises(ValueError):
        LegalEmotionDataset(
            str(path),
            'bert-base-uncased',
            ['anger', 'joy'],
            16,
            lexicon_path=str(DEFAULT_LEXICON_PATH),
            vad_lexicon_path=str(DEFAULT_VAD_LEXICON_PATH),
        )


def test_dataset_label_mask_partial_and_exhaustive(tmp_path):
    rows = [
        {
            'text': 'example text',
            'labels': ['anger'],
        },
        {
            'text': 'example text',
            'labels': [],
            'labels_exhaustive': True,
        },
        {
            'text': 'example text',
            'labels': [],
            'labels_negative': ['joy'],
        },
    ]
    path = make_tmp_jsonl(
        tmp_path,
        rows,
    )
    ds = LegalEmotionDataset(
        str(path),
        'hf-internal-testing/tiny-random-bert',
        ['anger', 'joy'],
        16,
        lexicon_path=str(DEFAULT_LEXICON_PATH),
        vad_lexicon_path=str(DEFAULT_VAD_LEXICON_PATH),
    )
    item0 = ds[0]
    item1 = ds[1]
    item2 = ds[2]
    assert item0['label_mask'].tolist() == [1.0, 0.0]
    assert item1['label_mask'].tolist() == [1.0, 1.0]
    assert item2['label_mask'].tolist() == [0.0, 1.0]
