import json
from pathlib import Path
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DEFAULT_LEXICON_PATH = DATA_DIR / 'NRC-Emotion-Lexicon'
DEFAULT_VAD_LEXICON_PATH = DATA_DIR / 'NRC-VAD-Lexicon-v2.1'


def _write_csv(
    path: Path,
    rows: list[dict],
) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(
        path,
        index=False,
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def test_sigmalaw_prepare_writes_splits(tmp_path):
    from legal_emotion.sentiment_prepare import (
        prepare_sigmalaw_absa,
    )

    csv_path = tmp_path / 'sigmalaw.csv'
    rows = []
    for i in range(10):
        s = [-1, 0, 1][i % 3]
        rows.append({
                'Sentence': f'example {i}',
                'Party': '[]',
                'Sentiment': '[]',
                'Overall Sentiment': s,
            })
    _write_csv(
        csv_path,
        rows,
    )
    out_dir = tmp_path / 'out'
    stats = prepare_sigmalaw_absa(
        out_dir=out_dir,
        csv_path=csv_path,
        dev_ratio=0.2,
        seed=13,
        download=False,
    )
    assert stats['n_total'] == 10
    assert stats['n_dev'] == 2
    assert (out_dir / 'train.jsonl').exists()
    assert (out_dir / 'dev.jsonl').exists()
    train = _read_jsonl(out_dir / 'train.jsonl')
    dev = _read_jsonl(out_dir / 'dev.jsonl')
    assert len(train) == 8
    assert len(dev) == 2
    assert {'text', 'label', 'sentiment', 'meta'} <= set(train[0].keys())


def test_sentiment_training_and_predict_smoke(tmp_path):
    from legal_emotion.sentiment_config import (
        SentimentConfig,
    )
    from legal_emotion.sentiment_predict import (
        predict_sentiment,
    )
    from legal_emotion.sentiment_report import (
        evaluate_sentiment_checkpoint,
    )
    from legal_emotion.sentiment_train import (
        run_sentiment_training,
    )

    train_path = tmp_path / 'train.jsonl'
    dev_path = tmp_path / 'dev.jsonl'
    train_rows = [
        {
            'text': 'good outcome',
            'label': 2,
        },
        {
            'text': 'bad outcome',
            'label': 0,
        },
        {
            'text': 'neutral outcome',
            'label': 1,
        },
        {
            'text': 'very good',
            'label': 2,
        },
    ]
    dev_rows = [
        {
            'text': 'very bad',
            'label': 0,
        },
        {
            'text': 'ok',
            'label': 1,
        },
    ]
    train_path.write_text(
        '\n'.join((json.dumps(r) for r in train_rows))
        + '\n',
        encoding='utf-8',
    )
    dev_path.write_text(
        '\n'.join((json.dumps(r) for r in dev_rows)) + '\n',
        encoding='utf-8',
    )
    setup = SentimentConfig(
        model_name='hf-internal-testing/tiny-random-bert',
        device='cpu',
        max_length=16,
        batch_size=2,
        num_epochs=1,
        warmup_steps=0,
        data_path=str(train_path),
        eval_path=str(dev_path),
        save_dir=str(tmp_path / 'ckpt'),
        log_every=0,
    )
    run_sentiment_training(setup)
    ckpt = Path(setup.save_dir) / 'sentiment.pt'
    assert ckpt.exists()
    (
        probs,
        guess_sent,
    ) = predict_sentiment(
        ['good', 'bad'],
        str(ckpt),
        setup,
    )
    assert probs.shape == (2, 3)
    assert guess_sent.shape == (2,)
    metrics = evaluate_sentiment_checkpoint(
        checkpoint=str(ckpt),
        data_path=str(dev_path),
        batch_size=2,
    )
    assert metrics['n_total'] == len(dev_rows)
    assert (
        metrics['confusion_matrix']
        and len(metrics['confusion_matrix']) == 3
    )


def test_sentiment_score_dir_smoke(tmp_path):
    from legal_emotion.sentiment_config import (
        SentimentConfig,
    )
    from legal_emotion.sentiment_score import (
        score_txt_dir_sentiment,
    )
    from legal_emotion.sentiment_train import (
        run_sentiment_training,
    )

    train_path = tmp_path / 'train.jsonl'
    dev_path = tmp_path / 'dev.jsonl'
    train_path.write_text(
        json.dumps({
                'text': 'good',
                'label': 2,
            })
        + '\n',
        encoding='utf-8',
    )
    dev_path.write_text(
        json.dumps({
                'text': 'bad',
                'label': 0,
            })
        + '\n',
        encoding='utf-8',
    )
    setup = SentimentConfig(
        model_name='hf-internal-testing/tiny-random-bert',
        device='cpu',
        max_length=16,
        batch_size=1,
        num_epochs=1,
        warmup_steps=0,
        data_path=str(train_path),
        eval_path=str(dev_path),
        save_dir=str(tmp_path / 'ckpt'),
        log_every=0,
    )
    run_sentiment_training(setup)
    ckpt = Path(setup.save_dir) / 'sentiment.pt'
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'a.txt').write_text(
        'good good good',
        encoding='utf-8',
    )
    out_jsonl = tmp_path / 'scores.jsonl'
    stats = score_txt_dir_sentiment(
        checkpoint=str(ckpt),
        input_dir=str(docs),
        output_jsonl=str(out_jsonl),
        cfg_path=None,
        recursive=True,
        stride=0,
        batch_size=1,
    )
    assert stats['written'] == 1
    rows = _read_jsonl(out_jsonl)
    assert (
        rows
        and 'probs' in rows[0]
        and (len(rows[0]['probs']) == 3)
    )


def test_init_encoder_from_sentiment_checkpoint_smoke(tmp_path):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
    )
    from legal_emotion.model import LegalEmotionModel
    from legal_emotion.transfer import (
        init_encoder_from_sentiment_checkpoint,
    )

    model_name = 'hf-internal-testing/tiny-random-bert'
    clf = (
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True,
        )
    )
    ckpt = tmp_path / 'sentiment.pt'
    torch.save(
        {
            'model': clf.state_dict(),
            'cfg': {'model_name': model_name},
        },
        ckpt,
    )
    emo = LegalEmotionModel(
        model_name,
        num_emotions=3,
    )
    info = init_encoder_from_sentiment_checkpoint(
        emo.encoder,
        str(ckpt),
        expected_model_name=model_name,
    )
    assert info['loaded_keys'] > 0
    info2 = init_encoder_from_sentiment_checkpoint(
        emo.encoder,
        str(ckpt),
        expected_model_name=f'{model_name}-other',
    )
    assert info2['loaded_keys'] > 0
    assert info2['model_name_match'] is False


def test_gold_eval_smoke(tmp_path):
    import torch
    from legal_emotion.gold_eval import evaluate_checkpoint
    from legal_emotion.model import LegalEmotionModel

    model_name = 'hf-internal-testing/tiny-random-bert'
    emotions = ['anger', 'fear', 'joy']
    model = LegalEmotionModel(
        model_name,
        num_emotions=len(emotions),
    )
    ckpt = tmp_path / 'model.pt'
    torch.save(
        {
            'model': model.state_dict(),
            'cfg': {
                'model_name': model_name,
                'emotions': emotions,
                'max_length': 16,
                'batch_size': 2,
                'device': 'cpu',
                'lexicon_path': str(DEFAULT_LEXICON_PATH),
                'vad_lexicon_path': str(DEFAULT_VAD_LEXICON_PATH),
            },
        },
        ckpt,
    )
    gold_path = tmp_path / 'gold.jsonl'
    rows = [
        {
            'text': 'The witness was evasive.',
            'labels': ['anger'],
            'vad': [-0.2, 0.3, 0.1],
        },
        {
            'text': 'The court welcomes the agreement.',
            'labels': ['joy'],
            'vad': [0.4, 0.2, 0.3],
        },
    ]
    gold_path.write_text(
        '\n'.join((json.dumps(r) for r in rows)) + '\n',
        encoding='utf-8',
    )
    metrics = evaluate_checkpoint(
        checkpoint=str(ckpt),
        data_path=str(gold_path),
        batch_size=2,
    )
    assert metrics['n_gold_labels'] == len(rows)
    assert 'f1_micro_gold' in metrics
