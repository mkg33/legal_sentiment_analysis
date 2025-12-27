import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .lexicon import (
    LexiconFeaturizer,
    load_lexicon,
    load_word_vad,
    resolve_vad_path,
)
from .utils import load_config


def make_silver(
    input_path: str,
    output_path: str,
    cfg_path: str = None,
    prior_strength: float = 1.0,
    truncate_to_max_length: bool = True,
):
    setup = load_config(cfg_path)
    lex = load_lexicon(
        setup.lexicon_path,
        setup.vad_lexicon_path,
        lexicon_vad_scale=getattr(
            setup,
            'lexicon_vad_scale',
            None,
        ),
        word_vad_scale=getattr(
            setup,
            'word_vad_scale',
            None,
        ),
        stopwords_path=getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        ),
        extra_path=getattr(
            setup,
            'lexicon_extra_path',
            None,
        ),
        intensity_path=getattr(
            setup,
            'lexicon_intensity_path',
            None,
        ),
        intensity_min=getattr(
            setup,
            'lexicon_intensity_min',
            0.0,
        ),
        min_vad_salience=getattr(
            setup,
            'lexicon_min_vad_salience',
            0.0,
        ),
        min_vad_arousal=getattr(
            setup,
            'lexicon_min_vad_arousal',
            0.0,
        ),
        require_word_vad=bool(getattr(
                setup,
                'lexicon_require_word_vad',
                False,
            )),
        allow_seed_only=bool(getattr(
                setup,
                'lexicon_allow_seed_only',
                False,
            )),
        allow_missing_vad=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    )
    vad_path = resolve_vad_path(
        getattr(
            setup,
            'vad_lexicon_path',
            None,
        ),
        allow_missing=bool(getattr(
                setup,
                'vad_allow_missing',
                False,
            )),
    )
    vad_terms = (
        load_word_vad(
            vad_path,
            vad_scale=getattr(
                setup,
                'word_vad_scale',
                None,
            ),
            stopwords_path=getattr(
                setup,
                'lexicon_stopwords_file',
                None,
            ),
        )
        if vad_path
        else {}
    )
    featurizer = LexiconFeaturizer(
        lex,
        setup.emotions,
        vad_lexicon=vad_terms,
        negation_window=getattr(
            setup,
            'lexicon_negation_window',
            0,
        ),
        negators=getattr(
            setup,
            'lexicon_negators',
            None,
        ),
        shared_term_weighting=getattr(
            setup,
            'lexicon_shared_term_weighting',
            'split',
        ),
    )
    uniform = torch.full(
        (len(setup.emotions),),
        1.0 / len(setup.emotions),
        dtype=torch.float,
    )
    tokenizer = None
    if truncate_to_max_length and setup.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            setup.model_name,
            use_fast=True,
        )
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(input_path) as in_f, open(
        output_path,
        'w',
    ) as out_f:
        for line in in_f:
            row = json.loads(line)
            text = row.get(
                'text',
                '',
            )
            text_for_lex = text
            if tokenizer is not None:
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=setup.max_length,
                    return_tensors='pt',
                )
                text_for_lex = tokenizer.decode(
                    enc['input_ids'][0].tolist(),
                    skip_special_tokens=True,
                )
            (
                counts,
                prior,
                vad,
            ) = featurizer.vectors(text_for_lex)
            if prior_strength < 1.0:
                prior = (
                    prior_strength * prior
                    + (1.0 - prior_strength) * uniform
                )
                prior = prior / prior.sum().clamp_min(1e-08)
            row['silver_labels'] = prior.tolist()
            row['silver_vad'] = vad.tolist()
            row['silver_counts'] = counts.tolist()
            if tokenizer is not None:
                row['silver_max_length'] = int(setup.max_length)
            out_f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
    )
    parser.add_argument(
        '--output',
        required=True,
    )
    parser.add_argument(
        '--config',
        default=None,
    )
    parser.add_argument(
        '--no_truncate',
        action='store_true',
    )
    args = parser.parse_args()
    make_silver(
        args.input,
        args.output,
        args.config,
        truncate_to_max_length=not args.no_truncate,
    )
