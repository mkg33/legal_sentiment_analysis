from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    model_name: str = 'nlpaueb/legal-bert-base-uncased'
    max_length: int = 256
    batch_size: int = 4
    eval_batch_size: int | None = None
    lr: float = 3e-05
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 0
    emotions: List[str] = field(default_factory=lambda: [
            'anger',
            'fear',
            'joy',
            'sadness',
            'trust',
            'disgust',
            'surprise',
            'anticipation',
        ])
    vad: List[str] = field(default_factory=lambda: [
            'valence',
            'arousal',
            'dominance',
        ])
    label_vad_scale: str | None = 'auto'
    device: str = 'auto'
    amp: str = 'none'
    tf32: bool = True
    grad_accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    compile: bool = False
    gradient_checkpointing: bool = False
    log_every: int = 20
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 30
    ot_mode: str = 'unbalanced_divergence'
    ot_cost: str = 'vad'
    ot_reg_m: float = 0.1
    alpha_sinkhorn: float = 0.5
    alpha_mass: float = 0.1
    alpha_vad: float = 1.0
    alpha_cls: float = 1.0
    threshold: float = 0.5
    pseudo_threshold: float = 0.8
    pseudo_class_thresholds: (
        List[float] | Dict[str, float] | None
    ) = None
    use_silver: bool = False
    silver_weight: float = 0.3
    grad_clip: float = 2.0
    save_dir: str = 'checkpoints'
    data_path: str = 'data/train.jsonl'
    eval_path: str = 'data/dev.jsonl'
    lexicon_path: str = 'data/NRC-Emotion-Lexicon'
    vad_lexicon_path: str = 'data/NRC-VAD-Lexicon-v2.1'
    lexicon_vad_scale: str | None = None
    word_vad_scale: str | None = None
    lexicon_stopwords_file: str | None = (
        'data/stopwords_legal_en.txt'
    )
    lexicon_negation_window: int = 3
    lexicon_negators: List[str] = field(default_factory=lambda: [
            'not',
            'no',
            'never',
            'without',
            'neither',
            'nor',
            'cannot',
            "can't",
            'dont',
            "don't",
            "won't",
        ])
    lexicon_shared_term_weighting: str = 'split'
    lexicon_extra_path: str | None = None
    lexicon_intensity_path: str | None = None
    lexicon_intensity_min: float = 0.0
    lexicon_min_vad_salience: float = 0.0
    lexicon_min_vad_arousal: float = 0.0
    lexicon_require_word_vad: bool = False
    lexicon_allow_seed_only: bool = False
    vad_allow_missing: bool = False
    unlabelled_path: str = None
    seed: int = 13
    count_pred_scale: str = 'counts'
    semantic_calibration: bool = True
    semantic_lexicon_strength_per_1k: float = 3.0
    semantic_lexicon_basis: str = 'chunks'
    semantic_min_signal_per_1k_words: float = 0.2
    semantic_low_signal_entropy_ratio: float = 0.9
    semantic_strict_lex_gate: bool = False
    semantic_min_lex_hits: int = 0
    semantic_min_lex_chunk_ratio: float = 0.0
    semantic_zero_low_signal: bool = False
    compare_drop_low_signal: bool = False
    compare_min_words: int = 0
    semantic_calibration_allow_seed: bool = False
    silver_teacher_model: str | None = None
    silver_teacher_batch_size: int = 16
    silver_teacher_max_length: int | None = None
    silver_force_has_lex: bool = False
    emotion_init_from_sentiment: bool = True
    init_from_sentiment_checkpoint: str | None = None
    token_ot_embed_model: str | None = (
        'BAAI/bge-base-en-v1.5'
    )
    token_ot_embed_backend: str = 'encoder'
    token_ot_embed_pooling: str = 'cls'
    token_ot_embed_batch_size: int = 64
    token_ot_embed_max_length: int = 32
    token_ot_embed_prompt_mode: str | None = 'none'
    token_ot_embed_prompt_text: str | None = None
    token_ot_allow_model_embed: bool = False
    token_emotional_vocab: str = 'lexicon'
    token_stopwords_file: str | None = (
        'data/stopwords_legal_en_token_ot.txt'
    )
    token_include_english_stopwords: bool = False
    token_allow_vad_stopwords: bool = False
    token_vad_imputed_weight: float = 0.0
    token_vad_imputed_path: str | None = None
    token_term_weights_path: str | None = None
    token_term_weight_power: float = 1.0
    token_term_weight_min: float = 0.0
    token_term_weight_max: float = 1.0
    token_term_weight_mix: float = 1.0
    token_term_weight_default: float = 1.0
    emotion_display_names: Dict[str, str] | None = None


def default_config() -> Config:
    return Config()
