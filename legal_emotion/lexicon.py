import json
import math
import os
import re
import unicodedata
import warnings
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
import torch

_TOKEN_RE = re.compile(
    "[^\\W\\d_]+(?:['-][^\\W\\d_]+)*",
    re.UNICODE,
)
_PUNCT_TRANSLATE = str.maketrans({
        '‘': "'",
        '’': "'",
        '‛': "'",
        '′': "'",
        '‐': '-',
        '‑': '-',
        '‒': '-',
        '–': '-',
        '—': '-',
        '−': '-',
    })
_UNSET = object()


def _canonical_term(term: str) -> str:
    return ' '.join(tokenize(term))


@lru_cache(maxsize=8)
def load_stopwords(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    out: set[str] = set()
    with p.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key = _canonical_term(line)
            if key:
                out.add(key)
    return out


def _default_data_dir() -> Optional[Path]:
    try:
        base = Path(__file__).resolve().parents[1] / 'data'
    except Exception:
        return None
    if base.exists():
        return base
    return None


def _resolve_default_lexicon_path(path: Optional[str]) -> Optional[str]:
    if path:
        return path
    base = _default_data_dir()
    if base is None:
        return None
    candidate = base / 'NRC-Emotion-Lexicon'
    if candidate.exists():
        return str(candidate)
    return None


def _resolve_default_vad_path(path: Optional[str]) -> Optional[str]:
    if path:
        return path
    base = _default_data_dir()
    if base is None:
        return None
    candidate = base / 'NRC-VAD-Lexicon-v2.1'
    if candidate.exists():
        return str(candidate)
    return None


def resolve_vad_path(
    path: Optional[str],
    *,
    allow_missing: bool = False,
) -> Optional[str]:
    if path:
        return path
    if allow_missing:
        return None
    return _resolve_default_vad_path(None)


def seed_entries() -> (
    Dict[str, List[Tuple[str, Tuple[float, float, float]]]]
):
    seeds = {
        'anger': [
            ('outraged', (-0.8, 0.7, 0.6)),
            ('protest', (-0.6, 0.6, 0.5)),
            ('resentment', (-0.7, 0.5, 0.5)),
            ('condemn', (-0.6, 0.4, 0.6)),
        ],
        'fear': [
            ('alarm', (-0.6, 0.7, 0.4)),
            ('threat', (-0.7, 0.8, 0.5)),
            ('apprehension', (-0.5, 0.5, 0.4)),
        ],
        'joy': [
            ('welcome', (0.7, 0.4, 0.5)),
            ('celebrate', (0.8, 0.5, 0.6)),
            ('rejoice', (0.8, 0.6, 0.6)),
            ('praise', (0.6, 0.3, 0.6)),
        ],
        'sadness': [
            ('regret', (-0.4, 0.3, 0.4)),
            ('lament', (-0.5, 0.4, 0.4)),
            ('grieve', (-0.6, 0.4, 0.3)),
            ('mourn', (-0.6, 0.4, 0.3)),
        ],
        'disgust': [
            ('disgust', (-0.7, 0.6, 0.4)),
            ('revulsion', (-0.7, 0.6, 0.3)),
            ('nausea', (-0.6, 0.5, 0.3)),
        ],
        'surprise': [
            ('surprised', (0.1, 0.8, 0.4)),
            ('astonished', (0.2, 0.7, 0.5)),
            ('unexpected', (0.0, 0.6, 0.4)),
        ],
        'anticipation': [
            ('anticipate', (0.4, 0.5, 0.6)),
            ('expect', (0.3, 0.4, 0.6)),
            ('hope', (0.6, 0.5, 0.6)),
        ],
        'trust': [
            ('support', (0.5, 0.4, 0.6)),
            ('solidarity', (0.5, 0.3, 0.6)),
            ('cooperate', (0.6, 0.4, 0.6)),
            ('partnership', (0.6, 0.3, 0.7)),
        ],
        'resentment': [
            ('objection', (-0.4, 0.3, 0.5)),
            ('disapproval', (-0.5, 0.3, 0.5)),
            ('dissent', (-0.5, 0.4, 0.5)),
        ],
    }
    return seeds


def _default_emotion_vad() -> (
    Dict[str, Tuple[float, float, float]]
):
    seeds = seed_entries()
    protos: Dict[str, Tuple[float, float, float]] = {}
    for emotion, pairs in seeds.items():
        vals = torch.tensor(
            [vad for _, vad in pairs],
            dtype=torch.float,
        )
        protos[emotion] = tuple(vals.mean(dim=0).tolist())
    protos.setdefault(
        'anticipation',
        (0.35, 0.45, 0.55),
    )
    protos.setdefault(
        'surprise',
        (0.05, 0.75, 0.45),
    )
    protos.setdefault(
        'disgust',
        (-0.55, 0.45, 0.35),
    )
    protos.setdefault(
        'positive',
        (0.55, 0.35, 0.6),
    )
    protos.setdefault(
        'negative',
        (-0.55, 0.45, 0.35),
    )
    return protos


def _normalise_vad_scale(scale: Optional[str]) -> Optional[str]:
    if scale is None:
        return None
    s = str(scale).strip().lower()
    aliases = {
        'signed': 'signed',
        'centred': 'signed',
        'zero_one': 'zero_one',
        '0_1': 'zero_one',
        '0-1': 'zero_one',
        'one_nine': 'one_nine',
        '1_9': 'one_nine',
        '1-9': 'one_nine',
    }
    if s in aliases:
        return aliases[s]
    raise ValueError('error: ValueError')


def normalise_vad_scale(scale: Optional[str]) -> Optional[str]:
    return _normalise_vad_scale(scale)


def detect_vad_scale(values: Iterable[Tuple[float, float, float]]) -> str:
    return _detect_vad_scale(values)


def _load_json_lexicon(
    path: Path,
    *,
    vad_scale: Optional[str] = None,
) -> Dict[
    str, List[Tuple[str, Tuple[float, float, float]]]
]:
    with path.open() as f:
        user = json.load(f)
    scale_override = _normalise_vad_scale(vad_scale)
    meta_scale = None
    if isinstance(
        user,
        dict,
    ):
        for meta_key in ('__meta__', '_meta'):
            if meta_key in user and isinstance(
                user[meta_key],
                dict,
            ):
                meta_scale = user[meta_key].get('vad_scale')
                user = {
                    k: v
                    for k, v in user.items()
                    if k not in (meta_key,)
                }
                break
    if scale_override is None:
        scale_override = _normalise_vad_scale(meta_scale)
    raw_entries: List[
        Tuple[str, str, Tuple[float, float, float]]
    ] = []
    vad_values: List[Tuple[float, float, float]] = []
    for emotion, items in user.items():
        for token, vals in items:
            if (
                not isinstance(
                    vals,
                    (list, tuple),
                )
                or len(vals) < 3
            ):
                raise ValueError('error: ValueError')
            v = (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
            )
            raw_entries.append((emotion, token, v))
            vad_values.append(v)
    mode = scale_override or _detect_vad_scale(vad_values)
    out: Dict[
        str, List[Tuple[str, Tuple[float, float, float]]]
    ] = {}
    for emotion, token, vals in raw_entries:
        out.setdefault(
            emotion,
            [],
        )
        out[emotion].append((token, _scale_vad_with_mode(
                vals,
                mode,
            )))
    return out


def _find_intensity_file(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.rglob('*.txt'))
    if not candidates:
        return None
    for cand in candidates:
        if (
            cand.name
            == 'NRC-Emotion-Intensity-Lexicon-v1.txt'
        ):
            return cand
    for cand in candidates:
        name = cand.name.lower()
        if (
            'intensity' in name
            and 'forvariouslanguages' not in name
        ):
            return cand
    return candidates[0] if candidates else None


def _load_emotion_intensity(path: Optional[str | Path]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    src = _find_intensity_file(p) if p.is_dir() else p
    if src is None or not src.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with src.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            (
                term_raw,
                emo_raw,
                score_raw,
            ) = (
                parts[0],
                parts[1],
                parts[2],
            )
            term = ' '.join(tokenize(term_raw))
            if not term:
                continue
            emotion = emo_raw.strip().lower()
            try:
                score = float(score_raw)
            except Exception:
                continue
            out.setdefault(
                term,
                {},
            )
            out[term][emotion] = max(
                out[term].get(
                    emotion,
                    0.0,
                ),
                score,
            )
    return out


def _vad_scale_stats(values: Iterable[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    min_seen = float('inf')
    max_seen = float('-inf')
    sum_seen = 0.0
    count_seen = 0
    for v, a, d in values:
        v = float(v)
        a = float(a)
        d = float(d)
        min_seen = min(
            min_seen,
            v,
            a,
            d,
        )
        max_seen = max(
            max_seen,
            v,
            a,
            d,
        )
        sum_seen += v + a + d
        count_seen += 3
    if min_seen == float('inf'):
        return (min_seen, max_seen, 0.0, 0.0)
    mean = float(sum_seen / float(max(
                count_seen,
                1,
            )))
    span = float(max_seen - min_seen)
    return (min_seen, max_seen, mean, span)


def _infer_vad_scale(
    min_seen: float,
    max_seen: float,
    mean: float,
    span: float,
) -> Tuple[str, Optional[str]]:
    tol = 1e-06
    if min_seen >= 0.0 - tol and max_seen <= 1.0 + tol:
        if span >= 0.5 and 0.25 <= mean <= 0.75:
            return ('zero_one', None)
        return ('signed', 'zero_one')
    if min_seen >= 1.0 - tol and max_seen <= 9.0 + tol:
        if span >= 4.0 and 3.5 <= mean <= 6.5:
            return ('one_nine', None)
        return ('signed', 'one_nine')
    return ('signed', None)


def _detect_vad_scale(values: Iterable[Tuple[float, float, float]]) -> str:
    (
        min_seen,
        max_seen,
        mean,
        span,
    ) = _vad_scale_stats(values)
    if min_seen == float('inf'):
        return 'signed'
    (
        mode,
        _,
    ) = _infer_vad_scale(
        min_seen,
        max_seen,
        mean,
        span,
    )
    return mode


def _scale_vad_with_mode(
    vals: Tuple[float, float, float],
    mode: str,
) -> Tuple[float, float, float]:
    (
        v,
        a,
        d,
    ) = (
        float(vals[0]),
        float(vals[1]),
        float(vals[2]),
    )
    if mode == 'zero_one':
        return tuple((
                max(
                    -1.0,
                    min(
                        1.0,
                        2.0 * x - 1.0,
                    ),
                )
                for x in (v, a, d)
            ))
    if mode == 'one_nine':
        return tuple((
                max(
                    -1.0,
                    min(
                        1.0,
                        (x - 5.0) / 4.0,
                    ),
                )
                for x in (v, a, d)
            ))
    return tuple((max(
            -1.0,
            min(
                1.0,
                x,
            ),
        ) for x in (v, a, d)))


def _scale_vad(
    v: float,
    a: float,
    d: float,
) -> Tuple[float, float, float]:
    vals = (float(v), float(a), float(d))
    if all((0.0 <= x <= 1.0 for x in vals)):
        return _scale_vad_with_mode(
            vals,
            'zero_one',
        )
    if all((1.0 <= x <= 9.0 for x in vals)):
        return _scale_vad_with_mode(
            vals,
            'one_nine',
        )
    return _scale_vad_with_mode(
        vals,
        'signed',
    )


def scale_vad(
    values: Iterable[float],
    *,
    scale: Optional[str] = None,
) -> Tuple[float, float, float]:
    vals = list(values)
    if len(vals) < 3:
        raise ValueError('error: ValueError')
    (
        v,
        a,
        d,
    ) = (
        float(vals[0]),
        float(vals[1]),
        float(vals[2]),
    )
    mode = (
        _normalise_vad_scale(scale)
        if scale is not None
        else None
    )
    if mode is None:
        return _scale_vad(
            v,
            a,
            d,
        )
    return _scale_vad_with_mode(
        (v, a, d),
        mode,
    )


def _load_word_vad(
    path: Path,
    *,
    vad_scale: Optional[str] = None,
    stopwords_path: Optional[str] = None,
) -> Dict[str, Tuple[float, float, float]]:
    candidates: List[Path] = []
    if path.is_dir():
        for pat in (
            'NRC-VAD-Lexicon*.txt',
            'NRC-VAD-Lexicon*.tsv',
            'NRC-VAD-Lexicon*.csv',
        ):
            direct = sorted(path.glob(pat))
            if direct:
                vad_file = direct[0]
                break
        else:
            vad_file = None
        patterns = [
            '*VAD*Lexicon*.txt',
            '*VAD*Lexicon*.tsv',
            '*VAD*Lexicon*.csv',
            '*vad*lexicon*.txt',
            '*vad*lexicon*.tsv',
            '*vad*lexicon*.csv',
            '*VAD*.txt',
            '*VAD*.tsv',
            '*VAD*.csv',
            '*vad*.txt',
            '*vad*.tsv',
            '*vad*.csv',
        ]
        for pat in patterns:
            candidates.extend(path.glob(pat))
        candidates = [p for p in candidates if p.is_file()]
        if vad_file is None:
            if not candidates:
                raise FileNotFoundError('error: FileNotFoundError')
            unique = list({
                    p.resolve(): p for p in candidates
                }.values())
            vad_file = max(
                unique,
                key=lambda p: p.stat().st_size,
            )
    else:
        vad_file = path
        if not vad_file.exists():
            raise FileNotFoundError('error: FileNotFoundError')
    stopwords = (
        load_stopwords(stopwords_path)
        if stopwords_path
        else set()
    )
    raw: Dict[str, Tuple[float, float, float]] = {}
    min_seen = float('inf')
    max_seen = float('-inf')
    sum_seen = 0.0
    count_seen = 0
    with vad_file.open(
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('word') and (
                'valence' in line.lower()
                or 'arousal' in line.lower()
            ):
                continue
            parts = line.split('\t')
            if len(parts) == 1:
                parts = line.split(',')
            if len(parts) == 1:
                parts = line.split()
            if len(parts) < 4:
                continue
            word = parts[0].strip()
            if not word:
                continue
            if stopwords:
                key = _canonical_term(word)
                if key in stopwords:
                    continue
            floats = None
            for triple in (parts[1:4], parts[-3:]):
                try:
                    (
                        v,
                        a,
                        d,
                    ) = (
                        float(triple[0]),
                        float(triple[1]),
                        float(triple[2]),
                    )
                except Exception:
                    continue
                floats = (v, a, d)
                break
            if floats is None:
                continue
            (
                v,
                a,
                d,
            ) = floats
            min_seen = min(
                min_seen,
                v,
                a,
                d,
            )
            max_seen = max(
                max_seen,
                v,
                a,
                d,
            )
            sum_seen += float(v + a + d)
            count_seen += 3
            raw[word.lower()] = (v, a, d)
    if not raw:
        raise ValueError('error: ValueError')
    mode = _normalise_vad_scale(vad_scale)
    if mode is None:
        mean = float(sum_seen / float(max(
                    count_seen,
                    1,
                )))
        span = float(max_seen - min_seen)
        (
            mode,
            ambiguous,
        ) = _infer_vad_scale(
            min_seen,
            max_seen,
            mean,
            span,
        )
        if ambiguous:
            range_label = (
                '0-1' if ambiguous == 'zero_one' else '1-9'
            )
            warnings.warn(
                'warn: vad scale',
                RuntimeWarning,
                stacklevel=2,
            )

    def _scale(vals: Tuple[float, float, float]) -> Tuple[float, float, float]:
        (
            v,
            a,
            d,
        ) = (
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
        )
        if mode == 'zero_one':
            return tuple((
                    max(
                        -1.0,
                        min(
                            1.0,
                            2.0 * x - 1.0,
                        ),
                    )
                    for x in (v, a, d)
                ))
        if mode == 'one_nine':
            return tuple((
                    max(
                        -1.0,
                        min(
                            1.0,
                            (x - 5.0) / 4.0,
                        ),
                    )
                    for x in (v, a, d)
                ))
        return tuple((max(
                -1.0,
                min(
                    1.0,
                    x,
                ),
            ) for x in (v, a, d)))

    out: Dict[str, Tuple[float, float, float]] = {
        term: _scale(vad) for term, vad in raw.items()
    }
    return out


@lru_cache(maxsize=8)
def load_word_vad(
    path: str,
    *,
    vad_scale: Optional[str] = None,
    stopwords_path: Optional[str] = None,
) -> Dict[str, Tuple[float, float, float]]:
    return _load_word_vad(
        Path(path),
        vad_scale=vad_scale,
        stopwords_path=stopwords_path,
    )


def _load_nrc_emolex_wordlevel(
    path: Path,
    emotion_vad: Dict[str, Tuple[float, float, float]],
    word_vad: Optional[
        Dict[str, Tuple[float, float, float]]
    ] = None,
):
    out: Dict[
        str, List[Tuple[str, Tuple[float, float, float]]]
    ] = {}
    with path.open(
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            (
                word,
                emotion,
                value,
            ) = parts
            if value != '1':
                continue
            word_key = word.strip().lower()
            emotion = emotion.strip().lower()
            vad = (
                word_vad.get(word_key) if word_vad else None
            )
            if vad is None:
                vad = emotion_vad.get(emotion)
            if vad is None:
                continue
            out.setdefault(
                emotion,
                [],
            ).append((word, vad))
    return out


def _load_nrc_emolex_one_file_per_emotion(
    dir_path: Path,
    emotion_vad: Dict[str, Tuple[float, float, float]],
    word_vad: Optional[
        Dict[str, Tuple[float, float, float]]
    ] = None,
):
    out: Dict[
        str, List[Tuple[str, Tuple[float, float, float]]]
    ] = {}
    base = dir_path / 'OneFilePerEmotion'
    if not base.exists():
        return out
    for p in sorted(base.glob('*-NRC-Emotion-Lexicon.txt')):
        emotion = (
            p.name.split('-NRC-Emotion-Lexicon.txt')[0]
            .strip()
            .lower()
        )
        feeling_default_vad = emotion_vad.get(emotion)
        if feeling_default_vad is None:
            continue
        with p.open(
            encoding='utf-8',
            errors='ignore',
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word = line.split(
                    '\t',
                    1,
                )[0]
                if word:
                    word_key = word.strip().lower()
                    vad = (
                        word_vad.get(word_key)
                        if word_vad
                        else None
                    )
                    if vad is None:
                        vad = feeling_default_vad
                    out.setdefault(
                        emotion,
                        [],
                    ).append((word, vad))
    return out


@lru_cache(maxsize=8)
def load_lexicon(
    path: str = None,
    vad_path: str = None,
    *,
    lexicon_vad_scale: Optional[str] = None,
    word_vad_scale: Optional[str] = None,
    stopwords_path: Optional[str] = None,
    word_vad_stopwords_path: (
        Optional[str] | object
    ) = _UNSET,
    extra_path: Optional[str] = None,
    intensity_path: Optional[str] = None,
    intensity_min: Optional[float] = None,
    min_vad_salience: Optional[float] = None,
    min_vad_arousal: Optional[float] = None,
    require_word_vad: bool = False,
    allow_seed_only: bool = False,
    allow_missing_vad: bool = False,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    resolved_default_lex = False
    if not allow_seed_only:
        resolved = _resolve_default_lexicon_path(path)
        if path is None and resolved is not None:
            warnings.warn(
                'warn: lexicon_path',
                RuntimeWarning,
                stacklevel=2,
            )
            resolved_default_lex = True
        path = resolved
    if not allow_missing_vad:
        resolved_vad = _resolve_default_vad_path(vad_path)
        if vad_path is None and resolved_vad is not None:
            warnings.warn(
                'warn: vad_lexicon_path',
                RuntimeWarning,
                stacklevel=2,
            )
        vad_path = resolved_vad
    data = seed_entries()
    feeling_vad = _default_emotion_vad()
    word_vad = None
    if vad_path:
        vad_stopwords_path = (
            stopwords_path
            if word_vad_stopwords_path is _UNSET
            else word_vad_stopwords_path
        )
        word_vad = load_word_vad(
            vad_path,
            vad_scale=word_vad_scale,
            stopwords_path=vad_stopwords_path,
        )
    stopwords = (
        load_stopwords(stopwords_path)
        if stopwords_path
        else set()
    )
    if path is None and (not resolved_default_lex):
        warnings.warn(
            'warn: seed',
            RuntimeWarning,
            stacklevel=2,
        )
    if require_word_vad and (not word_vad):
        warnings.warn(
            'warn: word_vad',
            RuntimeWarning,
            stacklevel=2,
        )
        require_word_vad = False
    min_sal = float(min_vad_salience or 0.0)
    min_ar = float(min_vad_arousal or 0.0)
    if min_sal < 0.0:
        min_sal = 0.0
    if min_ar < 0.0:
        min_ar = 0.0
    if path:
        p = Path(path)
        user_pairs: Dict[
            str,
            List[Tuple[str, Tuple[float, float, float]]],
        ] = {}
        if p.is_file() and p.suffix.lower() == '.json':
            user_pairs = _load_json_lexicon(
                p,
                vad_scale=lexicon_vad_scale,
            )
        elif p.is_dir():
            if word_vad is None:
                try:
                    word_vad = load_word_vad(
                        str(p),
                        vad_scale=word_vad_scale,
                        stopwords_path=stopwords_path,
                    )
                except Exception:
                    word_vad = None
            wordlevel = None
            candidates = sorted(p.glob('NRC-Emotion-Lexicon-Wordlevel*.txt'))
            if candidates:
                wordlevel = candidates[0]
            if wordlevel and wordlevel.exists():
                user_pairs = _load_nrc_emolex_wordlevel(
                    wordlevel,
                    feeling_vad,
                    word_vad,
                )
            else:
                user_pairs = (
                    _load_nrc_emolex_one_file_per_emotion(
                        p,
                        feeling_vad,
                        word_vad,
                    )
                )
        elif p.is_file():
            user_pairs = _load_nrc_emolex_wordlevel(
                p,
                feeling_vad,
                word_vad,
            )
        else:
            raise FileNotFoundError('error: FileNotFoundError')
        for emotion, items in user_pairs.items():
            data.setdefault(
                emotion,
                [],
            )
            for token, vad in items:
                data[emotion].append((token, tuple(vad)))
    if extra_path:
        extra_p = Path(extra_path)
        if extra_p.exists():
            extra_pairs = _load_json_lexicon(
                extra_p,
                vad_scale=lexicon_vad_scale,
            )
            for emotion, items in extra_pairs.items():
                data.setdefault(
                    emotion,
                    [],
                )
                for token, vad in items:
                    data[emotion].append((token, tuple(vad)))
    merged: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ] = {}
    intensity_map = _load_emotion_intensity(intensity_path)
    min_intensity = float(intensity_min or 0.0)
    if min_intensity < 0.0:
        min_intensity = 0.0
    for emotion, pairs in data.items():
        entries: Dict[str, Tuple[float, float, float]] = {}
        proto = feeling_vad.get(emotion)
        for token, vad in pairs:
            key = token.lower()
            if stopwords:
                term_key = _canonical_term(key)
                if term_key in stopwords:
                    continue
            if intensity_map and min_intensity > 0.0:
                term_key = _canonical_term(key)
                intensity = intensity_map.get(
                    term_key,
                    {},
                ).get(emotion)
                if (
                    intensity is None
                    or intensity < min_intensity
                ):
                    continue
            if (
                require_word_vad
                and word_vad is not None
                and (key not in word_vad)
            ):
                continue
            if min_sal > 0.0 or min_ar > 0.0:
                valence_abs = abs(float(vad[0]))
                arousal_pos = max(
                    0.0,
                    float(vad[1]),
                )
                if (
                    min_sal > 0.0
                    and max(
                        valence_abs,
                        arousal_pos,
                    )
                    < min_sal
                ):
                    continue
                if min_ar > 0.0 and arousal_pos < min_ar:
                    continue
            if key in entries and proto is not None:
                existing = entries[key]
                tol = 0.0001
                new_is_proto = (
                    max((
                            abs(float(vad[i])
                                - float(proto[i]))
                            for i in range(3)
                        ))
                    <= tol
                )
                existing_is_proto = (
                    max((
                            abs(float(existing[i])
                                - float(proto[i]))
                            for i in range(3)
                        ))
                    <= tol
                )
                if new_is_proto and (not existing_is_proto):
                    continue
            entries[key] = vad
        merged[emotion] = entries
    return merged


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    norm = unicodedata.normalize(
        'NFKC',
        text,
    )
    norm = norm.translate(_PUNCT_TRANSLATE)
    return _TOKEN_RE.findall(norm.lower())


NEGATION_BOUNDARIES = {
    'but',
    'however',
    'though',
    'although',
    'yet',
    'nevertheless',
    'nonetheless',
    'still',
    'except',
}
NEGATION_SKIP_NEXT = {'only', 'just', 'merely', 'simply'}


def is_negator_token(
    token: str,
    neg_tokens: set[str],
) -> bool:
    if token in neg_tokens:
        return True
    return token.endswith("n't") and len(token) > 3


def normalise_negators(negators: Optional[Iterable[str]]) -> Tuple[set[str], set[Tuple[str, ...]]]:
    tokens: set[str] = set()
    phrases: set[Tuple[str, ...]] = set()
    if not negators:
        return (tokens, phrases)
    if isinstance(
        negators,
        str,
    ):
        neg_iter: Iterable[str] = [negators]
    else:
        neg_iter = negators
    for term in neg_iter:
        if not term:
            continue
        parts = tuple(tokenize(str(term)))
        if not parts:
            continue
        if len(parts) == 1:
            tokens.add(parts[0])
        else:
            phrases.add(parts)
    return (tokens, phrases)


def _match_phrase(
    tokens: List[str],
    start: int,
    phrases: List[Tuple[str, ...]],
) -> int:
    if not phrases:
        return 0
    for phrase in phrases:
        n = len(phrase)
        if n == 0 or start + n > len(tokens):
            continue
        if tuple(tokens[start : start + n]) == phrase:
            return n
    return 0


def build_negation_mask(
    tokens: List[str],
    window: int,
    neg_tokens: set[str],
    neg_phrases: set[Tuple[str, ...]],
) -> List[bool]:
    if window <= 0 or (
        not neg_tokens and (not neg_phrases)
    ):
        return [False] * len(tokens)
    phrases_sorted = (
        sorted(
            neg_phrases,
            key=len,
            reverse=True,
        )
        if neg_phrases
        else []
    )
    mask = [False] * len(tokens)
    i = 0
    while i < len(tokens):
        span = _match_phrase(
            tokens,
            i,
            phrases_sorted,
        )
        if span == 0 and is_negator_token(
            tokens[i],
            neg_tokens,
        ):
            if (
                tokens[i] == 'not'
                and i + 1 < len(tokens)
                and (tokens[i + 1] in NEGATION_SKIP_NEXT)
            ):
                i += 1
                continue
            span = 1
        if span <= 0:
            i += 1
            continue
        start = i + span
        if start >= len(tokens):
            i += span
            continue
        end = min(
            len(tokens),
            start + int(window),
        )
        for j in range(
            start,
            end,
        ):
            if tokens[j] in NEGATION_BOUNDARIES:
                break
            mask[j] = True
        i += span
    return mask


def term_starts_with_negator(
    parts: Tuple[str, ...],
    neg_tokens: set[str],
    neg_phrases: set[Tuple[str, ...]],
) -> bool:
    if not parts:
        return False
    if parts[0] in neg_tokens:
        return True
    for phrase in neg_phrases:
        n = len(phrase)
        if (
            n
            and len(parts) >= n
            and (tuple(parts[:n]) == phrase)
        ):
            return True
    return False


def _build_emotion_term_maps(
    lexicon: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ],
    emotions: List[str],
    *,
    shared_term_weighting: str = 'split',
) -> Tuple[
    Dict[str, List[Tuple[int, torch.Tensor, float]]],
    Dict[
        Tuple[str, ...],
        List[Tuple[int, torch.Tensor, float]],
    ],
    int,
]:
    weighting = (
        (shared_term_weighting or 'split').lower().strip()
    )
    if weighting not in {'split', 'none'}:
        raise ValueError('error: ValueError')
    feeling_to_idx = {e: i for i, e in enumerate(emotions)}
    term_feelings: Dict[str, set[str]] = defaultdict(set)
    for emotion, entries in lexicon.items():
        if feeling_to_idx.get(emotion) is None:
            continue
        for token in entries.keys():
            toks = tokenize(token)
            if not toks:
                continue
            term_feelings[' '.join(toks)].add(emotion)
    term_weight: Dict[str, float] = {}
    if weighting == 'split':
        for term, emos in term_feelings.items():
            denom = max(
                1,
                len(emos),
            )
            term_weight[term] = 1.0 / float(denom)
    token_to_entries: Dict[
        str, List[Tuple[int, torch.Tensor, float]]
    ] = defaultdict(list)
    phrase_to_entries: Dict[
        Tuple[str, ...],
        List[Tuple[int, torch.Tensor, float]],
    ] = defaultdict(list)
    vad_tensor_cache: Dict[
        Tuple[float, float, float], torch.Tensor
    ] = {}
    max_n = 1
    for emotion, entries in lexicon.items():
        idx = feeling_to_idx.get(emotion)
        if idx is None:
            continue
        for token, vad in entries.items():
            toks = tokenize(token)
            if not toks:
                continue
            weight = term_weight.get(
                ' '.join(toks),
                1.0,
            )
            vad_key = (
                float(vad[0]),
                float(vad[1]),
                float(vad[2]),
            )
            vad_tensor = vad_tensor_cache.get(vad_key)
            if vad_tensor is None:
                vad_tensor = torch.tensor(
                    vad_key,
                    dtype=torch.float,
                )
                vad_tensor_cache[vad_key] = vad_tensor
            if len(toks) == 1:
                token_to_entries[toks[0]].append((idx, vad_tensor, weight))
            else:
                phrase_to_entries[tuple(toks)].append((idx, vad_tensor, weight))
                max_n = max(
                    max_n,
                    len(toks),
                )
    return (
        dict(token_to_entries),
        dict(phrase_to_entries),
        max_n,
    )


def _emotion_counts_from_terms(
    tokens: List[str],
    *,
    token_to_entries: Dict[
        str, List[Tuple[int, torch.Tensor, float]]
    ],
    phrase_to_entries: Dict[
        Tuple[str, ...],
        List[Tuple[int, torch.Tensor, float]],
    ],
    max_n: int,
    num_emotions: int,
    negation_window: int = 0,
    negator_tokens: Optional[set[str]] = None,
    negator_phrases: Optional[set[Tuple[str, ...]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
    counts = torch.zeros(
        num_emotions,
        dtype=torch.float,
    )
    vad_sum = torch.zeros(
        3,
        dtype=torch.float,
    )
    vad_hits = 0.0
    raw_hits = 0
    if not tokens:
        return (counts, vad_sum, vad_hits, raw_hits)
    max_n = min(
        max(
            1,
            int(max_n),
        ),
        len(tokens),
    )
    neg_tokens = negator_tokens or set()
    neg_phrases = negator_phrases or set()
    window = max(
        0,
        int(negation_window),
    )
    neg_mask = build_negation_mask(
        tokens,
        window,
        neg_tokens,
        neg_phrases,
    )

    def _is_negated(
        start: int,
        span: int = 1,
    ) -> bool:
        if not neg_mask:
            return False
        end = min(
            len(neg_mask),
            start + max(
                1,
                int(span),
            ),
        )
        return any(neg_mask[start:end])

    i = 0
    while i < len(tokens):
        matched = False
        if phrase_to_entries and max_n > 1:
            for n in range(
                max_n,
                1,
                -1,
            ):
                if i + n > len(tokens):
                    continue
                key = tuple(tokens[i : i + n])
                entries = phrase_to_entries.get(key)
                if entries is None:
                    continue
                if _is_negated(
                    i,
                    n,
                ) and (
                    not term_starts_with_negator(
                        key,
                        neg_tokens,
                        neg_phrases,
                    )
                ):
                    i += n
                    matched = True
                    break
                for entry in entries:
                    if len(entry) == 3:
                        (
                            idx,
                            vad,
                            weight,
                        ) = entry
                    else:
                        (
                            idx,
                            vad,
                        ) = entry
                        weight = 1.0
                    counts[idx] += float(weight)
                    vad_sum += vad * float(weight)
                    vad_hits += float(weight)
                    raw_hits += 1
                i += n
                matched = True
                break
        if matched:
            continue
        token = tokens[i]
        if not _is_negated(i) or term_starts_with_negator(
            (token,),
            neg_tokens,
            neg_phrases,
        ):
            for entry in token_to_entries.get(
                token,
                [],
            ):
                if len(entry) == 3:
                    (
                        idx,
                        vad,
                        weight,
                    ) = entry
                else:
                    (
                        idx,
                        vad,
                    ) = entry
                    weight = 1.0
                counts[idx] += float(weight)
                vad_sum += vad * float(weight)
                vad_hits += float(weight)
                raw_hits += 1
        i += 1
    return (counts, vad_sum, vad_hits, raw_hits)


def lexicon_vectors(
    text: str,
    lexicon: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ],
    emotions: List[str],
    *,
    negation_window: int = 0,
    negators: Optional[Iterable[str]] = None,
    shared_term_weighting: str = 'split',
):
    tokens = tokenize(text)
    (
        token_to_entries,
        phrase_to_entries,
        max_n,
    ) = (
        _build_emotion_term_maps(
            lexicon,
            emotions,
            shared_term_weighting=shared_term_weighting,
        )
    )
    (
        neg_tokens,
        neg_phrases,
    ) = normalise_negators(negators)
    (
        counts,
        vad_sum,
        vad_hits,
        _,
    ) = (
        _emotion_counts_from_terms(
            tokens,
            token_to_entries=token_to_entries,
            phrase_to_entries=phrase_to_entries,
            max_n=max_n,
            num_emotions=len(emotions),
            negation_window=negation_window,
            negator_tokens=neg_tokens,
            negator_phrases=neg_phrases,
        )
    )
    vad_avg = vad_sum / max(
        vad_hits,
        1,
    )
    prior = counts / counts.sum().clamp(min=1.0)
    if prior.sum() == 0:
        prior = torch.full_like(
            prior,
            1.0 / len(prior),
        )
    return (counts, prior, vad_avg)


class LexiconFeaturizer:

    def __init__(
        self,
        lexicon: Dict[
            str, Dict[str, Tuple[float, float, float]]
        ],
        emotions: List[str],
        *,
        vad_lexicon: Optional[
            Dict[str, Tuple[float, float, float]]
        ] = None,
        vad_max_ngram: int = 3,
        negation_window: int = 0,
        negators: Optional[Iterable[str]] = None,
        shared_term_weighting: str = 'split',
    ):
        self.emotions = list(emotions)
        (
            token_to_entries,
            phrase_to_entries,
            max_n,
        ) = (
            _build_emotion_term_maps(
                lexicon,
                self.emotions,
                shared_term_weighting=shared_term_weighting,
            )
        )
        self.token_to_entries = token_to_entries
        self.phrase_to_entries = phrase_to_entries
        self.lex_max_ngram = int(max_n)
        self.vad_max_ngram = int(vad_max_ngram)
        self._vad_terms = self._build_vad_terms(
            vad_lexicon or {},
            vad_max_ngram=self.vad_max_ngram,
        )
        self.negation_window = max(
            0,
            int(negation_window),
        )
        (
            neg_tokens,
            neg_phrases,
        ) = normalise_negators(negators)
        self.negator_tokens = neg_tokens
        self.negator_phrases = neg_phrases

    @staticmethod
    def _build_vad_terms(
        terms: Dict[str, Tuple[float, float, float]],
        *,
        vad_max_ngram: int,
    ) -> Dict[Tuple[str, ...], torch.Tensor]:
        if not terms:
            return {}
        max_n = int(vad_max_ngram)
        if max_n <= 0:
            raise ValueError('error: ValueError')
        vad_tensor_cache: Dict[
            Tuple[float, float, float], torch.Tensor
        ] = {}
        out: Dict[Tuple[str, ...], torch.Tensor] = {}
        for term, vad in terms.items():
            tokens = tokenize(term)
            if not tokens:
                continue
            if len(tokens) > max_n:
                continue
            vad_key = (
                float(vad[0]),
                float(vad[1]),
                float(vad[2]),
            )
            vad_tensor = vad_tensor_cache.get(vad_key)
            if vad_tensor is None:
                vad_tensor = torch.tensor(
                    vad_key,
                    dtype=torch.float,
                )
                vad_tensor_cache[vad_key] = vad_tensor
            out[tuple(tokens)] = vad_tensor
        return out

    def _vad_from_terms(
        self,
        tokens: List[str],
    ) -> Tuple[torch.Tensor, int]:
        if not self._vad_terms or not tokens:
            return (torch.zeros(
                3,
                dtype=torch.float,
            ), 0)
        max_n = min(
            self.vad_max_ngram,
            len(tokens),
        )
        neg_tokens = self.negator_tokens
        neg_phrases = self.negator_phrases
        neg_mask = build_negation_mask(
            tokens,
            self.negation_window,
            neg_tokens,
            neg_phrases,
        )

        def _is_negated(
            start: int,
            span: int = 1,
        ) -> bool:
            if not neg_mask:
                return False
            end = min(
                len(neg_mask),
                start + max(
                    1,
                    int(span),
                ),
            )
            return any(neg_mask[start:end])

        vad_sum = torch.zeros(
            3,
            dtype=torch.float,
        )
        hits = 0
        i = 0
        while i < len(tokens):
            matched = False
            for n in range(
                max_n,
                0,
                -1,
            ):
                if i + n > len(tokens):
                    continue
                key = tuple(tokens[i : i + n])
                vad = self._vad_terms.get(key)
                if vad is None:
                    continue
                if _is_negated(
                    i,
                    n,
                ) and (
                    not term_starts_with_negator(
                        key,
                        neg_tokens,
                        neg_phrases,
                    )
                ):
                    i += n
                    matched = True
                    break
                vad_sum += vad
                hits += 1
                i += n
                matched = True
                break
            if not matched:
                i += 1
        return (vad_sum, hits)

    def _emotion_from_terms(
        self,
        tokens: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        return _emotion_counts_from_terms(
            tokens,
            token_to_entries=self.token_to_entries,
            phrase_to_entries=self.phrase_to_entries,
            max_n=self.lex_max_ngram,
            num_emotions=len(self.emotions),
            negation_window=self.negation_window,
            negator_tokens=self.negator_tokens,
            negator_phrases=self.negator_phrases,
        )

    def vectors_with_stats(
        self,
        text: str,
    ):
        tokens = tokenize(text)
        (
            counts,
            vad_sum_emo,
            vad_hits_emo,
            raw_hits,
        ) = (
            self._emotion_from_terms(tokens)
        )
        (
            vad_sum_vad,
            vad_hits_vad,
        ) = self._vad_from_terms(tokens)
        if vad_hits_vad > 0:
            vad_avg = vad_sum_vad / max(
                vad_hits_vad,
                1,
            )
            vad_source = 'vad_lexicon'
            vad_hits = int(vad_hits_vad)
        elif vad_hits_emo > 0:
            vad_avg = vad_sum_emo / max(
                vad_hits_emo,
                1,
            )
            vad_source = 'emotion_lexicon'
            vad_hits = int(raw_hits)
        else:
            vad_avg = torch.zeros(
                3,
                dtype=torch.float,
            )
            vad_source = 'none'
            vad_hits = 0
        total = counts.sum()
        if total.item() <= 0:
            prior = torch.full_like(
                counts,
                1.0 / len(counts),
            )
        else:
            prior = counts / total
        stats = {
            'lex_hits': int(raw_hits),
            'lex_hits_weighted': float(total.item()),
            'vad_hits': int(vad_hits),
            'vad_source': vad_source,
        }
        return (counts, prior, vad_avg, stats)

    def vectors(
        self,
        text: str,
    ):
        (
            counts,
            prior,
            vad_avg,
            _,
        ) = self.vectors_with_stats(text)
        return (counts, prior, vad_avg)


def emotion_prototypes(
    lexicon: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ],
    emotions: List[str],
) -> torch.Tensor:
    protos = []
    fallback = _default_emotion_vad()
    fallback_vals = (
        torch.tensor(
            list(fallback.values()),
            dtype=torch.float,
        )
        if fallback
        else torch.zeros(
            (0, 3),
            dtype=torch.float,
        )
    )
    fallback_mean = (
        fallback_vals.mean(dim=0)
        if fallback_vals.numel()
        else torch.zeros(
            3,
            dtype=torch.float,
        )
    )
    missing: List[str] = []
    for emotion in emotions:
        entries = lexicon.get(
            emotion,
            {},
        )
        if entries:
            vals = torch.tensor(
                list(entries.values()),
                dtype=torch.float,
            )
            protos.append(vals.mean(dim=0))
        else:
            missing.append(emotion)
            proto = fallback.get(emotion)
            if proto is None:
                protos.append(fallback_mean.clone())
            else:
                protos.append(torch.tensor(
                        proto,
                        dtype=torch.float,
                    ))
    if missing:
        warnings.warn(
            'warn: vad_prototypes',
            RuntimeWarning,
            stacklevel=2,
        )
    return torch.stack(
        protos,
        dim=0,
    )


def blend_priors(
    prior: torch.Tensor,
    logits: torch.Tensor,
    strength: float = 0.3,
):
    probs = torch.softmax(
        logits,
        dim=-1,
    )
    mixed = (1 - strength) * probs + strength * prior
    return mixed / mixed.sum(
        dim=-1,
        keepdim=True,
    )


def expand_with_neighbours(
    base_tokens: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ],
    neighbours: Dict[str, List[str]],
):
    expanded = {}
    for emotion, entries in base_tokens.items():
        expanded[emotion] = dict(entries)
        for token, vad in entries.items():
            for n in neighbours.get(
                token,
                [],
            ):
                if n not in expanded[emotion]:
                    expanded[emotion][n] = vad
    return expanded
