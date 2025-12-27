from __future__ import annotations
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .corpus import read_text, strip_icj_boilerplate
from .losses import sinkhorn_cost_parts
from .utils import get_device, load_config
from .lexicon import (
    _default_emotion_vad,
    build_negation_mask as _lex_build_negation_mask,
    is_negator_token as _lex_is_negator_token,
    load_lexicon,
    load_word_vad,
    normalise_negators as _lex_normalise_negators,
    resolve_vad_path,
    term_starts_with_negator as _lex_term_starts_with_negator,
    tokenize as lex_tokenize,
)

try:
    from sklearn.feature_extraction.text import (
        ENGLISH_STOP_WORDS,
    )
except Exception:
    ENGLISH_STOP_WORDS = frozenset()


@dataclass(frozen=True)
class TokenCloud:
    terms: List[str]
    weights: torch.Tensor
    X: torch.Tensor
    vad: torch.Tensor
    vad_conf: Optional[torch.Tensor] = None
    C_embed_self: Optional[torch.Tensor] = None
    C_vad_self: Optional[torch.Tensor] = None


_VAD_MAX_DIST = math.sqrt(3.0) * 2.0


def _read_jsonl(
    path: str | Path,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(
        'r',
        encoding='utf-8',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= max(
                0,
                int(limit),
            ):
                break
    return rows


def _load_stopword_terms(path: Optional[str | Path]) -> Tuple[set[str], set[str]]:
    if path is None:
        return (set(), set())
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    uni: set[str] = set()
    phrases: set[str] = set()
    with p.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = lex_tokenize(line)
            if not toks:
                continue
            term = ' '.join(toks)
            if len(toks) == 1:
                uni.add(term)
            else:
                phrases.add(term)
    return (uni, phrases)


def _default_stopwords_path() -> Optional[Path]:
    base = Path(__file__).resolve().parents[1]
    p = base / 'data' / 'stopwords_legal_en_token_ot.txt'
    return p if p.exists() else None


def _canonical_term(term: str) -> str:
    return ' '.join(lex_tokenize(term))


def _load_term_weights(path: Optional[str | Path]) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    if not path:
        return ({}, None)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    payload = json.loads(p.read_text(encoding='utf-8'))
    meta = None
    data = payload
    if isinstance(
        payload,
        dict,
    ):
        meta = (
            payload.get('__meta__')
            or payload.get('_meta')
            or payload.get('meta')
        )
        if 'weights' in payload and isinstance(
            payload['weights'],
            (dict, list),
        ):
            data = payload['weights']
        elif meta is not None:
            data = {
                k: v
                for k, v in payload.items()
                if k not in {'__meta__', '_meta', 'meta'}
            }
    items = []
    if isinstance(
        data,
        dict,
    ):
        items = list(data.items())
    elif isinstance(
        data,
        list,
    ):
        items = data
    else:
        raise ValueError('error: ValueError')
    out: Dict[str, float] = {}
    for entry in items:
        term = None
        weight = None
        if (
            isinstance(
                entry,
                (list, tuple),
            )
            and len(entry) >= 2
        ):
            term = entry[0]
            weight = entry[1]
        elif isinstance(
            entry,
            dict,
        ):
            term = (
                entry.get('term')
                or entry.get('token')
                or entry.get('word')
            )
            weight = entry.get('weight')
            if weight is None:
                weight = entry.get('score')
        else:
            continue
        if term is None or weight is None:
            continue
        key = _canonical_term(str(term))
        if not key:
            continue
        try:
            weight_f = float(weight)
        except Exception:
            continue
        out[key] = weight_f
    return (out, meta)


def _apply_term_weights(
    terms: List[str],
    weights: List[float],
    term_weights: Dict[str, float],
    *,
    weight_default: float,
    weight_min: float,
    weight_max: float,
    weight_power: float,
    weight_mix: float,
) -> Tuple[List[float], List[float]]:
    if not terms or not weights or (not term_weights):
        return (
            list(weights),
            [float(weight_default) for _ in terms],
        )
    out_weights: List[float] = []
    out_mult: List[float] = []
    w_min = float(weight_min)
    w_max = float(weight_max)
    w_pow = float(weight_power)
    mix = float(weight_mix)
    if w_max < w_min:
        w_max = w_min
    mix = max(
        0.0,
        min(
            1.0,
            mix,
        ),
    )
    for term, base in zip(
        terms,
        weights,
    ):
        raw = float(term_weights.get(
                term,
                weight_default,
            ))
        clamped = max(
            w_min,
            min(
                w_max,
                raw,
            ),
        )
        scaled = clamped**w_pow
        mult = 1.0 - mix + mix * scaled
        out_weights.append(float(base) * float(mult))
        out_mult.append(float(mult))
    return (out_weights, out_mult)


def _load_imputed_vad_map(path: Optional[str | Path]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    if not path:
        return ({}, {})
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    payload = json.loads(p.read_text(encoding='utf-8'))
    data = payload
    if isinstance(
        payload,
        dict,
    ) and 'terms' in payload:
        data = payload['terms']
    if isinstance(
        data,
        dict,
    ):
        items = list(data.items())
    elif isinstance(
        data,
        list,
    ):
        items = data
    else:
        raise ValueError('error: ValueError')
    vad_map: Dict[str, torch.Tensor] = {}
    conf_map: Dict[str, float] = {}
    for entry in items:
        term = None
        vad = None
        conf = None
        if (
            isinstance(
                entry,
                (list, tuple),
            )
            and len(entry) >= 2
        ):
            term = entry[0]
            vad = entry[1]
            if len(entry) >= 3:
                conf = entry[2]
        elif isinstance(
            entry,
            dict,
        ):
            term = (
                entry.get('term')
                or entry.get('token')
                or entry.get('word')
            )
            vad = entry.get('vad') or entry.get('vals')
            conf = entry.get('conf')
        else:
            continue
        if term is None or vad is None:
            continue
        key = _canonical_term(str(term))
        if not key:
            continue
        if (
            not isinstance(
                vad,
                (list, tuple),
            )
            or len(vad) < 3
        ):
            continue
        try:
            vec = torch.tensor(
                [
                    float(vad[0]),
                    float(vad[1]),
                    float(vad[2]),
                ],
                dtype=torch.float,
            )
        except Exception:
            continue
        vad_map[key] = vec
        try:
            conf_f = (
                float(conf) if conf is not None else 0.0
            )
        except Exception:
            conf_f = 0.0
        conf_map[key] = conf_f
    return (vad_map, conf_map)


def _merge_imputed_vad(
    term_vad: Dict[str, torch.Tensor],
    term_vad_conf: Dict[str, float],
    imputed_vad: Dict[str, torch.Tensor],
    imputed_conf: Dict[str, float],
    *,
    default_conf: float,
) -> set[str]:
    imputed_terms: set[str] = set()
    fallback = float(default_conf)
    if fallback < 0.0:
        fallback = 0.0
    if fallback > 1.0:
        fallback = 1.0
    for term, vad in imputed_vad.items():
        conf = float(imputed_conf.get(
                term,
                fallback,
            ))
        if (
            term in term_vad
            and term_vad_conf.get(
                term,
                0.0,
            ) >= conf
        ):
            continue
        term_vad[term] = vad
        term_vad_conf[term] = conf
        imputed_terms.add(term)
    return imputed_terms


def _normalise_negators(negators: Optional[Iterable[str]]) -> Tuple[set[str], set[Tuple[str, ...]]]:
    return _lex_normalise_negators(negators)


_NEGATION_BOUNDARIES = {
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
_NEGATION_SKIP_NEXT = {'only', 'just', 'merely', 'simply'}


def _is_negator_token(
    token: str,
    neg_tokens: set[str],
) -> bool:
    return _lex_is_negator_token(
        token,
        neg_tokens,
    )


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


def _build_negation_mask(
    tokens: List[str],
    window: int,
    neg_tokens: set[str],
    neg_phrases: set[Tuple[str, ...]],
) -> List[bool]:
    return _lex_build_negation_mask(
        tokens,
        window,
        neg_tokens,
        neg_phrases,
    )


def _get_text(row: Dict[str, Any]) -> str:
    text = row.get('text')
    if isinstance(
        text,
        str,
    ):
        meta = row.get('meta')
        if isinstance(
            meta,
            dict,
        ):
            stem = str(meta.get('stem') or '')
            source = str(meta.get('source') or '')
            if (
                stem.startswith('ICJ_')
                or source.upper() == 'ICJ'
            ):
                return strip_icj_boilerplate(text)
        return text
    meta = row.get('meta')
    if isinstance(
        meta,
        dict,
    ) and isinstance(
        meta.get('path'),
        str,
    ):
        p = Path(meta['path'])
        if not p.exists():
            raise FileNotFoundError('error: FileNotFoundError')
        return read_text(p)
    raise ValueError('error: ValueError')


def _tokenize_docs(
    texts: Iterable[str],
    *,
    min_token_len: int,
    stopwords: Optional[set[str]],
) -> List[List[str]]:
    out: List[List[str]] = []
    sw = stopwords or set()
    min_len = max(
        1,
        int(min_token_len),
    )
    for t in texts:
        toks = [
            w
            for w in lex_tokenize(t)
            if len(w) >= min_len and w not in sw
        ]
        out.append(toks)
    return out


def _build_idf(docs: List[List[str]]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    n = len(docs)
    for toks in docs:
        for tok in set(toks):
            df[tok] += 1
    out: Dict[str, float] = {}
    for tok, dfi in df.items():
        out[tok] = (
            math.log((1.0 + float(n)) / (1.0 + float(dfi)))
            + 1.0
        )
    return out


def _doc_terms_and_weights(
    toks: List[str],
    *,
    idf: Dict[str, float],
    weight: str,
    max_terms: int,
    unk_token: str,
    allow_empty: bool = False,
    term_weights: Optional[Dict[str, float]] = None,
    term_weight_default: float = 1.0,
    term_weight_min: float = 0.0,
    term_weight_max: float = 1.0,
    term_weight_power: float = 1.0,
    term_weight_mix: float = 1.0,
) -> Tuple[List[str], List[float]]:
    max_k = int(max_terms)
    if max_k <= 0:
        raise ValueError('error: ValueError')
    tf = Counter(toks)
    weight_mode = (weight or 'tfidf').lower().strip()
    scored: List[Tuple[str, float, float]] = []
    if weight_mode == 'tfidf':
        for tok, cnt in tf.items():
            base = float(cnt) * float(idf.get(
                    tok,
                    1.0,
                ))
            scored.append((tok, base, base))
    elif weight_mode in {'tf', 'count', 'counts'}:
        for tok, cnt in tf.items():
            base = float(cnt)
            scored.append((tok, base, base))
    elif weight_mode in {'uniform', 'binary'}:
        for tok in tf.keys():
            scored.append((tok, 1.0, 1.0))
    else:
        raise ValueError('error: ValueError')
    if term_weights:
        w_min = float(term_weight_min)
        w_max = float(term_weight_max)
        w_pow = float(term_weight_power)
        mix = float(term_weight_mix)
        if w_max < w_min:
            w_max = w_min
        mix = max(
            0.0,
            min(
                1.0,
                mix,
            ),
        )
        out: List[Tuple[str, float, float]] = []
        for tok, base, _ in scored:
            raw = float(term_weights.get(
                    tok,
                    term_weight_default,
                ))
            clamped = max(
                w_min,
                min(
                    w_max,
                    raw,
                ),
            )
            scaled = clamped**w_pow
            mult = 1.0 - mix + mix * scaled
            out.append((tok, base, float(base) * float(mult)))
        scored = out
    scored = [
        (tok, base, sort_w)
        for tok, base, sort_w in scored
        if base > 0 and sort_w > 0
    ]
    scored.sort(
        key=lambda x: (x[2], x[0]),
        reverse=True,
    )
    scored = scored[:max_k]
    if not scored:
        if allow_empty:
            return ([], [])
        return ([unk_token], [1.0])
    terms = [t for t, _, _ in scored]
    weights = [w for _, w, _ in scored]
    return (terms, weights)


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )


@lru_cache(maxsize=2)
def _load_embedding_weight(model_name: str) -> torch.Tensor:
    model = AutoModel.from_pretrained(model_name)
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(
        emb,
        'weight',
    ):
        raise ValueError('error: ValueError')
    return emb.weight.detach().cpu()


def _load_encoder_model(
    model_name: str,
    *,
    device: torch.device,
    amp: str = 'none',
):
    device_str = str(device)
    amp_mode = (amp or 'none').lower().strip()
    if device.type != 'cuda':
        amp_mode = 'none'
    if amp_mode not in {'none', 'fp16', 'bf16'}:
        amp_mode = 'none'
    return _load_encoder_model_cached(
        model_name,
        device_str,
        amp_mode,
    )


@lru_cache(maxsize=4)
def _load_encoder_model_cached(
    model_name: str,
    device_str: str,
    amp_mode: str,
):
    device = torch.device(device_str)
    dtype = None
    if device.type == 'cuda':
        if amp_mode == 'fp16':
            dtype = torch.float16
        elif amp_mode == 'bf16':
            dtype = torch.bfloat16
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    return model


def _normalise_embed_prompt_mode(mode: Optional[str]) -> str:
    if not mode:
        return 'none'
    m = str(mode).strip().lower()
    if m in {'none', 'off', 'disabled', 'false'}:
        return 'none'
    if m in {'e5', 'e5-passage', 'passage'}:
        return 'e5-passage'
    if m in {'e5-query', 'query'}:
        return 'e5-query'
    if m in {
        'e5-mistral',
        'mistral',
        'e5-instruct',
        'instruct',
    }:
        return 'e5-mistral'
    raise ValueError('error: ValueError')


def _apply_embed_prompt(
    terms: List[str],
    *,
    mode: str,
    prompt_text: Optional[str],
) -> List[str]:
    if not terms or mode == 'none':
        return list(terms)
    if mode == 'e5-passage':
        return [f'passage: {t}' for t in terms]
    if mode == 'e5-query':
        return [f'query: {t}' for t in terms]
    if mode == 'e5-mistral':
        instruction = str(prompt_text
            or 'Represent the query for retrieval')
        return [
            f'Instruct: {instruction}\nQuery: {t}'
            for t in terms
        ]
    return list(terms)


def _embed_terms_from_input_embeddings(
    terms: List[str],
    *,
    terms_prompted: Optional[List[str]] = None,
    tokenizer,
    emb_weight_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    if not terms:
        return {}
    emb_weight = emb_weight_cpu.to(dtype=torch.float)
    prompt_terms = terms_prompted or terms
    out: Dict[str, torch.Tensor] = {}
    bs = max(
        1,
        int(batch_size),
    )
    with torch.inference_mode():
        for start in range(
            0,
            len(terms),
            bs,
        ):
            batch_terms = terms[start : start + bs]
            batch_prompt = prompt_terms[start : start + bs]
            enc = tokenizer(
                batch_prompt,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors='pt',
            )
            ids = enc['input_ids']
            mask = enc['attention_mask'].to(dtype=torch.float)
            seq_emb = emb_weight[ids]
            denom = (
                mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            )
            avg = (seq_emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            avg = F.normalize(
                avg,
                dim=-1,
                eps=1e-08,
            )
            for t, v in zip(
                batch_terms,
                avg,
            ):
                out[t] = v.detach().cpu()
    return out


def _embed_terms_from_encoder(
    terms: List[str],
    *,
    terms_prompted: Optional[List[str]] = None,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 32,
    pooling: str = 'cls',
) -> Dict[str, torch.Tensor]:
    if not terms:
        return {}
    bs = max(
        1,
        int(batch_size),
    )
    max_len = max(
        2,
        int(max_length),
    )
    pooling_mode = (pooling or 'cls').lower().strip()
    if pooling_mode not in {'cls', 'mean'}:
        raise ValueError('error: ValueError')
    out: Dict[str, torch.Tensor] = {}
    prompt_terms = terms_prompted or terms
    with torch.inference_mode():
        for start in range(
            0,
            len(terms),
            bs,
        ):
            batch_terms = terms[start : start + bs]
            batch_prompt = prompt_terms[start : start + bs]
            enc = tokenizer(
                batch_prompt,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_attention_mask=True,
                return_tensors='pt',
            )
            enc = {
                k: v.to(
                    device,
                    non_blocking=True,
                )
                for k, v in enc.items()
                if isinstance(
                    v,
                    torch.Tensor,
                )
            }
            outputs = model(
                **enc,
                return_dict=True,
            )
            last = outputs['last_hidden_state']
            if pooling_mode == 'cls':
                pooled = last[:, 0]
            else:
                mask = enc.get('attention_mask')
                if mask is None:
                    pooled = last.mean(dim=1)
                else:
                    mask_f = mask.to(dtype=last.dtype).unsqueeze(-1)
                    pooled = (last * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1e-08)
            pooled = pooled.to(dtype=torch.float)
            pooled = F.normalize(
                pooled,
                dim=-1,
                eps=1e-08,
            )
            for t, v in zip(
                batch_terms,
                pooled,
            ):
                out[t] = v.detach().cpu()
    return out


def _embed_terms(
    terms: List[str],
    *,
    model_name: str,
    device: torch.device,
    backend: str = 'encoder',
    pooling: str = 'cls',
    batch_size: int = 64,
    max_length: int = 32,
    amp: str = 'none',
    prompt_mode: str = 'none',
    prompt_text: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    backend_mode = (backend or 'encoder').lower().strip()
    if backend_mode not in {'encoder', 'input_embeddings'}:
        raise ValueError('error: ValueError')
    prompt_mode_norm = _normalise_embed_prompt_mode(prompt_mode)
    terms_prompted = _apply_embed_prompt(
        terms,
        mode=prompt_mode_norm,
        prompt_text=prompt_text,
    )
    tokenizer = _load_tokenizer(model_name)
    if backend_mode == 'input_embeddings':
        emb_weight_cpu = _load_embedding_weight(model_name)
        return _embed_terms_from_input_embeddings(
            terms,
            terms_prompted=terms_prompted,
            tokenizer=tokenizer,
            emb_weight_cpu=emb_weight_cpu,
            device=device,
            batch_size=int(batch_size),
        )
    model = _load_encoder_model(
        model_name,
        device=device,
        amp=amp,
    )
    return _embed_terms_from_encoder(
        terms,
        terms_prompted=terms_prompted,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=int(batch_size),
        max_length=int(max_length),
        pooling=pooling,
    )


def _cosine_cost(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    X = F.normalize(
        X.to(dtype=torch.float),
        dim=-1,
        eps=1e-08,
    )
    Y = F.normalize(
        Y.to(dtype=torch.float),
        dim=-1,
        eps=1e-08,
    )
    C = 0.5 * (1.0 - X @ Y.t())
    return C.clamp(
        0.0,
        1.0,
    )


def _vad_cost(
    Va: torch.Tensor,
    Vb: torch.Tensor,
    *,
    conf_a: Optional[torch.Tensor] = None,
    conf_b: Optional[torch.Tensor] = None,
    unknown_cost: Optional[float] = None,
) -> torch.Tensor:
    Va = Va.to(dtype=torch.float)
    Vb = Vb.to(dtype=torch.float)
    diffs = Va[:, None, :] - Vb[None, :, :]
    C = torch.sqrt(torch.sum(
            diffs * diffs,
            dim=-1,
        ) + 1e-12)
    C = (C / float(_VAD_MAX_DIST)).clamp(
        0.0,
        1.0,
    )
    if conf_a is not None and conf_b is not None:
        wa = (
            conf_a.to(dtype=torch.float)
            .clamp(
                0.0,
                1.0,
            )
            .view(
                -1,
                1,
            )
        )
        wb = (
            conf_b.to(dtype=torch.float)
            .clamp(
                0.0,
                1.0,
            )
            .view(
                1,
                -1,
            )
        )
        conf = wa * wb
        if unknown_cost is None:
            C = C * conf
        else:
            fill = float(unknown_cost)
            if fill < 0.0:
                fill = 0.0
            if fill > 1.0:
                fill = 1.0
            C = C * conf + fill * (1.0 - conf)
    return C


def _term_vad_from_lexicon(lexicon: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ]) -> Dict[str, torch.Tensor]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for _, entries in lexicon.items():
        for term, vad in entries.items():
            key = ' '.join(lex_tokenize(term))
            if not key:
                continue
            v = torch.tensor(
                [
                    float(vad[0]),
                    float(vad[1]),
                    float(vad[2]),
                ],
                dtype=torch.float,
            )
            if key in sums:
                sums[key] = sums[key] + v
                counts[key] += 1
            else:
                sums[key] = v
                counts[key] = 1
    out: Dict[str, torch.Tensor] = {}
    for term, v in sums.items():
        out[term] = v / float(max(
                1,
                counts[term],
            ))
    return out


def _build_term_vad_map(
    lexicon: Dict[
        str, Dict[str, Tuple[float, float, float]]
    ],
    word_vad: Dict[str, Tuple[float, float, float]],
    *,
    imputed_weight: float = 0.0,
    return_conf: bool = False,
) -> (
    Dict[str, torch.Tensor]
    | Tuple[Dict[str, torch.Tensor], Dict[str, float]]
):
    out = _term_vad_from_lexicon(lexicon)
    conf: Dict[str, float] = {}
    imputed = float(imputed_weight)
    if imputed < 0.0:
        imputed = 0.0
    if imputed > 1.0:
        imputed = 1.0
    if out:
        conf = {term: imputed for term in out}
        protos = _default_emotion_vad()
        tol = 0.0001
        for emotion, entries in lexicon.items():
            proto = protos.get(emotion)
            for term, vad in entries.items():
                key = ' '.join(lex_tokenize(term))
                if not key:
                    continue
                conf_entry = imputed
                if proto is None:
                    conf_entry = 1.0
                elif (
                    max((
                            abs(float(vad[i])
                                - float(proto[i]))
                            for i in range(3)
                        ))
                    > tol
                ):
                    conf_entry = 1.0
                conf[key] = max(
                    conf.get(
                        key,
                        imputed,
                    ),
                    conf_entry,
                )
    word_vad_norm: Dict[str, torch.Tensor] = {}
    for term, vad in (word_vad or {}).items():
        key = ' '.join(lex_tokenize(term))
        if not key:
            continue
        word_vad_norm[key] = torch.tensor(
            [float(vad[0]), float(vad[1]), float(vad[2])],
            dtype=torch.float,
        )
    for term, vad in word_vad_norm.items():
        out[term] = vad
        conf[term] = 1.0
    if return_conf:
        return (out, conf)
    return out


def _vad_salience(v: torch.Tensor) -> float:
    v = v.to(dtype=torch.float).view(-1)
    if v.numel() < 2:
        return float(torch.abs(v).max().item())
    valence = float(torch.abs(v[0]).clamp(
            0.0,
            1.0,
        ).item())
    arousal_pos = float(v[1].clamp(
            0.0,
            1.0,
        ).item())
    return max(
        valence,
        arousal_pos,
    )


def _candidate_lemmas(token: str) -> List[str]:
    t = token
    if len(t) < 3:
        return []
    out: List[str] = []
    irregular = {
        'better': 'good',
        'best': 'good',
        'worse': 'bad',
        'worst': 'bad',
        'less': 'little',
        'least': 'little',
        'more': 'many',
        'most': 'many',
    }
    if t in irregular:
        out.append(irregular[t])
    if t.endswith('ies') and len(t) > 4:
        out.append(t[:-3] + 'y')
    if t.endswith('ier') and len(t) > 4:
        out.append(t[:-3] + 'y')
    if t.endswith('iest') and len(t) > 5:
        out.append(t[:-4] + 'y')
    if t.endswith('es') and len(t) > 3:
        out.append(t[:-2])
    if (
        t.endswith('s')
        and (not t.endswith('ss'))
        and (len(t) > 3)
    ):
        out.append(t[:-1])
    if t.endswith('ed') and len(t) > 3:
        base = t[:-2]
        out.append(base)
        if not base.endswith('e'):
            out.append(base + 'e')
        if len(base) >= 2 and base[-1] == base[-2]:
            out.append(base[:-1])
    if t.endswith('ing') and len(t) > 4:
        base = t[:-3]
        out.append(base)
        if not base.endswith('e'):
            out.append(base + 'e')
        if len(base) >= 2 and base[-1] == base[-2]:
            out.append(base[:-1])
    if t.endswith('ly') and len(t) > 3:
        out.append(t[:-2])
    if t.endswith('er') and len(t) > 4:
        out.append(t[:-2])
    if t.endswith('est') and len(t) > 5:
        out.append(t[:-3])
    if t.endswith('ness') and len(t) > 6:
        out.append(t[:-4])
    return out


def _resolve_emotional_term(
    term: str,
    allowed: set[str],
) -> Optional[str]:
    if term in allowed:
        return term
    for cand in _candidate_lemmas(term):
        if cand in allowed:
            return cand
    return None


def _build_emotional_matcher(
    *,
    lexicon_terms: set[str],
    word_vad_terms: set[str],
    term_vad: Dict[str, torch.Tensor],
    term_vad_conf: Optional[Dict[str, float]] = None,
    vad_threshold: float,
    vad_min_arousal_vad_only: float,
    emotional_vocab: str,
    max_ngram: int,
    stopword_terms: set[str],
) -> Tuple[set[str], Dict[Tuple[str, ...], str]]:
    max_n = max(
        1,
        int(max_ngram),
    )
    thr = float(vad_threshold)
    ar_min = float(vad_min_arousal_vad_only)
    if ar_min < 0.0:
        ar_min = 0.0
    if ar_min > 1.0:
        ar_min = 1.0
    vocab_mode = (
        (emotional_vocab or 'lexicon_or_vad')
        .lower()
        .strip()
    )
    if vocab_mode not in {
        'lexicon',
        'vad',
        'lexicon_or_vad',
    }:
        raise ValueError('error: ValueError')
    if vocab_mode == 'lexicon':
        candidates = set(lexicon_terms)
    elif vocab_mode == 'vad':
        candidates = set(word_vad_terms)
    else:
        candidates = lexicon_terms | word_vad_terms
    allowed: set[str] = set()
    conf_map = term_vad_conf or {}
    for term in candidates:
        if term in stopword_terms:
            continue
        v = term_vad.get(term)
        if v is None:
            continue
        v = v.to(dtype=torch.float).view(-1)
        valence_abs = (
            float(torch.abs(v[0]).clamp(
                    0.0,
                    1.0,
                ).item())
            if v.numel() >= 1
            else 0.0
        )
        arousal_pos = (
            float(v[1].clamp(
                    0.0,
                    1.0,
                ).item())
            if v.numel() >= 2
            else 0.0
        )
        conf_val = float(conf_map.get(
                term,
                1.0,
            ))
        if term in lexicon_terms:
            if conf_val < 1.0:
                allowed.add(term)
            elif max(
                valence_abs,
                arousal_pos,
            ) >= thr:
                allowed.add(term)
        elif valence_abs >= thr and arousal_pos >= ar_min:
            allowed.add(term)
    ngram_to_term: Dict[Tuple[str, ...], str] = {}
    for term in allowed:
        parts = tuple(term.split())
        if 1 < len(parts) <= max_n:
            ngram_to_term[parts] = term
    return (allowed, ngram_to_term)


def _extract_emotional_terms(
    tokens: List[str],
    *,
    lexicon_terms: set[str],
    word_vad_terms: set[str],
    term_vad: Dict[str, torch.Tensor],
    term_vad_conf: Optional[Dict[str, float]] = None,
    vad_threshold: float,
    vad_min_arousal_vad_only: float,
    emotional_vocab: str,
    max_ngram: int,
    stopwords: set[str],
    stopword_terms: set[str],
    min_token_len: int,
    negation_window: int = 0,
    negator_tokens: Optional[set[str]] = None,
    negator_phrases: Optional[set[Tuple[str, ...]]] = None,
    allow_vad_terms_in_stopwords: bool = False,
    allowed_terms: Optional[set[str]] = None,
    ngram_to_term: Optional[
        Dict[Tuple[str, ...], str]
    ] = None,
) -> List[str]:
    min_len = max(
        1,
        int(min_token_len),
    )
    max_n = max(
        1,
        int(max_ngram),
    )
    if max_n <= 0:
        raise ValueError('error: ValueError')
    window = max(
        0,
        int(negation_window),
    )
    neg_tokens = negator_tokens or set()
    neg_phrases = negator_phrases or set()
    toks = list(tokens)
    if not toks:
        return []
    allowed = allowed_terms
    ngram_map = ngram_to_term
    if allowed is None:
        (
            allowed,
            ngram_map_built,
        ) = _build_emotional_matcher(
            lexicon_terms=lexicon_terms,
            word_vad_terms=word_vad_terms,
            term_vad=term_vad,
            term_vad_conf=term_vad_conf,
            vad_threshold=float(vad_threshold),
            vad_min_arousal_vad_only=float(vad_min_arousal_vad_only),
            emotional_vocab=str(emotional_vocab),
            max_ngram=int(max_ngram),
            stopword_terms=stopword_terms,
        )
        if ngram_map is None:
            ngram_map = ngram_map_built
    if ngram_map is None:
        ngram_map = {}
        for term in allowed:
            parts = tuple(term.split())
            if 1 < len(parts) <= max_n:
                ngram_map[parts] = term

    def _is_emotional_term(term: str) -> Optional[str]:
        return _resolve_emotional_term(
            term,
            allowed,
        )

    neg_mask = _build_negation_mask(
        toks,
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

    def _term_starts_with_negator(parts: Tuple[str, ...]) -> bool:
        return _lex_term_starts_with_negator(
            parts,
            neg_tokens,
            neg_phrases,
        )

    out: List[str] = []
    i = 0
    while i < len(toks):
        matched = False
        for n in range(
            min(
                max_n,
                len(toks) - i,
            ),
            1,
            -1,
        ):
            key = tuple(toks[i : i + n])
            term = ngram_map.get(key)
            if term is None:
                continue
            if term in stopword_terms:
                continue
            if _is_negated(
                i,
                n,
            ) and (
                not _term_starts_with_negator(key)
            ):
                i += n
                matched = True
                break
            if _is_emotional_term(term):
                out.append(term)
                i += n
                matched = True
                break
        if matched:
            continue
        term = toks[i]
        if term in stopword_terms:
            i += 1
            continue
        if len(term) >= min_len and (not _is_negated(i)):
            resolved = _is_emotional_term(term)
            if resolved and resolved not in stopword_terms:
                in_lex = resolved in lexicon_terms
                if (
                    term in stopwords
                    or resolved in stopwords
                ) and (
                    not (
                        allow_vad_terms_in_stopwords
                        and (not in_lex)
                    )
                ):
                    i += 1
                    continue
                out.append(resolved)
        i += 1
    return out


def _snippet_from_tokens(
    tokens: List[str],
    term: str,
    *,
    window: int = 20,
) -> Optional[str]:
    parts = term.split()
    if not parts:
        return None
    n = len(parts)
    if n == 0:
        return None
    for i in range(
        0,
        max(
            0,
            len(tokens) - n + 1,
        ),
    ):
        if tokens[i : i + n] == parts:
            lo = max(
                0,
                i - int(window),
            )
            hi = min(
                len(tokens),
                i + n + int(window),
            )
            return ' '.join(tokens[lo:hi])
    return None


def _normalise_weights_for_ot(
    weights: torch.Tensor,
    *,
    unbalanced: bool,
) -> torch.Tensor:
    w = weights.to(dtype=torch.float).clamp_min(1e-08)
    if unbalanced:
        return w
    return w / w.sum().clamp_min(1e-08)


def _top_transport_contrib_rect(
    plan: torch.Tensor,
    C: torch.Tensor,
    src_terms: List[str],
    tgt_terms: List[str],
    *,
    topk: int = 8,
    C_embed: Optional[torch.Tensor] = None,
    C_vad: Optional[torch.Tensor] = None,
    src_weights: Optional[torch.Tensor] = None,
    tgt_weights: Optional[torch.Tensor] = None,
    src_weights_norm: Optional[torch.Tensor] = None,
    tgt_weights_norm: Optional[torch.Tensor] = None,
    src_vad: Optional[torch.Tensor] = None,
    tgt_vad: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    if plan.dim() != 2 or C.dim() != 2:
        raise ValueError('error: ValueError')
    if plan.shape != C.shape:
        raise ValueError('error: ValueError')
    if plan.size(0) != len(src_terms) or plan.size(1) != len(tgt_terms):
        raise ValueError('error: ValueError')
    contrib = (
        (plan * C).detach().cpu().to(dtype=torch.float)
    )
    mass = plan.detach().cpu().to(dtype=torch.float)
    C_cpu = C.detach().cpu().to(dtype=torch.float)
    C_embed_cpu = (
        C_embed.detach().cpu().to(dtype=torch.float)
        if isinstance(
            C_embed,
            torch.Tensor,
        )
        else None
    )
    C_vad_cpu = (
        C_vad.detach().cpu().to(dtype=torch.float)
        if isinstance(
            C_vad,
            torch.Tensor,
        )
        else None
    )
    src_w_cpu = (
        src_weights.detach().cpu().to(dtype=torch.float)
        if isinstance(
            src_weights,
            torch.Tensor,
        )
        else None
    )
    tgt_w_cpu = (
        tgt_weights.detach().cpu().to(dtype=torch.float)
        if isinstance(
            tgt_weights,
            torch.Tensor,
        )
        else None
    )
    src_w_n_cpu = (
        src_weights_norm.detach()
        .cpu()
        .to(dtype=torch.float)
        if isinstance(
            src_weights_norm,
            torch.Tensor,
        )
        else None
    )
    tgt_w_n_cpu = (
        tgt_weights_norm.detach()
        .cpu()
        .to(dtype=torch.float)
        if isinstance(
            tgt_weights_norm,
            torch.Tensor,
        )
        else None
    )
    src_vad_cpu = (
        src_vad.detach().cpu().to(dtype=torch.float)
        if isinstance(
            src_vad,
            torch.Tensor,
        )
        else None
    )
    tgt_vad_cpu = (
        tgt_vad.detach().cpu().to(dtype=torch.float)
        if isinstance(
            tgt_vad,
            torch.Tensor,
        )
        else None
    )
    flat = contrib.view(-1)
    k = min(
        int(topk),
        flat.numel(),
    )
    if k <= 0:
        return []
    (
        vals,
        idx,
    ) = torch.topk(
        flat,
        k=k,
        largest=True,
    )
    out: List[Dict[str, Any]] = []
    n_tgt = int(contrib.size(1))
    for v, linear in zip(
        vals.tolist(),
        idx.tolist(),
    ):
        if v <= 0:
            continue
        i = int(linear // n_tgt)
        j = int(linear % n_tgt)
        out.append({
                'from': src_terms[i],
                'to': tgt_terms[j],
                'from_index': int(i),
                'to_index': int(j),
                'from_weight_raw': (
                    float(src_w_cpu[i].item())
                    if src_w_cpu is not None
                    else None
                ),
                'to_weight_raw': (
                    float(tgt_w_cpu[j].item())
                    if tgt_w_cpu is not None
                    else None
                ),
                'from_weight_norm': (
                    float(src_w_n_cpu[i].item())
                    if src_w_n_cpu is not None
                    else None
                ),
                'to_weight_norm': (
                    float(tgt_w_n_cpu[j].item())
                    if tgt_w_n_cpu is not None
                    else None
                ),
                'from_vad': (
                    src_vad_cpu[i].tolist()
                    if src_vad_cpu is not None
                    and src_vad_cpu.numel()
                    else None
                ),
                'to_vad': (
                    tgt_vad_cpu[j].tolist()
                    if tgt_vad_cpu is not None
                    and tgt_vad_cpu.numel()
                    else None
                ),
                'mass': float(mass[i, j].item()),
                'cost': float(C_cpu[i, j].item()),
                'cost_embed': (
                    float(C_embed_cpu[i, j].item())
                    if C_embed_cpu is not None
                    else None
                ),
                'cost_vad': (
                    float(C_vad_cpu[i, j].item())
                    if C_vad_cpu is not None
                    else None
                ),
                'contribution': float(v),
            })
    return out


def _top_transport_mass_rect(
    plan: torch.Tensor,
    C: torch.Tensor,
    src_terms: List[str],
    tgt_terms: List[str],
    *,
    topk: int = 8,
    C_embed: Optional[torch.Tensor] = None,
    C_vad: Optional[torch.Tensor] = None,
    src_weights: Optional[torch.Tensor] = None,
    tgt_weights: Optional[torch.Tensor] = None,
    src_weights_norm: Optional[torch.Tensor] = None,
    tgt_weights_norm: Optional[torch.Tensor] = None,
    src_vad: Optional[torch.Tensor] = None,
    tgt_vad: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    if plan.dim() != 2 or C.dim() != 2:
        raise ValueError('error: ValueError')
    if plan.shape != C.shape:
        raise ValueError('error: ValueError')
    if plan.size(0) != len(src_terms) or plan.size(1) != len(tgt_terms):
        raise ValueError('error: ValueError')
    mass = plan.detach().cpu().to(dtype=torch.float)
    C_cpu = C.detach().cpu().to(dtype=torch.float)
    contrib = (mass * C_cpu).to(dtype=torch.float)
    C_embed_cpu = (
        C_embed.detach().cpu().to(dtype=torch.float)
        if isinstance(
            C_embed,
            torch.Tensor,
        )
        else None
    )
    C_vad_cpu = (
        C_vad.detach().cpu().to(dtype=torch.float)
        if isinstance(
            C_vad,
            torch.Tensor,
        )
        else None
    )
    src_w_cpu = (
        src_weights.detach().cpu().to(dtype=torch.float)
        if isinstance(
            src_weights,
            torch.Tensor,
        )
        else None
    )
    tgt_w_cpu = (
        tgt_weights.detach().cpu().to(dtype=torch.float)
        if isinstance(
            tgt_weights,
            torch.Tensor,
        )
        else None
    )
    src_w_n_cpu = (
        src_weights_norm.detach()
        .cpu()
        .to(dtype=torch.float)
        if isinstance(
            src_weights_norm,
            torch.Tensor,
        )
        else None
    )
    tgt_w_n_cpu = (
        tgt_weights_norm.detach()
        .cpu()
        .to(dtype=torch.float)
        if isinstance(
            tgt_weights_norm,
            torch.Tensor,
        )
        else None
    )
    src_vad_cpu = (
        src_vad.detach().cpu().to(dtype=torch.float)
        if isinstance(
            src_vad,
            torch.Tensor,
        )
        else None
    )
    tgt_vad_cpu = (
        tgt_vad.detach().cpu().to(dtype=torch.float)
        if isinstance(
            tgt_vad,
            torch.Tensor,
        )
        else None
    )
    flat = mass.view(-1)
    k = min(
        int(topk),
        flat.numel(),
    )
    if k <= 0:
        return []
    (
        vals,
        idx,
    ) = torch.topk(
        flat,
        k=k,
        largest=True,
    )
    out: List[Dict[str, Any]] = []
    n_tgt = int(mass.size(1))
    for v, linear in zip(
        vals.tolist(),
        idx.tolist(),
    ):
        if v <= 0:
            continue
        i = int(linear // n_tgt)
        j = int(linear % n_tgt)
        out.append({
                'from': src_terms[i],
                'to': tgt_terms[j],
                'from_index': int(i),
                'to_index': int(j),
                'from_weight_raw': (
                    float(src_w_cpu[i].item())
                    if src_w_cpu is not None
                    else None
                ),
                'to_weight_raw': (
                    float(tgt_w_cpu[j].item())
                    if tgt_w_cpu is not None
                    else None
                ),
                'from_weight_norm': (
                    float(src_w_n_cpu[i].item())
                    if src_w_n_cpu is not None
                    else None
                ),
                'to_weight_norm': (
                    float(tgt_w_n_cpu[j].item())
                    if tgt_w_n_cpu is not None
                    else None
                ),
                'from_vad': (
                    src_vad_cpu[i].tolist()
                    if src_vad_cpu is not None
                    and src_vad_cpu.numel()
                    else None
                ),
                'to_vad': (
                    tgt_vad_cpu[j].tolist()
                    if tgt_vad_cpu is not None
                    and tgt_vad_cpu.numel()
                    else None
                ),
                'mass': float(mass[i, j].item()),
                'cost': float(C_cpu[i, j].item()),
                'cost_embed': (
                    float(C_embed_cpu[i, j].item())
                    if C_embed_cpu is not None
                    else None
                ),
                'cost_vad': (
                    float(C_vad_cpu[i, j].item())
                    if C_vad_cpu is not None
                    else None
                ),
                'contribution': float(contrib[i, j].item()),
            })
    return out


def _copy_primary_explain(explain: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(
        explain,
        dict,
    ):
        return None
    out = dict(explain)
    flows = explain.get('top_transport_cost_contrib')
    if isinstance(
        flows,
        list,
    ):
        out['top_transport_cost_contrib'] = [
            dict(f) if isinstance(
                f,
                dict,
            ) else f
            for f in flows
        ]
    flows_mass = explain.get('top_transport_mass')
    if isinstance(
        flows_mass,
        list,
    ):
        out['top_transport_mass'] = [
            dict(f) if isinstance(
                f,
                dict,
            ) else f
            for f in flows_mass
        ]
    return out


def _reverse_primary_explain(explain: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(explain)
    (
        out['mass_a'],
        out['mass_b'],
    ) = (
        out.get('mass_b'),
        out.get('mass_a'),
    )
    (
        out['kl_a'],
        out['kl_b'],
    ) = (
        out.get('kl_b'),
        out.get('kl_a'),
    )
    (
        out['cost_aa'],
        out['cost_bb'],
    ) = (
        out.get('cost_bb'),
        out.get('cost_aa'),
    )

    def _flip_list(key: str) -> List[Dict[str, Any]]:
        flows = explain.get(key)
        if not isinstance(
            flows,
            list,
        ):
            return []
        flipped: List[Dict[str, Any]] = []
        for f in flows:
            if not isinstance(
                f,
                dict,
            ):
                continue
            flipped.append({
                    **dict(f),
                    'from': f.get('to'),
                    'to': f.get('from'),
                    'from_index': f.get('to_index'),
                    'to_index': f.get('from_index'),
                    'from_weight_raw': f.get('to_weight_raw'),
                    'to_weight_raw': f.get('from_weight_raw'),
                    'from_weight_norm': f.get('to_weight_norm'),
                    'to_weight_norm': f.get('from_weight_norm'),
                    'from_vad': f.get('to_vad'),
                    'to_vad': f.get('from_vad'),
                    'from_context': f.get('to_context'),
                    'to_context': f.get('from_context'),
                })
        return flipped

    out['top_transport_cost_contrib'] = _flip_list('top_transport_cost_contrib')
    out['top_transport_mass'] = _flip_list('top_transport_mass')
    return out


def _pair_distance_and_explain(
    a: TokenCloud,
    b: TokenCloud,
    *,
    mode: str,
    cost: str,
    alpha_embed: float,
    beta_vad: float,
    epsilon: float,
    iters: int,
    reg_m: float,
    top_flows: int,
    include_plan: bool,
    device: torch.device,
    self_cost_a: Optional[float] = None,
    self_cost_b: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    mode_str = (
        (mode or 'sinkhorn_divergence').lower().strip()
    )
    use_div = False
    if mode_str.endswith('_divergence'):
        mode_base = mode_str[: -len('_divergence')]
        use_div = True
    else:
        mode_base = mode_str
    if mode_base not in {'sinkhorn', 'unbalanced'}:
        raise ValueError('error: ValueError')
    unbalanced = mode_base == 'unbalanced'
    cost_mode = (cost or 'embedding').lower().strip()
    if cost_mode not in {
        'embedding',
        'vad',
        'embedding_vad',
    }:
        raise ValueError('error: ValueError')
    a_w = a.weights.to(
        device=device,
        dtype=torch.float,
    )
    b_w = b.weights.to(
        device=device,
        dtype=torch.float,
    )
    Xa = a.X.to(
        device=device,
        dtype=torch.float,
    )
    Xb = b.X.to(
        device=device,
        dtype=torch.float,
    )
    Va = a.vad.to(
        device=device,
        dtype=torch.float,
    )
    Vb = b.vad.to(
        device=device,
        dtype=torch.float,
    )
    Va_conf = (
        a.vad_conf.to(
            device=device,
            dtype=torch.float,
        )
        if isinstance(
            a.vad_conf,
            torch.Tensor,
        )
        else None
    )
    Vb_conf = (
        b.vad_conf.to(
            device=device,
            dtype=torch.float,
        )
        if isinstance(
            b.vad_conf,
            torch.Tensor,
        )
        else None
    )
    C_ab_embed: Optional[torch.Tensor] = None
    C_ab_vad: Optional[torch.Tensor] = None
    if cost_mode in {'embedding', 'embedding_vad'}:
        C_ab_embed = _cosine_cost(
            Xa,
            Xb,
        )
    if cost_mode in {'vad', 'embedding_vad'}:
        vad_fill = 1.0 if cost_mode == 'vad' else 0.0
        C_ab_vad = _vad_cost(
            Va,
            Vb,
            conf_a=Va_conf,
            conf_b=Vb_conf,
            unknown_cost=vad_fill,
        )
    if cost_mode == 'embedding':
        assert C_ab_embed is not None
        C_ab = C_ab_embed
    elif cost_mode == 'vad':
        assert C_ab_vad is not None
        C_ab = C_ab_vad
    else:
        assert (
            C_ab_embed is not None and C_ab_vad is not None
        )
        C_ab = (
            float(alpha_embed) * C_ab_embed
            + float(beta_vad) * C_ab_vad
        )
    parts_ab = sinkhorn_cost_parts(
        a_w,
        b_w,
        C_ab,
        epsilon=float(epsilon),
        iters=int(iters),
        unbalanced=bool(unbalanced),
        reg_m=float(reg_m),
        return_plan=bool(include_plan),
    )
    cost_ab_f = float(parts_ab['total'].detach().cpu().item())
    cost_aa_f: Optional[float] = None
    cost_bb_f: Optional[float] = None
    divergence_calc_f: Optional[float] = None
    if use_div:
        if self_cost_a is None:
            if cost_mode == 'embedding':
                C_aa = _cosine_cost(
                    Xa,
                    Xa,
                )
            elif cost_mode == 'vad':
                C_aa = _vad_cost(
                    Va,
                    Va,
                    conf_a=Va_conf,
                    conf_b=Va_conf,
                    unknown_cost=1.0,
                )
            else:
                C_aa = float(alpha_embed) * _cosine_cost(
                    Xa,
                    Xa,
                ) + float(beta_vad) * _vad_cost(
                    Va,
                    Va,
                    conf_a=Va_conf,
                    conf_b=Va_conf,
                    unknown_cost=0.0,
                )
            C_aa.fill_diagonal_(0.0)
            parts_aa = sinkhorn_cost_parts(
                a_w,
                a_w,
                C_aa,
                epsilon=float(epsilon),
                iters=int(iters),
                unbalanced=bool(unbalanced),
                reg_m=float(reg_m),
                return_plan=False,
            )
            cost_aa_f = float(parts_aa['total'].detach().cpu().item())
        else:
            cost_aa_f = float(self_cost_a)
        if self_cost_b is None:
            if cost_mode == 'embedding':
                C_bb = _cosine_cost(
                    Xb,
                    Xb,
                )
            elif cost_mode == 'vad':
                C_bb = _vad_cost(
                    Vb,
                    Vb,
                    conf_a=Vb_conf,
                    conf_b=Vb_conf,
                    unknown_cost=1.0,
                )
            else:
                C_bb = float(alpha_embed) * _cosine_cost(
                    Xb,
                    Xb,
                ) + float(beta_vad) * _vad_cost(
                    Vb,
                    Vb,
                    conf_a=Vb_conf,
                    conf_b=Vb_conf,
                    unknown_cost=0.0,
                )
            C_bb.fill_diagonal_(0.0)
            parts_bb = sinkhorn_cost_parts(
                b_w,
                b_w,
                C_bb,
                epsilon=float(epsilon),
                iters=int(iters),
                unbalanced=bool(unbalanced),
                reg_m=float(reg_m),
                return_plan=False,
            )
            cost_bb_f = float(parts_bb['total'].detach().cpu().item())
        else:
            cost_bb_f = float(self_cost_b)
        divergence_calc_f = float(cost_ab_f
            - 0.5 * float(cost_aa_f)
            - 0.5 * float(cost_bb_f))
        dist_f = float(divergence_calc_f)
    else:
        dist_f = float(cost_ab_f)
    explain: Dict[str, Any] = {
        'mode': mode_str,
        'cost': cost_mode,
        'alpha_embed': float(alpha_embed),
        'beta_vad': float(beta_vad),
        'distance': float(dist_f),
        'cost_ab': float(cost_ab_f),
        'cost_aa': (
            float(cost_aa_f)
            if cost_aa_f is not None
            else None
        ),
        'cost_bb': (
            float(cost_bb_f)
            if cost_bb_f is not None
            else None
        ),
        'divergence_calc': (
            float(divergence_calc_f)
            if divergence_calc_f is not None
            else None
        ),
        'distance_minus_calc': (
            float(dist_f - float(divergence_calc_f))
            if divergence_calc_f is not None
            else None
        ),
        'transport_ab': float(parts_ab['transport'].detach().cpu().item()),
        'kl_a': float(parts_ab['kl_a'].detach().cpu().item()),
        'kl_b': float(parts_ab['kl_b'].detach().cpu().item()),
        'mass_a': float(a.weights.sum().detach().cpu().item()),
        'mass_b': float(b.weights.sum().detach().cpu().item()),
        'mass_plan': float(parts_ab['mass_plan'].detach().cpu().item()),
        'top_transport_cost_contrib': [],
        'top_transport_mass': [],
    }
    if include_plan:
        plan = parts_ab.get('plan')
        if isinstance(
            plan,
            torch.Tensor,
        ):
            a_w_n = _normalise_weights_for_ot(
                a_w,
                unbalanced=unbalanced,
            )
            b_w_n = _normalise_weights_for_ot(
                b_w,
                unbalanced=unbalanced,
            )
            explain['top_transport_cost_contrib'] = (
                _top_transport_contrib_rect(
                    plan,
                    C_ab,
                    a.terms,
                    b.terms,
                    topk=int(top_flows),
                    C_embed=C_ab_embed,
                    C_vad=C_ab_vad,
                    src_weights=a_w,
                    tgt_weights=b_w,
                    src_weights_norm=a_w_n,
                    tgt_weights_norm=b_w_n,
                    src_vad=Va,
                    tgt_vad=Vb,
                )
            )
            explain['top_transport_mass'] = (
                _top_transport_mass_rect(
                    plan,
                    C_ab,
                    a.terms,
                    b.terms,
                    topk=int(top_flows),
                    C_embed=C_ab_embed,
                    C_vad=C_ab_vad,
                    src_weights=a_w,
                    tgt_weights=b_w,
                    src_weights_norm=a_w_n,
                    tgt_weights_norm=b_w_n,
                    src_vad=Va,
                    tgt_vad=Vb,
                )
            )
    return (dist_f, explain)


def compare_token_clouds(
    *,
    input_jsonl: str,
    output_path: str,
    cfg_path: Optional[str] = None,
    fmt: str = 'neighbours',
    topk: int = 10,
    candidate_k: int = 0,
    mode: Optional[str] = None,
    focus: str = 'emotional',
    cost: str = 'embedding_vad',
    embed_model: Optional[str] = None,
    embed_backend: Optional[str] = None,
    embed_pooling: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    embed_max_length: Optional[int] = None,
    embed_prompt_mode: Optional[str] = None,
    embed_prompt_text: Optional[str] = None,
    alpha_embed: float = 0.5,
    beta_vad: float = 0.5,
    vad_threshold: float = 0.45,
    emotional_vocab: str = 'auto',
    vad_min_arousal_vad_only: float = 0.45,
    max_ngram: int = 0,
    epsilon: Optional[float] = None,
    iters: Optional[int] = None,
    reg_m: Optional[float] = None,
    weight: str = 'tfidf',
    term_weights_path: Optional[str] = None,
    term_weight_power: float = 1.0,
    term_weight_min: float = 0.0,
    term_weight_max: float = 1.0,
    term_weight_mix: float = 1.0,
    term_weight_default: float = 1.0,
    max_terms: int = 256,
    min_token_len: int = 2,
    stopwords: bool = True,
    stopwords_file: Optional[str] = None,
    negation_window: Optional[int] = None,
    negators: Optional[Iterable[str]] = None,
    drop_top_df: int = 100,
    include_explain: bool = True,
    top_flows: int = 8,
    limit: Optional[int] = None,
    vis: bool = False,
    vis_path: Optional[str] = None,
    vad_imputed_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup = load_config(cfg_path)
    rows = _read_jsonl(
        input_jsonl,
        limit=limit,
    )
    if not rows:
        raise ValueError('error: ValueError')
    mode = (mode or 'sinkhorn_divergence').lower()
    focus = (focus or 'emotional').lower().strip()
    if focus not in {'all', 'emotional'}:
        raise ValueError('error: ValueError')
    cost_default = (
        'embedding_vad'
        if focus == 'emotional'
        else 'embedding'
    )
    cost = (cost or cost_default).lower().strip()
    if (
        focus == 'all'
        and cost == 'vad'
        and (not getattr(
            setup,
            'vad_lexicon_path',
            None,
        ))
    ):
        warnings.warn(
            'warn: fallback',
            RuntimeWarning,
            stacklevel=2,
        )
        cost = 'embedding'
    epsilon = float(epsilon
        if epsilon is not None
        else setup.sinkhorn_epsilon)
    iters = int(iters if iters is not None else setup.sinkhorn_iters)
    reg_m = float(reg_m if reg_m is not None else setup.ot_reg_m)
    term_weights_path = term_weights_path or getattr(
        setup,
        'token_term_weights_path',
        None,
    )
    term_weight_power = float(term_weight_power
        if term_weight_power is not None
        else getattr(
            setup,
            'token_term_weight_power',
            1.0,
        ))
    term_weight_min = float(term_weight_min
        if term_weight_min is not None
        else getattr(
            setup,
            'token_term_weight_min',
            0.0,
        ))
    term_weight_max = float(term_weight_max
        if term_weight_max is not None
        else getattr(
            setup,
            'token_term_weight_max',
            1.0,
        ))
    term_weight_mix = float(term_weight_mix
        if term_weight_mix is not None
        else getattr(
            setup,
            'token_term_weight_mix',
            1.0,
        ))
    term_weight_default = float(term_weight_default
        if term_weight_default is not None
        else getattr(
            setup,
            'token_term_weight_default',
            1.0,
        ))
    (
        term_weights,
        term_weights_meta,
    ) = (
        _load_term_weights(term_weights_path)
        if term_weights_path
        else ({}, None)
    )
    vad_imputed_path = vad_imputed_path or getattr(
        setup,
        'token_vad_imputed_path',
        None,
    )
    vad_imputed_weight = float(getattr(
            setup,
            'token_vad_imputed_weight',
            0.0,
        )
        or 0.0)
    max_terms = int(max_terms)
    if max_terms <= 0:
        raise ValueError('error: ValueError')
    device = get_device(setup.device)
    embed_model_name = str(embed_model
        or getattr(
            setup,
            'token_ot_embed_model',
            None,
        )
        or getattr(
            setup,
            'embed_model_name',
            None,
        )
        or getattr(
            setup,
            'model_name',
            None,
        )
        or setup.model_name)
    embed_backend_mode = str(embed_backend
        or getattr(
            setup,
            'token_ot_embed_backend',
            None,
        )
        or 'encoder')
    embed_pooling_mode = str(embed_pooling
        or getattr(
            setup,
            'token_ot_embed_pooling',
            None,
        )
        or 'cls')
    embed_bs = int(embed_batch_size
        if embed_batch_size is not None
        else getattr(
            setup,
            'token_ot_embed_batch_size',
            64,
        ))
    embed_max_len = int(embed_max_length
        if embed_max_length is not None
        else getattr(
            setup,
            'token_ot_embed_max_length',
            32,
        ))
    embed_prompt_mode = str(embed_prompt_mode
        or getattr(
            setup,
            'token_ot_embed_prompt_mode',
            None,
        )
        or 'none')
    embed_prompt_text = (
        embed_prompt_text
        if embed_prompt_text is not None
        else getattr(
            setup,
            'token_ot_embed_prompt_text',
            None,
        )
    )
    unk = str(getattr(
            _load_tokenizer(embed_model_name),
            'unk_token',
            None,
        )
        or '[UNK]')
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for r in rows:
        texts.append(_get_text(r))
        meta = (
            r.get('meta')
            if isinstance(
                r.get('meta'),
                dict,
            )
            else {}
        )
        metas.append(meta)
    stopwords_default = False
    stopwords_file = stopwords_file or getattr(
        setup,
        'token_stopwords_file',
        None,
    )
    if stopwords and stopwords_file is None:
        default_sw = _default_stopwords_path()
        if default_sw is not None:
            stopwords_file = str(default_sw)
            stopwords_default = True
    vocab_mode = (emotional_vocab or 'auto').lower().strip()
    if vocab_mode == 'auto':
        vocab_mode = (
            str(getattr(
                    setup,
                    'token_emotional_vocab',
                    'lexicon',
                ))
            .lower()
            .strip()
        )
    emotional_vocab = vocab_mode
    allow_vad_stopwords = bool(getattr(
            setup,
            'token_allow_vad_stopwords',
            False,
        ))
    if not allow_vad_stopwords:
        allow_vad_stopwords = bool(stopwords_default and emotional_vocab == 'vad')
    include_english = bool(getattr(
            setup,
            'token_include_english_stopwords',
            False,
        ))
    sw: set[str] = set()
    if stopwords and include_english:
        sw |= set(ENGLISH_STOP_WORDS)
    (
        extra_sw,
        extra_phrase_sw,
    ) = (
        _load_stopword_terms(stopwords_file)
        if stopwords
        else (set(), set())
    )
    sw |= extra_sw
    neg_window = int(negation_window
        if negation_window is not None
        else getattr(
            setup,
            'lexicon_negation_window',
            0,
        ))
    negator_list = (
        negators
        if negators is not None
        else getattr(
            setup,
            'lexicon_negators',
            None,
        )
    )
    (
        neg_tokens,
        neg_phrases,
    ) = _normalise_negators(negator_list)
    negator_terms = (
        [str(n) for n in negator_list]
        if negator_list
        else []
    )
    term_vad: Dict[str, torch.Tensor] = {}
    term_vad_conf: Dict[str, float] = {}
    lexicon_terms: set[str] = set()
    word_vad_terms: set[str] = set()
    imputed_terms: set[str] = set()
    if focus == 'emotional' or cost in {
        'vad',
        'embedding_vad',
    }:
        lex_stopwords_path = getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        )
        word_vad_stopwords_path = (
            None
            if allow_vad_stopwords
            else lex_stopwords_path
        )
        if (
            focus == 'emotional'
            and (not getattr(
                setup,
                'lexicon_path',
                None,
            ))
            and (
                not getattr(
                    setup,
                    'vad_lexicon_path',
                    None,
                )
            )
        ):
            warnings.warn(
                'warn: seed',
                RuntimeWarning,
                stacklevel=2,
            )
        lex = load_lexicon(
            getattr(
                setup,
                'lexicon_path',
                None,
            ),
            getattr(
                setup,
                'vad_lexicon_path',
                None,
            ),
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
            stopwords_path=lex_stopwords_path,
            word_vad_stopwords_path=word_vad_stopwords_path,
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
        word_vad = (
            load_word_vad(
                vad_path,
                vad_scale=getattr(
                    setup,
                    'word_vad_scale',
                    None,
                ),
                stopwords_path=word_vad_stopwords_path,
            )
            if vad_path
            else {}
        )
        (
            term_vad,
            term_vad_conf,
        ) = _build_term_vad_map(
            lex,
            word_vad,
            imputed_weight=vad_imputed_weight,
            return_conf=True,
        )
        lexicon_terms = set(_term_vad_from_lexicon(lex).keys())
        word_vad_terms = set((
                ' '.join(lex_tokenize(t))
                for t in (word_vad or {}).keys()
                if ' '.join(lex_tokenize(t))
            ))
        if vad_imputed_path:
            (
                imputed_vad,
                imputed_conf,
            ) = (
                _load_imputed_vad_map(vad_imputed_path)
            )
            imputed_terms = _merge_imputed_vad(
                term_vad,
                term_vad_conf,
                imputed_vad,
                imputed_conf,
                default_conf=vad_imputed_weight,
            )
            if imputed_terms:
                word_vad_terms |= imputed_terms
    max_ngram_eff = (
        int(max_ngram) if max_ngram is not None else 0
    )
    if max_ngram_eff <= 0:
        lex_max = max(
            (len(t.split()) for t in lexicon_terms),
            default=1,
        )
        vad_max = max(
            (len(t.split()) for t in word_vad_terms),
            default=1,
        )
        max_ngram_eff = max(
            lex_max,
            vad_max,
            1,
        )
    vad_support = any((v > 0.0 for v in term_vad_conf.values()))
    if cost == 'vad' and (not vad_support):
        warnings.warn(
            'warn: vad',
            RuntimeWarning,
            stacklevel=2,
        )
        cost = 'embedding'
    k_drop = int(drop_top_df)
    raw_token_counts: List[int] = []
    docs_terms: List[List[str]] = []
    df: Optional[Counter[str]] = (
        Counter() if k_drop > 0 else None
    )
    min_len = int(min_token_len)
    allowed_terms: Optional[set[str]] = None
    ngram_map: Optional[Dict[Tuple[str, ...], str]] = None
    if focus != 'all':
        (
            allowed_terms,
            ngram_map,
        ) = _build_emotional_matcher(
            lexicon_terms=lexicon_terms,
            word_vad_terms=word_vad_terms,
            term_vad=term_vad,
            term_vad_conf=term_vad_conf,
            vad_threshold=float(vad_threshold),
            vad_min_arousal_vad_only=float(vad_min_arousal_vad_only),
            emotional_vocab=str(emotional_vocab),
            max_ngram=int(max_ngram_eff),
            stopword_terms=extra_phrase_sw,
        )
    for text in texts:
        toks_raw = lex_tokenize(text)
        raw_token_counts.append(int(len(toks_raw)))
        toks_all = [
            w
            for w in toks_raw
            if len(w) >= min_len and w not in sw
        ]
        if df is not None:
            for t in set(toks_all):
                df[t] += 1
        if focus == 'all':
            docs_terms.append(list(toks_all))
        else:
            docs_terms.append(_extract_emotional_terms(
                    toks_raw,
                    lexicon_terms=lexicon_terms,
                    word_vad_terms=word_vad_terms,
                    term_vad=term_vad,
                    term_vad_conf=term_vad_conf,
                    vad_threshold=float(vad_threshold),
                    vad_min_arousal_vad_only=float(vad_min_arousal_vad_only),
                    emotional_vocab=str(emotional_vocab),
                    max_ngram=int(max_ngram_eff),
                    stopwords=sw,
                    stopword_terms=extra_phrase_sw,
                    min_token_len=min_len,
                    negation_window=neg_window,
                    negator_tokens=neg_tokens,
                    negator_phrases=neg_phrases,
                    allow_vad_terms_in_stopwords=allow_vad_stopwords,
                    allowed_terms=allowed_terms,
                    ngram_to_term=ngram_map,
                ))
    if (
        focus == 'all'
        and cost == 'vad'
        and (not term_vad_conf)
    ):
        docs_terms = [
            [t for t in toks if t in term_vad]
            for toks in docs_terms
        ]
    dropped_top_df_terms: List[str] = []
    if k_drop > 0 and df is not None and df:
        items = sorted(
            df.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
        if len(items) > k_drop:
            dropped_top_df_terms = [
                t for t, _ in items[:k_drop]
            ]
            drop_set = set(dropped_top_df_terms)
            if drop_set:
                docs_terms = [
                    [t for t in toks if t not in drop_set]
                    for toks in docs_terms
                ]
    if cost == 'vad' and term_vad_conf:
        docs_terms = [
            [
                t
                for t in toks
                if term_vad_conf.get(
                    t,
                    0.0,
                ) > 0.0
            ]
            for toks in docs_terms
        ]
    selected_token_counts = [
        int(len(toks)) for toks in docs_terms
    ]
    selected_unique_counts = [
        int(len(set(toks))) for toks in docs_terms
    ]
    docs_with_zero_selected = int(sum((1 for c in selected_token_counts if c == 0)))
    total_raw = float(sum(raw_token_counts))
    total_selected = float(sum(selected_token_counts))
    coverage_total = float(total_selected / max(
            1.0,
            total_raw,
        ))
    idf = _build_idf(docs_terms)
    doc_terms: List[List[str]] = []
    doc_weights: List[List[float]] = []
    term_weights_active = bool(term_weights)
    doc_term_weights: Optional[List[List[float]]] = (
        [] if term_weights_active else None
    )
    for toks in docs_terms:
        (
            terms,
            weights_i,
        ) = _doc_terms_and_weights(
            toks,
            idf=idf,
            weight=weight,
            max_terms=max_terms,
            unk_token=unk,
            allow_empty=True,
            term_weights=(
                term_weights
                if term_weights_active
                else None
            ),
            term_weight_default=term_weight_default,
            term_weight_min=term_weight_min,
            term_weight_max=term_weight_max,
            term_weight_power=term_weight_power,
            term_weight_mix=term_weight_mix,
        )
        if term_weights_active:
            (
                weights_i,
                term_mult,
            ) = _apply_term_weights(
                terms,
                weights_i,
                term_weights,
                weight_default=term_weight_default,
                weight_min=term_weight_min,
                weight_max=term_weight_max,
                weight_power=term_weight_power,
                weight_mix=term_weight_mix,
            )
            if doc_term_weights is not None:
                doc_term_weights.append(term_mult)
        doc_terms.append(terms)
        doc_weights.append(weights_i)
    valid_mask = [len(terms) > 0 for terms in doc_terms]
    valid_indices = [
        i for i, ok in enumerate(valid_mask) if ok
    ]
    cost_mode = (cost or 'embedding').lower().strip()
    need_embed = cost_mode in {'embedding', 'embedding_vad'}
    vad_support = any((v > 0.0 for v in term_vad_conf.values()))
    if (
        cost_mode == 'embedding_vad'
        and (not vad_support)
        and (float(beta_vad) > 0.0)
    ):
        warnings.warn(
            'warn: beta_vad',
            RuntimeWarning,
            stacklevel=2,
        )
        beta_vad = 0.0
    if (
        need_embed
        and embed_model is None
        and (
            getattr(
                setup,
                'token_ot_embed_model',
                None,
            )
            is None
        )
        and (embed_model_name == setup.model_name)
    ):
        warnings.warn(
            'warn: base',
            RuntimeWarning,
            stacklevel=2,
        )
    allow_model_embed = bool(getattr(
            setup,
            'token_ot_allow_model_embed',
            False,
        ))
    if (
        focus == 'emotional'
        and need_embed
        and (embed_model is None)
        and (
            getattr(
                setup,
                'token_ot_embed_model',
                None,
            )
            is None
        )
        and (embed_model_name == setup.model_name)
        and (not allow_model_embed)
    ):
        if vad_support and float(beta_vad) > 0.0:
            warnings.warn(
                'warn: base',
                RuntimeWarning,
                stacklevel=2,
            )
            cost_mode = 'vad'
            cost = cost_mode
            need_embed = False
            alpha_embed = 0.0
        else:
            warnings.warn(
                'warn: weak',
                RuntimeWarning,
                stacklevel=2,
            )
    term_to_emb: Dict[str, torch.Tensor] = {}
    if need_embed:
        vocab = sorted({t for terms in doc_terms for t in terms})
        term_to_emb = _embed_terms(
            vocab,
            model_name=embed_model_name,
            device=device,
            backend=embed_backend_mode,
            pooling=embed_pooling_mode,
            batch_size=int(embed_bs),
            max_length=int(embed_max_len),
            amp=str(getattr(
                    setup,
                    'amp',
                    None,
                ) or 'none'),
            prompt_mode=embed_prompt_mode,
            prompt_text=embed_prompt_text,
        )
    clouds: List[TokenCloud] = []
    masses: List[float] = []
    embed_dim = 1
    if need_embed and term_to_emb:
        embed_dim = int(next(iter(term_to_emb.values())).numel())
    for terms, weights_i, ok in zip(
        doc_terms,
        doc_weights,
        valid_mask,
    ):
        if not ok:
            X = torch.zeros(
                (0, embed_dim),
                dtype=torch.float,
            )
            w = torch.zeros(
                (0,),
                dtype=torch.float,
            )
            V = torch.zeros(
                (0, 3),
                dtype=torch.float,
            )
            V_conf = (
                torch.zeros(
                    (0,),
                    dtype=torch.float,
                )
                if term_vad_conf
                else None
            )
            clouds.append(TokenCloud(
                    terms=terms,
                    weights=w,
                    X=X,
                    vad=V,
                    vad_conf=V_conf,
                ))
            masses.append(0.0)
            continue
        if need_embed:
            X = torch.stack(
                [term_to_emb[t] for t in terms],
                dim=0,
            ).to(dtype=torch.float)
        else:
            X = torch.zeros(
                (len(terms), 1),
                dtype=torch.float,
            )
        w = torch.tensor(
            weights_i,
            dtype=torch.float,
        ).clamp_min(1e-08)
        V = torch.stack(
            [
                term_vad.get(
                    t,
                    torch.zeros(
                        3,
                        dtype=torch.float,
                    ),
                )
                for t in terms
            ],
            dim=0,
        ).to(dtype=torch.float)
        V_conf = None
        if term_vad_conf:
            V_conf = torch.tensor(
                [term_vad_conf.get(
                    t,
                    0.0,
                ) for t in terms],
                dtype=torch.float,
            )
        clouds.append(TokenCloud(
                terms=terms,
                weights=w,
                X=X,
                vad=V,
                vad_conf=V_conf,
            ))
        masses.append(float(w.sum().item()))
    n = len(clouds)
    valid_n = len(valid_indices)
    idx_to_pos = {
        i: pos for pos, i in enumerate(valid_indices)
    }
    mode_str = (
        (mode or 'sinkhorn_divergence').lower().strip()
    )
    use_div = bool(mode_str.endswith('_divergence'))
    mode_base = (
        mode_str[: -len('_divergence')]
        if use_div
        else mode_str
    )
    if mode_base not in {'sinkhorn', 'unbalanced'}:
        raise ValueError('error: ValueError')
    unbalanced = mode_base == 'unbalanced'
    self_cost_cache: List[Optional[float]] = (
        [None] * n if use_div else []
    )

    def _get_self_cost(idx: int) -> float:
        if not use_div:
            raise RuntimeError('error: RuntimeError')
        cached = self_cost_cache[idx]
        if cached is not None:
            return float(cached)
        c = clouds[idx]
        w_i = c.weights.to(
            device=device,
            dtype=torch.float,
        )
        X_i = c.X.to(
            device=device,
            dtype=torch.float,
        )
        V_i = c.vad.to(
            device=device,
            dtype=torch.float,
        )
        V_i_conf = (
            c.vad_conf.to(
                device=device,
                dtype=torch.float,
            )
            if isinstance(
                c.vad_conf,
                torch.Tensor,
            )
            else None
        )
        if cost_mode == 'embedding':
            C_ii = _cosine_cost(
                X_i,
                X_i,
            )
        elif cost_mode == 'vad':
            C_ii = _vad_cost(
                V_i,
                V_i,
                conf_a=V_i_conf,
                conf_b=V_i_conf,
                unknown_cost=1.0,
            )
        else:
            C_ii = float(alpha_embed) * _cosine_cost(
                X_i,
                X_i,
            ) + float(beta_vad) * _vad_cost(
                V_i,
                V_i,
                conf_a=V_i_conf,
                conf_b=V_i_conf,
                unknown_cost=0.0,
            )
        C_ii.fill_diagonal_(0.0)
        parts = sinkhorn_cost_parts(
            w_i,
            w_i,
            C_ii,
            epsilon=float(epsilon),
            iters=int(iters),
            unbalanced=bool(unbalanced),
            reg_m=float(reg_m),
            return_plan=False,
        )
        total_f = float(parts['total'].detach().cpu().item())
        self_cost_cache[idx] = total_f
        return total_f

    fmt = (fmt or 'neighbours').lower().strip()
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    vis_enabled = bool(vis)
    vis_out_p: Optional[Path] = None
    if vis_enabled:
        vis_out_p = (
            Path(vis_path)
            if vis_path
            else out_p.with_suffix('.html')
        )

    def _term_source_label(term: str) -> str:
        if term == unk:
            return 'unk'
        in_lex = term in lexicon_terms
        in_vad = term in word_vad_terms
        in_imp = term in imputed_terms
        if in_lex and in_vad:
            return (
                'lexicon+vad_imputed'
                if in_imp
                else 'lexicon+vad'
            )
        if in_lex:
            return 'lexicon'
        if in_vad:
            return 'vad_imputed' if in_imp else 'vad'
        return 'other'

    if fmt == 'matrix':
        cand_k = int(candidate_k)
        if cand_k != 0:
            raise ValueError('error: ValueError')
        dist_list: List[List[Optional[float]]] = [
            [None for _ in range(n)] for _ in range(n)
        ]
        D_valid: Optional[torch.Tensor] = None
        if valid_n > 0:
            D_valid = torch.empty(
                (valid_n, valid_n),
                dtype=torch.float,
            )
            for pos_i, i in enumerate(valid_indices):
                D_valid[pos_i, pos_i] = (
                    0.0
                    if mode.endswith('_divergence')
                    else float(_pair_distance_and_explain(
                            clouds[i],
                            clouds[i],
                            mode=mode,
                            cost=cost,
                            alpha_embed=float(alpha_embed),
                            beta_vad=float(beta_vad),
                            epsilon=epsilon,
                            iters=iters,
                            reg_m=reg_m,
                            top_flows=0,
                            include_plan=False,
                            device=device,
                        )[0])
                )
            for pos_i, i in enumerate(valid_indices):
                for pos_j in range(
                    pos_i + 1,
                    valid_n,
                ):
                    j = valid_indices[pos_j]
                    self_i = (
                        _get_self_cost(i)
                        if use_div
                        else None
                    )
                    self_j = (
                        _get_self_cost(j)
                        if use_div
                        else None
                    )
                    (
                        dist,
                        _,
                    ) = _pair_distance_and_explain(
                        clouds[i],
                        clouds[j],
                        mode=mode,
                        cost=cost,
                        alpha_embed=float(alpha_embed),
                        beta_vad=float(beta_vad),
                        epsilon=epsilon,
                        iters=iters,
                        reg_m=reg_m,
                        top_flows=0,
                        include_plan=False,
                        device=device,
                        self_cost_a=self_i,
                        self_cost_b=self_j,
                    )
                    D_valid[pos_i, pos_j] = float(dist)
                    D_valid[pos_j, pos_i] = float(dist)
            for pos_i, i in enumerate(valid_indices):
                for pos_j, j in enumerate(valid_indices):
                    dist_list[i][j] = float(D_valid[pos_i, pos_j].item())
        docs_out: List[Dict[str, Any]] = []
        include_term_meta = bool(term_vad)
        for i in range(n):
            terms_out = list(clouds[i].terms)
            weights_out = [float(x) for x in doc_weights[i]]
            term_weight_out = (
                [float(x) for x in doc_term_weights[i]]
                if doc_term_weights is not None
                else None
            )
            d: Dict[str, Any] = {
                'index': i,
                'meta': metas[i],
                'mass': float(masses[i]),
                'n_terms': int(len(clouds[i].terms)),
                'n_raw_tokens': int(raw_token_counts[i]),
                'n_selected_tokens': int(selected_token_counts[i]),
                'n_selected_unique': int(selected_unique_counts[i]),
                'selected_ratio': float(float(selected_token_counts[i])
                    / max(
                        1.0,
                        float(raw_token_counts[i]),
                    )),
                'used_unk_fallback': bool(selected_token_counts[i] == 0),
                'no_emotional_terms': bool(not valid_mask[i]),
                'terms': terms_out,
                'weights': weights_out,
            }
            if term_weight_out is not None:
                d['term_weight'] = term_weight_out
            if include_term_meta:
                V_cpu = (
                    clouds[i]
                    .vad.detach()
                    .cpu()
                    .to(dtype=torch.float)
                )
                if V_cpu.numel():
                    val_abs = torch.abs(V_cpu[:, 0]).clamp(
                        0.0,
                        1.0,
                    )
                    aro_pos = V_cpu[:, 1].clamp(
                        0.0,
                        1.0,
                    )
                    sal = torch.maximum(
                        val_abs,
                        aro_pos,
                    ).tolist()
                    vad_out = V_cpu.tolist()
                else:
                    sal = []
                    vad_out = []
                sources_out: List[str] = []
                for t in terms_out:
                    sources_out.append(_term_source_label(t))
                d['term_vad'] = vad_out
                d['term_salience'] = [float(x) for x in sal]
                d['term_source'] = sources_out
                if term_vad_conf:
                    d['term_vad_conf'] = [
                        float(term_vad_conf.get(
                                t,
                                0.0,
                            ))
                        for t in terms_out
                    ]
            docs_out.append(d)
        payload = {
            'format': 'matrix',
            'mode': mode,
            'focus': focus,
            'cost': cost,
            'embed_model': (
                embed_model_name if need_embed else None
            ),
            'embed_backend': (
                embed_backend_mode if need_embed else None
            ),
            'embed_pooling': (
                embed_pooling_mode if need_embed else None
            ),
            'embed_batch_size': (
                int(embed_bs) if need_embed else None
            ),
            'embed_max_length': (
                int(embed_max_len) if need_embed else None
            ),
            'embed_prompt_mode': (
                str(embed_prompt_mode)
                if need_embed
                else None
            ),
            'embed_prompt_text': (
                str(embed_prompt_text)
                if need_embed and embed_prompt_text
                else None
            ),
            'alpha_embed': float(alpha_embed),
            'beta_vad': float(beta_vad),
            'vad_threshold': float(vad_threshold),
            'emotional_vocab': (
                str(emotional_vocab)
                if focus != 'all'
                else None
            ),
            'vad_min_arousal_vad_only': (
                float(vad_min_arousal_vad_only)
                if focus != 'all'
                else None
            ),
            'max_ngram': int(max_ngram_eff),
            'weight': weight,
            'term_weight_path': (
                str(term_weights_path)
                if term_weights_path
                else None
            ),
            'term_weight_meta': term_weights_meta,
            'term_weight_power': float(term_weight_power),
            'term_weight_min': float(term_weight_min),
            'term_weight_max': float(term_weight_max),
            'term_weight_mix': float(term_weight_mix),
            'term_weight_default': float(term_weight_default),
            'vad_imputed_path': (
                str(vad_imputed_path)
                if vad_imputed_path
                else None
            ),
            'epsilon': float(epsilon),
            'iters': int(iters),
            'reg_m': float(reg_m),
            'max_terms': int(max_terms),
            'stopwords': bool(stopwords),
            'stopwords_file': (
                str(stopwords_file)
                if stopwords_file is not None
                else None
            ),
            'negation_window': int(neg_window),
            'negators': list(negator_terms),
            'drop_top_df': int(k_drop),
            'dropped_top_df_terms': list(dropped_top_df_terms),
            'coverage_total': float(coverage_total),
            'docs_with_zero_selected': int(docs_with_zero_selected),
            'docs': docs_out,
            'distance': dist_list,
        }
        out_p.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        out_stats: Dict[str, Any] = {
            'docs': n,
            'format': 'matrix',
            'output': str(out_p),
            'drop_top_df': int(k_drop),
            'coverage_total': float(coverage_total),
            'docs_with_zero_selected': int(docs_with_zero_selected),
        }
        if vis_enabled and vis_out_p is not None:
            from .token_viz import (
                write_token_ot_html_report,
            )

            docs_payload = []
            include_term_meta = bool(term_vad)
            for i in range(n):
                terms_out = list(clouds[i].terms)
                d: Dict[str, Any] = {
                    'index': i,
                    'meta': metas[i],
                    'mass': float(masses[i]),
                    'n_terms': int(len(clouds[i].terms)),
                    'n_raw_tokens': int(raw_token_counts[i]),
                    'n_selected_tokens': int(selected_token_counts[i]),
                    'n_selected_unique': int(selected_unique_counts[i]),
                    'selected_ratio': float(float(selected_token_counts[i])
                        / max(
                            1.0,
                            float(raw_token_counts[i]),
                        )),
                    'used_unk_fallback': bool(selected_token_counts[i] == 0),
                    'no_emotional_terms': bool(not valid_mask[i]),
                    'terms': terms_out,
                    'weights': [
                        float(x) for x in doc_weights[i]
                    ],
                }
                if doc_term_weights is not None:
                    d['term_weight'] = [
                        float(x)
                        for x in doc_term_weights[i]
                    ]
                if include_term_meta:
                    V_cpu = (
                        clouds[i]
                        .vad.detach()
                        .cpu()
                        .to(dtype=torch.float)
                    )
                    if V_cpu.numel():
                        val_abs = torch.abs(V_cpu[:, 0]).clamp(
                            0.0,
                            1.0,
                        )
                        aro_pos = V_cpu[:, 1].clamp(
                            0.0,
                            1.0,
                        )
                        sal = torch.maximum(
                            val_abs,
                            aro_pos,
                        ).tolist()
                        vad_out = V_cpu.tolist()
                    else:
                        sal = []
                        vad_out = []
                    sources_out: List[str] = []
                    for t in terms_out:
                        sources_out.append(_term_source_label(t))
                    d['term_vad'] = vad_out
                    d['term_salience'] = [
                        float(x) for x in sal
                    ]
                    d['term_source'] = sources_out
                    if term_vad_conf:
                        d['term_vad_conf'] = [
                            float(term_vad_conf.get(
                                    t,
                                    0.0,
                                ))
                            for t in terms_out
                        ]
                docs_payload.append(d)
            k = min(
                int(topk),
                max(
                    0,
                    valid_n - 1,
                ),
            )
            neighbours_payload: List[Dict[str, Any]] = []
            for i in range(n):
                if (
                    not valid_mask[i]
                    or D_valid is None
                    or k <= 0
                ):
                    neighbours_payload.append({
                            'index': i,
                            'neighbours': [],
                        })
                    continue
                pos_i = idx_to_pos[i]
                d = D_valid[pos_i].clone()
                d[pos_i] = float('inf')
                (
                    vals,
                    idx,
                ) = torch.topk(
                    d,
                    k=k,
                    largest=False,
                )
                neigh_out: List[Dict[str, Any]] = []
                tokens_i = lex_tokenize(texts[i])
                for pos_j, v in zip(
                    idx.tolist(),
                    vals.tolist(),
                ):
                    j = valid_indices[int(pos_j)]
                    dist_f = float(v)
                    self_i = (
                        _get_self_cost(i)
                        if use_div
                        else None
                    )
                    self_j = (
                        _get_self_cost(j)
                        if use_div
                        else None
                    )
                    (
                        _,
                        primary_explain,
                    ) = (
                        _pair_distance_and_explain(
                            clouds[i],
                            clouds[j],
                            mode=mode,
                            cost=cost,
                            alpha_embed=float(alpha_embed),
                            beta_vad=float(beta_vad),
                            epsilon=epsilon,
                            iters=iters,
                            reg_m=reg_m,
                            top_flows=int(top_flows),
                            include_plan=True,
                            device=device,
                            self_cost_a=self_i,
                            self_cost_b=self_j,
                        )
                    )
                    if isinstance(
                        primary_explain,
                        dict,
                    ):
                        tokens_j = lex_tokenize(texts[j])
                        for key in (
                            'top_transport_cost_contrib',
                            'top_transport_mass',
                        ):
                            flows = primary_explain.get(key)
                            if not isinstance(
                                flows,
                                list,
                            ):
                                continue
                            for flow in flows:
                                if not isinstance(
                                    flow,
                                    dict,
                                ):
                                    continue
                                a_term = str(flow.get('from') or '')
                                b_term = str(flow.get('to') or '')
                                flow['from_context'] = (
                                    _snippet_from_tokens(
                                        tokens_i,
                                        a_term,
                                    )
                                    if a_term
                                    else None
                                )
                                flow['to_context'] = (
                                    _snippet_from_tokens(
                                        tokens_j,
                                        b_term,
                                    )
                                    if b_term
                                    else None
                                )
                    neigh_out.append({
                            'index': j,
                            'distance': dist_f,
                            'mass_diff': float(abs(masses[i] - masses[j])),
                            'primary_explain': primary_explain,
                        })
                neighbours_payload.append({
                        'index': i,
                        'neighbours': neigh_out,
                    })
            report = {
                'format': 'neighbours',
                'mode': mode,
                'focus': focus,
                'cost': cost,
                'embed_model': (
                    embed_model_name if need_embed else None
                ),
                'embed_backend': (
                    embed_backend_mode
                    if need_embed
                    else None
                ),
                'embed_pooling': (
                    embed_pooling_mode
                    if need_embed
                    else None
                ),
                'embed_batch_size': (
                    int(embed_bs) if need_embed else None
                ),
                'embed_max_length': (
                    int(embed_max_len)
                    if need_embed
                    else None
                ),
                'embed_prompt_mode': (
                    str(embed_prompt_mode)
                    if need_embed
                    else None
                ),
                'embed_prompt_text': (
                    str(embed_prompt_text)
                    if need_embed and embed_prompt_text
                    else None
                ),
                'alpha_embed': float(alpha_embed),
                'beta_vad': float(beta_vad),
                'vad_threshold': float(vad_threshold),
                'emotional_vocab': (
                    str(emotional_vocab)
                    if focus != 'all'
                    else None
                ),
                'vad_min_arousal_vad_only': (
                    float(vad_min_arousal_vad_only)
                    if focus != 'all'
                    else None
                ),
                'max_ngram': int(max_ngram_eff),
                'weight': weight,
                'term_weight_path': (
                    str(term_weights_path)
                    if term_weights_path
                    else None
                ),
                'term_weight_meta': term_weights_meta,
                'term_weight_power': float(term_weight_power),
                'term_weight_min': float(term_weight_min),
                'term_weight_max': float(term_weight_max),
                'term_weight_mix': float(term_weight_mix),
                'term_weight_default': float(term_weight_default),
                'vad_imputed_path': (
                    str(vad_imputed_path)
                    if vad_imputed_path
                    else None
                ),
                'epsilon': float(epsilon),
                'iters': int(iters),
                'reg_m': float(reg_m),
                'max_terms': int(max_terms),
                'stopwords': bool(stopwords),
                'stopwords_file': (
                    str(stopwords_file)
                    if stopwords_file is not None
                    else None
                ),
                'negation_window': int(neg_window),
                'negators': list(negator_terms),
                'drop_top_df': int(k_drop),
                'dropped_top_df_terms': list(dropped_top_df_terms),
                'coverage_total': float(coverage_total),
                'docs_with_zero_selected': int(docs_with_zero_selected),
                'docs': docs_payload,
                'neighbours': neighbours_payload,
                'topk': int(topk),
                'top_flows': int(top_flows),
            }
            out_stats['visualization'] = (
                write_token_ot_html_report(
                    output_path=vis_out_p,
                    payload=report,
                )
            )
        return out_stats
    if fmt != 'neighbours':
        raise ValueError('error: ValueError')
    if topk <= 0:
        raise ValueError('error: ValueError')
    if n < 2:
        raise ValueError('error: ValueError')
    cand_k = int(candidate_k)
    if cand_k < 0:
        raise ValueError('error: ValueError')
    cand_k = min(
        cand_k,
        max(
            0,
            valid_n - 1,
        ),
    )
    k = min(
        int(topk),
        max(
            0,
            valid_n - 1,
        ),
    )

    def _maybe_add_context(
        explain: Optional[Dict[str, Any]],
        *,
        src_tokens: List[str],
        tgt_tokens: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(
            explain,
            dict,
        ):
            return explain
        for key in (
            'top_transport_cost_contrib',
            'top_transport_mass',
        ):
            flows = explain.get(key)
            if not isinstance(
                flows,
                list,
            ):
                continue
            for flow in flows:
                if not isinstance(
                    flow,
                    dict,
                ):
                    continue
                a_term = str(flow.get('from') or '')
                b_term = str(flow.get('to') or '')
                flow['from_context'] = (
                    _snippet_from_tokens(
                        src_tokens,
                        a_term,
                    )
                    if a_term
                    else None
                )
                flow['to_context'] = (
                    _snippet_from_tokens(
                        tgt_tokens,
                        b_term,
                    )
                    if b_term
                    else None
                )
        return explain

    def _write_empty_neighbours() -> Dict[str, Any]:
        neighbours_payload: Optional[
            List[Dict[str, Any]]
        ] = ([] if vis_enabled else None)
        with out_p.open(
            'w',
            encoding='utf-8',
        ) as f:
            for i in range(n):
                neighbours_out: List[Dict[str, Any]] = []
                if neighbours_payload is not None:
                    neighbours_payload.append({
                            'index': i,
                            'neighbours': neighbours_out,
                        })
                f.write(json.dumps(
                        {
                            'index': i,
                            'meta': metas[i],
                            'mass': float(masses[i]),
                            'n_terms': int(len(clouds[i].terms)),
                            'n_raw_tokens': int(raw_token_counts[i]),
                            'n_selected_tokens': int(selected_token_counts[i]),
                            'n_selected_unique': int(selected_unique_counts[i]),
                            'selected_ratio': float(float(selected_token_counts[i])
                                / max(
                                    1.0,
                                    float(raw_token_counts[i]),
                                )),
                            'used_unk_fallback': bool(selected_token_counts[i]
                                == 0),
                            'no_emotional_terms': bool(not valid_mask[i]),
                            'neighbours': neighbours_out,
                            'term_weight': (
                                [
                                    float(x)
                                    for x in doc_term_weights[
                                        i
                                    ]
                                ]
                                if doc_term_weights
                                is not None
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + '\n')
        out_stats = {
            'docs': n,
            'format': 'neighbours',
            'topk': int(topk),
            'candidate_k': int(cand_k),
            'output': str(out_p),
            'drop_top_df': int(k_drop),
            'coverage_total': float(coverage_total),
            'docs_with_zero_selected': int(docs_with_zero_selected),
        }
        if (
            vis_enabled
            and vis_out_p is not None
            and (neighbours_payload is not None)
        ):
            from .token_viz import (
                write_token_ot_html_report,
            )

            docs_payload = []
            for i in range(n):
                docs_payload.append({
                        'index': i,
                        'meta': metas[i],
                        'mass': float(masses[i]),
                        'n_terms': int(len(clouds[i].terms)),
                        'n_raw_tokens': int(raw_token_counts[i]),
                        'n_selected_tokens': int(selected_token_counts[i]),
                        'n_selected_unique': int(selected_unique_counts[i]),
                        'selected_ratio': float(float(selected_token_counts[i])
                            / max(
                                1.0,
                                float(raw_token_counts[i]),
                            )),
                        'used_unk_fallback': bool(selected_token_counts[i] == 0),
                        'no_emotional_terms': bool(not valid_mask[i]),
                        'terms': list(clouds[i].terms),
                        'weights': [
                            float(x) for x in doc_weights[i]
                        ],
                        'term_weight': (
                            [
                                float(x)
                                for x in doc_term_weights[i]
                            ]
                            if doc_term_weights is not None
                            else None
                        ),
                    })
            report = {
                'format': 'neighbours',
                'mode': mode,
                'focus': focus,
                'cost': cost,
                'embed_model': (
                    embed_model_name if need_embed else None
                ),
                'embed_backend': (
                    embed_backend_mode
                    if need_embed
                    else None
                ),
                'embed_pooling': (
                    embed_pooling_mode
                    if need_embed
                    else None
                ),
                'embed_batch_size': (
                    int(embed_bs) if need_embed else None
                ),
                'embed_max_length': (
                    int(embed_max_len)
                    if need_embed
                    else None
                ),
                'embed_prompt_mode': (
                    str(embed_prompt_mode)
                    if need_embed
                    else None
                ),
                'embed_prompt_text': (
                    str(embed_prompt_text)
                    if need_embed and embed_prompt_text
                    else None
                ),
                'alpha_embed': float(alpha_embed),
                'beta_vad': float(beta_vad),
                'vad_threshold': float(vad_threshold),
                'emotional_vocab': (
                    str(emotional_vocab)
                    if focus != 'all'
                    else None
                ),
                'vad_min_arousal_vad_only': (
                    float(vad_min_arousal_vad_only)
                    if focus != 'all'
                    else None
                ),
                'max_ngram': int(max_ngram_eff),
                'weight': weight,
                'term_weight_path': (
                    str(term_weights_path)
                    if term_weights_path
                    else None
                ),
                'term_weight_meta': term_weights_meta,
                'term_weight_power': float(term_weight_power),
                'term_weight_min': float(term_weight_min),
                'term_weight_max': float(term_weight_max),
                'term_weight_mix': float(term_weight_mix),
                'term_weight_default': float(term_weight_default),
                'vad_imputed_path': (
                    str(vad_imputed_path)
                    if vad_imputed_path
                    else None
                ),
                'epsilon': float(epsilon),
                'iters': int(iters),
                'reg_m': float(reg_m),
                'max_terms': int(max_terms),
                'stopwords': bool(stopwords),
                'stopwords_file': (
                    str(stopwords_file)
                    if stopwords_file is not None
                    else None
                ),
                'negation_window': int(neg_window),
                'negators': list(negator_terms),
                'drop_top_df': int(k_drop),
                'dropped_top_df_terms': list(dropped_top_df_terms),
                'coverage_total': float(coverage_total),
                'docs_with_zero_selected': int(docs_with_zero_selected),
                'docs': docs_payload,
                'neighbours': neighbours_payload,
                'topk': int(topk),
                'top_flows': int(top_flows),
            }
            out_stats['visualization'] = (
                write_token_ot_html_report(
                    output_path=vis_out_p,
                    payload=report,
                )
            )
        return out_stats

    if valid_n < 2:
        return _write_empty_neighbours()
    if cand_k > 0:
        if cand_k < k:
            raise ValueError('error: ValueError')
        centroids: List[torch.Tensor] = []
        use_embed_for_candidates = bool(need_embed)
        valid_clouds = [clouds[i] for i in valid_indices]
        with torch.inference_mode():
            for c in valid_clouds:
                feats = (
                    c.X
                    if use_embed_for_candidates
                    else c.vad
                )
                w_n = c.weights.to(dtype=torch.float).clamp_min(1e-08)
                w_n = w_n / w_n.sum().clamp_min(1e-08)
                v = (
                    w_n.unsqueeze(1)
                    * feats.to(dtype=torch.float)
                ).sum(dim=0)
                v = v / v.norm().clamp_min(1e-12)
                centroids.append(v)
            C_cpu = torch.stack(
                centroids,
                dim=0,
            )
            if device.type != 'cpu':
                C_dev = C_cpu.to(
                    device=device,
                    dtype=torch.float,
                )
                sims = C_dev @ C_dev.T
                sims.fill_diagonal_(-1000000000.0)
                (
                    _,
                    cand_idx,
                ) = torch.topk(
                    sims,
                    k=cand_k,
                    largest=True,
                )
                cand_idx_cpu = cand_idx.detach().cpu()
            else:
                sims = C_cpu @ C_cpu.T
                sims.fill_diagonal_(-1000000000.0)
                (
                    _,
                    cand_idx_cpu,
                ) = torch.topk(
                    sims,
                    k=cand_k,
                    largest=True,
                )
        pairs: set[tuple[int, int]] = set()
        for pos_i, i in enumerate(valid_indices):
            for pos_j in cand_idx_cpu[pos_i].tolist():
                j = valid_indices[int(pos_j)]
                if j == i:
                    continue
                (
                    a,
                    b,
                ) = (i, j) if i < j else (j, i)
                pairs.add((a, b))
        pair_list = sorted(pairs)
        pair_to_dist: Dict[tuple[int, int], float] = {}
        for a, b in pair_list:
            self_a = _get_self_cost(a) if use_div else None
            self_b = _get_self_cost(b) if use_div else None
            (
                dist,
                _,
            ) = _pair_distance_and_explain(
                clouds[a],
                clouds[b],
                mode=mode,
                cost=cost,
                alpha_embed=float(alpha_embed),
                beta_vad=float(beta_vad),
                epsilon=epsilon,
                iters=iters,
                reg_m=reg_m,
                top_flows=0,
                include_plan=False,
                device=device,
                self_cost_a=self_a,
                self_cost_b=self_b,
            )
            pair_to_dist[a, b] = float(dist)
        explain_cache: Dict[
            tuple[int, int], Dict[str, Any]
        ] = {}
        neighbours_payload: Optional[
            List[Dict[str, Any]]
        ] = ([] if vis_enabled else None)
        with out_p.open(
            'w',
            encoding='utf-8',
        ) as f:
            for i in range(n):
                if not valid_mask[i]:
                    neighbours_out: List[Dict[str, Any]] = (
                        []
                    )
                    if neighbours_payload is not None:
                        neighbours_payload.append({
                                'index': i,
                                'neighbours': neighbours_out,
                            })
                    f.write(json.dumps(
                            {
                                'index': i,
                                'meta': metas[i],
                                'mass': float(masses[i]),
                                'n_terms': int(len(clouds[i].terms)),
                                'n_raw_tokens': int(raw_token_counts[i]),
                                'n_selected_tokens': int(selected_token_counts[i]),
                                'n_selected_unique': int(selected_unique_counts[
                                        i
                                    ]),
                                'selected_ratio': float(float(selected_token_counts[
                                            i
                                        ])
                                    / max(
                                        1.0,
                                        float(raw_token_counts[
                                                i
                                            ]),
                                    )),
                                'used_unk_fallback': bool(selected_token_counts[i]
                                    == 0),
                                'no_emotional_terms': bool(not valid_mask[i]),
                                'neighbours': neighbours_out,
                                'term_weight': (
                                    [
                                        float(x)
                                        for x in doc_term_weights[
                                            i
                                        ]
                                    ]
                                    if doc_term_weights
                                    is not None
                                    else None
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + '\n')
                    continue
                scored: List[tuple[float, int]] = []
                pos_i = idx_to_pos[i]
                for pos_j in cand_idx_cpu[pos_i].tolist():
                    j = valid_indices[int(pos_j)]
                    if j == i:
                        continue
                    (
                        a,
                        b,
                    ) = (i, j) if i < j else (j, i)
                    scored.append((float(pair_to_dist[a, b]), j))
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:k]
                tokens_i = lex_tokenize(texts[i])
                neighbours_out = []
                for dist_f, j in top:
                    j = int(j)
                    primary_explain = None
                    if include_explain:
                        (
                            a,
                            b,
                        ) = (i, j) if i < j else (j, i)
                        base = explain_cache.get((a, b))
                        if base is None:
                            self_a = (
                                _get_self_cost(a)
                                if use_div
                                else None
                            )
                            self_b = (
                                _get_self_cost(b)
                                if use_div
                                else None
                            )
                            (
                                _,
                                base,
                            ) = (
                                _pair_distance_and_explain(
                                    clouds[a],
                                    clouds[b],
                                    mode=mode,
                                    cost=cost,
                                    alpha_embed=float(alpha_embed),
                                    beta_vad=float(beta_vad),
                                    epsilon=epsilon,
                                    iters=iters,
                                    reg_m=reg_m,
                                    top_flows=int(top_flows),
                                    include_plan=True,
                                    device=device,
                                    self_cost_a=self_a,
                                    self_cost_b=self_b,
                                )
                            )
                            explain_cache[a, b] = base
                        if i == a:
                            primary_explain = (
                                _copy_primary_explain(base)
                            )
                        else:
                            primary_explain = (
                                _reverse_primary_explain(base)
                            )
                        tokens_j = lex_tokenize(texts[j])
                        primary_explain = (
                            _maybe_add_context(
                                primary_explain,
                                src_tokens=tokens_i,
                                tgt_tokens=tokens_j,
                            )
                        )
                    neighbours_out.append({
                            'index': j,
                            'distance': float(dist_f),
                            'mass_diff': float(abs(masses[i] - masses[j])),
                            'primary_explain': primary_explain,
                        })
                if neighbours_payload is not None:
                    neighbours_payload.append({
                            'index': i,
                            'neighbours': neighbours_out,
                        })
                f.write(json.dumps(
                        {
                            'index': i,
                            'meta': metas[i],
                            'mass': float(masses[i]),
                            'n_terms': int(len(clouds[i].terms)),
                            'n_raw_tokens': int(raw_token_counts[i]),
                            'n_selected_tokens': int(selected_token_counts[i]),
                            'n_selected_unique': int(selected_unique_counts[i]),
                            'selected_ratio': float(float(selected_token_counts[i])
                                / max(
                                    1.0,
                                    float(raw_token_counts[i]),
                                )),
                            'used_unk_fallback': bool(selected_token_counts[i]
                                == 0),
                            'no_emotional_terms': bool(not valid_mask[i]),
                            'neighbours': neighbours_out,
                            'term_weight': (
                                [
                                    float(x)
                                    for x in doc_term_weights[
                                        i
                                    ]
                                ]
                                if doc_term_weights
                                is not None
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + '\n')
        out_stats = {
            'docs': n,
            'format': 'neighbours',
            'topk': int(topk),
            'candidate_k': int(cand_k),
            'output': str(out_p),
            'drop_top_df': int(k_drop),
            'coverage_total': float(coverage_total),
            'docs_with_zero_selected': int(docs_with_zero_selected),
        }
        if (
            vis_enabled
            and vis_out_p is not None
            and (neighbours_payload is not None)
        ):
            from .token_viz import (
                write_token_ot_html_report,
            )

            docs_payload = []
            for i in range(n):
                docs_payload.append({
                        'index': i,
                        'meta': metas[i],
                        'mass': float(masses[i]),
                        'n_terms': int(len(clouds[i].terms)),
                        'n_raw_tokens': int(raw_token_counts[i]),
                        'n_selected_tokens': int(selected_token_counts[i]),
                        'n_selected_unique': int(selected_unique_counts[i]),
                        'selected_ratio': float(float(selected_token_counts[i])
                            / max(
                                1.0,
                                float(raw_token_counts[i]),
                            )),
                        'used_unk_fallback': bool(selected_token_counts[i] == 0),
                        'no_emotional_terms': bool(not valid_mask[i]),
                        'terms': list(clouds[i].terms),
                        'weights': [
                            float(x) for x in doc_weights[i]
                        ],
                        'term_weight': (
                            [
                                float(x)
                                for x in doc_term_weights[i]
                            ]
                            if doc_term_weights is not None
                            else None
                        ),
                    })
            report = {
                'format': 'neighbours',
                'mode': mode,
                'focus': focus,
                'cost': cost,
                'embed_model': (
                    embed_model_name if need_embed else None
                ),
                'embed_backend': (
                    embed_backend_mode
                    if need_embed
                    else None
                ),
                'embed_pooling': (
                    embed_pooling_mode
                    if need_embed
                    else None
                ),
                'embed_batch_size': (
                    int(embed_bs) if need_embed else None
                ),
                'embed_max_length': (
                    int(embed_max_len)
                    if need_embed
                    else None
                ),
                'embed_prompt_mode': (
                    str(embed_prompt_mode)
                    if need_embed
                    else None
                ),
                'embed_prompt_text': (
                    str(embed_prompt_text)
                    if need_embed and embed_prompt_text
                    else None
                ),
                'alpha_embed': float(alpha_embed),
                'beta_vad': float(beta_vad),
                'vad_threshold': float(vad_threshold),
                'emotional_vocab': (
                    str(emotional_vocab)
                    if focus != 'all'
                    else None
                ),
                'vad_min_arousal_vad_only': (
                    float(vad_min_arousal_vad_only)
                    if focus != 'all'
                    else None
                ),
                'max_ngram': int(max_ngram_eff),
                'weight': weight,
                'term_weight_path': (
                    str(term_weights_path)
                    if term_weights_path
                    else None
                ),
                'term_weight_meta': term_weights_meta,
                'term_weight_power': float(term_weight_power),
                'term_weight_min': float(term_weight_min),
                'term_weight_max': float(term_weight_max),
                'term_weight_mix': float(term_weight_mix),
                'term_weight_default': float(term_weight_default),
                'vad_imputed_path': (
                    str(vad_imputed_path)
                    if vad_imputed_path
                    else None
                ),
                'epsilon': float(epsilon),
                'iters': int(iters),
                'reg_m': float(reg_m),
                'max_terms': int(max_terms),
                'stopwords': bool(stopwords),
                'stopwords_file': (
                    str(stopwords_file)
                    if stopwords_file is not None
                    else None
                ),
                'negation_window': int(neg_window),
                'negators': list(negator_terms),
                'drop_top_df': int(k_drop),
                'dropped_top_df_terms': list(dropped_top_df_terms),
                'coverage_total': float(coverage_total),
                'docs_with_zero_selected': int(docs_with_zero_selected),
                'docs': docs_payload,
                'neighbours': neighbours_payload,
                'topk': int(topk),
                'top_flows': int(top_flows),
            }
            out_stats['visualization'] = (
                write_token_ot_html_report(
                    output_path=vis_out_p,
                    payload=report,
                )
            )
        return out_stats
    neighbours_payload: Optional[List[Dict[str, Any]]] = (
        [] if vis_enabled else None
    )
    D_valid: Optional[torch.Tensor] = None
    if valid_n > 0:
        D_valid = torch.empty(
            (valid_n, valid_n),
            dtype=torch.float,
        )
        for pos_i, i in enumerate(valid_indices):
            D_valid[pos_i, pos_i] = (
                0.0
                if mode.endswith('_divergence')
                else float(_pair_distance_and_explain(
                        clouds[i],
                        clouds[i],
                        mode=mode,
                        cost=cost,
                        alpha_embed=float(alpha_embed),
                        beta_vad=float(beta_vad),
                        epsilon=epsilon,
                        iters=iters,
                        reg_m=reg_m,
                        top_flows=0,
                        include_plan=False,
                        device=device,
                    )[0])
            )
        for pos_i, i in enumerate(valid_indices):
            for pos_j in range(
                pos_i + 1,
                valid_n,
            ):
                j = valid_indices[pos_j]
                self_i = (
                    _get_self_cost(i) if use_div else None
                )
                self_j = (
                    _get_self_cost(j) if use_div else None
                )
                (
                    dist,
                    _,
                ) = _pair_distance_and_explain(
                    clouds[i],
                    clouds[j],
                    mode=mode,
                    cost=cost,
                    alpha_embed=float(alpha_embed),
                    beta_vad=float(beta_vad),
                    epsilon=epsilon,
                    iters=iters,
                    reg_m=reg_m,
                    top_flows=0,
                    include_plan=False,
                    device=device,
                    self_cost_a=self_i,
                    self_cost_b=self_j,
                )
                D_valid[pos_i, pos_j] = float(dist)
                D_valid[pos_j, pos_i] = float(dist)
    with out_p.open(
        'w',
        encoding='utf-8',
    ) as f:
        for i in range(n):
            if (
                not valid_mask[i]
                or D_valid is None
                or k <= 0
            ):
                neighbours_out: List[Dict[str, Any]] = []
                if neighbours_payload is not None:
                    neighbours_payload.append({
                            'index': i,
                            'neighbours': neighbours_out,
                        })
                f.write(json.dumps(
                        {
                            'index': i,
                            'meta': metas[i],
                            'mass': float(masses[i]),
                            'n_terms': int(len(clouds[i].terms)),
                            'n_raw_tokens': int(raw_token_counts[i]),
                            'n_selected_tokens': int(selected_token_counts[i]),
                            'n_selected_unique': int(selected_unique_counts[i]),
                            'selected_ratio': float(float(selected_token_counts[i])
                                / max(
                                    1.0,
                                    float(raw_token_counts[i]),
                                )),
                            'used_unk_fallback': bool(selected_token_counts[i]
                                == 0),
                            'no_emotional_terms': bool(not valid_mask[i]),
                            'neighbours': neighbours_out,
                            'term_weight': (
                                [
                                    float(x)
                                    for x in doc_term_weights[
                                        i
                                    ]
                                ]
                                if doc_term_weights
                                is not None
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + '\n')
                continue
            pos_i = idx_to_pos[i]
            d = D_valid[pos_i].clone()
            d[pos_i] = float('inf')
            (
                vals,
                idx,
            ) = torch.topk(
                d,
                k=k,
                largest=False,
            )
            neighbours_out = []
            tokens_i = lex_tokenize(texts[i])
            for pos_j, v in zip(
                idx.tolist(),
                vals.tolist(),
            ):
                j = valid_indices[int(pos_j)]
                dist_f = float(v)
                primary_explain = None
                if include_explain:
                    self_i = (
                        _get_self_cost(i)
                        if use_div
                        else None
                    )
                    self_j = (
                        _get_self_cost(j)
                        if use_div
                        else None
                    )
                    (
                        _,
                        primary_explain,
                    ) = (
                        _pair_distance_and_explain(
                            clouds[i],
                            clouds[j],
                            mode=mode,
                            cost=cost,
                            alpha_embed=float(alpha_embed),
                            beta_vad=float(beta_vad),
                            epsilon=epsilon,
                            iters=iters,
                            reg_m=reg_m,
                            top_flows=int(top_flows),
                            include_plan=True,
                            device=device,
                            self_cost_a=self_i,
                            self_cost_b=self_j,
                        )
                    )
                    if isinstance(
                        primary_explain,
                        dict,
                    ):
                        tokens_j = lex_tokenize(texts[j])
                        for key in (
                            'top_transport_cost_contrib',
                            'top_transport_mass',
                        ):
                            flows = primary_explain.get(key)
                            if not isinstance(
                                flows,
                                list,
                            ):
                                continue
                            for flow in flows:
                                if not isinstance(
                                    flow,
                                    dict,
                                ):
                                    continue
                                a_term = str(flow.get('from') or '')
                                b_term = str(flow.get('to') or '')
                                flow['from_context'] = (
                                    _snippet_from_tokens(
                                        tokens_i,
                                        a_term,
                                    )
                                    if a_term
                                    else None
                                )
                                flow['to_context'] = (
                                    _snippet_from_tokens(
                                        tokens_j,
                                        b_term,
                                    )
                                    if b_term
                                    else None
                                )
                neighbours_out.append({
                        'index': j,
                        'distance': dist_f,
                        'mass_diff': float(abs(masses[i] - masses[j])),
                        'primary_explain': primary_explain,
                    })
            if neighbours_payload is not None:
                neighbours_payload.append({
                        'index': i,
                        'neighbours': neighbours_out,
                    })
            f.write(json.dumps(
                    {
                        'index': i,
                        'meta': metas[i],
                        'mass': float(masses[i]),
                        'n_terms': int(len(clouds[i].terms)),
                        'n_raw_tokens': int(raw_token_counts[i]),
                        'n_selected_tokens': int(selected_token_counts[i]),
                        'n_selected_unique': int(selected_unique_counts[i]),
                        'selected_ratio': float(float(selected_token_counts[i])
                            / max(
                                1.0,
                                float(raw_token_counts[i]),
                            )),
                        'used_unk_fallback': bool(selected_token_counts[i] == 0),
                        'no_emotional_terms': bool(not valid_mask[i]),
                        'neighbours': neighbours_out,
                        'term_weight': (
                            [
                                float(x)
                                for x in doc_term_weights[i]
                            ]
                            if doc_term_weights is not None
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + '\n')
    out_stats: Dict[str, Any] = {
        'docs': n,
        'format': 'neighbours',
        'topk': int(topk),
        'output': str(out_p),
        'drop_top_df': int(k_drop),
        'coverage_total': float(coverage_total),
        'docs_with_zero_selected': int(docs_with_zero_selected),
    }
    if (
        vis_enabled
        and vis_out_p is not None
        and (neighbours_payload is not None)
    ):
        from .token_viz import write_token_ot_html_report

        docs_payload = []
        include_term_meta = bool(term_vad)
        for i in range(n):
            terms_out = list(clouds[i].terms)
            d: Dict[str, Any] = {
                'index': i,
                'meta': metas[i],
                'mass': float(masses[i]),
                'n_terms': int(len(clouds[i].terms)),
                'n_raw_tokens': int(raw_token_counts[i]),
                'n_selected_tokens': int(selected_token_counts[i]),
                'n_selected_unique': int(selected_unique_counts[i]),
                'selected_ratio': float(float(selected_token_counts[i])
                    / max(
                        1.0,
                        float(raw_token_counts[i]),
                    )),
                'used_unk_fallback': bool(selected_token_counts[i] == 0),
                'no_emotional_terms': bool(not valid_mask[i]),
                'terms': terms_out,
                'weights': [
                    float(x) for x in doc_weights[i]
                ],
            }
            if doc_term_weights is not None:
                d['term_weight'] = [
                    float(x) for x in doc_term_weights[i]
                ]
            if include_term_meta:
                V_cpu = (
                    clouds[i]
                    .vad.detach()
                    .cpu()
                    .to(dtype=torch.float)
                )
                if V_cpu.numel():
                    val_abs = torch.abs(V_cpu[:, 0]).clamp(
                        0.0,
                        1.0,
                    )
                    aro_pos = V_cpu[:, 1].clamp(
                        0.0,
                        1.0,
                    )
                    sal = torch.maximum(
                        val_abs,
                        aro_pos,
                    ).tolist()
                    vad_out = V_cpu.tolist()
                else:
                    sal = []
                    vad_out = []
                sources_out: List[str] = []
                for t in terms_out:
                    sources_out.append(_term_source_label(t))
                d['term_vad'] = vad_out
                d['term_salience'] = [float(x) for x in sal]
                d['term_source'] = sources_out
                if term_vad_conf:
                    d['term_vad_conf'] = [
                        float(term_vad_conf.get(
                                t,
                                0.0,
                            ))
                        for t in terms_out
                    ]
            docs_payload.append(d)
        report = {
            'format': 'neighbours',
            'mode': mode,
            'focus': focus,
            'cost': cost,
            'embed_model': (
                embed_model_name if need_embed else None
            ),
            'embed_backend': (
                embed_backend_mode if need_embed else None
            ),
            'embed_pooling': (
                embed_pooling_mode if need_embed else None
            ),
            'embed_batch_size': (
                int(embed_bs) if need_embed else None
            ),
            'embed_max_length': (
                int(embed_max_len) if need_embed else None
            ),
            'embed_prompt_mode': (
                str(embed_prompt_mode)
                if need_embed
                else None
            ),
            'embed_prompt_text': (
                str(embed_prompt_text)
                if need_embed and embed_prompt_text
                else None
            ),
            'alpha_embed': float(alpha_embed),
            'beta_vad': float(beta_vad),
            'vad_threshold': float(vad_threshold),
            'emotional_vocab': (
                str(emotional_vocab)
                if focus != 'all'
                else None
            ),
            'vad_min_arousal_vad_only': (
                float(vad_min_arousal_vad_only)
                if focus != 'all'
                else None
            ),
            'max_ngram': int(max_ngram_eff),
            'weight': weight,
            'term_weight_path': (
                str(term_weights_path)
                if term_weights_path
                else None
            ),
            'term_weight_meta': term_weights_meta,
            'term_weight_power': float(term_weight_power),
            'term_weight_min': float(term_weight_min),
            'term_weight_max': float(term_weight_max),
            'term_weight_mix': float(term_weight_mix),
            'term_weight_default': float(term_weight_default),
            'vad_imputed_path': (
                str(vad_imputed_path)
                if vad_imputed_path
                else None
            ),
            'epsilon': float(epsilon),
            'iters': int(iters),
            'reg_m': float(reg_m),
            'max_terms': int(max_terms),
            'stopwords': bool(stopwords),
            'stopwords_file': (
                str(stopwords_file)
                if stopwords_file is not None
                else None
            ),
            'negation_window': int(neg_window),
            'negators': list(negator_terms),
            'drop_top_df': int(k_drop),
            'dropped_top_df_terms': list(dropped_top_df_terms),
            'coverage_total': float(coverage_total),
            'docs_with_zero_selected': int(docs_with_zero_selected),
            'docs': docs_payload,
            'neighbours': neighbours_payload,
            'topk': int(topk),
            'top_flows': int(top_flows),
        }
        out_stats['visualization'] = (
            write_token_ot_html_report(
                output_path=vis_out_p,
                payload=report,
            )
        )
    return out_stats


def explain_token_pair(
    *,
    input_jsonl: str,
    i: int,
    j: int,
    cfg_path: Optional[str] = None,
    mode: Optional[str] = None,
    focus: str = 'emotional',
    cost: str = 'embedding_vad',
    embed_model: Optional[str] = None,
    embed_backend: Optional[str] = None,
    embed_pooling: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    embed_max_length: Optional[int] = None,
    embed_prompt_mode: Optional[str] = None,
    embed_prompt_text: Optional[str] = None,
    alpha_embed: float = 0.5,
    beta_vad: float = 0.5,
    vad_threshold: float = 0.45,
    emotional_vocab: str = 'auto',
    vad_min_arousal_vad_only: float = 0.45,
    max_ngram: int = 0,
    epsilon: Optional[float] = None,
    iters: Optional[int] = None,
    reg_m: Optional[float] = None,
    weight: str = 'tfidf',
    term_weights_path: Optional[str] = None,
    term_weight_power: float = 1.0,
    term_weight_min: float = 0.0,
    term_weight_max: float = 1.0,
    term_weight_mix: float = 1.0,
    term_weight_default: float = 1.0,
    max_terms: int = 256,
    min_token_len: int = 2,
    stopwords: bool = True,
    stopwords_file: Optional[str] = None,
    negation_window: Optional[int] = None,
    negators: Optional[Iterable[str]] = None,
    drop_top_df: int = 100,
    vad_imputed_path: Optional[str] = None,
    top_flows: int = 8,
) -> Dict[str, Any]:
    setup = load_config(cfg_path)
    rows = _read_jsonl(
        input_jsonl,
        limit=None,
    )
    if not rows:
        raise ValueError('error: ValueError')
    n = len(rows)
    i = int(i)
    j = int(j)
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError('error: ValueError')
    if i == j:
        raise ValueError('error: ValueError')
    mode = (mode or 'sinkhorn_divergence').lower()
    focus = (focus or 'emotional').lower().strip()
    if focus not in {'all', 'emotional'}:
        raise ValueError('error: ValueError')
    cost_default = (
        'embedding_vad'
        if focus == 'emotional'
        else 'embedding'
    )
    cost = (cost or cost_default).lower().strip()
    if (
        focus == 'all'
        and cost == 'vad'
        and (not getattr(
            setup,
            'vad_lexicon_path',
            None,
        ))
    ):
        warnings.warn(
            'warn: fallback',
            RuntimeWarning,
            stacklevel=2,
        )
        cost = 'embedding'
    epsilon = float(epsilon
        if epsilon is not None
        else setup.sinkhorn_epsilon)
    iters = int(iters if iters is not None else setup.sinkhorn_iters)
    reg_m = float(reg_m if reg_m is not None else setup.ot_reg_m)
    term_weights_path = term_weights_path or getattr(
        setup,
        'token_term_weights_path',
        None,
    )
    term_weight_power = float(term_weight_power
        if term_weight_power is not None
        else getattr(
            setup,
            'token_term_weight_power',
            1.0,
        ))
    term_weight_min = float(term_weight_min
        if term_weight_min is not None
        else getattr(
            setup,
            'token_term_weight_min',
            0.0,
        ))
    term_weight_max = float(term_weight_max
        if term_weight_max is not None
        else getattr(
            setup,
            'token_term_weight_max',
            1.0,
        ))
    term_weight_mix = float(term_weight_mix
        if term_weight_mix is not None
        else getattr(
            setup,
            'token_term_weight_mix',
            1.0,
        ))
    term_weight_default = float(term_weight_default
        if term_weight_default is not None
        else getattr(
            setup,
            'token_term_weight_default',
            1.0,
        ))
    (
        term_weights,
        term_weights_meta,
    ) = (
        _load_term_weights(term_weights_path)
        if term_weights_path
        else ({}, None)
    )
    vad_imputed_path = vad_imputed_path or getattr(
        setup,
        'token_vad_imputed_path',
        None,
    )
    vad_imputed_weight = float(getattr(
            setup,
            'token_vad_imputed_weight',
            0.0,
        )
        or 0.0)
    device = get_device(setup.device)
    embed_model_name = str(embed_model
        or getattr(
            setup,
            'token_ot_embed_model',
            None,
        )
        or getattr(
            setup,
            'embed_model_name',
            None,
        )
        or getattr(
            setup,
            'model_name',
            None,
        )
        or setup.model_name)
    embed_backend_mode = str(embed_backend
        or getattr(
            setup,
            'token_ot_embed_backend',
            None,
        )
        or 'encoder')
    embed_pooling_mode = str(embed_pooling
        or getattr(
            setup,
            'token_ot_embed_pooling',
            None,
        )
        or 'cls')
    embed_bs = int(embed_batch_size
        if embed_batch_size is not None
        else getattr(
            setup,
            'token_ot_embed_batch_size',
            64,
        ))
    embed_max_len = int(embed_max_length
        if embed_max_length is not None
        else getattr(
            setup,
            'token_ot_embed_max_length',
            32,
        ))
    embed_prompt_mode = str(embed_prompt_mode
        or getattr(
            setup,
            'token_ot_embed_prompt_mode',
            None,
        )
        or 'none')
    embed_prompt_text = (
        embed_prompt_text
        if embed_prompt_text is not None
        else getattr(
            setup,
            'token_ot_embed_prompt_text',
            None,
        )
    )
    unk = str(getattr(
            _load_tokenizer(embed_model_name),
            'unk_token',
            None,
        )
        or '[UNK]')
    texts = [_get_text(r) for r in rows]
    metas = [
        (
            r.get('meta')
            if isinstance(
                r.get('meta'),
                dict,
            )
            else {}
        )
        for r in rows
    ]
    stopwords_default = False
    stopwords_file = stopwords_file or getattr(
        setup,
        'token_stopwords_file',
        None,
    )
    if stopwords and stopwords_file is None:
        default_sw = _default_stopwords_path()
        if default_sw is not None:
            stopwords_file = str(default_sw)
            stopwords_default = True
    vocab_mode = (emotional_vocab or 'auto').lower().strip()
    if vocab_mode == 'auto':
        vocab_mode = (
            str(getattr(
                    setup,
                    'token_emotional_vocab',
                    'lexicon',
                ))
            .lower()
            .strip()
        )
    emotional_vocab = vocab_mode
    allow_vad_stopwords = bool(getattr(
            setup,
            'token_allow_vad_stopwords',
            False,
        ))
    if not allow_vad_stopwords:
        allow_vad_stopwords = bool(stopwords_default and emotional_vocab == 'vad')
    include_english = bool(getattr(
            setup,
            'token_include_english_stopwords',
            False,
        ))
    sw: set[str] = set()
    if stopwords and include_english:
        sw |= set(ENGLISH_STOP_WORDS)
    (
        extra_sw,
        extra_phrase_sw,
    ) = (
        _load_stopword_terms(stopwords_file)
        if stopwords
        else (set(), set())
    )
    sw |= extra_sw
    neg_window = int(negation_window
        if negation_window is not None
        else getattr(
            setup,
            'lexicon_negation_window',
            0,
        ))
    negator_list = (
        negators
        if negators is not None
        else getattr(
            setup,
            'lexicon_negators',
            None,
        )
    )
    (
        neg_tokens,
        neg_phrases,
    ) = _normalise_negators(negator_list)
    negator_terms = (
        [str(n) for n in negator_list]
        if negator_list
        else []
    )
    term_vad: Dict[str, torch.Tensor] = {}
    term_vad_conf: Dict[str, float] = {}
    lexicon_terms: set[str] = set()
    word_vad_terms: set[str] = set()
    imputed_terms: set[str] = set()
    if focus == 'emotional' or cost in {
        'vad',
        'embedding_vad',
    }:
        lex_stopwords_path = getattr(
            setup,
            'lexicon_stopwords_file',
            None,
        )
        word_vad_stopwords_path = (
            None
            if allow_vad_stopwords
            else lex_stopwords_path
        )
        if (
            focus == 'emotional'
            and (not getattr(
                setup,
                'lexicon_path',
                None,
            ))
            and (
                not getattr(
                    setup,
                    'vad_lexicon_path',
                    None,
                )
            )
        ):
            warnings.warn(
                'warn: seed',
                RuntimeWarning,
                stacklevel=2,
            )
        lex = load_lexicon(
            getattr(
                setup,
                'lexicon_path',
                None,
            ),
            getattr(
                setup,
                'vad_lexicon_path',
                None,
            ),
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
            stopwords_path=lex_stopwords_path,
            word_vad_stopwords_path=word_vad_stopwords_path,
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
        word_vad = (
            load_word_vad(
                vad_path,
                vad_scale=getattr(
                    setup,
                    'word_vad_scale',
                    None,
                ),
                stopwords_path=word_vad_stopwords_path,
            )
            if vad_path
            else {}
        )
        (
            term_vad,
            term_vad_conf,
        ) = _build_term_vad_map(
            lex,
            word_vad,
            imputed_weight=vad_imputed_weight,
            return_conf=True,
        )
        lexicon_terms = set(_term_vad_from_lexicon(lex).keys())
        word_vad_terms = set((
                ' '.join(lex_tokenize(t))
                for t in (word_vad or {}).keys()
                if ' '.join(lex_tokenize(t))
            ))
        if vad_imputed_path:
            (
                imputed_vad,
                imputed_conf,
            ) = (
                _load_imputed_vad_map(vad_imputed_path)
            )
            imputed_terms = _merge_imputed_vad(
                term_vad,
                term_vad_conf,
                imputed_vad,
                imputed_conf,
                default_conf=vad_imputed_weight,
            )
            if imputed_terms:
                word_vad_terms |= imputed_terms
    max_ngram_eff = (
        int(max_ngram) if max_ngram is not None else 0
    )
    if max_ngram_eff <= 0:
        lex_max = max(
            (len(t.split()) for t in lexicon_terms),
            default=1,
        )
        vad_max = max(
            (len(t.split()) for t in word_vad_terms),
            default=1,
        )
        max_ngram_eff = max(
            lex_max,
            vad_max,
            1,
        )
    vad_support = any((v > 0.0 for v in term_vad_conf.values()))
    if cost == 'vad' and (not vad_support):
        warnings.warn(
            'warn: vad',
            RuntimeWarning,
            stacklevel=2,
        )
        cost = 'embedding'
    raw_doc_tokens = [lex_tokenize(t) for t in texts]
    raw_token_counts = [
        int(len(toks)) for toks in raw_doc_tokens
    ]
    docs_all_terms = [
        [
            w
            for w in toks
            if len(w) >= int(min_token_len) and w not in sw
        ]
        for toks in raw_doc_tokens
    ]
    docs_terms: List[List[str]] = []
    if focus == 'all':
        docs_terms = [list(toks) for toks in docs_all_terms]
    else:
        (
            allowed_terms,
            ngram_map,
        ) = _build_emotional_matcher(
            lexicon_terms=lexicon_terms,
            word_vad_terms=word_vad_terms,
            term_vad=term_vad,
            term_vad_conf=term_vad_conf,
            vad_threshold=float(vad_threshold),
            vad_min_arousal_vad_only=float(vad_min_arousal_vad_only),
            emotional_vocab=str(emotional_vocab),
            max_ngram=int(max_ngram_eff),
            stopword_terms=extra_phrase_sw,
        )
        for toks in raw_doc_tokens:
            docs_terms.append(_extract_emotional_terms(
                    toks,
                    lexicon_terms=lexicon_terms,
                    word_vad_terms=word_vad_terms,
                    term_vad=term_vad,
                    term_vad_conf=term_vad_conf,
                    vad_threshold=float(vad_threshold),
                    vad_min_arousal_vad_only=float(vad_min_arousal_vad_only),
                    emotional_vocab=str(emotional_vocab),
                    max_ngram=int(max_ngram_eff),
                    stopwords=sw,
                    stopword_terms=extra_phrase_sw,
                    min_token_len=int(min_token_len),
                    negation_window=neg_window,
                    negator_tokens=neg_tokens,
                    negator_phrases=neg_phrases,
                    allow_vad_terms_in_stopwords=allow_vad_stopwords,
                    allowed_terms=allowed_terms,
                    ngram_to_term=ngram_map,
                ))
    if (
        focus == 'all'
        and cost == 'vad'
        and (not term_vad_conf)
    ):
        docs_terms = [
            [t for t in toks if t in term_vad]
            for toks in docs_terms
        ]
    k_drop = int(drop_top_df)
    dropped_top_df_terms: List[str] = []
    if k_drop > 0:
        df: Counter[str] = Counter()
        for toks in docs_all_terms:
            for t in set(toks):
                df[t] += 1
        if df:
            items = sorted(
                df.items(),
                key=lambda x: (x[1], x[0]),
                reverse=True,
            )
            if len(items) > k_drop:
                dropped_top_df_terms = [
                    t for t, _ in items[:k_drop]
                ]
                drop_set = set(dropped_top_df_terms)
                if drop_set:
                    docs_terms = [
                        [
                            t
                            for t in toks
                            if t not in drop_set
                        ]
                        for toks in docs_terms
                    ]
    if cost == 'vad' and term_vad_conf:
        docs_terms = [
            [
                t
                for t in toks
                if term_vad_conf.get(
                    t,
                    0.0,
                ) > 0.0
            ]
            for toks in docs_terms
        ]
    selected_token_counts = [
        int(len(toks)) for toks in docs_terms
    ]
    selected_unique_counts = [
        int(len(set(toks))) for toks in docs_terms
    ]
    idf = _build_idf(docs_terms)
    (
        terms_i,
        weights_i,
    ) = _doc_terms_and_weights(
        docs_terms[i],
        idf=idf,
        weight=weight,
        max_terms=int(max_terms),
        unk_token=unk,
        allow_empty=True,
        term_weights=(
            term_weights if bool(term_weights) else None
        ),
        term_weight_default=term_weight_default,
        term_weight_min=term_weight_min,
        term_weight_max=term_weight_max,
        term_weight_power=term_weight_power,
        term_weight_mix=term_weight_mix,
    )
    (
        terms_j,
        weights_j,
    ) = _doc_terms_and_weights(
        docs_terms[j],
        idf=idf,
        weight=weight,
        max_terms=int(max_terms),
        unk_token=unk,
        allow_empty=True,
        term_weights=(
            term_weights if bool(term_weights) else None
        ),
        term_weight_default=term_weight_default,
        term_weight_min=term_weight_min,
        term_weight_max=term_weight_max,
        term_weight_power=term_weight_power,
        term_weight_mix=term_weight_mix,
    )
    term_weights_active = bool(term_weights)
    term_weight_i = None
    term_weight_j = None
    if term_weights_active:
        (
            weights_i,
            term_weight_i,
        ) = _apply_term_weights(
            terms_i,
            weights_i,
            term_weights,
            weight_default=term_weight_default,
            weight_min=term_weight_min,
            weight_max=term_weight_max,
            weight_power=term_weight_power,
            weight_mix=term_weight_mix,
        )
        (
            weights_j,
            term_weight_j,
        ) = _apply_term_weights(
            terms_j,
            weights_j,
            term_weights,
            weight_default=term_weight_default,
            weight_min=term_weight_min,
            weight_max=term_weight_max,
            weight_power=term_weight_power,
            weight_mix=term_weight_mix,
        )
    cost_mode = (cost or 'embedding').lower().strip()
    need_embed = cost_mode in {'embedding', 'embedding_vad'}
    vad_support = any((v > 0.0 for v in term_vad_conf.values()))
    if (
        cost_mode == 'embedding_vad'
        and (not vad_support)
        and (float(beta_vad) > 0.0)
    ):
        warnings.warn(
            'warn: beta_vad',
            RuntimeWarning,
            stacklevel=2,
        )
        beta_vad = 0.0
    if (
        need_embed
        and embed_model is None
        and (
            getattr(
                setup,
                'token_ot_embed_model',
                None,
            )
            is None
        )
        and (embed_model_name == setup.model_name)
    ):
        warnings.warn(
            'warn: base',
            RuntimeWarning,
            stacklevel=2,
        )
    allow_model_embed = bool(getattr(
            setup,
            'token_ot_allow_model_embed',
            False,
        ))
    if (
        focus == 'emotional'
        and need_embed
        and (embed_model is None)
        and (
            getattr(
                setup,
                'token_ot_embed_model',
                None,
            )
            is None
        )
        and (embed_model_name == setup.model_name)
        and (not allow_model_embed)
    ):
        if vad_support and float(beta_vad) > 0.0:
            warnings.warn(
                'warn: base',
                RuntimeWarning,
                stacklevel=2,
            )
            cost_mode = 'vad'
            cost = cost_mode
            need_embed = False
            alpha_embed = 0.0
        else:
            warnings.warn(
                'warn: weak',
                RuntimeWarning,
                stacklevel=2,
            )
    if not terms_i or not terms_j:

        def _doc_payload(
            idx: int,
            terms: List[str],
            weights: List[float],
            term_weight: Optional[List[float]],
        ) -> Dict[str, Any]:
            return {
                'index': idx,
                'meta': metas[idx],
                'n_raw_tokens': int(raw_token_counts[idx]),
                'n_selected_tokens': int(selected_token_counts[idx]),
                'n_selected_unique': int(selected_unique_counts[idx]),
                'selected_ratio': float(float(selected_token_counts[idx])
                    / max(
                        1.0,
                        float(raw_token_counts[idx]),
                    )),
                'used_unk_fallback': bool(selected_token_counts[idx] == 0),
                'no_emotional_terms': bool(selected_token_counts[idx] == 0),
                'terms': terms,
                'weights': [float(x) for x in weights],
                'term_weight': (
                    [float(x) for x in term_weight]
                    if term_weight
                    else None
                ),
                'term_vad': None,
                'term_salience': None,
                'term_source': None,
                'term_vad_conf': None,
            }

        missing = []
        if not terms_i:
            missing.append(f'i={i}')
        if not terms_j:
            missing.append(f'j={j}')
        return {
            'settings': {
                'mode': mode,
                'focus': focus,
                'cost': cost,
                'embed_model': (
                    embed_model_name if need_embed else None
                ),
                'embed_backend': (
                    embed_backend_mode
                    if need_embed
                    else None
                ),
                'embed_pooling': (
                    embed_pooling_mode
                    if need_embed
                    else None
                ),
                'embed_batch_size': (
                    int(embed_bs) if need_embed else None
                ),
                'embed_max_length': (
                    int(embed_max_len)
                    if need_embed
                    else None
                ),
                'embed_prompt_mode': (
                    str(embed_prompt_mode)
                    if need_embed
                    else None
                ),
                'embed_prompt_text': (
                    str(embed_prompt_text)
                    if need_embed and embed_prompt_text
                    else None
                ),
                'alpha_embed': float(alpha_embed),
                'beta_vad': float(beta_vad),
                'vad_threshold': float(vad_threshold),
                'emotional_vocab': (
                    str(emotional_vocab)
                    if focus != 'all'
                    else None
                ),
                'vad_min_arousal_vad_only': (
                    float(vad_min_arousal_vad_only)
                    if focus != 'all'
                    else None
                ),
                'max_ngram': int(max_ngram_eff),
                'weight': weight,
                'term_weight_path': (
                    str(term_weights_path)
                    if term_weights_path
                    else None
                ),
                'term_weight_meta': term_weights_meta,
                'term_weight_power': float(term_weight_power),
                'term_weight_min': float(term_weight_min),
                'term_weight_max': float(term_weight_max),
                'term_weight_mix': float(term_weight_mix),
                'term_weight_default': float(term_weight_default),
                'vad_imputed_path': (
                    str(vad_imputed_path)
                    if vad_imputed_path
                    else None
                ),
                'epsilon': float(epsilon),
                'iters': int(iters),
                'reg_m': float(reg_m),
                'max_terms': int(max_terms),
                'min_token_len': int(min_token_len),
                'stopwords': bool(stopwords),
                'stopwords_file': (
                    str(stopwords_file)
                    if stopwords_file is not None
                    else None
                ),
                'negation_window': int(neg_window),
                'negators': list(negator_terms),
                'drop_top_df': int(k_drop),
                'dropped_top_df_terms': list(dropped_top_df_terms),
            },
            'i': _doc_payload(
                i,
                terms_i,
                weights_i,
                term_weight_i,
            ),
            'j': _doc_payload(
                j,
                terms_j,
                weights_j,
                term_weight_j,
            ),
            'primary': {
                'error': f'no emotional terms selected for {', '.join(missing)}',
                'distance': None,
            },
        }
    term_to_emb: Dict[str, torch.Tensor] = {}
    if need_embed:
        vocab = sorted(set(terms_i) | set(terms_j))
        term_to_emb = _embed_terms(
            vocab,
            model_name=embed_model_name,
            device=device,
            backend=embed_backend_mode,
            pooling=embed_pooling_mode,
            batch_size=int(embed_bs),
            max_length=int(embed_max_len),
            amp=str(getattr(
                    setup,
                    'amp',
                    None,
                ) or 'none'),
            prompt_mode=embed_prompt_mode,
            prompt_text=embed_prompt_text,
        )
    if need_embed:
        Xi = torch.stack(
            [term_to_emb[t] for t in terms_i],
            dim=0,
        ).to(device=device)
        Xj = torch.stack(
            [term_to_emb[t] for t in terms_j],
            dim=0,
        ).to(device=device)
        Ci_embed = _cosine_cost(
            Xi,
            Xi,
        )
        Cj_embed = _cosine_cost(
            Xj,
            Xj,
        )
        Ci_embed.fill_diagonal_(0.0)
        Cj_embed.fill_diagonal_(0.0)
    else:
        Xi = torch.zeros(
            (len(terms_i), 1),
            dtype=torch.float,
            device=device,
        )
        Xj = torch.zeros(
            (len(terms_j), 1),
            dtype=torch.float,
            device=device,
        )
        Ci_embed = torch.zeros(
            (len(terms_i), len(terms_i)),
            dtype=torch.float,
            device=device,
        )
        Cj_embed = torch.zeros(
            (len(terms_j), len(terms_j)),
            dtype=torch.float,
            device=device,
        )
    wi = torch.tensor(
        weights_i,
        device=device,
        dtype=torch.float,
    ).clamp_min(1e-08)
    wj = torch.tensor(
        weights_j,
        device=device,
        dtype=torch.float,
    ).clamp_min(1e-08)
    Vi = torch.stack(
        [
            term_vad.get(
                t,
                torch.zeros(
                    3,
                    dtype=torch.float,
                ),
            )
            for t in terms_i
        ],
        dim=0,
    ).to(device=device)
    Vj = torch.stack(
        [
            term_vad.get(
                t,
                torch.zeros(
                    3,
                    dtype=torch.float,
                ),
            )
            for t in terms_j
        ],
        dim=0,
    ).to(device=device)
    Vi_conf = None
    Vj_conf = None
    if term_vad_conf:
        Vi_conf = torch.tensor(
            [term_vad_conf.get(
                t,
                0.0,
            ) for t in terms_i],
            dtype=torch.float,
            device=device,
        )
        Vj_conf = torch.tensor(
            [term_vad_conf.get(
                t,
                0.0,
            ) for t in terms_j],
            dtype=torch.float,
            device=device,
        )
    vad_fill = 1.0 if cost_mode == 'vad' else 0.0
    Ci_vad = _vad_cost(
        Vi,
        Vi,
        conf_a=Vi_conf,
        conf_b=Vi_conf,
        unknown_cost=vad_fill,
    )
    Cj_vad = _vad_cost(
        Vj,
        Vj,
        conf_a=Vj_conf,
        conf_b=Vj_conf,
        unknown_cost=vad_fill,
    )
    Ci_vad.fill_diagonal_(0.0)
    Cj_vad.fill_diagonal_(0.0)
    cloud_i = TokenCloud(
        terms=terms_i,
        weights=wi,
        X=Xi,
        vad=Vi,
        vad_conf=Vi_conf,
        C_embed_self=Ci_embed,
        C_vad_self=Ci_vad,
    )
    cloud_j = TokenCloud(
        terms=terms_j,
        weights=wj,
        X=Xj,
        vad=Vj,
        vad_conf=Vj_conf,
        C_embed_self=Cj_embed,
        C_vad_self=Cj_vad,
    )
    (
        dist,
        explain,
    ) = _pair_distance_and_explain(
        cloud_i,
        cloud_j,
        mode=mode,
        cost=cost,
        alpha_embed=float(alpha_embed),
        beta_vad=float(beta_vad),
        epsilon=epsilon,
        iters=iters,
        reg_m=reg_m,
        top_flows=int(top_flows),
        include_plan=True,
        device=device,
    )
    if isinstance(
        explain,
        dict,
    ):
        for key in (
            'top_transport_cost_contrib',
            'top_transport_mass',
        ):
            flows = explain.get(key)
            if not isinstance(
                flows,
                list,
            ):
                continue
            for flow in flows:
                if not isinstance(
                    flow,
                    dict,
                ):
                    continue
                a_term = str(flow.get('from') or '')
                b_term = str(flow.get('to') or '')
                flow['from_context'] = (
                    _snippet_from_tokens(
                        raw_doc_tokens[i],
                        a_term,
                    )
                    if a_term
                    else None
                )
                flow['to_context'] = (
                    _snippet_from_tokens(
                        raw_doc_tokens[j],
                        b_term,
                    )
                    if b_term
                    else None
                )

    def _term_source_label(term: str) -> str:
        if term == unk:
            return 'unk'
        in_lex = term in lexicon_terms
        in_vad = term in word_vad_terms
        in_imp = term in imputed_terms
        if in_lex and in_vad:
            return (
                'lexicon+vad_imputed'
                if in_imp
                else 'lexicon+vad'
            )
        if in_lex:
            return 'lexicon'
        if in_vad:
            return 'vad_imputed' if in_imp else 'vad'
        return 'other'

    include_term_meta = bool(term_vad)
    if include_term_meta:
        Vi_cpu = Vi.detach().cpu().to(dtype=torch.float)
        Vj_cpu = Vj.detach().cpu().to(dtype=torch.float)

        def _salience_list(V_cpu: torch.Tensor) -> List[float]:
            if not V_cpu.numel():
                return []
            val_abs = torch.abs(V_cpu[:, 0]).clamp(
                0.0,
                1.0,
            )
            aro_pos = V_cpu[:, 1].clamp(
                0.0,
                1.0,
            )
            return [
                float(x)
                for x in torch.maximum(
                    val_abs,
                    aro_pos,
                ).tolist()
            ]

        sources_i = [_term_source_label(t) for t in terms_i]
        sources_j = [_term_source_label(t) for t in terms_j]
    return {
        'settings': {
            'mode': mode,
            'focus': focus,
            'cost': cost,
            'embed_model': (
                embed_model_name if need_embed else None
            ),
            'embed_backend': (
                embed_backend_mode if need_embed else None
            ),
            'embed_pooling': (
                embed_pooling_mode if need_embed else None
            ),
            'embed_batch_size': (
                int(embed_bs) if need_embed else None
            ),
            'embed_max_length': (
                int(embed_max_len) if need_embed else None
            ),
            'embed_prompt_mode': (
                str(embed_prompt_mode)
                if need_embed
                else None
            ),
            'embed_prompt_text': (
                str(embed_prompt_text)
                if need_embed and embed_prompt_text
                else None
            ),
            'alpha_embed': float(alpha_embed),
            'beta_vad': float(beta_vad),
            'vad_threshold': float(vad_threshold),
            'emotional_vocab': (
                str(emotional_vocab)
                if focus != 'all'
                else None
            ),
            'vad_min_arousal_vad_only': (
                float(vad_min_arousal_vad_only)
                if focus != 'all'
                else None
            ),
            'max_ngram': int(max_ngram_eff),
            'weight': weight,
            'term_weight_path': (
                str(term_weights_path)
                if term_weights_path
                else None
            ),
            'term_weight_meta': term_weights_meta,
            'term_weight_power': float(term_weight_power),
            'term_weight_min': float(term_weight_min),
            'term_weight_max': float(term_weight_max),
            'term_weight_mix': float(term_weight_mix),
            'term_weight_default': float(term_weight_default),
            'vad_imputed_path': (
                str(vad_imputed_path)
                if vad_imputed_path
                else None
            ),
            'epsilon': float(epsilon),
            'iters': int(iters),
            'reg_m': float(reg_m),
            'max_terms': int(max_terms),
            'min_token_len': int(min_token_len),
            'stopwords': bool(stopwords),
            'stopwords_file': (
                str(stopwords_file)
                if stopwords_file is not None
                else None
            ),
            'negation_window': int(neg_window),
            'negators': list(negator_terms),
            'drop_top_df': int(k_drop),
            'dropped_top_df_terms': list(dropped_top_df_terms),
        },
        'i': {
            'index': i,
            'meta': metas[i],
            'n_raw_tokens': int(raw_token_counts[i]),
            'n_selected_tokens': int(selected_token_counts[i]),
            'n_selected_unique': int(selected_unique_counts[i]),
            'selected_ratio': float(float(selected_token_counts[i])
                / max(
                    1.0,
                    float(raw_token_counts[i]),
                )),
            'used_unk_fallback': bool(selected_token_counts[i] == 0),
            'no_emotional_terms': bool(selected_token_counts[i] == 0),
            'terms': terms_i,
            'weights': [float(x) for x in weights_i],
            'term_weight': (
                [float(x) for x in term_weight_i]
                if term_weight_i
                else None
            ),
            'term_vad': (
                Vi_cpu.tolist()
                if include_term_meta
                else None
            ),
            'term_salience': (
                _salience_list(Vi_cpu)
                if include_term_meta
                else None
            ),
            'term_source': (
                sources_i if include_term_meta else None
            ),
            'term_vad_conf': (
                [
                    float(term_vad_conf.get(
                            t,
                            0.0,
                        ))
                    for t in terms_i
                ]
                if include_term_meta and term_vad_conf
                else None
            ),
        },
        'j': {
            'index': j,
            'meta': metas[j],
            'n_raw_tokens': int(raw_token_counts[j]),
            'n_selected_tokens': int(selected_token_counts[j]),
            'n_selected_unique': int(selected_unique_counts[j]),
            'selected_ratio': float(float(selected_token_counts[j])
                / max(
                    1.0,
                    float(raw_token_counts[j]),
                )),
            'used_unk_fallback': bool(selected_token_counts[j] == 0),
            'no_emotional_terms': bool(selected_token_counts[j] == 0),
            'terms': terms_j,
            'weights': [float(x) for x in weights_j],
            'term_weight': (
                [float(x) for x in term_weight_j]
                if term_weight_j
                else None
            ),
            'term_vad': (
                Vj_cpu.tolist()
                if include_term_meta
                else None
            ),
            'term_salience': (
                _salience_list(Vj_cpu)
                if include_term_meta
                else None
            ),
            'term_source': (
                sources_j if include_term_meta else None
            ),
            'term_vad_conf': (
                [
                    float(term_vad_conf.get(
                            t,
                            0.0,
                        ))
                    for t in terms_j
                ]
                if include_term_meta and term_vad_conf
                else None
            ),
        },
        'primary': explain,
    }
