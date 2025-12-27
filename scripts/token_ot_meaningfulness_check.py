#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(_ROOT),
)
from legal_emotion.corpus import iter_text_paths, read_text
from legal_emotion.lexicon import tokenize as lex_tokenize
from legal_emotion.token_compare import (
    compare_token_clouds,
    explain_token_pair,
)
from legal_emotion.utils import load_config


@dataclass(frozen=True)
class SelectedDoc:
    path: Path
    category: str
    doc_id: str


LEGAL_BOILERPLATE_HINTS = {
    'obligation',
    'obligations',
    'jurisdiction',
    'jurisdictional',
    'procedure',
    'procedural',
    'proceedings',
    'application',
    'applicant',
    'respondent',
    'article',
    'paragraph',
    'annex',
    'statute',
    'convention',
    'treaty',
    'agreement',
    'charter',
    'court',
    'tribunal',
    'judge',
    'judgment',
    'order',
    'resolution',
    'security council',
    'general assembly',
    'international',
    'states',
    'state',
}
EMOTION_ANCHORS = {
    'attack',
    'attacks',
    'massacre',
    'massacres',
    'genocide',
    'atrocity',
    'atrocities',
    'rape',
    'rapes',
    'torture',
    'murder',
    'murders',
    'kill',
    'killed',
    'killing',
    'violence',
    'suffer',
    'suffering',
    'harm',
    'threat',
    'threats',
    'danger',
    'terror',
    'terrorism',
    'terrorist',
    'fear',
    'condemn',
    'regret',
    'grieve',
    'lament',
    'outrage',
    'humiliation',
}


def _default_categories() -> Dict[str, List[str]]:
    return {
        'genocide': [
            'ApplicationGenocideConvention',
            'Genocide',
        ],
        'armed_conflict': [
            'ArmedActivities',
            'UseOfForce',
            'Military',
            'Hostilities',
        ],
        'terrorism': [
            'TerrorismFinancing',
            'Terrorism',
            'Terrorist',
        ],
        'nuclear': [
            'NuclearTests',
            'Nuclear',
            'Radioactive',
        ],
        'maritime': [
            'MaritimeDelimitation',
            'ContinentalShelf',
            'Fisheries',
            'Maritime',
        ],
        'procedural': [
            'PreliminaryObjections',
            'ProvisionalMeasures',
            'Jurisdiction',
            'Revision',
            'Interpretation',
        ],
    }


def _matches_any(
    stem: str,
    needles: Sequence[str],
) -> bool:
    s = stem.lower()
    return any((n.lower() in s for n in needles))


def _select_subset(
    all_paths: Sequence[Path],
    *,
    categories: Dict[str, List[str]],
    n_per_category: int,
    random_n: int,
    seed: int,
) -> List[SelectedDoc]:
    rng = random.Random(int(seed))
    chosen: List[SelectedDoc] = []
    used: set[Path] = set()
    paths_sorted = sorted(
        all_paths,
        key=lambda p: p.name,
    )
    for cat, needles in categories.items():
        matches = [
            p
            for p in paths_sorted
            if _matches_any(
                p.stem,
                needles,
            )
        ]
        if not matches:
            continue
        rng.shuffle(matches)
        for p in matches[: max(
            0,
            int(n_per_category),
        )]:
            if p in used:
                continue
            used.add(p)
            chosen.append(SelectedDoc(
                    path=p,
                    category=cat,
                    doc_id=p.stem,
                ))
    remaining = [p for p in paths_sorted if p not in used]
    rng.shuffle(remaining)
    for p in remaining[: max(
        0,
        int(random_n),
    )]:
        used.add(p)
        chosen.append(SelectedDoc(
                path=p,
                category='random',
                doc_id=p.stem,
            ))
    chosen.sort(key=lambda d: (d.category, d.path.name))
    return chosen


def _write_jsonl(
    path: Path,
    rows: Iterable[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        'w',
        encoding='utf-8',
    ) as f:
        for r in rows:
            f.write(json.dumps(
                    r,
                    ensure_ascii=False,
                ) + '\n')


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


def _read_selected_jsonl(path: Path) -> List[SelectedDoc]:
    selected: List[SelectedDoc] = []
    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError('error: ValueError') from e
            if not isinstance(
                row,
                dict,
            ):
                continue
            meta = (
                row.get('meta')
                if isinstance(
                    row.get('meta'),
                    dict,
                )
                else {}
            )
            p_raw = meta.get('path') or row.get('path')
            cat_raw = meta.get('category') or row.get('category')
            id_raw = meta.get('id') or row.get('id')
            if (
                not isinstance(
                    p_raw,
                    str,
                )
                or not p_raw.strip()
            ):
                raise ValueError('error: ValueError')
            p = Path(p_raw)
            if not p.exists():
                raise FileNotFoundError('error: FileNotFoundError')
            cat = (
                str(cat_raw).strip()
                if cat_raw is not None
                else ''
            )
            if not cat:
                cat = 'unknown'
            doc_id = (
                str(id_raw).strip()
                if isinstance(
                    id_raw,
                    str,
                )
                and id_raw.strip()
                else p.stem
            )
            selected.append(SelectedDoc(
                    path=p,
                    category=cat,
                    doc_id=doc_id,
                ))
    selected.sort(key=lambda d: (d.category, d.path.name))
    return selected


def _load_stopword_terms(path: Optional[str | Path]) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
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
            toks = lex_tokenize(line)
            if not toks:
                continue
            out.add(' '.join(toks))
    return out


def _anchor_hit(term: str) -> bool:
    parts = term.split()
    for w in parts:
        if w in EMOTION_ANCHORS:
            return True
        if (
            len(w) > 3
            and w.endswith('s')
            and (w[:-1] in EMOTION_ANCHORS)
        ):
            return True
    return False


def _assess(
    *,
    payload: dict,
    ratios: List[float],
    top1_acc: Optional[float],
    top1_denom: int,
    intra: List[float],
    inter: List[float],
    explained_flows: List[dict],
    stopword_terms: set[str],
) -> dict:
    coverage_total = float(payload.get('coverage_total') or 0.0)
    docs_with_zero = int(payload.get('docs_with_zero_selected') or 0)
    median_ratio = (
        float(statistics.median(ratios)) if ratios else 0.0
    )
    endpoints: List[str] = []
    for flow in explained_flows:
        a = str(flow.get('from') or '')
        b = str(flow.get('to') or '')
        if a:
            endpoints.append(a)
        if b:
            endpoints.append(b)
    boilerplate_set = set(stopword_terms) | set(LEGAL_BOILERPLATE_HINTS)
    boilerplate_hits = sum((1 for t in endpoints if t in boilerplate_set))
    anchor_hits = sum((1 for t in endpoints if _anchor_hit(t)))
    boilerplate_rate = float(boilerplate_hits / max(
            1,
            len(endpoints),
        ))
    anchor_rate = float(anchor_hits / max(
            1,
            len(endpoints),
        ))
    sep_effect = None
    sep_delta = None
    if intra and inter:
        mu_intra = float(statistics.mean(intra))
        mu_inter = float(statistics.mean(inter))
        sep_delta = mu_inter - mu_intra
        sd_intra = (
            float(statistics.pstdev(intra))
            if len(intra) > 1
            else 0.0
        )
        sd_inter = (
            float(statistics.pstdev(inter))
            if len(inter) > 1
            else 0.0
        )
        pooled = (
            0.5
            * (sd_intra * sd_intra + sd_inter * sd_inter)
        ) ** 0.5
        sep_effect = (
            float(sep_delta / pooled)
            if pooled > 1e-12
            else None
        )
    target_cov = 0.05
    coverage_score = max(
        0.0,
        min(
            35.0,
            35.0 * coverage_total / target_cov,
        ),
    )
    if docs_with_zero > 0:
        coverage_score = max(
            0.0,
            coverage_score - 10.0,
        )
    target_top1 = 0.6
    if top1_acc is None:
        top1_score = 10.0
    else:
        top1_score = max(
            0.0,
            min(
                20.0,
                20.0 * float(top1_acc) / target_top1,
            ),
        )
    if sep_effect is None:
        separation_score = 12.0
    else:
        separation_score = max(
            0.0,
            min(
                25.0,
                25.0 * max(
                    0.0,
                    float(sep_effect),
                ) / 0.5,
            ),
        )
    explanation_score = 20.0 * (
        0.5 * anchor_rate + 0.5 * (1.0 - boilerplate_rate)
    )
    explanation_score = max(
        0.0,
        min(
            20.0,
            explanation_score,
        ),
    )
    score = float(coverage_score
        + top1_score
        + separation_score
        + explanation_score)
    reasons: List[str] = []
    recs: List[str] = []
    if docs_with_zero > 0:
        reasons.append(f'{docs_with_zero} docs had 0 selected emotional terms (UNK fallback).')
        recs.append('Lower `--vad_threshold`, reduce `--drop_top_df`, or relax `--stopwords_file` so each doc has some selected emotional terms.')
    if coverage_total < 0.02 or median_ratio < 0.02:
        reasons.append(f'Low coverage (coverage_total={coverage_total:.3f}, median selected_ratio={median_ratio:.3f}).')
        recs.append('Lower `--vad_threshold` (e.g. by 0.05-0.15) and/or lower `--drop_top_df`; consider increasing `--max_terms`.')
    if (
        intra
        and inter
        and (sep_delta is not None)
        and (sep_delta <= 0)
    ):
        reasons.append('No separation: intra-category distances are not smaller than inter-category distances.')
        recs.append('Increase `--alpha_embed` (more semantic), verify embedding model, and expand/adjust `data/stopwords_legal_en_token_ot.txt` to remove legal boilerplate.')
    if top1_acc is None:
        reasons.append('Nearest-neighbour category coherence not computed (need at least 2 docs per category).')
        recs.append('Increase `--n_per_category` (e.g. >=2) so top-1 coherence can be assessed.')
    elif top1_acc < 0.3:
        reasons.append(f'Low nearest-neighbour category coherence (top1={top1_acc:.2f}).')
        recs.append('Raise `--vad_threshold` and/or increase `--drop_top_df` to reduce boilerplate; also consider `--mode unbalanced_divergence` with `--weight tf`.')
    if boilerplate_rate > 0.35:
        reasons.append(f'Top transport flows still include lots of boilerplate (boilerplate_rate={boilerplate_rate:.2f}).')
        recs.append('Add the repeated non-emotional terms you see in the flows to `data/stopwords_legal_en_token_ot.txt` and re-run; or raise `--drop_top_df`.')
    if anchor_rate < 0.15:
        reasons.append(f'Few affective/violence anchor terms in top flows (anchor_rate={anchor_rate:.2f}).')
        recs.append('Inspect `top_transport_cost_contrib` in the report; if it's abstract/legal, raise `--vad_threshold` and expand stopwords; if it's too sparse, lower `--vad_threshold`.')
    if coverage_total < 0.01 or score < 40.0:
        verdict = 'fail'
    elif (
        coverage_total < 0.03
        or (top1_acc is not None and top1_acc < 0.4)
        or boilerplate_rate > 0.45
        or (score < 70.0)
    ):
        verdict = 'warn'
    else:
        verdict = 'pass'
    metrics = {
        'score': score,
        'verdict': verdict,
        'coverage_total': coverage_total,
        'docs_with_zero_selected': docs_with_zero,
        'selected_ratio_median': median_ratio,
        'top1_same_category_rate': (
            float(top1_acc)
            if top1_acc is not None
            else None
        ),
        'top1_denom': int(top1_denom),
        'intra_mean': (
            float(statistics.mean(intra)) if intra else None
        ),
        'inter_mean': (
            float(statistics.mean(inter)) if inter else None
        ),
        'separation_delta': (
            float(sep_delta)
            if sep_delta is not None
            else None
        ),
        'separation_effect': (
            float(sep_effect)
            if sep_effect is not None
            else None
        ),
        'boilerplate_rate': float(boilerplate_rate),
        'anchor_rate': float(anchor_rate),
    }
    return {
        'verdict': verdict,
        'score': score,
        'metrics': metrics,
        'reasons': reasons,
        'recommendations': recs,
    }


def _is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return 'out of memory' in msg or 'cuda oom' in msg


def _empty_device_cache() -> None:
    try:
        import torch

        if (
            hasattr(
                torch,
                'cuda',
            )
            and hasattr(
                torch.cuda,
                'is_initialized',
            )
            and torch.cuda.is_initialized()
        ):
            torch.cuda.empty_cache()
        if hasattr(
            torch,
            'mps',
        ) and hasattr(
            torch.mps,
            'empty_cache',
        ):
            torch.mps.empty_cache()
    except Exception:
        return


def _compare_with_oom_retry(**kwargs):
    bs = kwargs.get('embed_batch_size')
    if bs is None:
        return (compare_token_clouds(**kwargs), bs)
    bs_i = max(
        1,
        int(bs),
    )
    while True:
        try:
            kwargs['embed_batch_size'] = int(bs_i)
            return (
                compare_token_clouds(**kwargs),
                int(bs_i),
            )
        except RuntimeError as e:
            if not _is_oom_error(e) or bs_i <= 1:
                raise
            _empty_device_cache()
            bs_i = max(
                1,
                bs_i // 2,
            )


def _explain_with_oom_retry(**kwargs):
    bs = kwargs.get('embed_batch_size')
    if bs is None:
        return (explain_token_pair(**kwargs), bs)
    bs_i = max(
        1,
        int(bs),
    )
    while True:
        try:
            kwargs['embed_batch_size'] = int(bs_i)
            return (explain_token_pair(**kwargs), int(bs_i))
        except RuntimeError as e:
            if not _is_oom_error(e) or bs_i <= 1:
                raise
            _empty_device_cache()
            bs_i = max(
                1,
                bs_i // 2,
            )


def _nearest_neighbours(D: List[List[float]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    n = len(D)
    for i in range(n):
        best_j = None
        best = float('inf')
        for j in range(n):
            if i == j:
                continue
            raw = D[i][j]
            if raw is None:
                continue
            try:
                d = float(raw)
            except Exception:
                continue
            if d < best:
                best = d
                best_j = j
        if best_j is None:
            out.append((i, float('inf')))
        else:
            out.append((best_j, float(best)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='token-ot quick check')
    ap.add_argument(
        '--input_dir',
        type=str,
        default='data/EN_TXT_BEST_FULL',
    )
    ap.add_argument(
        '--selected_jsonl',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--config',
        type=str,
        default='config.icj.gpu.yaml',
    )
    ap.add_argument(
        '--out_dir',
        type=str,
        default='outputs/token_ot_check',
    )
    ap.add_argument(
        '--n_per_category',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--random_n',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--seed',
        type=int,
        default=13,
    )
    ap.add_argument(
        '--format',
        type=str,
        default='matrix',
        choices=('matrix', 'neighbours'),
    )
    ap.add_argument(
        '--topk',
        type=int,
        default=10,
    )
    ap.add_argument(
        '--candidate_k',
        type=int,
        default=50,
    )
    ap.add_argument(
        '--mode',
        type=str,
        default='sinkhorn_divergence',
    )
    ap.add_argument(
        '--focus',
        type=str,
        default='emotional',
        choices=('all', 'emotional'),
    )
    ap.add_argument(
        '--cost',
        type=str,
        default='embedding_vad',
        choices=('embedding', 'vad', 'embedding_vad'),
    )
    ap.add_argument(
        '--embed_model',
        type=str,
        default='BAAI/bge-large-en-v1.5',
    )
    ap.add_argument(
        '--embed_backend',
        type=str,
        default='encoder',
        choices=('encoder', 'input_embeddings'),
    )
    ap.add_argument(
        '--embed_pooling',
        type=str,
        default='cls',
        choices=('cls', 'mean'),
    )
    ap.add_argument(
        '--embed_batch_size',
        type=int,
        default=32,
    )
    ap.add_argument(
        '--embed_max_length',
        type=int,
        default=32,
    )
    ap.add_argument(
        '--embed_prompt_mode',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--embed_prompt_text',
        type=str,
        default=None,
    )
    ap.add_argument(
        '--alpha_embed',
        type=float,
        default=0.8,
    )
    ap.add_argument(
        '--beta_vad',
        type=float,
        default=0.2,
    )
    ap.add_argument(
        '--vad_threshold',
        type=float,
        default=0.55,
    )
    ap.add_argument(
        '--emotional_vocab',
        type=str,
        default='lexicon_or_vad',
        choices=('lexicon', 'vad', 'lexicon_or_vad'),
    )
    ap.add_argument(
        '--vad_min_arousal_vad_only',
        type=float,
        default=0.35,
    )
    ap.add_argument(
        '--max_ngram',
        type=int,
        default=3,
    )
    ap.add_argument(
        '--weight',
        type=str,
        default='tfidf',
    )
    ap.add_argument(
        '--max_terms',
        type=int,
        default=128,
    )
    ap.add_argument(
        '--min_token_len',
        type=int,
        default=2,
    )
    ap.add_argument(
        '--no_stopwords',
        action='store_true',
    )
    ap.add_argument(
        '--stopwords_file',
        type=str,
        default='data/stopwords_legal_en_token_ot.txt',
    )
    ap.add_argument(
        '--drop_top_df',
        type=int,
        default=100,
    )
    ap.add_argument(
        '--top_flows',
        type=int,
        default=12,
    )
    ap.add_argument(
        '--explain_pairs',
        type=int,
        default=6,
    )
    ap.add_argument(
        '--explain_mode',
        type=str,
        default='closest',
        choices=('closest', 'per_doc'),
    )
    ap.add_argument(
        '--strict',
        action='store_true',
    )
    ap.add_argument(
        '--quiet',
        action='store_true',
    )
    opts = ap.parse_args()

    def _arg_provided(flag: str) -> bool:
        for a in sys.argv[1:]:
            if a == flag or a.startswith(flag + '='):
                return True
        return False

    def log(msg: str) -> None:
        if bool(opts.quiet):
            return
        print(
            msg,
            file=sys.stderr,
            flush=True,
        )

    setup = load_config(str(opts.config))
    if not _arg_provided('--embed_model'):
        embed_model_cfg = getattr(
            setup,
            'token_ot_embed_model',
            None,
        )
        if embed_model_cfg:
            opts.embed_model = str(embed_model_cfg)
    if not _arg_provided('--embed_backend'):
        embed_backend_cfg = getattr(
            setup,
            'token_ot_embed_backend',
            None,
        )
        if embed_backend_cfg:
            opts.embed_backend = str(embed_backend_cfg)
    if not _arg_provided('--embed_pooling'):
        embed_pooling_cfg = getattr(
            setup,
            'token_ot_embed_pooling',
            None,
        )
        if embed_pooling_cfg:
            opts.embed_pooling = str(embed_pooling_cfg)
    if not _arg_provided('--embed_batch_size'):
        embed_bs_cfg = getattr(
            setup,
            'token_ot_embed_batch_size',
            None,
        )
        if embed_bs_cfg:
            opts.embed_batch_size = int(embed_bs_cfg)
    if not _arg_provided('--embed_max_length'):
        embed_len_cfg = getattr(
            setup,
            'token_ot_embed_max_length',
            None,
        )
        if embed_len_cfg:
            opts.embed_max_length = int(embed_len_cfg)
    if not _arg_provided('--embed_prompt_mode'):
        embed_prompt_cfg = getattr(
            setup,
            'token_ot_embed_prompt_mode',
            None,
        )
        if embed_prompt_cfg:
            opts.embed_prompt_mode = str(embed_prompt_cfg)
    if not _arg_provided('--embed_prompt_text'):
        embed_prompt_text_cfg = getattr(
            setup,
            'token_ot_embed_prompt_text',
            None,
        )
        if embed_prompt_text_cfg:
            opts.embed_prompt_text = str(embed_prompt_text_cfg)
    in_dir = Path(opts.input_dir)
    out_dir = Path(opts.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    if opts.selected_jsonl:
        sel_path = Path(opts.selected_jsonl)
        log(f'selected {sel_path}')
        selected = _read_selected_jsonl(sel_path)
    else:
        log(f'input {in_dir}')
        all_paths = list(iter_text_paths(
                in_dir,
                recursive=True,
            ))
        if not all_paths:
            raise SystemExit('error: SystemExit')
        log(f'found {len(all_paths)} files')
        categories = _default_categories()
        selected = _select_subset(
            all_paths,
            categories=categories,
            n_per_category=opts.n_per_category,
            random_n=opts.random_n,
            seed=opts.seed,
        )
    if len(selected) < 2:
        raise SystemExit('error: SystemExit')
    selected_counts = Counter((d.category for d in selected))
    log(f'selected {len(selected)}')
    selected_jsonl = out_dir / 'selected_docs.jsonl'
    log(f'write selected {selected_jsonl}')
    _write_jsonl(
        selected_jsonl,
        [
            {
                'meta': {
                    'id': d.doc_id,
                    'category': d.category,
                    'path': str(d.path),
                }
            }
            for d in selected
        ],
    )
    matrix_path = out_dir / 'matrix.json'
    fmt = str(opts.format or 'matrix').strip().lower()
    if fmt == 'matrix':
        log('build matrix')
        (
            compare_stats,
            used_embed_bs,
        ) = (
            _compare_with_oom_retry(
                input_jsonl=str(selected_jsonl),
                output_path=str(matrix_path),
                cfg_path=str(opts.config),
                fmt='matrix',
                mode=opts.mode,
                focus=opts.focus,
                cost=opts.cost,
                embed_model=(
                    str(opts.embed_model)
                    if opts.embed_model
                    else None
                ),
                embed_backend=(
                    str(opts.embed_backend)
                    if opts.embed_backend
                    else None
                ),
                embed_pooling=(
                    str(opts.embed_pooling)
                    if opts.embed_pooling
                    else None
                ),
                embed_batch_size=(
                    int(opts.embed_batch_size)
                    if opts.embed_batch_size
                    else None
                ),
                embed_max_length=(
                    int(opts.embed_max_length)
                    if opts.embed_max_length
                    else None
                ),
                embed_prompt_mode=(
                    str(opts.embed_prompt_mode)
                    if opts.embed_prompt_mode
                    else None
                ),
                embed_prompt_text=(
                    str(opts.embed_prompt_text)
                    if opts.embed_prompt_text
                    else None
                ),
                alpha_embed=float(opts.alpha_embed),
                beta_vad=float(opts.beta_vad),
                vad_threshold=float(opts.vad_threshold),
                emotional_vocab=str(opts.emotional_vocab),
                vad_min_arousal_vad_only=float(opts.vad_min_arousal_vad_only),
                max_ngram=int(opts.max_ngram),
                weight=str(opts.weight),
                max_terms=int(opts.max_terms),
                min_token_len=int(opts.min_token_len),
                stopwords=not bool(opts.no_stopwords),
                stopwords_file=(
                    str(opts.stopwords_file)
                    if opts.stopwords_file
                    else None
                ),
                drop_top_df=int(opts.drop_top_df),
                include_explain=False,
            )
        )
        log('matrix done')
        payload = json.loads(matrix_path.read_text(encoding='utf-8'))
        docs = payload['docs']
        D = payload['distance']
        idx_to_meta = {
            int(d['index']): d.get(
                'meta',
                {},
            ) for d in docs
        }
        idx_to_cat = {
            i: idx_to_meta[i].get('category') or 'unknown'
            for i in range(len(docs))
        }
        idx_to_id = {
            i: idx_to_meta[i].get('id') or f'doc_{i}'
            for i in range(len(docs))
        }
        ratios = [
            float(d.get(
                    'selected_ratio',
                    0.0,
                ))
            for d in docs
        ]
        nn = _nearest_neighbours(D)
        intra: List[float] = []
        inter: List[float] = []
        n = len(D)
        for i in range(n):
            for j in range(
                i + 1,
                n,
            ):
                ci = idx_to_cat.get(
                    i,
                    'unknown',
                )
                cj = idx_to_cat.get(
                    j,
                    'unknown',
                )
                if 'random' in {ci, cj}:
                    continue
                raw = D[i][j]
                if raw is None:
                    continue
                try:
                    dist = float(raw)
                except Exception:
                    continue
                (intra if ci == cj else inter).append(dist)
        out_target = matrix_path
        payload_for_assess = payload
    else:
        neighbours_path = out_dir / 'neighbours.jsonl'
        log('build neighbours')
        (
            compare_stats,
            used_embed_bs,
        ) = (
            _compare_with_oom_retry(
                input_jsonl=str(selected_jsonl),
                output_path=str(neighbours_path),
                cfg_path=str(opts.config),
                fmt='neighbours',
                topk=int(opts.topk),
                candidate_k=int(opts.candidate_k),
                mode=opts.mode,
                focus=opts.focus,
                cost=opts.cost,
                embed_model=(
                    str(opts.embed_model)
                    if opts.embed_model
                    else None
                ),
                embed_backend=(
                    str(opts.embed_backend)
                    if opts.embed_backend
                    else None
                ),
                embed_pooling=(
                    str(opts.embed_pooling)
                    if opts.embed_pooling
                    else None
                ),
                embed_batch_size=(
                    int(opts.embed_batch_size)
                    if opts.embed_batch_size
                    else None
                ),
                embed_max_length=(
                    int(opts.embed_max_length)
                    if opts.embed_max_length
                    else None
                ),
                embed_prompt_mode=(
                    str(opts.embed_prompt_mode)
                    if opts.embed_prompt_mode
                    else None
                ),
                embed_prompt_text=(
                    str(opts.embed_prompt_text)
                    if opts.embed_prompt_text
                    else None
                ),
                alpha_embed=float(opts.alpha_embed),
                beta_vad=float(opts.beta_vad),
                vad_threshold=float(opts.vad_threshold),
                emotional_vocab=str(opts.emotional_vocab),
                vad_min_arousal_vad_only=float(opts.vad_min_arousal_vad_only),
                max_ngram=int(opts.max_ngram),
                weight=str(opts.weight),
                max_terms=int(opts.max_terms),
                min_token_len=int(opts.min_token_len),
                stopwords=not bool(opts.no_stopwords),
                stopwords_file=(
                    str(opts.stopwords_file)
                    if opts.stopwords_file
                    else None
                ),
                drop_top_df=int(opts.drop_top_df),
                include_explain=False,
            )
        )
        log('neighbours done')
        docs_by_idx: List[dict] = []
        by_idx: Dict[int, dict] = {}
        with neighbours_path.open(
            'r',
            encoding='utf-8',
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(
                    row,
                    dict,
                ):
                    continue
                idx = row.get('index')
                if isinstance(
                    idx,
                    int,
                ):
                    by_idx[int(idx)] = row
                    docs_by_idx.append(row)
        n_docs = max(
            by_idx.keys(),
            default=-1,
        ) + 1
        docs = [
            by_idx.get(
                i,
                {
                    'index': i,
                    'meta': {},
                    'neighbours': [],
                    'selected_ratio': 0.0,
                },
            )
            for i in range(n_docs)
        ]
        idx_to_meta = {
            int(d.get('index') or 0): (
                d.get('meta')
                if isinstance(
                    d.get('meta'),
                    dict,
                )
                else {}
            )
            for d in docs
        }
        idx_to_cat = {
            i: idx_to_meta[i].get('category') or 'unknown'
            for i in range(len(docs))
        }
        idx_to_id = {
            i: idx_to_meta[i].get('id') or f'doc_{i}'
            for i in range(len(docs))
        }
        ratios = [
            float(d.get(
                    'selected_ratio',
                    0.0,
                ) or 0.0)
            for d in docs
        ]
        nn: List[Tuple[int, float]] = []
        intra = []
        inter = []
        for i, d in enumerate(docs):
            neigh = d.get('neighbours')
            if isinstance(
                neigh,
                list,
            ) and neigh:
                first = (
                    neigh[0]
                    if isinstance(
                        neigh[0],
                        dict,
                    )
                    else {}
                )
                j = (
                    int(first.get('index'))
                    if isinstance(
                        first.get('index'),
                        int,
                    )
                    else int(i)
                )
                dist_raw = first.get('distance')
                dist = (
                    float(dist_raw)
                    if isinstance(
                        dist_raw,
                        (int, float),
                    )
                    else float('inf')
                )
                nn.append((j, dist))
            else:
                nn.append((int(i), float('inf')))
            ci = idx_to_cat.get(
                i,
                'unknown',
            )
            if ci == 'random':
                continue
            if isinstance(
                neigh,
                list,
            ):
                for nb in neigh:
                    if not isinstance(
                        nb,
                        dict,
                    ):
                        continue
                    j2 = nb.get('index')
                    if not isinstance(
                        j2,
                        int,
                    ):
                        continue
                    cj = idx_to_cat.get(
                        int(j2),
                        'unknown',
                    )
                    if cj == 'random':
                        continue
                    dist_raw2 = nb.get('distance')
                    dist2 = (
                        float(dist_raw2)
                        if isinstance(
                            dist_raw2,
                            (int, float),
                        )
                        else float('inf')
                    )
                    (intra if ci == cj else inter).append(dist2)
        out_target = neighbours_path
        payload_for_assess = {
            'coverage_total': float(compare_stats.get('coverage_total') or 0.0),
            'docs_with_zero_selected': int(compare_stats.get('docs_with_zero_selected')
                or 0),
        }

    def cat_match(
        i: int,
        j: int,
    ) -> bool:
        a = idx_to_cat.get(
            i,
            'unknown',
        )
        b = idx_to_cat.get(
            j,
            'unknown',
        )
        if 'random' in {a, b}:
            return False
        return a == b

    matches = [
        bool(cat_match(
                i,
                j,
            )
            and j != i
            and (dist != float('inf')))
        for i, (j, dist) in enumerate(nn)
    ]
    cat_counts = Counter((c for c in idx_to_cat.values() if c != 'random'))
    eligible = [
        i
        for i in range(len(docs))
        if idx_to_cat.get(i) != 'random'
        and nn[i][0] != i
        and (nn[i][1] != float('inf'))
        and (
            cat_counts.get(
                idx_to_cat.get(
                    i,
                    'unknown',
                ),
                0,
            )
            >= 2
        )
    ]
    top1_denom = int(len(eligible))
    top1_acc = (
        float(sum((1 for i in eligible if matches[i]))
            / float(top1_denom))
        if top1_denom > 0
        else None
    )
    report_path = out_dir / 'report.md'
    summary_path = out_dir / 'summary.json'
    log(f'write report {report_path}')
    with report_path.open(
        'w',
        encoding='utf-8',
    ) as f:
        f.write('# Token-cloud OT meaningfulness check\n\n')
        f.write(f'- Selected docs: {len(docs)} (written to `{selected_jsonl}`)\n')
        f.write(f'- Output ({fmt}): `{out_target}`\n')
        f.write(f'- Config: `{opts.config}`\n\n')
        if used_embed_bs is not None:
            f.write(f'- embed_batch_size used: {int(used_embed_bs)}\n\n')
        f.write('## Coverage\n')
        f.write(f'- coverage_total: {payload_for_assess.get('coverage_total'):.4f}\n')
        f.write(f'- docs_with_zero_selected: {payload_for_assess.get('docs_with_zero_selected')}\n')
        f.write(f'- selected_ratio min/median/max: {min(ratios):.4f} / {statistics.median(ratios):.4f} / {max(ratios):.4f}\n\n')
        if intra and inter:
            f.write('## Separation\n')
            f.write(f'- intra-category mean distance: {statistics.mean(intra):.4f} (n={len(intra)})\n')
            f.write(f'- inter-category mean distance: {statistics.mean(inter):.4f} (n={len(inter)})\n\n')
        f.write('## Nearest-neighbour category check\n')
        if top1_acc is None:
            f.write('- top-1 same-category rate: n/a (need at least 2 docs per non-random category)\n\n')
        else:
            f.write(f"- top-1 same-category rate (excluding 'random', categories with <2 docs): {top1_acc:.3f} (denom={top1_denom})\n\n")
        f.write('| idx | category | id | selected_ratio | nn_idx | nn_category | nn_id | dist |\n')
        f.write('|---:|---|---|---:|---:|---|---|---:|\n')
        for i, (j, dist) in enumerate(nn):
            f.write(f'| {i} | {idx_to_cat.get(i)} | {idx_to_id.get(i)} | {ratios[i]:.4f} | {j} | {idx_to_cat.get(j)} | {idx_to_id.get(j)} | {dist:.4f} |\n')
        f.write('\n')
        f.write('## Pair explanations (nearest neighbours)\n')
        k = min(
            int(opts.explain_pairs),
            len(docs),
        )
        explain_mode = (
            str(opts.explain_mode or 'closest')
            .strip()
            .lower()
        )
        if explain_mode == 'per_doc':
            pairs = [
                (i, nn[i][0])
                for i in range(len(docs))
                if nn[i][0] != i
                and nn[i][1] != float('inf')
            ]
            pairs = pairs[:k]
        else:
            pairs = [
                (i, nn[i][0])
                for i in range(len(docs))
                if nn[i][0] != i
                and nn[i][1] != float('inf')
            ]
            pairs.sort(key=lambda ij: float(nn[ij[0]][1]))
            pairs = pairs[:k]
        explained_flow_terms: List[dict] = []
        for i, j in pairs:
            log(f'explain {i} {j}')
            f.write(f'### {i} ({idx_to_cat[i]}: {idx_to_id[i]}) vs {j} ({idx_to_cat[j]}: {idx_to_id[j]})\n\n')
            (
                exp,
                used_embed_bs,
            ) = _explain_with_oom_retry(
                input_jsonl=str(selected_jsonl),
                i=i,
                j=j,
                cfg_path=str(opts.config),
                mode=opts.mode,
                focus=opts.focus,
                cost=opts.cost,
                embed_model=(
                    str(opts.embed_model)
                    if opts.embed_model
                    else None
                ),
                embed_backend=(
                    str(opts.embed_backend)
                    if opts.embed_backend
                    else None
                ),
                embed_pooling=(
                    str(opts.embed_pooling)
                    if opts.embed_pooling
                    else None
                ),
                embed_batch_size=(
                    int(used_embed_bs
                        if used_embed_bs is not None
                        else opts.embed_batch_size)
                    if used_embed_bs
                    or opts.embed_batch_size
                    else None
                ),
                embed_max_length=(
                    int(opts.embed_max_length)
                    if opts.embed_max_length
                    else None
                ),
                embed_prompt_mode=(
                    str(opts.embed_prompt_mode)
                    if opts.embed_prompt_mode
                    else None
                ),
                embed_prompt_text=(
                    str(opts.embed_prompt_text)
                    if opts.embed_prompt_text
                    else None
                ),
                alpha_embed=float(opts.alpha_embed),
                beta_vad=float(opts.beta_vad),
                vad_threshold=float(opts.vad_threshold),
                emotional_vocab=str(opts.emotional_vocab),
                vad_min_arousal_vad_only=float(opts.vad_min_arousal_vad_only),
                max_ngram=int(opts.max_ngram),
                weight=str(opts.weight),
                max_terms=int(opts.max_terms),
                min_token_len=int(opts.min_token_len),
                stopwords=not bool(opts.no_stopwords),
                stopwords_file=(
                    str(opts.stopwords_file)
                    if opts.stopwords_file
                    else None
                ),
                drop_top_df=int(opts.drop_top_df),
                top_flows=int(opts.top_flows),
            )
            prim = exp['primary']
            f.write(f"- distance: {prim['distance']:.6f}\n")
            f.write('- top terms i: '
                + ', '.join(exp['i']['terms'][:15])
                + '\n')
            f.write('- top terms j: '
                + ', '.join(exp['j']['terms'][:15])
                + '\n')
            f.write('\nTop transport cost contributions:\n\n')
            for flow in prim.get(
                'top_transport_cost_contrib',
                [],
            )[: int(opts.top_flows)]:
                explained_flow_terms.append(flow)
                f.write(f"- {flow['from']} -> {flow['to']} (contrib={flow['contribution']:.4f}, mass={flow['mass']:.4f}, cost={flow['cost']:.4f})\n")
            try:
                toks_i = lex_tokenize(read_text(Path(exp['i']['meta']['path'])))
                toks_j = lex_tokenize(read_text(Path(exp['j']['meta']['path'])))
                f.write('\nTokenized context snippets (first matches):\n\n')
                seen_terms: set[str] = set()
                for flow in prim.get(
                    'top_transport_cost_contrib',
                    [],
                )[:6]:
                    for side, toks, term in (
                        ('i', toks_i, flow['from']),
                        ('j', toks_j, flow['to']),
                    ):
                        if term in seen_terms:
                            continue
                        seen_terms.add(term)
                        snip = _snippet_from_tokens(
                            toks,
                            term,
                            window=18,
                        )
                        if snip:
                            f.write(f'- {side}:{term}: ... {snip} ...\n')
            except Exception:
                pass
            f.write('\n')
    stopword_terms = _load_stopword_terms(opts.stopwords_file)
    evaluation = _assess(
        payload=payload_for_assess,
        ratios=ratios,
        top1_acc=top1_acc,
        top1_denom=top1_denom,
        intra=intra,
        inter=inter,
        explained_flows=explained_flow_terms,
        stopword_terms=stopword_terms,
    )
    summary_path.write_text(
        json.dumps(
            evaluation,
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    log(f"evaluation {evaluation.get('verdict')} {evaluation.get('score')}")
    with report_path.open(
        'a',
        encoding='utf-8',
    ) as f:
        f.write('## Automatic evaluation\n')
        f.write(f"- verdict: `{evaluation['verdict']}`\n")
        f.write(f"- score: {evaluation['score']:.1f}/100\n")
        f.write(f'- summary: `{summary_path}`\n')
        if evaluation['reasons']:
            f.write('\nReasons:\n\n')
            for r in evaluation['reasons']:
                f.write(f'- {r}\n')
        if evaluation['recommendations']:
            f.write('\nRecommendations:\n\n')
            for r in evaluation['recommendations']:
                f.write(f'- {r}\n')
        f.write('\n')
    print(json.dumps(
            {
                'matrix_stats': compare_stats,
                'report': str(report_path),
                'summary': str(summary_path),
                'selected': str(selected_jsonl),
                'evaluation': evaluation,
            },
            indent=2,
            ensure_ascii=False,
        ))
    if (
        bool(opts.strict)
        and evaluation['verdict'] != 'pass'
    ):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
