from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import torch


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(
            path,
            map_location='cpu',
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location='cpu',
        )


def _unpack_checkpoint_payload(payload: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any] | None]:
    if isinstance(
        payload,
        dict,
    ) and 'model' in payload:
        state = payload['model']
        setup = (
            payload.get('cfg')
            if isinstance(
                payload.get('cfg'),
                dict,
            )
            else None
        )
    elif isinstance(
        payload,
        dict,
    ):
        state = payload
        setup = None
    else:
        raise ValueError('error: ValueError')
    if not isinstance(
        state,
        dict,
    ) or not state:
        raise ValueError('error: ValueError')
    return (state, setup)


def _best_prefix_strip(
    state: Dict[str, torch.Tensor],
    encoder_keys: set[str],
) -> Tuple[str | None, Dict[str, torch.Tensor]]:
    best_prefix: str | None = None
    best_state: Dict[str, torch.Tensor] = state
    best_hits = sum((1 for k in state.keys() if k in encoder_keys))
    prefixes = {
        k.split(
            '.',
            1,
        )[0] for k in state.keys() if '.' in k
    }
    for p in prefixes:
        stripped = {
            k[len(p) + 1 :]: v
            for k, v in state.items()
            if k.startswith(p + '.')
        }
        hits = sum((
                1
                for k in stripped.keys()
                if k in encoder_keys
            ))
        if hits > best_hits:
            best_hits = hits
            best_prefix = p
            best_state = stripped
    return (best_prefix, best_state)


def init_encoder_from_sentiment_checkpoint(
    encoder: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_model_name: str | None = None,
) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError('error: FileNotFoundError')
    payload = _torch_load(str(ckpt))
    (
        state,
        saved_cfg,
    ) = _unpack_checkpoint_payload(payload)
    saved_name: str | None = None
    if saved_cfg is not None:
        maybe_name = saved_cfg.get('model_name')
        if isinstance(
            maybe_name,
            str,
        ) and maybe_name:
            saved_name = maybe_name
    expected_name = (
        expected_model_name
        if isinstance(
            expected_model_name,
            str,
        )
        and expected_model_name
        else None
    )
    model_name_match = (
        None
        if saved_name is None or expected_name is None
        else saved_name == expected_name
    )
    encoder_sd = encoder.state_dict()
    encoder_keys = set(encoder_sd.keys())
    (
        prefix,
        stripped,
    ) = _best_prefix_strip(
        state,
        encoder_keys,
    )
    filtered = {
        k: v for k, v in stripped.items() if k in encoder_sd
    }
    coverage = float(len(filtered)) / float(max(
            1,
            len(encoder_sd),
        ))
    if model_name_match is False and coverage < 0.5:
        raise ValueError('error: ValueError')
    (
        missing,
        unexpected,
    ) = encoder.load_state_dict(
        filtered,
        strict=False,
    )
    return {
        'checkpoint': str(ckpt),
        'saved_model_name': saved_name,
        'expected_model_name': expected_name,
        'model_name_match': model_name_match,
        'prefix': prefix,
        'loaded_keys': int(len(filtered)),
        'coverage': float(coverage),
        'missing_keys': int(len(missing)),
        'unexpected_keys': int(len(unexpected)),
    }
