import os
import random
import warnings
import numpy as np
import torch
import yaml
from .config import Config


def _coerce_to_default_type(
    default,
    value,
):
    if value is None:
        return None
    if isinstance(
        default,
        bool,
    ):
        if isinstance(
            value,
            str,
        ):
            v = value.strip().lower()
            if v in {'true', 'yes', 'y', '1'}:
                return True
            if v in {'false', 'no', 'n', '0'}:
                return False
        return bool(value)
    if isinstance(
        default,
        int,
    ) and (
        not isinstance(
            default,
            bool,
        )
    ):
        if isinstance(
            value,
            str,
        ):
            v = value.strip()
            try:
                return int(v)
            except ValueError:
                return int(float(v))
        return int(value)
    if isinstance(
        default,
        float,
    ):
        if isinstance(
            value,
            str,
        ):
            return float(value.strip())
        return float(value)
    return value


def load_config(path: str = None):
    setup = Config()
    if not path:
        return setup
    with open(path) as f:
        data = yaml.safe_load(f)
    for k, v in data.items():
        if hasattr(
            setup,
            k,
        ):
            default = getattr(
                setup,
                k,
            )
            setattr(
                setup,
                k,
                _coerce_to_default_type(
                    default,
                    v,
                ),
            )
    return setup


def _env_flag(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    if isinstance(
        raw,
        str,
    ):
        v = raw.strip().lower()
        if v in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if v in {'0', 'false', 'no', 'n', 'off'}:
            return False
    return bool(raw)


def _has_nvidia_device_nodes() -> bool:
    for p in (
        '/dev/nvidia0',
        '/dev/nvidiactl',
        '/dev/nvidia-uvm',
        '/dev/nvidia-uvm-tools',
    ):
        if os.path.exists(p):
            return True
    return False


def get_device(preference: str = 'auto') -> torch.device:
    override = os.environ.get('CLDS_DEVICE_OVERRIDE') or os.environ.get('CLDS_DEVICE')
    if override:
        preference = override
    pref = (preference or 'auto').strip().lower()
    if pref == 'gpu':
        pref = 'cuda'
    has_mps = bool(getattr(
            torch.backends,
            'mps',
            None,
        )
        and torch.backends.mps.is_available())
    cuda_built = torch.version.cuda is not None
    cuda_disabled = _env_flag(
        'CLDS_DISABLE_CUDA',
        False,
    )
    if pref == 'auto':
        if (
            not cuda_disabled
            and cuda_built
            and _has_nvidia_device_nodes()
        ):
            return torch.device('cuda')
        if has_mps:
            return torch.device('mps')
        return torch.device('cpu')
    if pref.startswith('cuda'):
        if cuda_disabled:
            warnings.warn(
                'warn: cpu',
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device('cpu')
        if not cuda_built:
            warnings.warn(
                'warn: cpu',
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device('cpu')
        return torch.device(pref)
    if pref == 'mps':
        if has_mps:
            return torch.device('mps')
        warnings.warn(
            'warn: cpu',
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device('cpu')
    return torch.device(pref)


def set_seed(
    seed: int = 13,
    device: torch.device | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (
        device is not None
        and getattr(
            device,
            'type',
            None,
        ) == 'cuda'
    ):
        if not _env_flag(
            'CLDS_SEED_CUDA',
            False,
        ):
            return
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            return
