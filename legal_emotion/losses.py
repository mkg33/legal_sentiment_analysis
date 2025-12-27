import torch
import torch.nn.functional as F

try:
    import ot as ot_base
except Exception:
    ot_base = None


def cost_matrix(
    size: int,
    device,
    emotion_vad: torch.Tensor | None = None,
):
    if emotion_vad is None:
        eye = torch.eye(
            size,
            device=device,
        )
        return torch.ones(
            size,
            size,
            device=device,
        ) - eye
    emotion_vad = emotion_vad.to(
        device=device,
        dtype=torch.float,
    )
    if emotion_vad.shape != (size, 3):
        raise ValueError('error: ValueError')
    diffs = (
        emotion_vad[:, None, :] - emotion_vad[None, :, :]
    )
    C = torch.sqrt(torch.sum(
            diffs * diffs,
            dim=-1,
        ) + 1e-12)
    mask = ~torch.eye(
        size,
        device=device,
        dtype=torch.bool,
    )
    scale = C[mask].mean().clamp_min(1e-08)
    C = C / scale
    C.fill_diagonal_(0.0)
    return C


def _normalise_simplex(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-08)
    return x / x.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-08)


def _normalise_measure(
    x: torch.Tensor,
    *,
    simplex: bool,
) -> torch.Tensor:
    if simplex:
        return _normalise_simplex(x)
    return x.clamp_min(1e-08)


def _kl_measure(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    p = p.clamp_min(1e-08)
    q = q.clamp_min(1e-08)
    return torch.sum(
        p * (p.log() - q.log()) - p + q,
        dim=-1,
    )


def _sinkhorn_plan(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float,
    iters: int,
    *,
    unbalanced: bool,
    reg_m: float,
):
    if epsilon <= 0:
        raise ValueError('error: ValueError')
    if iters <= 0:
        raise ValueError('error: ValueError')
    squeeze = False
    if a.dim() == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        squeeze = True
    loga = a.log()
    logb = b.log()
    logK = (-C / epsilon).unsqueeze(0)
    logK_t = logK.transpose(
        -2,
        -1,
    )
    logu = torch.zeros_like(a)
    logv = torch.zeros_like(b)
    tau = 1.0
    if unbalanced:
        if reg_m <= 0:
            raise ValueError('error: ValueError')
        tau = reg_m / (reg_m + epsilon)
    for _ in range(iters):
        logu = loga - torch.logsumexp(
            logK + logv.unsqueeze(1),
            dim=-1,
        )
        if unbalanced:
            logu = tau * logu
        logv = logb - torch.logsumexp(
            logK_t + logu.unsqueeze(1),
            dim=-1,
        )
        if unbalanced:
            logv = tau * logv
    logP = logu.unsqueeze(-1) + logK + logv.unsqueeze(1)
    P = logP.exp()
    if squeeze:
        P = P.squeeze(0)
    return P


def _sinkhorn_cost(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float,
    iters: int,
    *,
    unbalanced: bool,
    reg_m: float,
) -> torch.Tensor:
    P = _sinkhorn_plan(
        a,
        b,
        C,
        epsilon,
        iters,
        unbalanced=unbalanced,
        reg_m=reg_m,
    )
    if P.dim() == 2:
        transport = torch.sum(P * C)
        if not unbalanced:
            return transport
        marg_a = P.sum(dim=-1)
        marg_b = P.sum(dim=-2)
        return transport + reg_m * (
            _kl_measure(
                marg_a,
                a,
            ) + _kl_measure(
                marg_b,
                b,
            )
        )
    transport = torch.sum(
        P * C,
        dim=(-2, -1),
    )
    if not unbalanced:
        return transport
    marg_a = P.sum(dim=-1)
    marg_b = P.sum(dim=-2)
    return transport + reg_m * (
        _kl_measure(
            marg_a,
            a,
        ) + _kl_measure(
            marg_b,
            b,
        )
    )


def _sinkhorn_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float,
    iters: int,
    *,
    unbalanced: bool,
    reg_m: float,
) -> torch.Tensor:
    cost_ab = _sinkhorn_cost(
        a,
        b,
        C,
        epsilon,
        iters,
        unbalanced=unbalanced,
        reg_m=reg_m,
    )
    cost_aa = _sinkhorn_cost(
        a,
        a,
        C,
        epsilon,
        iters,
        unbalanced=unbalanced,
        reg_m=reg_m,
    )
    cost_bb = _sinkhorn_cost(
        b,
        b,
        C,
        epsilon,
        iters,
        unbalanced=unbalanced,
        reg_m=reg_m,
    )
    return cost_ab - 0.5 * cost_aa - 0.5 * cost_bb


def sinkhorn_plan(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    *,
    epsilon: float,
    iters: int,
    unbalanced: bool = False,
    reg_m: float = 0.1,
) -> torch.Tensor:
    a = _normalise_measure(
        a,
        simplex=not unbalanced,
    )
    b = _normalise_measure(
        b,
        simplex=not unbalanced,
    )
    return _sinkhorn_plan(
        a,
        b,
        C,
        float(epsilon),
        int(iters),
        unbalanced=bool(unbalanced),
        reg_m=float(reg_m),
    )


def sinkhorn_cost_parts(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    *,
    epsilon: float,
    iters: int,
    unbalanced: bool = False,
    reg_m: float = 0.1,
    return_plan: bool = False,
) -> dict:
    a_n = _normalise_measure(
        a,
        simplex=not unbalanced,
    )
    b_n = _normalise_measure(
        b,
        simplex=not unbalanced,
    )
    P = _sinkhorn_plan(
        a_n,
        b_n,
        C,
        float(epsilon),
        int(iters),
        unbalanced=bool(unbalanced),
        reg_m=float(reg_m),
    )
    if P.dim() == 2:
        transport = torch.sum(P * C)
        marg_a = P.sum(dim=-1)
        marg_b = P.sum(dim=-2)
        kl_a = torch.zeros(
            (),
            device=P.device,
            dtype=P.dtype,
        )
        kl_b = torch.zeros(
            (),
            device=P.device,
            dtype=P.dtype,
        )
        if unbalanced:
            kl_a = _kl_measure(
                marg_a,
                a_n,
            )
            kl_b = _kl_measure(
                marg_b,
                b_n,
            )
            total = transport + float(reg_m) * (kl_a + kl_b)
        else:
            total = transport
        out = {
            'total': total,
            'transport': transport,
            'kl_a': kl_a,
            'kl_b': kl_b,
            'mass_a': a_n.sum(),
            'mass_b': b_n.sum(),
            'mass_plan': P.sum(),
            'marg_a': marg_a,
            'marg_b': marg_b,
        }
        if return_plan:
            out['plan'] = P
        return out
    transport = torch.sum(
        P * C,
        dim=(-2, -1),
    )
    marg_a = P.sum(dim=-1)
    marg_b = P.sum(dim=-2)
    kl_a = torch.zeros(
        (P.size(0),),
        device=P.device,
        dtype=P.dtype,
    )
    kl_b = torch.zeros(
        (P.size(0),),
        device=P.device,
        dtype=P.dtype,
    )
    if unbalanced:
        kl_a = _kl_measure(
            marg_a,
            a_n,
        )
        kl_b = _kl_measure(
            marg_b,
            b_n,
        )
        total = transport + float(reg_m) * (kl_a + kl_b)
    else:
        total = transport
    out = {
        'total': total,
        'transport': transport,
        'kl_a': kl_a,
        'kl_b': kl_b,
        'mass_a': a_n.sum(dim=-1),
        'mass_b': b_n.sum(dim=-1),
        'mass_plan': P.sum(dim=(-2, -1)),
        'marg_a': marg_a,
        'marg_b': marg_b,
    }
    if return_plan:
        out['plan'] = P
    return out


def ot_loss(
    probs,
    prior,
    C,
    epsilon,
    iters,
    mode='sinkhorn',
    reg_m=0.1,
):
    mode = (mode or 'sinkhorn').lower()
    use_divergence = False
    if mode.endswith('_divergence'):
        mode = mode[: -len('_divergence')]
        use_divergence = True
    unbalanced = mode == 'unbalanced'
    probs = _normalise_measure(
        probs,
        simplex=not unbalanced,
    )
    prior = _normalise_measure(
        prior,
        simplex=not unbalanced,
    )
    if mode == 'emd':
        if ot_base is None or not hasattr(
            ot_base,
            'emd2',
        ):
            raise ValueError('error: ValueError')
        if probs.dim() == 1:
            cost = ot_base.emd2(
                prior.detach().cpu().numpy(),
                probs.detach().cpu().numpy(),
                C.detach().cpu().numpy(),
            )
            return torch.tensor(
                cost,
                device=probs.device,
                dtype=probs.dtype,
            )
        costs = [
            ot_base.emd2(
                prior[i].detach().cpu().numpy(),
                probs[i].detach().cpu().numpy(),
                C.detach().cpu().numpy(),
            )
            for i in range(probs.size(0))
        ]
        return torch.tensor(
            costs,
            device=probs.device,
            dtype=probs.dtype,
        )
    if use_divergence:
        return _sinkhorn_divergence(
            prior,
            probs,
            C,
            epsilon,
            iters,
            unbalanced=unbalanced,
            reg_m=reg_m,
        )
    return _sinkhorn_cost(
        prior,
        probs,
        C,
        epsilon,
        iters,
        unbalanced=unbalanced,
        reg_m=reg_m,
    )


def emotion_loss(
    logits,
    vad_pred,
    batch,
    alpha_cls,
    alpha_vad,
    alpha_ot,
    epsilon,
    iters,
    silver_weight=0.3,
    use_silver=False,
    ot_mode='sinkhorn',
    ot_reg_m=0.1,
    C=None,
    *,
    ot_pred: torch.Tensor | None = None,
    count_pred: torch.Tensor | None = None,
    alpha_mass: float = 0.0,
    count_pred_scale: str | None = None,
):
    labels = batch['labels'].to(logits.device)
    silver_labels = batch['silver_labels'].to(logits.device)
    vad_target = batch['vad'].to(logits.device)
    silver_vad = batch['silver_vad'].to(logits.device)
    prior = batch['lex_prior'].to(logits.device)
    guess_sigmoid = torch.sigmoid(logits)
    guess_dist = (
        ot_pred if ot_pred is not None else guess_sigmoid
    ).to(logits.device)
    if 'has_labels' in batch:
        has_labels = (
            batch['has_labels']
            .to(logits.device)
            .view(
                -1,
                1,
            )
            .float()
        )
    else:
        has_labels = (
            labels.sum(
                dim=1,
                keepdim=True,
            ) > 0
        ).float()
    if 'has_vad' in batch:
        has_vad = (
            batch['has_vad']
            .to(logits.device)
            .view(
                -1,
                1,
            )
            .float()
        )
    else:
        has_vad = torch.ones_like(has_labels)
    if 'has_lex' in batch:
        has_lex = (
            batch['has_lex']
            .to(logits.device)
            .view(
                -1,
                1,
            )
            .float()
        )
    else:
        has_lex = torch.ones_like(has_labels)
    if 'has_lex_vad' in batch:
        has_lex_vad = (
            batch['has_lex_vad']
            .to(logits.device)
            .view(
                -1,
                1,
            )
            .float()
        )
    else:
        has_lex_vad = torch.ones_like(has_labels)
    label_mask = batch.get('label_mask')
    label_mask_missing = label_mask is None
    if label_mask_missing:
        label_mask = torch.ones_like(labels)
    else:
        label_mask = label_mask.to(logits.device).float()
        if label_mask.shape != labels.shape:
            raise ValueError('error: ValueError')
    if use_silver:
        if label_mask_missing:
            silver_mask = (1.0 - has_labels) * has_lex
            cls_target = (
                has_labels * labels
                + (1.0 - has_labels) * silver_labels
            )
            cls_weight = has_labels + silver_mask * float(silver_weight)
        else:
            unknown_mask = (1.0 - label_mask).clamp_min(0.0)
            silver_mask = unknown_mask * has_lex
            cls_target = (
                label_mask * labels
                + unknown_mask * silver_labels
            )
            cls_weight = (
                label_mask * has_labels
                + silver_mask * float(silver_weight)
            )
    else:
        cls_target = labels
        cls_weight = label_mask * has_labels
    cls_loss_vec = F.binary_cross_entropy_with_logits(
        logits,
        cls_target,
        reduction='none',
    )
    if cls_weight.shape != cls_loss_vec.shape:
        cls_weight = cls_weight.expand_as(cls_loss_vec)
    cls_denom = cls_weight.sum().clamp_min(1.0)
    cls_loss = (cls_loss_vec * cls_weight).sum() / cls_denom
    if use_silver:
        silver_vad_mask = (1.0 - has_vad) * has_lex_vad
        vad_target_mix = (
            has_vad * vad_target
            + (1.0 - has_vad) * silver_vad
        )
        vad_weight = has_vad + silver_vad_mask * float(silver_weight)
    else:
        vad_target_mix = vad_target
        vad_weight = has_vad
    vad_loss_vec = torch.mean(
        (vad_pred - vad_target_mix) ** 2,
        dim=-1,
        keepdim=True,
    )
    vad_denom = vad_weight.sum().clamp_min(1.0)
    vad_loss = (vad_loss_vec * vad_weight).sum() / vad_denom
    if C is None:
        C = cost_matrix(
            logits.size(-1),
            logits.device,
        )
    else:
        C = C.to(
            device=logits.device,
            dtype=logits.dtype,
        )
    mode = (ot_mode or 'sinkhorn').lower()
    count_scale = (
        (count_pred_scale or 'counts').lower().strip()
    )
    count_is_density = count_scale in {
        'density',
        'per_token',
        'per_tok',
    }
    token_denom = None
    if count_is_density and 'n_tokens' in batch:
        token_denom = (
            batch['n_tokens']
            .to(logits.device)
            .view(
                -1,
                1,
            )
            .clamp_min(1.0)
        )
    if (
        count_is_density
        and token_denom is None
        and ('lex_counts' in batch)
    ):
        raise ValueError('error: ValueError')
    if (
        mode.startswith('unbalanced')
        and 'lex_counts' in batch
    ):
        target = batch['lex_counts'].to(logits.device)
        if count_is_density and token_denom is not None:
            target = target / token_denom
    else:
        target = prior
    mass_guess = None
    if mode.startswith('unbalanced'):
        if ot_pred is None and count_pred is not None:
            guess_dist = count_pred.to(logits.device).clamp_min(1e-08)
        if count_pred is not None:
            mass_guess = count_pred.to(logits.device).sum(
                dim=-1,
                keepdim=True,
            )
        else:
            mass_guess = guess_dist.sum(
                dim=-1,
                keepdim=True,
            )
        mass_guess = mass_guess.clamp_min(1e-08)
        dist_denom = guess_dist.sum(
            dim=-1,
            keepdim=True,
        )
        dist_norm = guess_dist / dist_denom.clamp_min(1e-08)
        if torch.any(dist_denom <= 1e-08):
            uniform = torch.full_like(
                guess_dist,
                1.0 / guess_dist.size(-1),
            )
            dist_norm = torch.where(
                dist_denom <= 1e-08,
                uniform,
                dist_norm,
            )
        ot_input = dist_norm * mass_guess
    else:
        ot_input = guess_dist
    ot_vals = ot_loss(
        ot_input,
        target,
        C,
        epsilon,
        iters,
        ot_mode,
        ot_reg_m,
    )
    if ot_vals.dim() == 0:
        ot_vals = ot_vals.unsqueeze(0)
    ot_weight = has_lex.squeeze(-1)
    ot_denom = ot_weight.sum().clamp_min(1.0)
    ot_loss_val = (ot_vals * ot_weight).sum() / ot_denom
    mass_loss_val = torch.zeros(
        (),
        device=logits.device,
        dtype=logits.dtype,
    )
    if (
        alpha_mass
        and count_pred is not None
        and ('lex_counts' in batch)
    ):
        if mass_guess is None:
            mass_guess = (
                count_pred.to(logits.device)
                .sum(
                    dim=-1,
                    keepdim=True,
                )
                .clamp_min(1e-08)
            )
        mass_target = (
            batch['lex_counts']
            .to(logits.device)
            .sum(
                dim=-1,
                keepdim=True,
            )
        )
        if count_is_density and token_denom is not None:
            mass_target = mass_target / token_denom
        mass_loss_vec = (mass_guess - mass_target) ** 2
        mass_loss_val = (
            mass_loss_vec * has_lex
        ).sum() / has_lex.sum().clamp_min(1.0)
    total = (
        alpha_cls * cls_loss
        + alpha_vad * vad_loss
        + alpha_ot * ot_loss_val
        + float(alpha_mass) * mass_loss_val
    )
    return (
        total,
        {
            'cls': cls_loss.item(),
            'vad': vad_loss.item(),
            'ot': ot_loss_val.item(),
            'mass': mass_loss_val.item(),
        },
    )
