import torch
import pytest

try:
    from legal_emotion.losses import (
        cost_matrix,
        emotion_loss,
        ot_loss,
    )
    from legal_emotion.config import default_config
    from legal_emotion.model import LegalEmotionModel
except ImportError as e:
    pytest.skip(
        str(e),
        allow_module_level=True,
    )


def dummy_batch(cfg):
    bsz = 2
    seq = 8
    input_ids = torch.randint(
        0,
        100,
        (bsz, seq),
    )
    attn = torch.ones(
        (bsz, seq),
        dtype=torch.long,
    )
    labels = torch.zeros((bsz, len(cfg.emotions)))
    labels[0, 0] = 1
    vad = torch.zeros((bsz, 3))
    lex_counts = torch.ones((bsz, len(cfg.emotions)))
    lex_prior = torch.softmax(
        torch.rand((bsz, len(cfg.emotions))),
        dim=-1,
    )
    lex_vad = torch.zeros((bsz, 3))
    return {
        'input_ids': input_ids,
        'attention_mask': attn,
        'labels': labels,
        'vad': vad,
        'lex_counts': lex_counts,
        'lex_prior': lex_prior,
        'lex_vad': lex_vad,
        'silver_labels': lex_prior,
        'silver_vad': lex_vad,
        'text': ['a', 'b'],
        'meta': [{}, {}],
    }


def test_cost_matrix():
    C = cost_matrix(
        3,
        torch.device('cpu'),
    )
    assert torch.allclose(
        torch.diag(C),
        torch.zeros(3),
    )
    assert C.shape == (3, 3)


def test_emotion_loss_balanced_forward():
    setup = default_config()
    setup.model_name = (
        'hf-internal-testing/tiny-random-bert'
    )
    setup.ot_mode = 'sinkhorn_divergence'
    setup.alpha_mass = 0.0
    batch = dummy_batch(setup)
    model = LegalEmotionModel(
        setup.model_name,
        len(setup.emotions),
    )
    (
        logits,
        vad,
        count_guess,
    ) = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        lex_counts=batch['lex_counts'],
        lex_prior=batch['lex_prior'],
        lex_vad=batch['lex_vad'],
    )
    (
        loss,
        parts,
    ) = emotion_loss(
        logits,
        vad,
        batch,
        setup.alpha_cls,
        setup.alpha_vad,
        setup.alpha_sinkhorn,
        setup.sinkhorn_epsilon,
        setup.sinkhorn_iters,
        setup.silver_weight,
        setup.use_silver,
        setup.ot_mode,
        setup.ot_reg_m,
        count_pred=count_guess,
        alpha_mass=setup.alpha_mass,
    )
    assert loss.item() > 0
    assert parts['cls'] > 0
    assert parts['vad'] >= 0
    assert parts['ot'] >= 0
    assert parts['mass'] >= 0


def test_unbalanced_ot_penalizes_mass_mismatch():
    C = cost_matrix(
        3,
        torch.device('cpu'),
    )
    a = torch.tensor([0.2, 0.3, 0.5])
    b = a * 3.0
    loss_same = ot_loss(
        a,
        a,
        C,
        epsilon=0.1,
        iters=50,
        mode='unbalanced_divergence',
        reg_m=0.5,
    ).item()
    loss_mass = ot_loss(
        a,
        b,
        C,
        epsilon=0.1,
        iters=50,
        mode='unbalanced_divergence',
        reg_m=0.5,
    ).item()
    assert abs(loss_same) < 1e-06
    assert loss_mass > 0.0001


def test_sinkhorn_plan_marginals_match_balanced_inputs():
    from legal_emotion.losses import sinkhorn_plan

    C = cost_matrix(
        3,
        torch.device('cpu'),
    )
    a = torch.tensor([0.2, 0.3, 0.5])
    b = torch.tensor([0.5, 0.2, 0.3])
    P = sinkhorn_plan(
        a,
        b,
        C,
        epsilon=0.1,
        iters=200,
        unbalanced=False,
    )
    assert P.shape == (3, 3)
    assert torch.allclose(
        P.sum(dim=-1),
        a / a.sum(),
        atol=0.005,
    )
    assert torch.allclose(
        P.sum(dim=-2),
        b / b.sum(),
        atol=0.005,
    )


def test_sinkhorn_cost_parts_unbalanced_includes_kl():
    from legal_emotion.losses import sinkhorn_cost_parts

    C = cost_matrix(
        3,
        torch.device('cpu'),
    )
    a = torch.tensor([0.2, 0.3, 0.5])
    b = a * 3.0
    parts = sinkhorn_cost_parts(
        a,
        b,
        C,
        epsilon=0.1,
        iters=100,
        unbalanced=True,
        reg_m=0.5,
        return_plan=False,
    )
    assert parts['total'].item() > 0
    assert parts['transport'].item() >= 0
    assert parts['kl_a'].item() >= 0
    assert parts['kl_b'].item() >= 0
