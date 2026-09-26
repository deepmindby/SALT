"""Structural checks of the SALT transition (run with ``pytest tests``)."""

import torch

from salt.models.transition import StateAffineTransition

D, R = 192, 16


def make():
    torch.manual_seed(0)
    m = StateAffineTransition(dim=D, cond_dim=D, num_modes=R)
    with torch.no_grad():  # move away from the zero initialization
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.eval()


def test_parameter_count_matches_paper():
    n = sum(p.numel() for p in StateAffineTransition(dim=D, cond_dim=D, num_modes=R).parameters())
    assert n == 703_872


def test_initialization_is_near_identity():
    m = StateAffineTransition(dim=D, cond_dim=D, num_modes=R).eval()
    z, c = torch.randn(2, 3, D), torch.randn(2, 3, D)
    assert torch.allclose(m(z, c), torch.tanh(torch.tensor(3.0)) * z, atol=1e-5)


def test_state_affine_property():
    m = make()
    z, c, delta = torch.randn(4, 2, D, dtype=torch.float64), torch.randn(4, 2, D, dtype=torch.float64), torch.randn(4, 2, D, dtype=torch.float64)
    m = m.double()
    lhs = m(z + delta, c) - m(z, c)
    rhs = torch.einsum("btij,btj->bti", m.transition_matrix(c), delta)
    assert torch.allclose(lhs, rhs, atol=1e-9)


def test_base_matrix_is_contractive():
    m = make()
    sigma = torch.linalg.matrix_norm(m.base_matrix(), ord=2)
    assert sigma < 1.0
    assert torch.allclose(sigma, torch.tanh(m.s).abs().max(), atol=1e-4)


def test_legacy_state_dict_keys_load():
    m = make()
    legacy = {
        "skew_u": m.W_u, "skew_v": m.W_v, "s": m.s, "G_c.weight": m.W_g.weight, "C": m.N,
        "B.weight": m.B.weight, "B.bias": m.B.bias,
    }
    other = StateAffineTransition(dim=D, cond_dim=D, num_modes=R).eval()
    other.load_state_dict({k: v.detach().clone() for k, v in legacy.items()}, strict=True)
    z, c = torch.randn(2, 3, D), torch.randn(2, 3, D)
    assert torch.allclose(m(z, c), other(z, c), atol=1e-6)
