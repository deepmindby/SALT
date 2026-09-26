"""Action-conditioned state-affine transition (SALT).

    F(z, c) = A(c) z + B c + b,      A(c) = A_0 + sum_r (W_g c)_r N_r

The transition is affine in the latent state z for every action condition c,
so its state Jacobian A(c) is independent of z. The shared base matrix is
parameterized as A_0 = U diag(tanh s) V^T with U = exp(W_u - W_u^T) and
V = exp(W_v - W_v^T), which keeps sigma_max(A_0) < 1.
"""

import torch
from torch import nn

# Parameter names used by checkpoints produced with the pre-release code base.
_LEGACY_KEYS = {
    "skew_u": "W_u",
    "skew_v": "W_v",
    "G_c.weight": "W_g.weight",
    "C": "N",
}


class StateAffineTransition(nn.Module):
    """Per-step state-affine predictor.

    Args:
        dim: latent dimension d.
        cond_dim: action-condition dimension (equal to d in the paper).
        num_modes: number of modulation matrices R (16 in the paper).
        spectral: parameterize A_0 through the constrained form above.
            When False, A_0 is a free dense matrix (Appendix C.4).
        s_init: initial value of the singular-value parameters s.
    """

    def __init__(
        self,
        dim: int = 192,
        cond_dim: int = 192,
        num_modes: int = 16,
        spectral: bool = True,
        s_init: float = 3.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_modes = num_modes
        self.spectral = spectral

        if spectral:
            self.W_u = nn.Parameter(torch.zeros(dim, dim))
            self.W_v = nn.Parameter(torch.zeros(dim, dim))
            self.s = nn.Parameter(torch.full((dim,), float(s_init)))
        else:
            gain = float(torch.tanh(torch.tensor(s_init)))
            self.A_free = nn.Parameter(gain * torch.eye(dim))

        if num_modes > 0:
            self.W_g = nn.Linear(cond_dim, num_modes, bias=False)
            self.N = nn.Parameter(torch.zeros(num_modes, dim, dim))

        self.B = nn.Linear(cond_dim, dim, bias=True)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

        self._A0_cache = None

    @staticmethod
    def _orthogonal(w: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(w - w.transpose(-1, -2))

    def base_matrix(self) -> torch.Tensor:
        """Shared base transition A_0."""
        if not self.spectral:
            return self.A_free
        u = self._orthogonal(self.W_u)
        v = self._orthogonal(self.W_v)
        return u @ torch.diag(torch.tanh(self.s)) @ v.transpose(-1, -2)

    def _base(self) -> torch.Tensor:
        if self.training:
            return self.base_matrix()
        if self._A0_cache is None:
            self._A0_cache = self.base_matrix().detach()
        return self._A0_cache

    def train(self, mode: bool = True):
        self._A0_cache = None
        return super().train(mode)

    def transition_matrix(self, c: torch.Tensor) -> torch.Tensor:
        """Action-conditioned state Jacobian A(c), shape (..., d, d)."""
        a = self._base().expand(*c.shape[:-1], self.dim, self.dim)
        if self.num_modes > 0:
            a = a + torch.einsum("...r,rij->...ij", self.W_g(c), self.N)
        return a

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict z_{t+1} = F(z_t, c_t) position-wise.

        z: (B, T, d) latent states, c: (B, T, cond_dim) action conditions.
        """
        out = z @ self._base().transpose(-1, -2) + self.B(c)
        if self.num_modes > 0:
            out = out + torch.einsum("...r,rij,...j->...i", self.W_g(c), self.N, z)
        return out

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for old, new in _LEGACY_KEYS.items():
            if prefix + old in state_dict:
                state_dict[prefix + new] = state_dict.pop(prefix + old)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
