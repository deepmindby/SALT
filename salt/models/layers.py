"""Shared modules: action encoder and projection head (as in LeWM)."""

from torch import nn


class ActionEncoder(nn.Module):
    """Maps an action block (frameskip x action_dim) to a d-dimensional condition."""

    def __init__(self, input_dim: int, emb_dim: int, smoothed_dim: int = 10, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """x: (B, T, input_dim) -> (B, T, emb_dim)"""
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """Two-layer MLP with a normalization layer and activation in between."""

    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """x: (N, input_dim)"""
        return self.net(x)
