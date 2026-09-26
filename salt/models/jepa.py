"""Joint-embedding world model: encoder, action encoder and latent predictor."""

import torch
from einops import rearrange
from torch import nn


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    """Shared wrapper for SALT and the reproduced LeWM baseline.

    Args:
        encoder: visual backbone returning ``last_hidden_state`` (CLS token is used).
        predictor: latent predictor ``(B, T, d), (B, T, d) -> (B, T, d)``.
        action_encoder: maps action blocks to d-dimensional conditions.
        projector: projection head applied to the CLS token.
        pred_proj: optional projection after the predictor (LeWM only; ``None`` for SALT,
            whose transition acts directly in the regularized latent space).
    """

    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: dict) -> dict:
        """Encode ``info['pixels']`` (B, T, C, H, W) into ``info['emb']`` (B, T, d)."""
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        emb = self.projector(output.last_hidden_state[:, 0])
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Next-latent prediction at every position. emb, act_emb: (B, T, d)."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info: dict, action_sequence: torch.Tensor, history_size: int = 3) -> dict:
        """Autoregressive latent rollout used by the planner.

        info['pixels']: (B, S, H, C, H_img, W_img) with H history frames.
        action_sequence: (B, S, T, action_dim); the first H actions belong to the history.
        Writes info['predicted_emb']: (B, S, T + 1, d), i.e. the H encoded history latents
        followed by T + 1 - H predicted latents; the planner scores the last one.
        """
        n_hist = info["pixels"].size(2)
        b, s, t = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [n_hist, t - n_hist], dim=2)
        info["action"] = act_0
        n_steps = t - n_hist

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)
        emb = info["emb"] = init["emb"].unsqueeze(1).expand(b, s, -1, -1)
        init = {k: _detach_clone(v) for k, v in init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        for step in range(n_steps + 1):
            act_emb = self.action_encoder(act)
            pred = self.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            if step < n_steps:
                act = torch.cat([act, act_future[:, step : step + 1]], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=b, s=s)
        return info
