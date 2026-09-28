"""
Physics-guided multimodal spatiotemporal Transformer for MeghPrahari v0.3.

Input:
    x: [batch, time, features]
    The feature vector is split into satellite, meteorology and terrain groups.
    Static terrain variables are broadcast over the sequence.

Architecture:
    modality projections -> explicit cross-modal attention -> temporal Transformer
    -> shared representation -> 3 hazard heads x 3 lead-time heads.

This is a lightweight research/prototype model, not a claim of operational skill.
Train/evaluate on event-ledger labels before using it for real alerts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


HAZARDS = ("ts", "cb", "ff")
LEADS = ("0_2", "2_4", "4_6")


@dataclass(frozen=True)
class MTLConfig:
    n_features: int
    sat_idx: tuple[int, ...]
    met_idx: tuple[int, ...]
    terrain_idx: tuple[int, ...]
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.10
    max_seq_len: int = 12


class CrossModalBlock(nn.Module):
    """Bidirectional cross-modal attention followed by residual fusion."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.sat_to_met = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.met_to_sat = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_sat = nn.LayerNorm(d_model)
        self.norm_met = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, sat, met):
        sat_attn, _ = self.sat_to_met(sat, met, met, need_weights=False)
        met_attn, _ = self.met_to_sat(met, sat, sat, need_weights=False)
        sat = self.norm_sat(sat + sat_attn)
        met = self.norm_met(met + met_attn)
        fused = self.ff(torch.cat([sat, met], dim=-1))
        return self.norm_out(fused + (sat + met) / 2.0)


class PhysicsGuidedMTLTransformer(nn.Module):
    """Small shared encoder with simultaneous TS/CB/FF predictions for 0-2/2-4/4-6h."""

    def __init__(self, cfg: MTLConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.sat_proj = nn.Sequential(nn.Linear(len(cfg.sat_idx), cfg.d_model), nn.GELU())
        self.met_proj = nn.Sequential(nn.Linear(len(cfg.met_idx), cfg.d_model), nn.GELU())
        self.terrain_proj = nn.Sequential(nn.Linear(len(cfg.terrain_idx), cfg.d_model), nn.GELU())

        self.cross_modal = CrossModalBlock(cfg.d_model, cfg.n_heads, cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)

        self.shared_norm = nn.LayerNorm(cfg.d_model)
        self.hazard_heads = nn.ModuleDict({
            h: nn.Linear(cfg.d_model, len(LEADS)) for h in HAZARDS
        })

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, features]")
        b, t, f = x.shape
        if f != self.cfg.n_features:
            raise ValueError(f"expected {self.cfg.n_features} features, got {f}")
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}")

        sat = self.sat_proj(x[..., list(self.cfg.sat_idx)])
        met = self.met_proj(x[..., list(self.cfg.met_idx)])
        terrain = self.terrain_proj(x[..., list(self.cfg.terrain_idx)])

        shared = self.cross_modal(sat, met) + terrain
        shared = shared + self.pos[:, :t]
        z = self.temporal(shared)
        z = self.shared_norm(z[:, -1])  # latest timestep is the forecast anchor

        logits = torch.cat([self.hazard_heads[h](z) for h in HAZARDS], dim=-1)
        return logits.reshape(b, len(HAZARDS), len(LEADS))

    @torch.no_grad()
    def probabilities(self, x):
        return torch.sigmoid(self.forward(x))


def multitask_bce(logits, targets, pos_weight=None):
    """Masked BCE for [B, 3 hazards, 3 lead bins]. targets may contain NaN for unknown labels."""
    mask = torch.isfinite(targets)
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none", pos_weight=pos_weight
    )
    return loss[mask].mean() if mask.any() else loss.mean() * 0.0


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
