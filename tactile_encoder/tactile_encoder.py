"""
Tactile Encoder Architecture (v2 — with temporal consistency)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Focal Loss
# =========================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce

        if mask is not None:
            focal = focal * mask
            if mask.sum() > 0:
                return focal.sum() / mask.sum()
            else:
                return focal.sum() * 0.0
        else:
            if self.reduction == "mean":
                return focal.mean()
            elif self.reduction == "sum":
                return focal.sum()
            else:
                return focal


# =========================================================================
# Encoder Architecture (unchanged from v1)
# =========================================================================

class TactileEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        window_size: int = 16,
        hidden_dim: int = 64,
        z_dim: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, window_size])

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, window_size])

        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([hidden_dim, window_size])

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln_proj = nn.LayerNorm(hidden_dim)

        self.to_z = nn.Linear(hidden_dim, z_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = F.gelu(self.ln1(self.conv1(x)))
        x = F.gelu(self.ln2(self.conv2(x)))
        x = F.gelu(self.ln3(self.conv3(x)))
        h = x.mean(dim=-1)
        h_t = F.gelu(self.ln_proj(self.proj(h)))
        z_t = self.to_z(h_t)
        return h_t, z_t


# =========================================================================
# Auxiliary heads (unchanged from v1)
# =========================================================================

class AuxiliaryHeads(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_phases: int = 4):
        super().__init__()

        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.symmetry_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_phases),
        )
        self.force_mag_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.GELU(), nn.Linear(16, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_t: torch.Tensor):
        return dict(
            contact_logit=self.contact_head(h_t).squeeze(-1),
            symmetry_logit=self.symmetry_head(h_t).squeeze(-1),
            stability_logit=self.stability_head(h_t).squeeze(-1),
            phase_logits=self.phase_head(h_t),
            force_mag=self.force_mag_head(h_t).squeeze(-1),
        )


# =========================================================================
# Combined module for pre-training
# =========================================================================

class TactileEncoderWithHeads(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        window_size: int = 16,
        hidden_dim: int = 64,
        z_dim: int = 16,
        n_phases: int = 4,
    ):
        super().__init__()
        self.encoder = TactileEncoder(
            in_channels=in_channels,
            window_size=window_size,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
        )
        self.heads = AuxiliaryHeads(hidden_dim=hidden_dim, n_phases=n_phases)

    def forward(self, x: torch.Tensor):
        h_t, z_t = self.encoder(x)
        preds = self.heads(h_t)
        preds["h_t"] = h_t
        preds["z_t"] = z_t
        return preds


# =========================================================================
# Multi-task loss WITH temporal consistency
# =========================================================================

class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss with temporal consistency regularization.
    """
    def __init__(
        self,
        w_contact: float = 1.0,
        w_symmetry: float = 2.0,
        w_stability: float = 1.0,
        w_phase: float = 0.5,
        w_force_mag: float = 0.3,
        w_temporal: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        contact_pos_weight: float = 0.24,
    ):
        super().__init__()
        self.w_contact = w_contact
        self.w_symmetry = w_symmetry
        self.w_stability = w_stability
        self.w_phase = w_phase
        self.w_force_mag = w_force_mag
        self.w_temporal = w_temporal

        self.register_buffer(
            "contact_pos_weight",
            torch.tensor(contact_pos_weight, dtype=torch.float32),
        )
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, preds: dict, labels: dict,
                preds_prev: Optional[dict] = None,
                same_phase: Optional[torch.Tensor] = None):
        """
        Args:
            preds:      dict from TactileEncoderWithHeads for window at t
            labels:     dict with target labels at t
            preds_prev: dict from TactileEncoderWithHeads for window at t-1
                        (None disables temporal loss)
            same_phase: [B] float, 1.0 where phase(t)==phase(t-1)
                        (None disables temporal loss)
        Returns:
            total_loss, loss_components dict
        """
        # --- Head 1: Contact ---
        loss_contact = F.binary_cross_entropy_with_logits(
            preds["contact_logit"],
            labels["contact"],
            pos_weight=self.contact_pos_weight,
        )

        # --- Head 2: Symmetry (masked by contact) ---
        contact_mask = labels["contact"]
        symmetry_pred = torch.sigmoid(preds["symmetry_logit"])
        sym_sq_err = (symmetry_pred - labels["symmetry"]) ** 2
        if contact_mask.sum() > 0:
            loss_symmetry = (sym_sq_err * contact_mask).sum() / contact_mask.sum()
        else:
            loss_symmetry = sym_sq_err.mean() * 0.0

        # --- Head 3: Stability (focal, masked by contact) ---
        loss_stability = self.focal(
            preds["stability_logit"],
            labels["stability"],
            mask=contact_mask,
        )

        # --- Head 4: Phase ---
        loss_phase = F.cross_entropy(
            preds["phase_logits"],
            labels["phase"].long(),
        )

        # --- Head 5: Force magnitude ---
        loss_force_mag = F.mse_loss(preds["force_mag"], labels["force_mag"])

        # --- Head 6: Temporal consistency (margin-based) ---
        # Only penalizes z_t changes exceeding the margin.
        # L = same_phase * max(0, ||z_curr - z_prev|| - margin)^2
        if preds_prev is not None and same_phase is not None:
            z_curr = preds["z_t"]
            z_prev = preds_prev["z_t"]

            # Per-sample L2 distance
            z_dist = torch.norm(z_curr - z_prev, dim=-1)  # [B]

            # Hinge: only penalize beyond the margin
            margin = 0.5
            excess = torch.clamp(z_dist - margin, min=0.0)  # [B]
            hinge_sq = excess ** 2  # [B]

            # Mask to same-phase timesteps
            masked = hinge_sq * same_phase

            if same_phase.sum() > 0:
                loss_temporal = masked.sum() / same_phase.sum()
            else:
                loss_temporal = masked.sum() * 0.0
        else:
            loss_temporal = torch.tensor(0.0, device=preds["contact_logit"].device)

        # --- Total ---
        total = (
            self.w_contact * loss_contact
            + self.w_symmetry * loss_symmetry
            + self.w_stability * loss_stability
            + self.w_phase * loss_phase
            + self.w_force_mag * loss_force_mag
            + self.w_temporal * loss_temporal
        )

        components = dict(
            contact=loss_contact.detach().item(),
            symmetry=loss_symmetry.detach().item(),
            stability=loss_stability.detach().item(),
            phase=loss_phase.detach().item(),
            force_mag=loss_force_mag.detach().item(),
            temporal=loss_temporal.detach().item(),
            total=total.detach().item(),
        )
        return total, components


# =========================================================================
# Sanity check
# =========================================================================

if __name__ == "__main__":
    B, K, C = 32, 16, 6
    x_curr = torch.randn(B, K, C)
    x_prev = torch.randn(B, K, C)

    model = TactileEncoderWithHeads()
    preds_curr = model(x_curr)
    preds_prev = model(x_prev)

    print("=== TactileEncoderWithHeads (v2 — temporal consistency) ===")
    print(f"Input shape:       {tuple(x_curr.shape)}")
    print(f"h_t shape:         {tuple(preds_curr['h_t'].shape)}")
    print(f"z_t shape:         {tuple(preds_curr['z_t'].shape)}")

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.heads.parameters())
    print(f"\nEncoder parameters: {enc_params:,}")
    print(f"Heads parameters:   {head_params:,}")
    print(f"Total parameters:   {enc_params + head_params:,}")

    # Test loss with temporal consistency
    labels = dict(
        contact=torch.randint(0, 2, (B,)).float(),
        symmetry=torch.rand(B),
        stability=torch.randint(0, 2, (B,)).float(),
        phase=torch.randint(0, 4, (B,)).long(),
        force_mag=torch.randn(B),
    )
    same_phase = (torch.rand(B) > 0.3).float()  # 70% same phase

    loss_fn = MultiTaskLoss(w_temporal=0.5)

    # With temporal loss
    total_loss, components = loss_fn(preds_curr, labels, preds_prev, same_phase)
    print(f"\nLoss components (with temporal): {components}")
    print(f"Total loss: {total_loss.item():.4f}")

    # Without temporal loss (backward compatible)
    total_loss2, components2 = loss_fn(preds_curr, labels)
    print(f"\nLoss components (without temporal): {components2}")
    print(f"Total loss: {total_loss2.item():.4f}")
    print(f"\nTemporal loss contribution: {components['temporal']:.4f}")