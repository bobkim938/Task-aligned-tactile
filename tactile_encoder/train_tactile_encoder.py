"""
Tactile Encoder Pre-training Script (v2 — with temporal consistency)
=====================================================================
Trains the phase-aware tactile encoder with temporal consistency loss.

Key change from v1: each training step forwards TWO windows (t and t-1)
through the encoder and penalizes z_t differences when the phase hasn't
changed. This produces temporally smooth representations.

Usage:
    python tactile_encoder/train_tactile_encoder.py \
        --data-dir data/encoder_pretraining \
        --output-dir runs/tactile_encoder_v2 \
        --epochs 30 \
        --batch-size 512 \
        --lr 3e-4 \
        --w-temporal 0.5
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tactile_encoder import TactileEncoderWithHeads, MultiTaskLoss
from encoder_dataset import TactileWindowDataset, collate_fn


# =========================================================================
# Evaluation (includes temporal consistency metric)
# =========================================================================

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    all_contact_pred, all_contact_true = [], []
    all_symmetry_pred, all_symmetry_true, all_symmetry_mask = [], [], []
    all_stability_pred, all_stability_true, all_stability_mask = [], [], []
    all_phase_pred, all_phase_true = [], []
    all_force_mag_pred, all_force_mag_true = [], []
    all_z_delta = []  # track z_t temporal smoothness

    total_loss = 0.0
    total_components = {k: 0.0 for k in
                        ["contact", "symmetry", "stability", "phase",
                         "force_mag", "temporal", "total"]}
    n_batches = 0

    for batch in loader:
        x_curr = batch["force_window"].to(device)
        x_prev = batch["force_window_prev"].to(device)
        same_phase = batch["same_phase"].to(device)

        labels = {
            "contact": batch["contact"].to(device),
            "symmetry": batch["symmetry"].to(device),
            "stability": batch["stability"].to(device),
            "phase": batch["phase"].to(device),
            "force_mag": batch["force_mag"].to(device),
        }

        preds_curr = model(x_curr)
        preds_prev = model(x_prev)

        loss, components = loss_fn(preds_curr, labels, preds_prev, same_phase)

        total_loss += loss.item()
        for k, v in components.items():
            total_components[k] += v
        n_batches += 1

        # z_t temporal delta (for monitoring smoothness)
        z_delta = torch.norm(
            preds_curr["z_t"] - preds_prev["z_t"], dim=-1
        )  # [B]
        # Only measure delta where phase is the same (steady-state)
        steady_mask = same_phase > 0.5
        if steady_mask.sum() > 0:
            all_z_delta.append(z_delta[steady_mask].cpu().numpy())

        # Accumulate classification/regression metrics
        all_contact_pred.append(
            (torch.sigmoid(preds_curr["contact_logit"]) > 0.5).cpu().numpy()
        )
        all_contact_true.append(labels["contact"].cpu().numpy())

        all_symmetry_pred.append(
            torch.sigmoid(preds_curr["symmetry_logit"]).cpu().numpy()
        )
        all_symmetry_true.append(labels["symmetry"].cpu().numpy())
        all_symmetry_mask.append(labels["contact"].cpu().numpy())

        all_stability_pred.append(
            (torch.sigmoid(preds_curr["stability_logit"]) > 0.5).cpu().numpy()
        )
        all_stability_true.append(labels["stability"].cpu().numpy())
        all_stability_mask.append(labels["contact"].cpu().numpy())

        all_phase_pred.append(
            preds_curr["phase_logits"].argmax(dim=-1).cpu().numpy()
        )
        all_phase_true.append(labels["phase"].cpu().numpy())

        all_force_mag_pred.append(preds_curr["force_mag"].cpu().numpy())
        all_force_mag_true.append(labels["force_mag"].cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in total_components.items()}

    # Concatenate
    contact_pred = np.concatenate(all_contact_pred)
    contact_true = np.concatenate(all_contact_true)
    symmetry_pred = np.concatenate(all_symmetry_pred)
    symmetry_true = np.concatenate(all_symmetry_true)
    symmetry_mask = np.concatenate(all_symmetry_mask)
    stability_pred = np.concatenate(all_stability_pred)
    stability_true = np.concatenate(all_stability_true)
    stability_mask = np.concatenate(all_stability_mask)
    phase_pred = np.concatenate(all_phase_pred)
    phase_true = np.concatenate(all_phase_true)
    force_mag_pred = np.concatenate(all_force_mag_pred)
    force_mag_true = np.concatenate(all_force_mag_true)

    metrics = {}
    metrics["contact_acc"] = accuracy_score(contact_true > 0.5, contact_pred)
    metrics["contact_f1"] = f1_score(contact_true > 0.5, contact_pred, zero_division=0)

    if symmetry_mask.sum() > 0:
        m = symmetry_mask > 0.5
        if m.sum() > 1 and symmetry_true[m].std() > 0:
            metrics["symmetry_r2"] = r2_score(symmetry_true[m], symmetry_pred[m])
        else:
            metrics["symmetry_r2"] = 0.0
        metrics["symmetry_mae"] = float(np.mean(np.abs(symmetry_pred[m] - symmetry_true[m])))
    else:
        metrics["symmetry_r2"] = 0.0
        metrics["symmetry_mae"] = 0.0

    if stability_mask.sum() > 0:
        m = stability_mask > 0.5
        metrics["stability_acc"] = accuracy_score(stability_true[m] > 0.5, stability_pred[m])
        metrics["stability_f1_unstable"] = f1_score(
            (stability_true[m] < 0.5), (stability_pred[m] < 0.5), zero_division=0
        )
    else:
        metrics["stability_acc"] = 0.0
        metrics["stability_f1_unstable"] = 0.0

    metrics["phase_acc"] = accuracy_score(phase_true, phase_pred)

    if force_mag_true.std() > 0:
        metrics["force_mag_r2"] = r2_score(force_mag_true, force_mag_pred)
    else:
        metrics["force_mag_r2"] = 0.0

    # Temporal smoothness metric: mean z_t delta during steady-state
    if all_z_delta:
        z_delta_all = np.concatenate(all_z_delta)
        metrics["z_delta_mean"] = float(z_delta_all.mean())
        metrics["z_delta_p95"] = float(np.percentile(z_delta_all, 95))
    else:
        metrics["z_delta_mean"] = 0.0
        metrics["z_delta_p95"] = 0.0

    return avg_loss, avg_components, metrics


# =========================================================================
# Training loop
# =========================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    print("Loading datasets...")
    train_ds = TactileWindowDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    val_ds = TactileWindowDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # --- Model ---
    model = TactileEncoderWithHeads(
        in_channels=6,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        n_phases=4,
    ).to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.heads.parameters())
    print(f"Encoder params: {enc_params:,}")
    print(f"Heads params:   {head_params:,}")
    print(f"Total params:   {enc_params + head_params:,}")

    # --- Loss (with temporal consistency) ---
    loss_fn = MultiTaskLoss(
        w_contact=args.w_contact,
        w_symmetry=args.w_symmetry,
        w_stability=args.w_stability,
        w_phase=args.w_phase,
        w_force_mag=args.w_force_mag,
        w_temporal=args.w_temporal,
    ).to(device)

    print(f"\nLoss weights:")
    print(f"  contact={args.w_contact}, symmetry={args.w_symmetry}, "
          f"stability={args.w_stability}")
    print(f"  phase={args.w_phase}, force_mag={args.w_force_mag}, "
          f"temporal={args.w_temporal}")

    # --- Optimizer / scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- CSV log ---
    log_path = output_dir / "train_log.csv"
    log_fields = [
        "epoch", "lr",
        "train_loss", "train_contact", "train_symmetry", "train_stability",
        "train_phase", "train_force_mag", "train_temporal",
        "val_loss", "val_contact", "val_symmetry", "val_stability",
        "val_phase", "val_force_mag", "val_temporal",
        "val_contact_acc", "val_contact_f1",
        "val_symmetry_r2", "val_symmetry_mae",
        "val_stability_acc", "val_stability_f1_unstable",
        "val_phase_acc", "val_force_mag_r2",
        "val_z_delta_mean", "val_z_delta_p95",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs "
          f"({len(train_loader)} train batches/epoch, "
          f"{len(val_loader)} val batches/epoch)")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        running = {k: 0.0 for k in
                   ["contact", "symmetry", "stability", "phase",
                    "force_mag", "temporal", "total"]}
        n_batches = 0

        for step, batch in enumerate(train_loader):
            x_curr = batch["force_window"].to(device, non_blocking=True)
            x_prev = batch["force_window_prev"].to(device, non_blocking=True)
            same_phase = batch["same_phase"].to(device, non_blocking=True)

            labels = {
                "contact": batch["contact"].to(device, non_blocking=True),
                "symmetry": batch["symmetry"].to(device, non_blocking=True),
                "stability": batch["stability"].to(device, non_blocking=True),
                "phase": batch["phase"].to(device, non_blocking=True),
                "force_mag": batch["force_mag"].to(device, non_blocking=True),
            }

            # Forward BOTH windows through the encoder
            preds_curr = model(x_curr)
            preds_prev = model(x_prev)

            # Loss includes temporal consistency between z_t(t) and z_t(t-1)
            loss, components = loss_fn(
                preds_curr, labels, preds_prev, same_phase
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            for k, v in components.items():
                running[k] += v
            n_batches += 1

            if (step + 1) % args.log_every == 0:
                avg = {k: v / max(n_batches, 1) for k, v in running.items()}
                print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)}  "
                      f"loss={avg['total']:.4f}  "
                      f"con={avg['contact']:.3f}  "
                      f"sym={avg['symmetry']:.3f}  "
                      f"sta={avg['stability']:.3f}  "
                      f"pha={avg['phase']:.3f}  "
                      f"fmg={avg['force_mag']:.3f}  "
                      f"tmp={avg['temporal']:.3f}")

        train_avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        # --- Validate ---
        val_loss, val_components, val_metrics = evaluate(
            model, val_loader, loss_fn, device
        )
        elapsed = time.time() - t0

        # --- Log ---
        log_row = {
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_avg["total"],
            "train_contact": train_avg["contact"],
            "train_symmetry": train_avg["symmetry"],
            "train_stability": train_avg["stability"],
            "train_phase": train_avg["phase"],
            "train_force_mag": train_avg["force_mag"],
            "train_temporal": train_avg["temporal"],
            "val_loss": val_loss,
            "val_contact": val_components["contact"],
            "val_symmetry": val_components["symmetry"],
            "val_stability": val_components["stability"],
            "val_phase": val_components["phase"],
            "val_force_mag": val_components["force_mag"],
            "val_temporal": val_components["temporal"],
            "val_contact_acc": val_metrics["contact_acc"],
            "val_contact_f1": val_metrics["contact_f1"],
            "val_symmetry_r2": val_metrics["symmetry_r2"],
            "val_symmetry_mae": val_metrics["symmetry_mae"],
            "val_stability_acc": val_metrics["stability_acc"],
            "val_stability_f1_unstable": val_metrics["stability_f1_unstable"],
            "val_phase_acc": val_metrics["phase_acc"],
            "val_force_mag_r2": val_metrics["force_mag_r2"],
            "val_z_delta_mean": val_metrics["z_delta_mean"],
            "val_z_delta_p95": val_metrics["z_delta_p95"],
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(log_row)

        print(f"\n=== Epoch {epoch+1}/{args.epochs}  ({elapsed:.1f}s) ===")
        print(f"  Train loss:  {train_avg['total']:.4f}  "
              f"(temporal: {train_avg['temporal']:.4f})")
        print(f"  Val loss:    {val_loss:.4f}  "
              f"(temporal: {val_components['temporal']:.4f})")
        print(f"  Contact:     acc={val_metrics['contact_acc']:.3f}  "
              f"f1={val_metrics['contact_f1']:.3f}")
        print(f"  Symmetry:    R²={val_metrics['symmetry_r2']:.3f}  "
              f"MAE={val_metrics['symmetry_mae']:.3f}")
        print(f"  Stability:   acc={val_metrics['stability_acc']:.3f}  "
              f"f1_unst={val_metrics['stability_f1_unstable']:.3f}")
        print(f"  Phase:       acc={val_metrics['phase_acc']:.3f}")
        print(f"  Force mag:   R²={val_metrics['force_mag_r2']:.3f}")
        print(f"  z_t smooth:  mean_Δ={val_metrics['z_delta_mean']:.4f}  "
              f"p95_Δ={val_metrics['z_delta_p95']:.4f}")
        print(f"  LR:          {scheduler.get_last_lr()[0]:.2e}")

        # --- Save ---
        torch.save(model.encoder.state_dict(), output_dir / "encoder_only.pt")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
        }, output_dir / "full_checkpoint.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.encoder.state_dict(), output_dir / "best_encoder.pt")
            print(f"  ★ New best val_loss={val_loss:.4f} → saved best_encoder.pt")

    print(f"\n{'='*60}")
    print("Training complete")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Outputs in:    {output_dir}")

    train_ds.close()
    val_ds.close()


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/encoder_pretraining")
    parser.add_argument("--output-dir", type=str, default="runs/tactile_encoder_v2")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=16)
    # Loss weights
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--w-symmetry", type=float, default=2.0)
    parser.add_argument("--w-stability", type=float, default=1.0)
    parser.add_argument("--w-phase", type=float, default=0.5)
    parser.add_argument("--w-force-mag", type=float, default=0.3)
    parser.add_argument("--w-temporal", type=float, default=0.5)
    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)