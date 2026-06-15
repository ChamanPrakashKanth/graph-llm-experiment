import os
import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

# Ensure matplotlib runs headlessly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vlcm.trainer import load_vlcm_checkpoint
from reasoning_dataset import ReasoningDataset, ReasoningCollator
from torch.utils.data import DataLoader

def main():
    print("=== Loading Checkpoints ===")
    baseline_path = Path("checkpoints/vlcm_mech_baseline/vlcm_epoch_30.pt")
    second_order_path = Path("checkpoints/vlcm_mech_2nd_order/vlcm_epoch_30.pt")

    if not baseline_path.exists() or not second_order_path.exists():
        print(f"Error: Make sure both checkpoints exist:\n  - {baseline_path}\n  - {second_order_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_b = load_vlcm_checkpoint(baseline_path, device=device)
    model_b = checkpoint_b["model"]
    vocab = checkpoint_b["vocab"]
    tokenizer = checkpoint_b["tokenizer"]

    checkpoint_so = load_vlcm_checkpoint(second_order_path, device=device)
    model_so = checkpoint_so["model"]

    dataset_path = "data/mechanical_engineering_dataset.json"
    print(f"=== Loading Dataset: {dataset_path} ===")
    dataset = ReasoningDataset(
        dataset_path,
        max_length=64,
        max_path_length=model_b.config["path_length"],
        tokenizer=tokenizer,
        vocab=vocab,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=ReasoningCollator(),
    )

    pad_id = vocab.pad_id

    # Accumulate metrics
    print("=== Running Inference & Calculating Smoothness Metrics ===")
    totals_b = {"vel_l2": 0.0, "accel_l2": 0.0, "vel_sq": 0.0, "accel_sq": 0.0, "count_vel": 0, "count_accel": 0}
    totals_so = {"vel_l2": 0.0, "accel_l2": 0.0, "vel_sq": 0.0, "accel_sq": 0.0, "count_vel": 0, "count_accel": 0}

    # Store a few interesting samples for plotting
    plot_samples = []
    target_keywords = ["buckle", "thermal stress", "truss", "refrigeration"]

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_paths = batch["path_ids"].to(device)

        # Run Baseline Model
        with torch.no_grad():
            out_b = model_b(input_ids, attention_mask)
            logits_b = out_b["path_logits"]
            probs_b = F.softmax(logits_b, dim=-1)

            # Run Second-Order Model
            out_so = model_so(input_ids, attention_mask)
            logits_so = out_so["path_logits"]
            probs_so = F.softmax(logits_so, dim=-1)

        # Gather samples for plotting
        for i, q_text in enumerate(batch["questions"]):
            if any(kw in q_text.lower() for kw in target_keywords) and len(plot_samples) < 3:
                # Avoid duplicates
                if not any(s["question"] == q_text for s in plot_samples):
                    plot_samples.append({
                        "question": q_text,
                        "target_path": [vocab.id_to_concept.get(pid, "UNKNOWN") for pid in target_paths[i].tolist() if pid != pad_id],
                        "probs_b": probs_b[i].cpu().numpy(),
                        "probs_so": probs_so[i].cpu().numpy(),
                        "pred_b": [vocab.id_to_concept.get(pid, "UNKNOWN") for pid in out_b["predicted_path"][i].tolist() if pid != pad_id],
                        "pred_so": [vocab.id_to_concept.get(pid, "UNKNOWN") for pid in out_so["predicted_path"][i].tolist() if pid != pad_id],
                    })


        # Calculate metrics
        for probs, totals in [(probs_b, totals_b), (probs_so, totals_so)]:
            # velocity (first difference)
            first_diff = probs[:, 1:, :] - probs[:, :-1, :]  # B x (T-1) x V
            vel_l2 = torch.norm(first_diff, p=2, dim=-1)     # B x (T-1)
            vel_sq = (first_diff ** 2).sum(dim=-1)           # B x (T-1)

            # acceleration (second difference)
            second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]  # B x (T-2) x V
            accel_l2 = torch.norm(second_diff, p=2, dim=-1)             # B x (T-2)
            accel_sq = (second_diff ** 2).sum(dim=-1)                   # B x (T-2)

            # Valid step masks
            valid_vel = (target_paths[:, :-1] != pad_id) & (target_paths[:, 1:] != pad_id)
            valid_accel = (target_paths[:, :-2] != pad_id) & (target_paths[:, 1:-1] != pad_id) & (target_paths[:, 2:] != pad_id)

            totals["vel_l2"] += vel_l2[valid_vel].sum().item()
            totals["vel_sq"] += vel_sq[valid_vel].sum().item()
            totals["count_vel"] += valid_vel.sum().item()

            totals["accel_l2"] += accel_l2[valid_accel].sum().item()
            totals["accel_sq"] += accel_sq[valid_accel].sum().item()
            totals["count_accel"] += valid_accel.sum().item()

    # Compute averages
    avg_vel_l2_b = totals_b["vel_l2"] / max(totals_b["count_vel"], 1)
    avg_accel_l2_b = totals_b["accel_l2"] / max(totals_b["count_accel"], 1)
    avg_vel_sq_b = totals_b["vel_sq"] / max(totals_b["count_vel"], 1)
    avg_accel_sq_b = totals_b["accel_sq"] / max(totals_b["count_accel"], 1)

    avg_vel_l2_so = totals_so["vel_l2"] / max(totals_so["count_vel"], 1)
    avg_accel_l2_so = totals_so["accel_l2"] / max(totals_so["count_accel"], 1)
    avg_vel_sq_so = totals_so["vel_sq"] / max(totals_so["count_vel"], 1)
    avg_accel_sq_so = totals_so["accel_sq"] / max(totals_so["count_accel"], 1)

    print("\n=== Trajectory Smoothness Results ===")
    print(f"| Metric | Baseline (weight=0.0) | Second-Order (weight=0.1) | Reduction % |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| **Mean Velocity (L2)** | {avg_vel_l2_b:.5f} | {avg_vel_l2_so:.5f} | {(avg_vel_l2_b - avg_vel_l2_so)/avg_vel_l2_b * 100:.2f}% |")
    print(f"| **Mean Acceleration (L2)** | {avg_accel_l2_b:.5f} | {avg_accel_l2_so:.5f} | {(avg_accel_l2_b - avg_accel_l2_so)/avg_accel_l2_b * 100:.2f}% |")
    print(f"| **Mean Velocity (Squared)** | {avg_vel_sq_b:.5f} | {avg_vel_sq_so:.5f} | {(avg_vel_sq_b - avg_vel_sq_so)/avg_vel_sq_b * 100:.2f}% |")
    print(f"| **Mean Acceleration (Squared)** | {avg_accel_sq_b:.5f} | {avg_accel_sq_so:.5f} | {(avg_accel_sq_b - avg_accel_sq_so)/avg_accel_sq_b * 100:.2f}% |")
    print()

    # === Plot Trajectories ===
    print(f"=== Plotting {len(plot_samples)} Trajectories ===")
    fig, axes = plt.subplots(len(plot_samples), 1, figsize=(10, 4 * len(plot_samples)), squeeze=False)
    
    for idx, sample in enumerate(plot_samples):
        ax = axes[idx, 0]
        probs_b = sample["probs_b"]  # T x V
        probs_so = sample["probs_so"]  # T x V
        
        # Fit PCA on both to get a shared projection space
        combined = np.vstack([probs_b, probs_so])  # 2T x V
        pca = PCA(n_components=2)
        projected = pca.fit_transform(combined)
        
        coords_b = projected[:len(probs_b)]  # T x 2
        coords_so = projected[len(probs_b):]  # T x 2

        # Plot Baseline
        ax.plot(coords_b[:, 0], coords_b[:, 1], "o-", label="Baseline (w=0.0)", color="#ef4444", linewidth=2, markersize=6)
        # Plot Second-Order
        ax.plot(coords_so[:, 0], coords_so[:, 1], "s-", label="Second-Order (w=0.1)", color="#3b82f6", linewidth=2, markersize=6)

        # Annotate nodes
        for t in range(len(probs_b)):
            # Label baseline steps
            pred_lbl_b = sample["pred_b"][t] if t < len(sample["pred_b"]) else "PAD"
            ax.annotate(f"B{t}: {pred_lbl_b}", (coords_b[t, 0], coords_b[t, 1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, color="#ef4444")
            
            # Label second-order steps
            pred_lbl_so = sample["pred_so"][t] if t < len(sample["pred_so"]) else "PAD"
            ax.annotate(f"SO{t}: {pred_lbl_so}", (coords_so[t, 0], coords_so[t, 1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=7, color="#3b82f6")

        # Visual highlights
        ax.set_title(f"Trajectory Comparison: \"{sample['question'][:55]}...\"", fontsize=10, fontweight="bold")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    report_path = Path("reports/trajectory_comparison.png")
    fig.savefig(report_path, dpi=150)
    print(f"Saved trajectory visualization to: {report_path.resolve()}")

if __name__ == "__main__":
    main()
