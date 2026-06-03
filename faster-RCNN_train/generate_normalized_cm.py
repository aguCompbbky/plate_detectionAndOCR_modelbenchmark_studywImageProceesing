import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_confusion_matrix(cm, save_path, seed, epoch_label="Final", normalize=False):
    """Save a confusion matrix visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    matrix = np.array([
        [cm["TP"], cm["FP"]],
        [cm["FN"], 0]
    ], dtype=float)

    if normalize:
        total = cm["TP"] + cm["FP"] + cm["FN"]
        if total > 0:
            matrix = matrix / total

    labels_text = [["TP", "FP"], ["FN", "TN\n(N/A)"]]

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive (GT)", "Negative (GT)"])
    ax.set_yticklabels(["Positive (Pred)", "Negative (Pred)"])

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            if i == 1 and j == 1:
                text = f"{labels_text[i][j]}\n0"
            else:
                text = f"{labels_text[i][j]}\n{val:.3f}" if normalize else f"{labels_text[i][j]}\n{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if val > (matrix.max() / 2) else "black")

    title_prefix = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(f"{title_prefix} — Seed {seed} ({epoch_label})\n"
                 f"Precision: {cm['precision']:.4f} | Recall: {cm['recall']:.4f}",
                 fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

base_dir = "/home/muk/Masaüstü/python_scripts/plaka2/faster-RCNN_train"
report_path = os.path.join(base_dir, "metrics_report.json")
evidences_dir = os.path.join(base_dir, "evidences")

with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

for seed_str, data in report.get("seeds", {}).items():
    seed = seed_str.replace("seed_", "")
    seed_evidence_dir = os.path.join(evidences_dir, seed_str)
    os.makedirs(seed_evidence_dir, exist_ok=True)
    
    final_cm = data.get("final_confusion_matrix")
    if final_cm:
        save_path = os.path.join(seed_evidence_dir, f"confusion_matrix_normalized_seed{seed}.png")
        save_confusion_matrix(final_cm, save_path, seed, epoch_label="Final", normalize=True)
        print(f"Generated normalized confusion matrix for seed {seed} -> {save_path}")
