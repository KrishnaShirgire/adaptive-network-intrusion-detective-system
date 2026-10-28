
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 1. Load model comparison results
results_path = Path("results/model_comparison.csv")
output_dir = Path("results/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(results_path)

print("Loaded model comparison results:")
print(df.to_string(index=False))


# 2. Define the metrics to compare
metrics = {
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1-score": "F1_score",
    "False-positive rate": "False_positive_rate",
}


# 3. Create comparison graphs for classification metrics
for title, column in metrics.items():
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        df["Model"],
        df[column] * 100,
    )

    ax.set_title(f"Model Comparison: {title}")
    ax.set_ylabel(f"{title} (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, df[column] * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    filename = title.lower().replace("-", "_").replace(" ", "_")
    fig.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# 4. Compare the number of selected features
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    df["Model"],
    df["Selected_features"],
)

ax.set_title("Number of Input Features")
ax.set_ylabel("Number of features")
ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{int(bar.get_height())}",
        ha="center",
        va="bottom",
    )

fig.tight_layout()
fig.savefig(
    output_dir / "feature_count.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# 5. Compare prediction time
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    df["Model"],
    df["Prediction_time_seconds"],
)

ax.set_title("Batch Prediction Time Comparison")
ax.set_ylabel("Time (seconds)")
ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.3f} s",
        ha="center",
        va="bottom",
    )

fig.tight_layout()
fig.savefig(
    output_dir / "prediction_time.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


print("\nAll graphs created successfully!")
print("Graph folder:", output_dir.resolve())
