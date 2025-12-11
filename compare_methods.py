import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ============================================================
def parse_summary(file_path):
    method_name = os.path.basename(file_path).replace("_snr_summary.txt", "")

    # 替换 restormer → LMD-Net
    if method_name.lower() == "restormer":
        method_name = "LMD-Net"

    mean_snr = None
    mean_psnr = None
    mean_ssim = None

    with open(file_path, "r") as f:
        for line in f:
            if "Mean SNR" in line:
                mean_snr = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            elif "Mean PSNR" in line:
                mean_psnr = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            elif "Mean SSIM" in line:
                mean_ssim = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

    return {
        "Method": method_name,
        "Mean SNR": mean_snr,
        "Mean PSNR": mean_psnr,
        "Mean SSIM": mean_ssim
    }


summary_files = [
    "dataset/DCE_only_snr_summary.txt",
    "dataset/histogram_snr_summary.txt",
    "dataset/restormer_snr_summary.txt",
    "dataset/richardsonLucy_snr_summary.txt"
]

results = []
for file in summary_files:
    if not os.path.exists(file):
        print(f"❌ no file: {file}")
        continue
    print(f"read: {file}")
    results.append(parse_summary(file))

df = pd.DataFrame(results)
print("\n==================================")
print(df)
print("==================================\n")

df.to_csv("method_comparison.csv", index=False)
print("method_comparison.csv saved\n")


# ============================================================
# ============================================================
def plot_metric(df, metric, save_name):
    plt.figure(figsize=(8, 5))

    x = np.arange(len(df["Method"]))
    y = df[metric].values

    colors = ["#4a90e2", "#50e3c2", "#f5a623", "#d0021b"]
    bars = plt.bar(x, y, color=colors)

    for bar, value in zip(bars, y):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.4f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    xticks = []
    for m in df["Method"]:
        if m == "LMD-Net":
            xticks.append(r"$\mathbf{LMD\text{-}Net}$")
        else:
            xticks.append(m)

    plt.xticks(x, xticks, rotation=20)
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison Across Methods")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print(f"Saved → {save_name}")


# ============================================================
# ============================================================
def plot_radar(df, save_name="comparison_radar.png"):
    metrics = ["Mean SNR", "Mean PSNR", "Mean SSIM"]
    methods = df["Method"].tolist()

    data = df[metrics].values
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for i, method in enumerate(methods):
        values = data_norm[i].tolist() + [data_norm[i][0]]

        lbl = r"$\mathbf{LMD\text{-}Net}$" if method == "LMD-Net" else method

        ax.plot(angles, values, linewidth=2, label=lbl)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.title("Radar Comparison Across Methods")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print("Saved →", save_name)


# ============================================================
# ============================================================
def plot_line(df, save_name="comparison_line.png"):
    metrics = ["Mean SNR", "Mean PSNR", "Mean SSIM"]
    x = np.arange(len(metrics))

    plt.figure(figsize=(8, 5))

    for _, row in df.iterrows():
        label = r"$\mathbf{LMD\text{-}Net}$" if row["Method"] == "LMD-Net" else row["Method"]
        plt.plot(x, row[metrics], marker='o', linewidth=2, label=label)

    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Line Comparison Across Methods")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print("Saved →", save_name)


# ============================================================
# ============================================================
def plot_heatmap(df, save_name="comparison_heatmap.png"):
    metrics = ["Mean SNR", "Mean PSNR", "Mean SSIM"]
    data = df.set_index("Method")[metrics]

    plt.figure(figsize=(6, 4))
    im = plt.imshow(data, cmap="coolwarm", aspect="auto")

    plt.colorbar(im, fraction=0.046, pad=0.04)

    # X
    plt.xticks(range(len(metrics)), metrics, rotation=20)

    # Y  LMD-Net
    yticks = []
    for m in data.index:
        if m == "LMD-Net":
            yticks.append(r"$\mathbf{LMD\text{-}Net}$")
        else:
            yticks.append(m)
    plt.yticks(range(len(data.index)), yticks)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(
                j, i, f"{data.iloc[i, j]:.4f}",
                ha="center", va="center", fontsize=8
            )

    plt.title("Metric Heatmap Across Methods")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print("Saved →", save_name)


# ============================================================
# ============================================================
plot_metric(df, "Mean SNR", "comparison_snr.png")
plot_metric(df, "Mean PSNR", "comparison_psnr.png")
plot_metric(df, "Mean SSIM", "comparison_ssim.png")

plot_radar(df, "comparison_radar.png")
plot_line(df, "comparison_line.png")
plot_heatmap(df, "comparison_heatmap.png")

print("over！")
