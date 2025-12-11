import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_summary(file_path):
    method_name = os.path.basename(file_path).replace("_snr_summary.txt", "")

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

# DataFrame
df = pd.DataFrame(results)
print("\n==================================")
print(df)
print("================================================\n")

df.to_csv("method_comparison.csv", index=False)
print(" method_comparison.csv")

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

    plt.xticks(x, df["Method"], rotation=20)
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison Across Methods")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

    print(f"Saved → {save_name}")


# ------------------------------------------------------
#（SNR / PSNR / SSIM）
# ------------------------------------------------------
plot_metric(df, "Mean SNR", "comparison_snr.png")
plot_metric(df, "Mean PSNR", "comparison_psnr.png")
plot_metric(df, "Mean SSIM", "comparison_ssim.png")

print("over！")
