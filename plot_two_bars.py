import matplotlib.pyplot as plt
import numpy as np

def plot_two_bars(category1_name, value1, category2_name, value2, 
                  ylabel="Value", title="Two Bar Comparison", 
                  save_name="two_bars.png", colors=None):
    """
    Generate a bar chart with two bars.
    
    Args:
        category1_name: Name for the first bar
        value1: Value for the first bar
        category2_name: Name for the second bar
        value2: Value for the second bar
        ylabel: Label for y-axis
        title: Title of the plot
        save_name: Name of the output file
        colors: List of two colors for the bars (default: ["#4a90e2", "#50e3c2"])
    """
    plt.figure(figsize=(8, 5))
    
    categories = [category1_name, category2_name]
    values = [value1, value2]
    
    if colors is None:
        colors = ["#4a90e2", "#50e3c2"]
    
    x = np.arange(len(categories))
    bars = plt.bar(x, values, color=colors)
    
    # Calculate y-axis range to emphasize differences
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    
    # Set y-axis to start slightly below min and extend above max
    # Use a percentage of the range as padding (e.g., 20% of range)
    padding = max(value_range * 0.2, value_range * 0.1) if value_range > 0 else max_value * 0.01
    y_min = 8
    y_max = 10

    
    plt.ylim(y_min, y_max)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.4f}",
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Add difference annotation
    diff = max_value - min_value
    diff_percent = (diff / min_value) * 100 if min_value > 0 else 0
    plt.text(
        0.5, 0.98,
        f"Difference: {diff:.4f} ({diff_percent:.2f}%)",
        transform=plt.gca().transAxes,
        ha='center',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.xticks(x, categories, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print(f"Saved → {save_name}")


if __name__ == "__main__":

    plot_two_bars(
        category1_name="LMD-Net",
        value1=9.4430,
        category2_name="Change the order",
        value2=9.0563,
        ylabel="Mean SNR",
        title="LMD-Net vs Change the order",
        save_name="lmd_net_vs_change_the_order.png"
    )
