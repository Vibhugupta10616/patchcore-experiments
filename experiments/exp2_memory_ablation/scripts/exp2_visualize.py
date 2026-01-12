"""
Experiment 2: Simple Visualizations
Compares v1 (random) vs v2 (variance-weighted) coreset sampling
"""

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Exp2Visualizer:
    """Simple, clean visualizations for Exp2"""

    def __init__(self, base_path=None):
        if base_path is None:
            base_path = Path(__file__).resolve().parents[1]
        else:
            base_path = Path(base_path)

        self.base = base_path
        self.results_csv = self.base / "results" / "results_all_methods.csv"
        self.vis_dir = self.base / "visualizations"
        self.log_dir = self.base / "logs"

        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.results_csv)

    def plot_comprehensive_analysis(self) -> Path:
        """Comprehensive visualization with all three graphs on one canvas"""
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.35)

        # 1. Main comparison - AUROC vs compression
        ax1 = fig.add_subplot(gs[0, 0])
        for method in ["random_knn", "variance_weighted_knn"]:
            df_m = self.df[self.df["method"] == method]
            grouped = df_m.groupby("coreset_size_ratio")["image_auroc"].mean().sort_index()

            label = "v1: Random K-Center" if method == "random_knn" else "v2: Variance-Weighted"
            color = "#1f77b4" if method == "random_knn" else "#ff7f0e"
            marker = "o" if method == "random_knn" else "s"

            ax1.plot(
                grouped.index * 100,
                grouped.values,
                marker=marker,
                label=label,
                linewidth=2.5,
                markersize=8,
                color=color,
            )

        ax1.set_xlabel("Coreset Size (%)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("AUROC", fontsize=11, fontweight="bold")
        ax1.set_title("v1 vs v2 Performance", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.90, 0.96])

        # 2. Category performance bars
        ax2 = fig.add_subplot(gs[0, 1])
        categories = sorted(self.df["category"].unique())
        v1 = [
            self.df[(self.df["category"] == c) & (self.df["method"] == "random_knn")][
                "image_auroc"
            ].mean()
            for c in categories
        ]
        v2 = [
            self.df[(self.df["category"] == c) & (self.df["method"] == "variance_weighted_knn")][
                "image_auroc"
            ].mean()
            for c in categories
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, v1, width, label="v1: Random", color="#1f77b4", alpha=0.8)
        bars2 = ax2.bar(x + width / 2, v2, width, label="v2: Variance-Weighted", color="#ff7f0e", alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        ax2.set_xlabel("Category", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean AUROC", fontsize=11, fontweight="bold")
        ax2.set_title("Per-Category Performance", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45, ha="right")
        ax2.legend(fontsize=10)
        ax2.set_ylim([0.90, 0.96])
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Improvement percentage
        ax3 = fig.add_subplot(gs[0, 2])
        improvements = []
        for c in categories:
            v1_mean = self.df[(self.df["category"] == c) & (self.df["method"] == "random_knn")][
                "image_auroc"
            ].mean()
            v2_mean = self.df[(self.df["category"] == c) & (self.df["method"] == "variance_weighted_knn")][
                "image_auroc"
            ].mean()
            improvements.append((v2_mean - v1_mean) * 100)

        colors = ["#ff7f0e" if x > 0 else "#1f77b4" for x in improvements]
        bars = ax3.bar(categories, improvements, color=colors, alpha=0.8, edgecolor="black", linewidth=1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

        ax3.axhline(0, color="black", linewidth=1)
        ax3.set_xlabel("Category", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Improvement (%)", fontsize=11, fontweight="bold")
        ax3.set_title("v2 Improvement over v1", fontsize=12, fontweight="bold")
        x_pos = np.arange(len(categories))
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories, rotation=45, ha="right")
        ax3.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Experiment 2: Adaptive Coreset Sampling Analysis", 
                     fontsize=14, fontweight="bold", y=1.00)

        fig.tight_layout()
        path = self.vis_dir / "exp2_comprehensive_analysis.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_main_comparison(self) -> Path:
        """Simple line plot: AUROC vs compression for both methods"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in ["random_knn", "variance_weighted_knn"]:
            df_m = self.df[self.df["method"] == method]
            grouped = df_m.groupby("coreset_size_ratio")["image_auroc"].mean().sort_index()

            label = "v1: Random K-Center" if method == "random_knn" else "v2: Variance-Weighted"
            color = "#1f77b4" if method == "random_knn" else "#ff7f0e"
            marker = "o" if method == "random_knn" else "s"

            ax.plot(
                grouped.index * 100,
                grouped.values,
                marker=marker,
                label=label,
                linewidth=2.5,
                markersize=8,
                color=color,
            )

        ax.set_xlabel("Coreset Size (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("AUROC", fontsize=12, fontweight="bold")
        ax.set_title("Experiment 2: v1 vs v2 Performance", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.90, 0.96])

        fig.tight_layout()
        path = self.vis_dir / "01_main_comparison.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_category_bars(self) -> Path:
        """Simple bar chart: average AUROC per category"""
        fig, ax = plt.subplots(figsize=(10, 5))

        categories = sorted(self.df["category"].unique())
        v1 = [
            self.df[(self.df["category"] == c) & (self.df["method"] == "random_knn")][
                "image_auroc"
            ].mean()
            for c in categories
        ]
        v2 = [
            self.df[(self.df["category"] == c) & (self.df["method"] == "variance_weighted_knn")][
                "image_auroc"
            ].mean()
            for c in categories
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, v1, width, label="v1: Random", color="#1f77b4", alpha=0.8)
        ax.bar(x + width / 2, v2, width, label="v2: Variance-Weighted", color="#ff7f0e", alpha=0.8)

        ax.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean AUROC", fontsize=12, fontweight="bold")
        ax.set_title("Per-Category Performance", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.set_ylim([0.90, 0.96])
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        path = self.vis_dir / "02_category_performance.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_improvement(self) -> Path:
        """Simple bar chart: improvement % of v2 over v1"""
        fig, ax = plt.subplots(figsize=(10, 5))

        categories = sorted(self.df["category"].unique())
        improvements = []

        for c in categories:
            v1_mean = self.df[(self.df["category"] == c) & (self.df["method"] == "random_knn")][
                "image_auroc"
            ].mean()
            v2_mean = self.df[(self.df["category"] == c) & (self.df["method"] == "variance_weighted_knn")][
                "image_auroc"
            ].mean()
            improvements.append((v2_mean - v1_mean) * 100)

        colors = ["#51CF66" if x > 0 else "#FF6B6B" for x in improvements]
        ax.bar(categories, improvements, color=colors, alpha=0.8, edgecolor="black", linewidth=1)
        ax.axhline(0, color="black", linewidth=1)

        ax.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
        ax.set_title("v2 Improvement over v1", fontsize=13, fontweight="bold")
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        path = self.vis_dir / "03_improvement_percentage.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_summary(self) -> Path:
        """Generate simple text summary"""
        lines = [
            "=" * 70,
            "EXPERIMENT 2: ADAPTIVE CORESET SAMPLING - RESULTS SUMMARY",
            "=" * 70,
            "",
            "COMPARISON: Random K-Center (v1) vs Variance-Weighted K-Center (v2)",
            "",
            "OVERALL STATISTICS:",
        ]

        v1_mean = self.df[self.df["method"] == "random_knn"]["image_auroc"].mean()
        v2_mean = self.df[self.df["method"] == "variance_weighted_knn"]["image_auroc"].mean()
        overall_improvement = (v2_mean - v1_mean) * 100

        lines.extend([
            f"  v1 Mean AUROC: {v1_mean:.4f}",
            f"  v2 Mean AUROC: {v2_mean:.4f}",
            f"  Improvement:   {overall_improvement:+.2f}%",
            "",
            "BY CATEGORY:",
        ])

        for cat in sorted(self.df["category"].unique()):
            df_cat = self.df[self.df["category"] == cat]
            v1 = df_cat[df_cat["method"] == "random_knn"]["image_auroc"].mean()
            v2 = df_cat[df_cat["method"] == "variance_weighted_knn"]["image_auroc"].mean()
            imp = (v2 - v1) * 100
            lines.append(f"  {cat:12s}: v1={v1:.4f}  v2={v2:.4f}  improvement={imp:+.2f}%")

        lines.extend([
            "",
            "BY COMPRESSION LEVEL:",
        ])

        for size in sorted(self.df["coreset_size_ratio"].unique()):
            df_size = self.df[self.df["coreset_size_ratio"] == size]
            v1 = df_size[df_size["method"] == "random_knn"]["image_auroc"].mean()
            v2 = df_size[df_size["method"] == "variance_weighted_knn"]["image_auroc"].mean()
            imp = (v2 - v1) * 100
            lines.append(f"  {size*100:5.1f}%: v1={v1:.4f}  v2={v2:.4f}  improvement={imp:+.2f}%")

        lines.extend([
            "",
            "=" * 70,
        ])

        path = self.log_dir / "exp2_summary.txt"
        path.write_text("\n".join(lines))
        return path

    def run_all(self):
        """Generate all visualizations"""
        print("\n" + "=" * 60)
        print("GENERATING EXP2 VISUALIZATIONS")
        print("=" * 60)

        p0 = self.plot_comprehensive_analysis()
        print(f"✓ Comprehensive analysis: {p0.name}")

        p4 = self.generate_summary()
        print(f"✓ Summary report: {p4.name}")

        print("=" * 60)
        print("✅ DONE!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    viz = Exp2Visualizer()
    viz.run_all()
