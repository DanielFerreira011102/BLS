import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

from utils.helpers import setup_logging
from rq1.formula.impl.stats import (
    cohens_d,
    overlap_coefficient,
    kl_divergence,
    js_divergence,
    mutual_information,
    statistical_test,
)

# Initialize logging
logger = setup_logging()


class MetricValidator:
    """Validates metrics by computing separation statistics between simple and expert texts."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with DataFrame."""
        self.df = df

    def analyze(self) -> pd.DataFrame:
        """Compute separation stats for all metrics."""
        logger.info("Analyzing metrics")
        if self.df.empty:
            logger.warning("Empty data provided for metric analysis")
            return pd.DataFrame()

        metrics = [c for c in self.df.columns if c != "sample_id" and self.df[c].dtype in (np.float64, np.int64)]
        if not metrics:
            logger.warning("No metrics found for analysis")
            return pd.DataFrame()
        
        stats = [self._metric_stats(m) for m in metrics if self._valid_data(m)]
        df = pd.DataFrame(stats).sort_values("cohens_d", key=abs, ascending=False)
        logger.info(f"Analyzed {len(df)} metrics")
        return df

    def _valid_data(self, metric: str) -> bool:
        """Check if metric has enough data for analysis."""
        simple_scores, expert_scores = self._get_values(metric)
        return len(simple_scores) >= 2 and len(expert_scores) >= 2

    def _get_values(self, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve simple and expert values for a metric."""
        simple_scores = self.df[self.df["text_type"] == "simple"][metric].dropna().values
        expert_scores = self.df[self.df["text_type"] == "expert"][metric].dropna().values
        return simple_scores, expert_scores

    def _metric_stats(self, metric: str) -> Dict[str, Union[str, float]]:
        """Compute separation statistics for a single metric."""
        simple_scores, expert_scores = self._get_values(metric)
        scores = np.concatenate([simple_scores, expert_scores])
        # Create labels: 0 for simple, 1 for expert
        labels = np.array([0] * len(simple_scores) + [1] * len(expert_scores))
        
        # Determine if scores need inversion for ROC
        simple_mean, expert_mean = np.mean(simple_scores), np.mean(expert_scores)
        invert_scores = simple_mean > expert_mean
        scores_for_roc = -scores if invert_scores else scores

        auc = roc_auc_score(labels, scores_for_roc)
        fpr, tpr, thresholds = roc_curve(labels, scores_for_roc)
        j_scores = tpr - fpr  # Youden's J statistic
        best_idx = np.argmax(j_scores)
        best_j = j_scores[best_idx]
        best_threshold = thresholds[best_idx] if not invert_scores else -thresholds[best_idx]

        # Perform statistical test
        stat, pval, test_name = statistical_test(simple_scores, expert_scores)

        return {
            "metric": metric,
            "mean_difference": expert_mean - simple_mean,
            "simple_mean": simple_mean,
            "expert_mean": expert_mean,
            "simple_std": np.std(simple_scores),
            "expert_std": np.std(expert_scores),
            "cohens_d": cohens_d(simple_scores, expert_scores),
            "auc": auc,
            "best_threshold": best_threshold,
            "best_j": best_j,
            "overlap_coefficient": overlap_coefficient(simple_scores, expert_scores),
            "kl_simple_to_expert": kl_divergence(simple_scores, expert_scores),
            "kl_expert_to_simple": kl_divergence(expert_scores, simple_scores),
            "kl_divergence_min": min(kl_divergence(simple_scores, expert_scores),
                                     kl_divergence(expert_scores, simple_scores)),
            "kl_divergence_max": max(kl_divergence(simple_scores, expert_scores),
                                     kl_divergence(expert_scores, simple_scores)),
            "js_divergence": js_divergence(simple_scores, expert_scores),
            "mutual_information": mutual_information(scores, labels),
            "test_used": test_name,
            "test_statistic": stat,
            "p_value": pval
        }


class MetricVisualizer:
    """Generates visualizations for metric validation results."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory for plots."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set the Seaborn style globally
        sns.set_theme(style="whitegrid")

    def plot_distributions(self, df: pd.DataFrame, stats: pd.DataFrame, top_n: int = 5) -> str:
        """Plot distributions for top metrics with statistics."""
        if stats.empty:
            logger.warning("Empty stats; skipping distribution plot.")
            return ""

        top = stats.head(top_n)
        fig, axes = plt.subplots(top_n, 1, figsize=(15, 4 * top_n))
        axes = [axes] if top_n == 1 else axes

        for ax, (_, row) in zip(axes, top.iterrows()):
            m = row["metric"]
            simple_scores = df[df["text_type"] == "simple"][m].dropna()
            expert_scores = df[df["text_type"] == "expert"][m].dropna()

            # Plot histograms using seaborn
            sns.histplot(simple_scores, bins=20, alpha=0.5, color="green", edgecolor="none", label="Simple", ax=ax)
            sns.histplot(expert_scores, bins=20, alpha=0.5, color="red", edgecolor="none", label="Expert", ax=ax)
            
            # Add mean lines
            ax.axvline(np.mean(simple_scores), color="green", linestyle="--", linewidth=1.5, label="Simple Mean")
            ax.axvline(np.mean(expert_scores), color="red", linestyle="--", linewidth=1.5, label="Expert Mean")

            # Add stats annotation
            self._add_stats_annotation(ax, row)

            # Style plot
            self._style_plot(ax, f"{m}", "Value", "Frequency")

        filepath = self.output_dir / "metric_distribution.png"
        self._save_plot(fig, filepath)
        return str(filepath)

    def plot_roc(self, df: pd.DataFrame, stats: pd.DataFrame, top_n: int = 5) -> str:
        """Plot ROC curves for top metrics."""
        if stats.empty:
            logger.warning("Empty stats; skipping ROC plot.")
            return ""

        top = stats.head(top_n)
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot random baseline
        sns.lineplot(x=[0, 1], y=[0, 1], color="k", linestyle="--", 
                    label="Random (AUC = 0.5)", linewidth=1, ax=ax)

        colors = sns.color_palette("husl", n_colors=top_n)
        for i, (_, row) in enumerate(top.iterrows()):
            m = row["metric"]
            simple_scores = df[df["text_type"] == "simple"][m].dropna()
            expert_scores = df[df["text_type"] == "expert"][m].dropna()
            scores = np.concatenate([simple_scores, expert_scores])
            # Create labels: 0 for simple, 1 for expert
            labels = np.array([0] * len(simple_scores) + [1] * len(expert_scores))
            
            # Adjust scores for ROC if needed
            simple_mean, expert_mean = np.mean(simple_scores), np.mean(expert_scores)
            invert_scores = simple_mean > expert_mean
            scores_for_roc = -scores if invert_scores else scores

            fpr, tpr, _ = roc_curve(labels, scores_for_roc)
            sns.lineplot(
                x=fpr,
                y=tpr,
                label=f"{m} (AUC = {row['auc']:.3f})",
                color=colors[i],
                linewidth=1,
                ax=ax
            )

        # Style and save
        self._style_plot(ax, "ROC Curves", "False Positive Rate", "True Positive Rate")
        
        filepath = self.output_dir / "metric_roc.png"
        self._save_plot(fig, filepath)
        return str(filepath)

    def _add_stats_annotation(self, ax, stats):
        """Add statistical annotation to plot."""
        # Determine which statistical test was used
        test_info = (f"M-W p: {stats.get('p_value', 0):.3e}" if stats.get('test_used') == 'mann_whitney_u'
                    else f"t-test p: {stats.get('p_value', 0):.3e}")
        
        # Create text with key statistics
        info_text = (
            f"Mean Diff: {stats.get('mean_difference', 0):.3f}\n"
            f"Cohen's d: {stats.get('cohens_d', 0):.3f}\n"
            f"Overlap: {stats.get('overlap_coefficient', 0):.3f}\n"
            f"AUC: {stats.get('auc', 0):.3f}\n"
            f"JS Div: {stats.get('js_divergence', 0):.3f}\n"
            f"{test_info}"
        )
        
        # Add annotation to the plot
        ax.annotate(
            info_text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=9,
            backgroundcolor="white",
            va="top"
        )

    def _style_plot(self, ax, title, xlabel, ylabel):
        """Apply consistent styling to plots."""
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def _save_plot(self, fig, filepath):
        """Save plot to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot to {filepath}")


class Phase2Pipeline:
    """Runs Phase 2 analysis pipeline with full statistics."""

    def __init__(self, phase1_dir: Union[str, Path], output_dir: Union[str, Path] = None) -> None:
        """Initialize with input and output directories."""
        self.phase1_dir = Path(phase1_dir)
        self.output_dir = Path(output_dir) if output_dir else self.phase1_dir / "phase2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.DataFrame()

    def run(self) -> Dict:
        """Execute pipeline and return statistics."""
        logger.info("Running Phase 2")

        # Step 1: Load data from Phase 1
        self._load()
        if self.df.empty:
            logger.warning("No data loaded")
            return {}

        # Step 2: Analyze metrics
        stats = self._analyze()

        # Step 3: Save results
        self._save_results(stats)

        # Step 4: Visualize results
        self._visualize(stats)

        return {"stats": stats}

    def _load(self) -> None:
        """Load normalized metrics data from Phase 1."""
        norm_path = self.phase1_dir / "normalized_df_z_score.csv"
        logger.info(f"Loading data from {norm_path}")
        
        if not norm_path.exists():
            logger.warning(f"Normalized metrics file not found: {norm_path}")
            return
            
        self.df = pd.read_csv(norm_path)
        logger.info(f"Loaded normalized data with {len(self.df)} rows")

    def _analyze(self) -> pd.DataFrame:
        """Analyze metric distributions."""
        validator = MetricValidator(self.df)
        return validator.analyze()

    def _save_results(self, stats: pd.DataFrame) -> None:
        """Save analysis stats."""
        if stats.empty:
            logger.warning("No stats to save")
            return
        stats_path = self.output_dir / "stats_df.csv"
        stats.to_csv(stats_path, index=False)
        logger.info(f"Saved stats to {stats_path}")

    def _visualize(self, stats: pd.DataFrame) -> None:
        """Generate visualizations for top metrics."""
        if stats.empty:
            logger.warning("No stats to visualize")
            return
        visualizer = MetricVisualizer(self.output_dir / "vis")
        visualizer.plot_distributions(self.df, stats)
        visualizer.plot_roc(self.df, stats)


def main() -> None:
    """Command-line interface for running Phase 2."""
    parser = argparse.ArgumentParser(description="Phase 2 of text complexity analysis pipeline")
    parser.add_argument("--phase1-dir", required=True, help="Directory with Phase 1 results")
    parser.add_argument("--output-dir", help="Directory for Phase 2 outputs")
    args = parser.parse_args()
    
    pipeline = Phase2Pipeline(
        phase1_dir=args.phase1_dir,
        output_dir=args.output_dir
    )
    pipeline.run()


if __name__ == "__main__":
    main()