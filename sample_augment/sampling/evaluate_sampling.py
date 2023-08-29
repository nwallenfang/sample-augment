from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sample_augment.core import step
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.sampling.compare_strategies import SynthComparisonReport
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot


@step
def create_strategy_f1_plot(synth_report: SynthComparisonReport, strategies: List[str]):
    def idx_to_name(_idx):
        return strategies[_idx - 1] if _idx > 0 else "Baseline"

    reports = [  # just poor-naming-conventions shenanigans happening here
        synth_report.baseline_report.report,
        *[synth.report for synth in synth_report.synth_reports]
    ]

    all_data = []
    for idx, report in enumerate(reports):
        for class_name in GC10_CLASSES:
            all_data.append({
                "Training Regime": idx_to_name(idx),
                "Class": class_name,
                "F1 Score": report[class_name]["f1-score"]
            })

    df = pd.DataFrame(all_data)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    prepare_latex_plot()
    palette = ["grey"] + list(sns.color_palette("deep", n_colors=len(reports) - 1))

    # Use stripplot for the baseline-configs with '|' marker
    baseline_data = df[df["Training Regime"] == "Baseline"]
    sns.stripplot(x="F1 Score", y="Class", data=baseline_data, marker='|', jitter=False, size=15,
                  color=palette[0], linewidth=2)

    # Plot the other regimes using stripplot with 'o' marker
    for idx, regime in enumerate(df["Training Regime"].unique()[1:]):
        regime_data = df[df["Training Regime"] == regime]
        sns.stripplot(x="F1 Score", y="Class", data=regime_data, marker='o', jitter=False,
                      color=palette[idx + 1], size=8, linewidth=1.0, edgecolor="gray")

    handles = [plt.Line2D([0], [0], color=palette[i], marker=('|' if i == 0 else 'o'),
                          markersize=(15 if i == 0 else 8), linestyle='', label=idx_to_name(i),
                          linewidth=(2 if i == 0 else 1))
               for i in range(len(reports))]
    plt.legend(handles=handles, title="Strategy", loc='best')
    # plt.title("Comparison of Class-wise F1 scores for Different Training Regimes")
    plt.savefig(shared_dir / "figures" / f'strategies_f1_{synth_report.configs["name"]}.pdf', bbox_inches="tight")
