import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sample_augment.models.evaluate_classifier import k_fold_plot_loss_over_epochs
from sample_augment.utils.path_utils import root_dir
from sample_augment.utils.plot import prepare_latex_plot

figures_dir = root_dir.parent / "experiments" / "figures"


def check_best_epoch(names, metrics, _reports):
    all_epochs = []

    sns_colors = sns.color_palette("tab10", 5)
    color_groups = {
        sns_colors[0]: ['baseline'],
        sns_colors[1]: ['low-lr', 'high-lr', 'lr-scheduling', 'lr-scheduling-gamma'],
        sns_colors[2]: ['small-batch', 'large-batch'],
        sns_colors[3]: ['color-aug', 'flip-aug', 'geom-aug', 'full-aug']
    }

    color_list = []
    for run_name, run_metrics in zip(names, metrics):
        run_epochs = []

        for fold_metrics in run_metrics:
            # Assuming `fold_metrics` is an object and epoch is an attribute
            best_epoch = fold_metrics.epoch  # Adjust according to your actual data structure
            run_epochs.append(best_epoch)

        for color, group_names in color_groups.items():
            if run_name in group_names:
                color_list.append(color)
                break

        all_epochs.append(run_epochs)

    # Prepare the plot
    prepare_latex_plot()
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.swarmplot(data=all_epochs, ax=ax, palette=color_list)

    ax.set_xticklabels(names)

    plt.xticks(rotation=45)
    plt.ylabel('Epoch of Best Model (max Val. F1)')
    plt.tight_layout()
    plt.savefig(figures_dir / "best_epoch_comparison.pdf", bbox_inches="tight")


def check_macro_f1(names, _metrics, reports):
    all_f1s = []

    sns_colors = sns.color_palette("tab10", 5)  # Fetch 4 colors from the "deep" palette
    color_groups = {
        sns_colors[0]: ['baseline'],
        sns_colors[1]: ['low-lr', 'high-lr', 'lr-scheduling', 'lr-scheduling-gamma'],
        sns_colors[2]: ['small-batch', 'large-batch'],
        sns_colors[3]: ['color-aug', 'flip-aug', 'geom-aug', 'full-aug']
    }
    color_list = []

    for run_name, run_reports in zip(names, reports):
        macro_f1_scores = []

        for fold_report in run_reports:
            macro_f1_scores.append(fold_report['macro avg']['f1-score'])
        all_f1s.append(macro_f1_scores)

        # Assign a color to each run_name based on its group
        for color, group_names in color_groups.items():
            if run_name in group_names:
                color_list.append(color)
                break

    prepare_latex_plot()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.swarmplot(data=all_f1s, ax=ax, palette=color_list)

    baseline_mean_f1 = np.mean(
        [reports[0][i]['macro avg']['f1-score'] for i in range(5)])

    ax.axhline(baseline_mean_f1, color=sns_colors[0], linestyle='--', linewidth=0.9)

    for i, f1_scores in enumerate(all_f1s):
        mean_f1 = np.mean(f1_scores)
        ax.axhline(mean_f1, xmin=(i + 0.3) / len(all_f1s), xmax=(i + 0.7) / len(all_f1s), color='gray')

    ax.set_ylim([0.76, 0.86])
    ax.set_xticklabels(names)
    plt.xticks(rotation=45)
    plt.ylabel('Macro Average F1 Score')

    plt.tight_layout()
    plt.savefig(figures_dir / "f1_comparison.pdf", bbox_inches="tight")


def check_lr_losses(names, metrics, _reports):
    idx_baseline = names.index('baseline')
    # idx_lr_high = names.index('high-lr')
    idx_lr_schedule = names.index('lr-scheduling')

    metrics_baseline = metrics[idx_baseline]
    # metrics_lr_high = metrics[idx_lr_high]
    metrics_lr_schedule = metrics[idx_lr_schedule]

    k_fold_plot_loss_over_epochs(
        {"baseline": metrics_baseline, "lr-schedule": metrics_lr_schedule},
        figures_dir, "lr_comparison"
    )


def subplot_lr_losses(names, metrics, _reports):
    prepare_latex_plot()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))

    idx_baseline = names.index('baseline')
    idx_lr_schedule = names.index('lr-scheduling')
    idx_full_aug = names.index('full-aug')

    metrics_baseline = metrics[idx_baseline]
    metrics_lr_schedule = metrics[idx_lr_schedule]
    metrics_full_aug = metrics[idx_full_aug]

    yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ylims = [0.0, 0.5]
    deep_colors = sns.color_palette("tab10", 10)
    custom_palette = {'baseline': deep_colors[0], 'lr-schedule': deep_colors[1], 'full-aug': deep_colors[3]}

    k_fold_plot_loss_over_epochs({"baseline": metrics_baseline, "lr-schedule": metrics_lr_schedule}, figures_dir,
                                 "lr_comparison", ax=axes[0], yticks=yticks, ylim=ylims, color_dict=custom_palette)
    k_fold_plot_loss_over_epochs({"baseline": metrics_baseline, "full-aug": metrics_full_aug}, figures_dir,
                                 "aug_comparison", ax=axes[1], yticks=yticks, ylim=ylims, color_dict=custom_palette)

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "losses_lr_fullaug.pdf", bbox_inches='tight')
