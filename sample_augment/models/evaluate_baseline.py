import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sample_augment.models.evaluate_classifier import k_fold_plot_loss_over_epochs
from sample_augment.utils.path_utils import root_dir
from sample_augment.utils.plot import prepare_latex_plot

figures_dir = root_dir.parent / "experiments" / "figures"

palette = sns.color_palette("tab10", 5)
name_to_color = {
    'baseline': palette[0],
    'low-lr': palette[1],
    'high-lr': palette[1],
    'lr-scheduling': palette[1],
    'lr-scheduling-gamma': palette[1],
    'small-batch': palette[2],
    'large-batch': palette[2],
    'color-aug': palette[4],
    'flip-aug': palette[4],
    'geom-aug': palette[4],
    'full-aug': palette[4],
    'full-aug-lr': palette[4],
    'full-aug-strength': palette[4]
}


def check_best_epoch(names, metrics, _reports):
    all_epochs = []

    color_list = []
    for run_name, run_metrics in zip(names, metrics):
        run_epochs = [fold_metric.epoch for fold_metric in run_metrics]
        color_list.append(name_to_color.get(run_name, palette[4]))
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
    # hardcode fix: only take the first 11 for our plot now
    names = names[:11]
    reports = reports[:11]
    all_f1s = []
    color_list = []

    for run_name, run_reports in zip(names, reports):
        macro_f1_scores = []

        for fold_report in run_reports:
            macro_f1_scores.append(fold_report['macro avg']['f1-score'])
        all_f1s.append(macro_f1_scores)

        # Assign a color to each run_name based on its group
        color_list.append(name_to_color.get(run_name, palette[4]))

    prepare_latex_plot()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Convert your nested list of F1 scores to a DataFrame suitable for Seaborn
    data_list = []
    for name, f1_scores in zip(names, all_f1s):
        for f1 in f1_scores:
            data_list.append({'name': name, 'f1': f1})
    df = pd.DataFrame(data_list)

    sns.swarmplot(x='name', y='f1', data=df, ax=ax, palette=color_list, dodge=False)
    baseline_mean_f1 = np.mean(
        [reports[0][i]['macro avg']['f1-score'] for i in range(5)])

    ax.axhline(baseline_mean_f1, color=name_to_color['baseline'], linestyle='--', linewidth=0.9)

    for i, f1_scores in enumerate(all_f1s):
        mean_f1 = np.mean(f1_scores)
        min_f1 = np.min(f1_scores)
        max_f1 = np.max(f1_scores)
        # draw means
        ax.hlines(mean_f1, xmin=i - 0.2, xmax=i + 0.2, colors=color_list[i], linewidth=2, zorder=5)
        # draw std/errorbar with whiskers
        std_f1 = np.std(f1_scores)
        lower_std = np.clip(mean_f1 - std_f1, min_f1, mean_f1)
        upper_std = np.clip(mean_f1 + std_f1, mean_f1, max_f1)

        # draw std/errorbar with whiskers
        ax.errorbar(i, mean_f1, yerr=[[mean_f1 - lower_std], [upper_std - mean_f1]], fmt='none', color='gray',
                    linewidth=1, capsize=5, zorder=4)

    ax.set_ylim([0.76, 0.86])
    ax.set_xticklabels(names)
    plt.xticks(rotation=45)
    plt.ylabel('Macro Average F1 Score')
    plt.xlabel('Training Run')

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


def calc_auc(names, metrics, reports):
    # AUC
    # if not torch.cuda.is_available():
    #     log.info('no cuda - no auc metric')
    # predictions = predict_validation_set(classifier, val_set, batch_size=32).predictions
    # auc = roc_auc_score(val_set.label_tensor.numpy(), predictions, average='macro', multi_class='ovr')
    # auc_scores.append(auc)

    # augment_dataset: AugmentDataset = AugmentDataset.from_name('dataset_f00581')
    # bundle = create_train_test_val(augment_dataset, baseline_config.random_seed, baseline_config.test_ratio,
    #                                baseline_config.val_ratio, baseline_config.min_instances)
    # _val_set = bundle.val
    # TODO
    pass


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
    # deep_colors = sns.color_palette("tab10", 10)
    # custom_palette = {'baseline': deep_colors[0], 'lr-schedule': deep_colors[1], 'full-aug': deep_colors[3]}

    k_fold_plot_loss_over_epochs({"baseline": metrics_baseline, "lr-scheduling": metrics_lr_schedule}, figures_dir,
                                 "lr_comparison", ax=axes[0], yticks=yticks, ylim=ylims, color_dict=name_to_color)
    k_fold_plot_loss_over_epochs({"baseline": metrics_baseline, "full-aug": metrics_full_aug}, figures_dir,
                                 "aug_comparison", ax=axes[1], yticks=yticks, ylim=ylims, color_dict=name_to_color)

    plt.tight_layout()
    plt.savefig(figures_dir / "losses_lr_fullaug.pdf", bbox_inches='tight')
