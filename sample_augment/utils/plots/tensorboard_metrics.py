from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sample_augment.utils.path_utils import root_dir, shared_dir
from sample_augment.utils.plot import prepare_latex_plot
import seaborn as sns


def get_event_accumulator(event_file_path: str):
    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']

    print("Available scalar tags:", scalar_tags)
    return ea


def extract_metrics(ea: EventAccumulator, metric_tag: str):
    """
    Extracts metrics from StyleGAN training from a given TensorBoard EventAccumulator.

    Args:
        ea: Tensorboard EventAccumulator
        metric_tag (str): Tag of the metric to be extracted, e.g., 'Metrics/fid50k_full'.

    Returns:
        tuple: Two lists - steps and values corresponding to the given metric tag.

    """
    scalar_data = ea.Scalars(metric_tag)
    steps = [entry.step for entry in scalar_data]
    values = [entry.value for entry in scalar_data]

    return steps, values


def metric_comparison(steps, metric1, metric2, name):
    # Convert the steps and values into a DataFrame
    # Assuming steps for both metrics are the same; if not, additional preprocessing may be needed.
    data = pd.DataFrame({
        'Training Steps': steps,
        'Real Images': metric1,
        'Fake Images': metric2
    })

    # Plot the data using Seaborn
    plt.figure(figsize=(.85 * 8.88242, .85 * 5.8476))
    prepare_latex_plot()

    sns.lineplot(x='Training Steps', y='value', hue='variable',
                 data=pd.melt(data, ['Training Steps']))
    plt.ylabel('Discriminator Score')
    legend = plt.legend()
    legend.get_title().set_text('Input Source')
    plt.ylim(-5, 5)
    # plt.title('Discriminator Real vs. Fake Scores over Training Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(shared_dir / 'figures' / f'{name}.pdf', bbox_inches="tight")


def plot_single_metric_for_runs(event_files, metric_key, y_label, name, ylim=None):
    plt.figure(figsize=(.85 * 8.88242, .85 * 5.8476))
    prepare_latex_plot()

    for run_name, ea in event_files.items():
        steps = [e.step for e in ea.Scalars(metric_key)]
        metric_values = [e.value for e in ea.Scalars(metric_key)]

        # Convert the steps and values into a DataFrame
        data = pd.DataFrame({
            'Training Steps': steps,
            y_label: metric_values
        })

        sns.lineplot(x='Training Steps', y=y_label, data=data, label=run_name)

    plt.grid(True)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.legend(title="Run")
    plt.savefig(shared_dir / 'figures' / f'{name}.pdf', bbox_inches="tight")


def plot_real_vs_fake_loss(ea: Dict[str, event_accumulator.EventAccumulator]):
    """
    Plots the real vs. fake discriminator loss from a TensorBoard event accumulator.
    """
    # Extract the metrics for both real and fake scores
    steps_real, values_real = extract_metrics(ea['apa-12k-merged'], 'Loss/scores/real')
    steps_fake, values_fake = extract_metrics(ea['apa-12k-merged'], 'Loss/scores/fake')
    metric_comparison(steps_real, values_real, values_fake, 'stylegan_real_vs_fake_scores')


def plot_fid(event_files: Dict[str, EventAccumulator]):
    """
    Plots the FID metric using data from a TensorBoard event file.
    """

    # Setting up seaborn for aesthetics
    sns.set_style("whitegrid")
    # sns.set_context("talk")

    # Your latex configurations.
    prepare_latex_plot()

    plt.figure(figsize=(0.55 * 9.28242, 0.55 * 5.8476))

    # Loop through event_files to plot all the curves on one graph.
    steps = []
    for run_name, ea in event_files.items():
        steps, values = extract_metrics(ea, 'Metrics/fid50k_full')
        steps = [step / 1000 for step in steps]
        # Gaussian smoothing
        smoothed_values = gaussian_filter1d(values, sigma=2)

        # Plot the smoothed curve
        sns.lineplot(x=steps, y=values, label=f"{run_name}", linestyle='-')

        # Plot the original values using sns.scatterplot with adjusted zorder
        # sns.scatterplot(x=steps, y=values, label=run_name, marker='o', zorder=3, legend=False)

    # Adjusting legend to group together
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [1, 0, 3, 2]  # Assuming you have two runs. Adjust order if more runs.
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right", title="Runs",
    #            frameon=True)

    plt.xlim([0, 12.5])
    plt.xlabel('Total Images Processed by Discriminator [millions]')
    plt.ylabel('FID')
    plt.ylim([30, 80])
    plt.legend(loc="upper right", title="Runs", frameon=True)
    plt.tight_layout()
    plt.savefig(shared_dir / 'figures' / 'stylegan_fid.pdf', bbox_inches="tight")


def main():
    event_files = {
        'ada-5k': get_event_accumulator(
            str(root_dir / 'TrainedStyleGAN' / 'ada-018.out.tfevents.1688552824.ipt-d-0432.23820.0')),
        'apa-12k': get_event_accumulator(
            str(root_dir / 'TrainedStyleGAN' / 'wdataaug-028.out.tfevents.1690380831.ipt-d-0432.12244.0')),
        'apa-12k-merged': get_event_accumulator(
            str(root_dir / 'TrainedStyleGAN' / 'unified-030.out.tfevents.1692698037.ipt-d-0432.14936.0')),
        # 'resume-fail': get_event_accumulator(str(root_dir / 'TrainedStyleGAN' / 'resume-fail-024.out.tfevents.1689928298.ipt-d-0432.32864.0'))
    }
    # plot_real_vs_fake_loss(event_files)
    plot_fid(event_files)
    # plot_single_metric_for_runs(event_files, "Loss/pl_penalty", "PL Length Penalty", "style_gan_pl_length", ylim=[0.0, 1.0])
    # plot_single_metric_for_runs(event_files, "Loss/r1_penalty", "R1 Penalty", "stylegan_r1_penalty",
    #                             ylim=[0.0, 1.0])
    # plot_single_metric_for_runs(event_files, "Progress/augment", "Augment", "stylegan_augment_score",
    #                             ylim=[0.0, 1.0])


if __name__ == '__main__':
    main()
