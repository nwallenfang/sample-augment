import pandas as pd
from matplotlib import pyplot as plt
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
    steps = [entry[1] for entry in scalar_data]
    values = [entry[2] for entry in scalar_data]

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


def plot_real_vs_fake_loss(ea: event_accumulator.EventAccumulator):
    """
    Plots the real vs. fake discriminator loss from a TensorBoard event accumulator.
    """
    # Extract the metrics for both real and fake scores
    steps_real, values_real = extract_metrics(ea, 'Loss/scores/real')
    steps_fake, values_fake = extract_metrics(ea, 'Loss/scores/fake')
    metric_comparison(steps_real, values_real, values_fake, 'stylegan_real_vs_fake_scores')


def plot_fid(ea: EventAccumulator):
    """
    Plots the FID metric using data from a TensorBoard event file.
    """
    steps, values = extract_metrics(ea, 'Metrics/fid50k_full')

    prepare_latex_plot()
    plt.figure(figsize=(0.7 * 8.88242, 0.7 * 5.8476))
    plt.plot(steps, values)
    # plt.title(f'FID over training steps')
    plt.xlabel('Training Steps')
    plt.ylabel('FID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(shared_dir / 'figures' / 'stylegan_fid.pdf', bbox_inches="tight")


def main():
    ea = get_event_accumulator(
        str(root_dir / 'TrainedStyleGAN' / 'wdataaug-028.out.tfevents.1690380831.ipt-d-0432.12244.0'))
    # Sample usage
    plot_real_vs_fake_loss(ea)
    # plot_fid(ea)


if __name__ == '__main__':
    main()
