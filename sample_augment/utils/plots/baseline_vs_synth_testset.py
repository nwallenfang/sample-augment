import pandas as pd
from matplotlib import pyplot as plt

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import create_train_test_val
from sample_augment.models.evaluate_baseline import name_to_color
from sample_augment.models.evaluate_on_test_set import TestSetFullPredictions
from sample_augment.models.generator import GC10_CLASSES_TEXT, GC10_CLASSES
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot
import seaborn as sns


def create_baseline_vs_synth_plot(baseline_reports, synth_reports):
    # COPIED IN PART FROM create_baseline_vs_fullaug_plot()
    all_data = []

    # Collecting baseline data
    for baseline_report in baseline_reports:
        for class_text, class_name in zip(GC10_CLASSES_TEXT, GC10_CLASSES):
            all_data.append({
                "Configuration": "Baseline",
                "Class": class_text,
                "F1 Score": baseline_report[class_name]["f1-score"]
            })

    # Collecting synthetic data
    for synth_report in synth_reports:
        for class_text, class_name in zip(GC10_CLASSES_TEXT, GC10_CLASSES):
            all_data.append({
                "Configuration": "Synthetic",
                "Class": class_text,
                "F1 Score": synth_report[class_name]["f1-score"]
            })

    df = pd.DataFrame(all_data)

    # Define your color palette
    palette = {
        "Baseline": name_to_color['baseline'],
        "Synthetic": name_to_color['synthetic']
    }

    # The plotting code remains largely the same
    sns.set_style("whitegrid")
    prepare_latex_plot()
    plt.figure(figsize=(0.5 * 10, 0.5 * 8))

    ax = sns.boxplot(x="F1 Score", y="Class", hue="Configuration", data=df, palette=palette, linewidth=1.5,
                     whis=[0, 100], saturation=0.6)

    # Custom coloring for outliers
    for i, artist in enumerate(ax.artists):
        col = artist.get_facecolor()
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            if j % 6 == 5:
                line.set_color(col)
                line.set_markeredgecolor(col)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="gray")
    plt.legend(title="Run", loc='best')
    plt.savefig(shared_dir / "figures" / f'baseline_vs_synth.pdf', bbox_inches="tight")


if __name__ == '__main__':
    test_full_predictions = TestSetFullPredictions.from_name('noname_44136f')
    complete_dataset = AugmentDataset.from_name('dataset_f00581')
    bundle = create_train_test_val(complete_dataset, random_seed=100, test_ratio=0.2, val_ratio=0.1, min_instances=10)

    baseline_rep, synth_rep = test_full_predictions.generate_reports(bundle.test)
    create_baseline_vs_synth_plot(baseline_rep, synth_rep)
