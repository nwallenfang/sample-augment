import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sample_augment.models.evaluate_baseline import name_to_color
from sample_augment.models.evaluate_classifier import KFoldClassificationReport
from sample_augment.models.generator import GC10_CLASSES, GC10_CLASSES_TEXT
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot


def create_baseline_vs_fullaug_plot():
    # Prepare data
    # Data Preparation
    baseline_reports = KFoldClassificationReport.from_name('b00-baseline_98e9bf').reports
    fullaug_reports = KFoldClassificationReport.from_name('b10-full-aug_b30770').reports

    all_data = []
    for baseline_report in baseline_reports:
        for class_text, class_name in zip(GC10_CLASSES_TEXT, GC10_CLASSES):
            all_data.append({
                "Configuration": "baseline",
                "Class": class_text,
                "F1 Score": baseline_report.report[class_name]["f1-score"]
            })
    for fullaug_report in fullaug_reports:
        for class_text, class_name in zip(GC10_CLASSES_TEXT, GC10_CLASSES):
            all_data.append({
                "Configuration": "full-aug",
                "Class": class_text,
                "F1 Score": fullaug_report.report[class_name]["f1-score"]
            })

    df = pd.DataFrame(all_data)

    # Define your color palette
    palette = {
        "baseline": name_to_color['baseline'],
        "full-aug": name_to_color['full-aug']
    }

    # Plotting
    sns.set_style("whitegrid")
    prepare_latex_plot()
    plt.figure(figsize=(0.5 * 10, 0.5 * 8))

    # Create stripplot
    # sns.stripplot(x="F1 Score", y="Class", hue="Configuration", data=df, marker='o',
    #               jitter=True, palette=palette, edgecolor='gray')
    ax = sns.boxplot(x="F1 Score", y="Class", hue="Configuration", data=df, palette=palette, linewidth=1.5,
                     whis=[0, 100], saturation=0.6)
    # Custom coloring for outliers
    for i, artist in enumerate(ax.artists):
        col = artist.get_facecolor()
        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            if j % 6 == 5:  # The last line object associated with a box is its 'fliers' (outliers)
                line.set_color(col)
                line.set_markeredgecolor(col)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="gray")
    plt.legend(title="Run", loc='best')
    plt.savefig(shared_dir / "figures" / f'baseline_vs_fullaug.pdf', bbox_inches="tight")


if __name__ == '__main__':
    create_baseline_vs_fullaug_plot()
