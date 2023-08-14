import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D

from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot
import numpy as np

# Sample data: list of tuples (Model Name, GFLOPS, ImageNet Accuracy, No. of Parameters)
data = [
    ("DenseNet201", 4.29, 76.896, 20e6),
    ("EfficientNetV2-L", 56.08, 85.808, 118.5e6),
    ("EfficientNetV2-S", 8.37, 84.228, 21.5e6),
    ("ResNet50", 4.09, 80.858, 25.6e6),
    ("VisionTransformer-B-16", 17.5, 81.072, 86.6e6),
]

# Unzip the data for easier plotting
model_names, gflops, accuracy, params = zip(*data)

# Adjust the scale for parameters so they are visually distinguishable in the plot
dot_size = [p / 10000 for p in params]

sns.set_style("whitegrid")
prepare_latex_plot()

# Create the scatter plot
plt.figure(figsize=(0.55 * 8.88, 0.55 * 4.85))  # 5.8476, 8.88242

plot = sns.scatterplot(x=gflops, y=accuracy, size=dot_size, sizes=(100, 750), hue=model_names, legend=False,
                       palette="Set1", edgecolor="w", linewidth=0.5, zorder=1)

y_max = max(accuracy) + 1  # Add 2% for margin, adjust as needed
plt.ylim(min(accuracy) - 1, y_max)

# Remove hue legend and handle size legend
handles, labels = plot.get_legend_handles_labels()
# Determine where your size legend starts and ends. This will depend on how many unique sizes you have.


# TODO: https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn
texts = []
for i, model_name in enumerate(model_names):
    texts.append(plt.text(gflops[i], accuracy[i], model_name))

# Use adjust_text to resolve overlaps
# , force_points=2.5, force_text=0.8, expand_points=(1.2, 1.2), expand_text=(1.1, 1.1))
adjust_text(texts, force_points=2.22)

# Labels and title
plt.xlabel("GFLOPS")
plt.ylabel("ImageNet TOP-1 Accuracy (%)")
# plt.title("Model Complexity Comparison: GFLOPS vs. ImageNet Accuracy")
plt.tight_layout()

# Show the plot
plt.savefig(shared_dir / 'figures' / 'model_complexity.pdf')
