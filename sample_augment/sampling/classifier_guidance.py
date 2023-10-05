import enum
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from sample_augment.data.synth_data import SynthData
from sample_augment.data.train_test_split import TrainSet
from sample_augment.models.classifier import VisionTransformer
from sample_augment.models.generator import StyleGANGenerator
from sample_augment.models.train_classifier import TrainedClassifier
from sample_augment.models.train_classifier import plain_transforms
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir, shared_dir
from PIL import Image

from sample_augment.utils.plot import multihot_to_classnames


def l2_distance(predicted_scores, actual_labels):
    # potential for selecting based on other metrics such as two-model or entropy-based
    similarity = np.linalg.norm(predicted_scores.cpu().numpy() - actual_labels.cpu().numpy(), axis=1)
    return similarity


def shannon_entropy(predicted_scores, _):  # _, not using actual_labels
    eps = 1e-10  # to avoid log(0)
    entropy = -torch.sum(predicted_scores * torch.log(predicted_scores + eps), dim=1)
    return entropy.cpu().numpy()


class GuidanceMetric(enum.Enum):
    L2Distance = l2_distance
    Entropy = shannon_entropy


def classifier_guided(training_set: TrainSet, generator_name: str, random_seed: int,
                      classifier: TrainedClassifier, guidance_metric: GuidanceMetric) -> SynthData:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    classifier.model.to(device)
    classifier.model.eval()
    label_matrix = training_set.label_tensor.to(device)

    # resize for the vision transformer
    resize = transforms.Resize((224, 224), antialias=True)
    preprocess = transforms.Compose(plain_transforms)

    unique_label_combinations, _ = torch.unique(label_matrix, dim=0, return_inverse=True)

    n_generate = 50  # generate 50 synthetic instances per label combination
    n_select = 10  # select 10 best instances per label combination
    n_combinations = len(unique_label_combinations)

    synthetic_imgs_tensor = torch.empty((n_select * n_combinations, 3, 256, 256), dtype=torch.uint8, device=device)
    synthetic_labels_tensor = torch.empty((n_select * n_combinations, label_matrix.size(1)), device=device,
                                          dtype=torch.float32)

    generator = StyleGANGenerator.load_from_name(generator_name, random_seed)
    visualize_best_worst_images(training_set, generator_name, random_seed, classifier, GuidanceMetric.Entropy)
    visualize_best_worst_images(training_set, generator_name, random_seed, classifier, GuidanceMetric.L2Distance)
    # sys.exit(0)
    for label_idx, label_comb in enumerate(unique_label_combinations):
        c = label_comb.repeat(n_generate, 1)
        synth_raw = generator.generate(c=c).permute(0, 3, 1, 2)
        synth_processed = preprocess(synth_raw)
        if isinstance(classifier.model, VisionTransformer):
            synth_processed = resize(preprocess(synth_processed))
        else:
            synth_processed = preprocess(synth_raw)

        with torch.no_grad():
            scores = classifier.model(synth_processed)
            scores = torch.sigmoid(scores)

        # noinspection PyCallingNonCallable
        metric = guidance_metric(scores, c)

        # selected the top n_select instances based on metric
        top_idx = metric.argsort()[-n_select:]
        selected_instances = synth_raw[top_idx]

        synthetic_imgs_tensor[label_idx * n_select:(label_idx + 1) * n_select] = selected_instances
        synthetic_labels_tensor[label_idx * n_select:(label_idx + 1) * n_select] = c[:n_select]

        # show_image_tensor(synthetic_imgs_tensor[label_idx * n_select], str(label_comb))

    synthetic_imgs_tensor = synthetic_imgs_tensor.cpu()
    synthetic_labels_tensor = synthetic_labels_tensor.cpu()

    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor, multi_label=True)


def visualize_best_worst_images(training_set: TrainSet, generator_name: str, random_seed: int,
                                classifier: TrainedClassifier, guidance_metric: GuidanceMetric,
                                n_combinations_to_plot=3,
                                output_dir=shared_dir / 'generated' / 'cguidance'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 1. Select Label Combinations
    resize = transforms.Resize((224, 224), antialias=True)
    preprocess = transforms.Compose(plain_transforms)

    unique_label_combinations, _ = torch.unique(training_set.label_tensor, dim=0, return_inverse=True)
    names = [(i, multihot_to_classnames(comb.cpu().numpy())) for i, comb in enumerate(unique_label_combinations)]
    print(names)
    selected_combinations = unique_label_combinations[[27, 34, 37]]

    # Define the functions for generating images and computing metric
    def generate_images_and_metric(_label_comb):
        c = _label_comb.repeat(50, 1)  # you defined n_generate = 50
        _synth_raw = StyleGANGenerator.load_from_name(generator_name, random_seed).generate(c=c).permute(0, 3, 1, 2)
        with torch.no_grad():
            scores = classifier.model(resize(preprocess(_synth_raw)))
            scores = torch.sigmoid(scores)

        # noinspection PyCallingNonCallable
        _metric = guidance_metric(scores, c)
        return _synth_raw, _metric

    # 2. Generate Images and Compute Metric
    for idx, label_comb in enumerate(selected_combinations):
        synth_raw, metric = generate_images_and_metric(label_comb)

        np.save(os.path.join(output_dir, f'metric_{guidance_metric.__name__}_comb_{idx}.npy'), metric)

        # 3. Visualize Results
        fig, axs = plt.subplots(2, 6, figsize=(15, 6))
        top_idx = metric.argsort()[-6:]
        bottom_idx = metric.argsort()[:6]

        top_images = []
        bottom_images = []

        # Save Top and Bottom Images and Their Metrics
        for i, image_idx in enumerate(top_idx):
            img_np = synth_raw[image_idx].permute(1, 2, 0).cpu().numpy()
            # print('minmax', np.min(img_np), np.max(img_np))
            # img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(output_dir, f'top_{i + 1}_{guidance_metric.__name__}_comb_{idx}.png'))
            top_images.append(img_np)

        for i, image_idx in enumerate(bottom_idx):
            img_np = synth_raw[image_idx].permute(1, 2, 0).cpu().numpy()
            # img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(output_dir, f'bottom_{i + 1}_{guidance_metric.__name__}_comb_{idx}.png'))
            bottom_images.append(img_np)

        # Save Labels for this Combination
        np.save(os.path.join(output_dir, f'{guidance_metric.__name__}_comb_{idx}.npy'), label_comb.cpu().numpy())

        for i, image_idx in enumerate(top_idx):
            axs[0, i].imshow(synth_raw[image_idx].permute(1, 2, 0).cpu().numpy())
            axs[0, i].set_title(f"Top {i + 1} (metric={metric[image_idx]})")
            axs[0, i].axis('off')

        for i, image_idx in enumerate(bottom_idx):
            axs[1, i].imshow(synth_raw[image_idx].permute(1, 2, 0).cpu().numpy())
            axs[1, i].set_title(f"Bottom {i + 1} (metric={metric[image_idx]})")
            axs[1, i].axis('off')

        plt.suptitle(f'{guidance_metric.__name__} - {label_comb}')

        # 4. Save the Plots
        # plt.show()
        log.info(f'Saving plot {guidance_metric.__name__} :)')
        file_name = f"{guidance_metric.__name__}_label_comb_{idx}.png"
        fig.savefig(f"{str(root_dir.parent / 'experiments/figures')}/{file_name}")
        plt.close(fig)
