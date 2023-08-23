from sample_augment.data.synth_data import SynthData
from sample_augment.models.generator import StyleGANGenerator
import torch
import numpy as np

from sample_augment.data.train_test_split import TrainSet


def classifier_guided(training_set: TrainSet, generator_name: str, classifier: TrainedClassifier) -> SynthData:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    label_matrix = training_set.label_tensor.to(device)
    unique_label_combinations, _ = torch.unique(label_matrix, dim=0, return_inverse=True)

    n_generate = 50  # generate 50 synthetic instances per label combination
    n_select = 10  # select 10 best instances per label combination
    n_combinations = len(unique_label_combinations)

    synthetic_imgs_tensor = torch.empty((n_select * n_combinations, 3, 256, 256), dtype=torch.uint8, device=device)
    synthetic_labels_tensor = torch.empty((n_select * n_combinations, label_matrix.size(1)), device=device,
                                          dtype=torch.float32)

    generator = StyleGANGenerator.load_from_name(generator_name)

    for label_idx, label_comb in enumerate(unique_label_combinations):
        c = label_comb.repeat(n_generate, 1)
        synthetic_instances = generator.generate(c=c)
        predicted_scores = classifier.model(synthetic_instances)
        predicted_scores = torch.sigmoid(predicted_scores)  # If needed

        similarity_metric = calculate_similarity(predicted_scores, c)

        # Selecting the top 10 instances based on similarity_metric
        top_idx = similarity_metric.argsort()[-n_select:]
        selected_instances = synthetic_instances[top_idx]

        synthetic_imgs_tensor[label_idx * n_select:(label_idx + 1) * n_select] = selected_instances.permute(0, 3, 1, 2)
        synthetic_labels_tensor[label_idx * n_select:(label_idx + 1) * n_select] = c[:n_select]

    # Move everything to CPU before returning
    synthetic_imgs_tensor = synthetic_imgs_tensor.cpu()
    synthetic_labels_tensor = synthetic_labels_tensor.cpu()

    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor, multi_label=True)


def calculate_similarity(predicted_scores, actual_labels):
    # Implement the similarity or uncertainty metric here based on the predicted_scores and actual_labels
    # For example, you can use Euclidean distance, cosine similarity, or a custom metric
    similarity = np.linalg.norm(predicted_scores.cpu().numpy() - actual_labels.cpu().numpy(), axis=1)
    return similarity
