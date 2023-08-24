from torchvision.transforms import transforms

from sample_augment.data.synth_data import SynthData
from sample_augment.models.classifier import VisionTransformer
from sample_augment.models.generator import StyleGANGenerator
from sample_augment.data.train_test_split import TrainSet
from sample_augment.models.train_classifier import TrainedClassifier
import torch
import numpy as np

from sample_augment.models.train_classifier import plain_transforms


def classifier_guided(training_set: TrainSet, generator_name: str, classifier: TrainedClassifier) -> SynthData:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    classifier.model.to(device)
    classifier.model.eval()
    label_matrix = training_set.label_tensor.to(device)

    resize = transforms.Resize((224, 224), antialias=True)
    preprocess = transforms.Compose(plain_transforms)

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
        synth_raw = generator.generate(c=c).permute(0, 3, 1, 2)
        synth_processed = preprocess(synth_raw)
        if isinstance(classifier.model, VisionTransformer):
            synth_processed = resize(preprocess(synth_processed))
        else:
            synth_processed = preprocess(synth_raw)

        with torch.no_grad():  # Context where gradients won't be computed
            scores = classifier.model(synth_processed)
            scores = torch.sigmoid(scores)

        metric = calculate_metric(scores, c)

        # Selecting the top 10 instances based on metric
        top_idx = metric.argsort()[-n_select:]
        selected_instances = synth_raw[top_idx]

        synthetic_imgs_tensor[label_idx * n_select:(label_idx + 1) * n_select] = selected_instances
        synthetic_labels_tensor[label_idx * n_select:(label_idx + 1) * n_select] = c[:n_select]

    # Move everything to CPU before returning
    synthetic_imgs_tensor = synthetic_imgs_tensor.cpu()
    synthetic_labels_tensor = synthetic_labels_tensor.cpu()

    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor, multi_label=True)


def calculate_metric(predicted_scores, actual_labels):
    # Implement the similarity or uncertainty metric here based on the predicted_scores and actual_labels
    # For example, you can use Euclidean distance, cosine similarity, or a custom metric
    similarity = np.linalg.norm(predicted_scores.cpu().numpy() - actual_labels.cpu().numpy(), axis=1)
    return similarity
