from sample_augment.core import step
from sample_augment.data.synth_data import SyntheticBundle
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.utils.plot import show_image_tensor


def label_tensor_to_class_names(class_names, label_tensor):
    assert len(class_names) == len(label_tensor), "Length mismatch between class_names and label_tensor"
    return [class_name for class_name, label in zip(class_names, label_tensor) if label == 1]


@step
def check_synth_bundle(bundle: SyntheticBundle):
    strategies = bundle.configs['strategies']
    for d, s in zip(bundle.synthetic_datasets, strategies):
        print(s)
        # log.info("mmmmello!")

    cguided_data = bundle.synthetic_datasets[3]
    proj_data = bundle.synthetic_datasets[2]
    cguided_imgs = cguided_data.synthetic_images
    cguided_lbls = cguided_data.synthetic_labels

    proj_imgs = proj_data.synthetic_images
    proj_lbls = proj_data.synthetic_labels

    # first step, look at like 10 images
    for i in range(100):
        img = proj_imgs[i]

        show_image_tensor(img, f"label: {label_tensor_to_class_names(GC10_CLASSES, proj_lbls[i])}")
