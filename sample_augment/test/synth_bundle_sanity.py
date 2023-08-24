from typing import List

from sample_augment.data.synth_data import SynthData, SyntheticBundle


def synth_bundle_sanity_check():
    bundle = SyntheticBundle.from_name("synth-comparison-4-strats_acf63b")

    strats: List[str] = bundle.configs['strategies']

    print(strats)
    projection: SynthData = bundle.synthetic_datasets[0]
    classifier_guided: SynthData = bundle.synthetic_datasets[1]

    print(projection.synthetic_labels.shape)
    print(projection.synthetic_images.shape)
    print(classifier_guided.synthetic_images.shape)
    print(classifier_guided.synthetic_labels.shape)


if __name__ == '__main__':
    synth_bundle_sanity_check()
