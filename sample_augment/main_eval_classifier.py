import json
from pathlib import Path

from sample_augment.models.evaluate_classifier import k_fold_plot_loss_over_epochs
from sample_augment.models.train_classifier import KFoldTrainedClassifiers

if __name__ == '__main__':
    kfold_trained_path = Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data"
                              r"\KFoldTrainedClassifiers\aug-01_ecc814.json")
    with open(kfold_trained_path) as kfold_trained_file:
        kfold_trained_data = json.load(kfold_trained_file)

    k_fold_plot_loss_over_epochs(KFoldTrainedClassifiers.from_dict(kfold_trained_data),
                                 Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\shared"),
                                 "aug-01")
