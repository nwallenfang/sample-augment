from pathlib import Path

from sample_augment.core import Experiment, Store
from sample_augment.core.step import find_steps
from sample_augment.core.config import read_config


if __name__ == '__main__':
    # config = read_config(Path(r'C:\Users\Nils\Documents\Masterarbeit\sample-augment\config.json'))

    find_steps(include=['test', 'data', 'models'], exclude=['models.stylegan2'])

    # fixed store with trained classifier from colab
    store, stored_config = Store.load_from(store_path=Path(
        "C:\\Users\\Nils\\Documents\\Masterarbeit\\sample-augment\\data\\colab_2b2dc.json"),
        root_directory=Path("C:\\Users\\Nils\\Documents\\Masterarbeit\\sample-augment\\data\\"))

    # create Experiment instance
    experiment = Experiment(stored_config, store=store, save_store=True)
    # experiment.run("run_metrics_on_predictions_file")
    experiment.run("check_class_distributions")

