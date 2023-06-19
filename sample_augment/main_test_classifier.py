from pathlib import Path

from sample_augment.core import Experiment, Store
from sample_augment.core.step import find_steps
from sample_augment.core.config import read_config


def main():
    pass


if __name__ == '__main__':
    config = read_config(Path(r'C:\Users\Nils\Documents\Masterarbeit\sample-augment\config.json'))

    find_steps(include=['test', 'data', 'models'], exclude=['models.stylegan2'])

    # fixed store with trained classifier from colab
    store = Store.load_from(store_path=Path(
        "C:\\Users\\Nils\\Documents\\Masterarbeit\\sample-augment\\data\\colab_1883c.json"),
        root_directory=Path("C:\\Users\\Nils\\Documents\\Masterarbeit\\sample-augment\\data\\"))

    # create Experiment instance
    experiment = Experiment(config, store=store, save_store=False)
    experiment.run("plot_loss_over_epochs")
