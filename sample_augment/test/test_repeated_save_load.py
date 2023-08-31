"""
It seems that the training sets shrink after serialization and then loading again.
Look into this
"""
from pathlib import Path

from sample_augment.core import Store
from sample_augment.data.dataset import AugmentDataset
from sample_augment.test.test_stratified_split import create_dummy_dataset


def test_repeated_save_load(tmpdir):
    tmpdir = Path(tmpdir)
    n = 100
    ones = int(0.4 * n)
    dataset = create_dummy_dataset(n, ones, root_dir=Path(tmpdir))
    assert len(dataset) == n
    store = Store(artifacts={'AugmentDataset': dataset})

    store1_path = store.save('save1.json', run_identifier='test')

    store2 = Store.load_from(store1_path)

    dataset2: AugmentDataset = store2.artifacts['AugmentDataset']
    print(len(dataset2))
    assert len(dataset2) == n
