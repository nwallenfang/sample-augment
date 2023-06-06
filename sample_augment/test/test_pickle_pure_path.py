import pickle
import tempfile
from pathlib import PureWindowsPath

from utils.paths import project_path


def test_pickle_pure_path():
    win_path = PureWindowsPath("C:\\Users\\Nils")

    tmp = tempfile.NamedTemporaryFile()

    with tmp as f:
        pickle.dump(win_path, f)
        f.seek(0)
        read_win_path = pickle.load(f)

    print(read_win_path)


def test_load_metafile():
    print()
    _working_path = project_path('data_package/ds_data/gc10_test_meta.pkl')
    with open(_working_path, 'rb') as pickle_file:
        root, ids, paths = pickle.load(pickle_file)
        print(paths[:10])
