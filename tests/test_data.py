import os

import numpy as np

from cordmap.data import SUVIImageDataset

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_create_loader():
    """ Load a small dataset and check the number of images and that each is properly formatted"""
    index_path = os.path.join(TESTDATA_DIR, "index.csv")
    dataset = SUVIImageDataset(index_path, TESTDATA_DIR)
    assert len(dataset) == 4
    for i in range(4):
        cube, label = dataset[i]
        assert isinstance(cube, np.ndarray)
        assert cube.shape == (6, 1280, 1280)
        assert isinstance(label, np.ndarray)
        assert label.shape == (1280, 1280)


