import os
from pathlib import Path

import numpy as np
import pytest

from cordmap.data import SUVIImageDataset, SUVI_CHANNEL_KEYS

THIS_DIR = Path(__file__).parent
TESTDATA_DIR = str(THIS_DIR / "data")


def test_create_loader():
    """ Load a small dataset and check the number of images and that each is properly formatted"""
    index_path = os.path.join(TESTDATA_DIR, "index.csv")
    test_dataset = SUVIImageDataset(index_path, TESTDATA_DIR)

    assert len(test_dataset) == 4
    for i in range(4):
        cube, label = test_dataset[i]
        assert isinstance(cube, np.ndarray)
        assert cube.shape == (6, 1280, 1280)
        assert isinstance(label, np.ndarray)
        assert label.shape == (1280, 1280)


def test_resize():
    """ Load a small dataset with a set image_dim and confirm the images are resized"""
    index_path = os.path.join(TESTDATA_DIR, "index.csv")
    test_dataset = SUVIImageDataset(index_path, TESTDATA_DIR, image_dim=(1000, 800))

    assert len(test_dataset) == 4
    for i in range(4):
        cube, label = test_dataset[i]
        assert isinstance(cube, np.ndarray)
        assert cube.shape == (6, 1000, 800)
        assert isinstance(label, np.ndarray)
        assert label.shape == (1000, 800)


@pytest.mark.parametrize("channels", [(SUVI_CHANNEL_KEYS[0], ), (SUVI_CHANNEL_KEYS[0], SUVI_CHANNEL_KEYS[1])])
def test_select_channels(channels):
    """ Load a small dataset with a set image_dim and confirm only the requested channels are used"""
    index_path = os.path.join(TESTDATA_DIR, "index.csv")
    test_dataset = SUVIImageDataset(index_path, TESTDATA_DIR, channels=channels)

    assert len(test_dataset) == 4
    for i in range(4):
        cube, label = test_dataset[i]
        assert isinstance(cube, np.ndarray)
        assert cube.shape == (len(channels), 1280, 1280)
        assert isinstance(label, np.ndarray)
        assert label.shape == (1280, 1280)
