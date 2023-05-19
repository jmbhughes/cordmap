import os
from typing import Tuple

import numpy as np
from astropy.io import fits
import pandas as pd
from torch.utils.data import Dataset

SUVI_CHANNELS_KEYS = ("Product.suvi_l2_ci094",
                      "Product.suvi_l2_ci131",
                      "Product.suvi_l2_ci171",
                      "Product.suvi_l2_ci195",
                      "Product.suvi_l2_ci284",
                      "Product.suvi_l2_ci304")

THMAP_KEY = "Product.suvi_l2_thmap"


class SUVIImageDataset(Dataset):
    """ A pytorch dataset for the SUVI composite images with labels of thematic maps"""
    def __init__(self, index_path: str, img_dir: str):
        """ Create a SUVIImageDataset
        Parameters
        ----------
        index_path : str
            where the index CSV file that groups the images together is
        img_dir : str
            the path to the data folder
        """
        self.index = pd.read_csv(index_path)
        self.img_dir = img_dir

    def __len__(self):
        """ Number of groups in the dataset

        Returns
        -------
        int
            the number of composite image/thematic map groups in the dataset
        """
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarry, np.ndarray]:
        """ Loads the group at index=idx

        Parameters
        ----------
        idx : int
            what index value (the first column in the CSV) to load

        Returns
        -------
        image_cube, label as tuple of ndarray
            the image_cube is the stacked composite, label is the thematic map

        """
        # Load the row and convert to full paths
        img_paths = dict(self.index.loc[idx])
        img_paths = {kind: os.path.join(self.img_dir, str(path)) for kind, path in img_paths.items()
                     if kind in SUVI_CHANNELS_KEYS or kind == THMAP_KEY}

        # Open the images
        images = {kind: fits.open(path) for kind, path in img_paths.items()}

        # SUVI images are in HDU=1, pytorch wants the channels as the first dimensino
        image_cube = np.stack([images[key][1].data for key in SUVI_CHANNELS_KEYS], axis=0)
        label = images[THMAP_KEY][0].data  # thematic maps are in HDU = 0
        return image_cube, label
