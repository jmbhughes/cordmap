import os
from typing import Tuple, List

import numpy as np
from astropy.io import fits
import pandas as pd
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sunpy.visualization.colormaps.color_tables import suvi_color_table
import astropy.units as u


SUVI_CHANNELS_KEYS = ("Product.suvi_l2_ci094",
                      "Product.suvi_l2_ci131",
                      "Product.suvi_l2_ci171",
                      "Product.suvi_l2_ci195",
                      "Product.suvi_l2_ci284",
                      "Product.suvi_l2_ci304")

THMAP_KEY = "Product.suvi_l2_thmap"

THEMATIC_MAP_COLORS = ["white",
                       "black",
                       "white",  # this index is not actually used for any theme
                       "#F0E442",
                       "#D55E00",
                       "#E69F00",
                       "#009E73",
                       "#0072B2",
                       "#56B4E9",
                       "#CC79A7"]


class SUVIImageDataset(Dataset):
    """ A pytorch dataset for the SUVI composite images with labels of thematic maps"""
    def __init__(self,
                 index_path: str,
                 img_dir: str,
                 image_dim: Tuple[int, int],
                 split_range: Tuple[float, float]):
        """ Create a SUVIImageDataset
        Parameters
        ----------
        index_path : str
            where the index CSV file that groups the images together is
        img_dir : str
            the path to the data folder
        """
        self.index = pd.read_csv(index_path, index_col=0)
        self.img_dir = img_dir

    def __len__(self):
        """ Number of groups in the dataset

        Returns
        -------
        int
            the number of composite image/thematic map groups in the dataset
        """
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
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

    def visualize(self, idx: int) -> None:
        """

        Parameters
        ----------
        idx

        Returns
        -------

        """
        cube, label = self[idx]

        # set up the colorbatl
        cmap = matplotlib.colors.ListedColormap(THEMATIC_MAP_COLORS)

        # do actual plotting
        fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(8, 16))

        for i, key in enumerate(SUVI_CHANNELS_KEYS):
            wavelength = int(key[-3:]) * u.angstrom
            row, col = i // 2, i % 2
            low, high = np.nanpercentile(cube[i], 3), np.nanpercentile(cube[i], 99)
            axs[row, col].imshow(cube[i], origin='lower', cmap=suvi_color_table(wavelength), vmin=low, vmax=high)
            axs[row, col].set_title(key)
            axs[row, col].set_axis_off()

        axs[-1, -1].imshow(label, origin='lower', cmap=cmap, vmin=-1, vmax=len(THEMATIC_MAP_COLORS), interpolation='none')
        axs[-1, -1].set_axis_off()
        axs[-1, -1].set_title("Thematic map")

        axs[-1, 0].set_axis_off()

        plt.suptitle(idx)

        fig.tight_layout()
        return fig, axs


if __name__ == "__main__":
    d = SUVIImageDataset("../data/index.csv", "../data")
    fig, axs = d.visualize(9)
    fig.show()
