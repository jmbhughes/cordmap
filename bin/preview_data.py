import os
from glob import glob

import pandas as pd

from cordmap.data import SUVIImageDataset


if __name__ == "__main__":
    preview_dir = "/Users/jhughes/Desktop/repos/cordmap/previews/"
    img_dir = "/Users/jhughes/Desktop/repos/cordmap/data/"
    index_path = os.path.join(img_dir, "index.csv")

    d = SUVIImageDataset(index_path, img_dir)
    for i in d.index.index:
        try:
            fig, axs = d.visualize(i)
            fig.savefig(os.path.join(preview_dir, f"{i:04}.png"))
        except:
            print(f"{i} failed.")


