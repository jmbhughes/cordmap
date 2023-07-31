import os
from glob import glob
import pandas as pd

if __name__ == "__main__":
    index_path = "../data/index.csv"
    new_index_path = "../data/good_index.csv"
    previews = "../previews/*.png"

    index = pd.read_csv(index_path, index_col=0)
    preview_ids = sorted([int(os.path.basename(fn)[:-4]) for fn in glob(previews)])
    index = index[index.index.isin(preview_ids)]
    index.to_csv(new_index_path)