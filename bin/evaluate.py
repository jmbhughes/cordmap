from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor
import numpy as np
import zarr
from transformers import SamProcessor
import matplotlib.pyplot as plt
from cordmap.model import CORDNN
from cordmap.data import SUVIDataset, THEMATIC_MAP_COLORS, THEMATIC_MAP_CMAP

if __name__ == "__main__":
  """Plots results for the validation set for the specified model"""
  chosen_channels = np.array([2, 4, 5])  # channels = 171, 284, 304
  device = "cuda" if torch.cuda.is_available() else "cpu"
  cmap = THEMATIC_MAP_CMAP
  model = CORDNN.load("/home/mhughes/full_model_50/")
  
  # load images and masks
  images = np.array(zarr.open("/d0/mhughes/thmap_suvi_valid_x.zarr"))
  images = images[:, chosen_channels, :, :]
  masks = np.array(zarr.open("/d0/mhughes/thmap_suvi_valid_y.zarr"))
  
  # prepare dataset
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  train_dataset = SUVIDataset(images, masks, processor=processor)
  train_dataloader = DataLoader(train_dataset, 
                              batch_size=1, 
                              shuffle=False)
  
  i = 0
  for batch in tqdm(train_dataloader):
    cm = model.predict(batch)
      
    # Make a figure for each image in the validation set!
    fig, axs = plt.subplots(ncols=3, figsize=(15, 7))
    axs[1].imshow(np.moveaxis(np.array(images[i]), 0, -1))
    axs[1].set_title("Input")
    axs[0].imshow(np.array(batch["ground_truth_mask"]).squeeze(), cmap=cmap,
                        vmin=-1,
                        vmax=len(THEMATIC_MAP_COLORS),
                        interpolation='none')
    axs[0].set_title('"Ground truth"')
    axs[2].imshow(cm, cmap=cmap,
                        vmin=-1,
                        vmax=len(THEMATIC_MAP_COLORS),
                        interpolation='none')
    axs[2].set_title("Prediction")

    for ax in axs:
        ax.set_axis_off()
  
    fig.savefig(f"/home/mhughes/repos/cordmap/full_model_50_results/result_{i:03d}.png")
    plt.close()
    i += 1

  