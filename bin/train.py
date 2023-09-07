import numpy as np
import zarr
import wandb
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import SamProcessor
import numpy as np
import zarr
from transformers import SamProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from datetime import datetime

from cordmap.model import CORDNN
from cordmap.data import SUVIDataset, THEMATIC_MAP_COLORS, THEMATIC_MAP_CMAP
from cordmap.model import CORDNN


if __name__ == "__main__":
    config = {'chosen_channels': np.array([2, 4, 5]),  # channels = 171, 284, 304
              'zarr_path': "/d0/mhughes/thmap_suvi_daily.zarr",
               'train_months': [1, 2, 3, 4, 5, 6, 7, 8],
               'validation_months': [9, 10],
               'save_path': f"/d0/mhughes/thmap_suvi_models/months_training_quick_{datetime.now()}/",
               'num_epochs': 1,
               'model': 'CORDNN-one SAM per theme'
              }
    
    wandb.init(project="cordmap",
               config=config)
    
    augmentations = transforms.Compose(
        [
        transforms.ColorJitter(contrast=0.1, saturation=0.1, brightness=0.1),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ]
    )
    
    model = CORDNN()
    model.train(config['zarr_path'], config['train_months'], config['validation_months'], 
                num_epochs=config['num_epochs'],
                augmentations=augmentations)
    model.save(config['save_path'])
        
    # Display a single result
    chosen_channels = np.array([2, 4, 5])  # channels = 171, 284, 304
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmap = THEMATIC_MAP_CMAP
    
    # prepare dataset
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    valid_dataset = SUVIDataset(config['zarr_path'], 
                                months=config['validation_months'],
                                processor=processor)
    valid_dataloader = DataLoader(valid_dataset, 
                                batch_size=1, 
                                shuffle=False)
    
    # select a frame to visualize
    iterator = iter(valid_dataloader)
    for i in range(83):
        batch = next(iterator)
        
    result = model.predict_probabilities(batch)
    thmap = model.predict(batch)
    color_image = np.stack([result['coronal_hole'], result['prominence'], result['bright_region']], axis=-1)
    
    # generate the four panel figure
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12))
    axs[0, 0].imshow(np.moveaxis(np.array(batch['image'].squeeze()), 0, -1))
    axs[0, 0].set_title("Input")

    axs[1, 0].imshow(np.array(batch["ground_truth_mask"]).squeeze(), cmap=cmap,
                        vmin=-1,
                        vmax=len(THEMATIC_MAP_COLORS),
                        interpolation='none')
    axs[1, 0].set_title('"Ground truth"')

    axs[1, 1].imshow(thmap, cmap=cmap,
                        vmin=-1,
                        vmax=len(THEMATIC_MAP_COLORS),
                        interpolation='none')
    axs[1, 1].set_title("Prediction")

    axs[0, 1].imshow(color_image)
    axs[0, 1].set_title("Colormap showing probabilities")

    for ax in axs.flatten():
        ax.set_axis_off()   
        
    # generate the probability figure
    fig2, axs = plt.subplots(ncols=3, figsize=(15, 5))
    for ax, theme in zip(axs, ['coronal_hole', 'prominence', 'bright_region']):
        im = ax.imshow(result[theme], vmin=0, vmax=1, cmap='seismic', interpolation='none')
        ax.set_title(theme)
        ax.set_axis_off()
    fig2.colorbar(im, ax=axs, orientation='horizontal')    
    
    wandb.log({'epoch': config['num_epochs'], 
               'four-panel-fig': fig, 
               'probability-fig': fig2}) 
        
    wandb.finish()  

