import numpy as np
import zarr
import wandb
import torchvision.transforms.v2 as transforms

from cordmap.model import CORDNN


if __name__ == "__main__":
    config = {'chosen_channels': np.array([2, 4, 5]),  # channels = 171, 284, 304
              'training_zarr_x_path': "/d0/mhughes/thmap_suvi_train_x.zarr",
               'training_zarr_y_path': "/d0/mhughes/thmap_suvi_train_y.zarr",
               'validation_zarr_x_path': "/d0/mhughes/thmap_suvi_valid_x.zarr",
               'validation_zarr_y_path': "/d0/mhughes/thmap_suvi_valid_y.zarr", 
               'save_path': "/d0/mhughes/thmap_suvi_models/augmentation_run/",
               'num_epochs': 10,
               'model': 'CORDNN-one SAM per theme'
              }
    
    wandb.init(project="cordmap",
               config=config)
    
    images = zarr.open(config['training_zarr_x_path'])
    masks = zarr.open(config['training_zarr_y_path'])
    
    images = np.array(images)[:, config['chosen_channels'], :, :]
    masks = np.array(masks)
    
    valid_images = zarr.open(config['validation_zarr_x_path'])
    valid_masks = zarr.open(config['validation_zarr_y_path'])
    
    valid_images = np.array(valid_images)[:, config['chosen_channels'], :, :]
    valid_masks = np.array(valid_masks)
    
    augmentations = transforms.Compose(
        [
        transforms.ColorJitter(contrast=0.1, saturation=0.1, brightness=0.1),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ]
    )
    
    model = CORDNN()
    model.train(images, masks, valid_images, valid_masks, 
                num_epochs=config['num_epochs'],
                augmentations=augmentations)
    model.save(config['save_path'])
    
    wandb.finish()

