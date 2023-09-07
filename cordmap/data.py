import os
from typing import Tuple, List, Optional
import numpy as np
from astropy.io import fits
import pandas as pd
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sunpy.visualization.colormaps.color_tables import suvi_color_table
import astropy.units as u
import cv2
import zarr
from datetime import timedelta
from cordmap.prompt import get_suvi_prompt_box

THMAP_SIZE = 256
DIAM_SUN_ESTIMATE = 760 * 256/1280   # in an output cordmap of (256, 256) shape


SUVI_CHANNEL_KEYS = ("Product.suvi_l2_ci094",
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

THEMATIC_MAP_CMAP = matplotlib.colors.ListedColormap(THEMATIC_MAP_COLORS)

THMEMATIC_MAP_THEMES = {'unlabeled': 0,
                        'empty_outer_space': 1,
                        'bright_region': 3,
                        'filament': 4, 
                        'prominence': 5,
                        'coronal_hole': 6, 
                        'quiet_sun': 7, 
                        'limb': 8,
                        'flare': 9}

SCALE_POWER = 0.25


def scale_data(cube: np.ndarray, lows: [float], highs: [float]) -> np.ndarray:
    """Scales data to the 0.25 power, linearizes between 0 and 255, and clips

    Args:
        cube (np.ndarray): cube of observations
        lows (list[floats]): low percentile values for each channel
        highs (list[floats]): high percentile values for each channel

    Returns:
        np.ndarray: 
            scaled data cube
    """    
    cube = np.sign(cube) * np.power(np.abs(cube), SCALE_POWER)
        
    for i, (low, high) in enumerate(zip(lows, highs)):
        cube[i, :, :] = np.clip((cube[i, :, :] - low) / (high - low) * 255, 0, 255)
    return cube.astype(np.uint8)


class SUVIDataset(Dataset):
    def __init__(self, filename: str, 
                 processor, 
                 months: Optional[Tuple[int]] = None, 
                 augmentations=None,
                 x_channels=(2, 4, 5),  # channels = 171, 284, 304
                 zarr_group='data',
                 x_lower_scale=(0, 0.0, 0.26), 
                 x_upper_scale=(1.28, 1.14, 1.82),
                 y_dims=(256, 256),
                 cadence: timedelta = timedelta(days=1), 
                 cadence_epsilon: timedelta = timedelta(minutes=60)):
        self.x_lower_scale = x_lower_scale
        self.x_upper_scale = x_upper_scale
        self.y_dims = y_dims
        self.cadence_window = (cadence - cadence_epsilon, cadence + cadence_epsilon)
        self.zarr_group = zarr_group
        self.data = zarr.open(filename)
        self.channels = np.array(x_channels)
        self.processor = processor
        self.augmentations = augmentations

        # select only observations in the requested months
        if months is None:
            months = list(range(1, 13))
        self.t_obs =  pd.to_datetime(self.data['t_obs'])
        selected_times = np.array([(i, t) for i, t in 
                                   enumerate(self.t_obs) if t.month in months])
        
        # filter to make sure that each x observation has a corresponding valid y
        t_diff = np.median(np.diff(self.t_obs))
        step = int(np.round(np.timedelta64(cadence)/t_diff))
        self.x_index = [i for (i, t) in selected_times 
         if i+step < len(self.t_obs) 
            and (self.t_obs[i+step] - t) > self.cadence_window[0] 
            and (self.t_obs[i+step] - t) < self.cadence_window[1]]

    def __len__(self):
        return len(self.x_index)

    def __getitem__(self, idx):
        xi = self.x_index[idx]
        image = self.data[self.zarr_group][xi][self.channels]
        ground_truth_mask = self.data[self.zarr_group][xi][-1]
        
        # scale the image appropriately
        image = scale_data(image, self.x_lower_scale, self.x_upper_scale)
        
        # make the ground truth the expected size
        ground_truth_mask = cv2.resize(ground_truth_mask, 
                                       dsize=self.y_dims, 
                                       interpolation=cv2.INTER_NEAREST)
        
        # apply any specified augmentations
        if self.augmentations is not None:
            image, ground_truth_mask = self.augmentations(image, ground_truth_mask)

        # get bounding box prompt
        prompt = get_suvi_prompt_box()

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        inputs['image'] = image
        return inputs

    def visualize(self, idx: int) -> Tuple[plt.Figure, plt.Axes]:
        """ Creates a simple plot of the data at idx

        Parameters
        ----------
        idx : int
            what index value (the first column in the CSV) to load

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            the created figure and its axes

        """
        inputs = self[idx]
        cube = inputs["image"]
        label = np.array(inputs["label"])

        # set up the color map
        cmap = matplotlib.colors.ListedColormap(THEMATIC_MAP_COLORS)

        # do actual plotting
        fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(8, 16))

        for i, key in enumerate(SUVI_CHANNEL_KEYS):
            wavelength = int(key[-3:]) * u.angstrom
            row, col = i // 2, i % 2
            axs[row, col].imshow(cube[i], origin='lower', 
                                 cmap=suvi_color_table(wavelength))
            axs[row, col].set_title(key)
            axs[row, col].set_axis_off()

        axs[-1, -1].imshow(label, origin='lower',
                        cmap=cmap,
                        vmin=-1,
                        vmax=len(THEMATIC_MAP_COLORS),
                        interpolation='none')
        axs[-1, -1].set_axis_off()
        axs[-1, -1].set_title("Thematic map")

        axs[-1, 0].set_axis_off()

        plt.suptitle(idx)

        fig.tight_layout()
        return fig, axs

def create_mask(radius, image_size):
    # Define image center
    center_x = (image_size[0] / 2) - 0.5
    center_y = (image_size[1] / 2) - 0.5

    # Create mesh grid of image coordinates
    xm, ym = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    #np.linspace(0, image_size[0] - 1, num=image_size[0]), np.linspace(0, image_size[1] - 1, num=image_size[1]))

    # Center each mesh grid (zero at the center)
    xm_c = xm - center_x
    ym_c = ym - center_y

    # Create array of radii
    rad = np.sqrt(xm_c ** 2 + ym_c ** 2)

    # Create empty mask of same size as the image
    mask = np.zeros((image_size[0], image_size[1]))

    # Apply the mask as true for anything within a radius
    mask[(rad < radius)] = 1

    # Return the mask
    return mask.astype('bool')


def create_thmap_template(limb_thickness=6):
    # Get the solar radius with class function
    solar_radius = DIAM_SUN_ESTIMATE / 2 # get_solar_radius(image_set)

    # Define end of disk and end of limb radii
    disk_radius = solar_radius - (limb_thickness / 2)
    limb_radius = solar_radius + (limb_thickness / 2)

    # Create concentric layers for disk, limb, and outer space
    # First template layer, outer space (value 1) with same size as composites
    imagesize = (THMAP_SIZE, THMAP_SIZE) 
    thmap_data = np.ones(imagesize)
    
    # Mask out the limb (value 8)
    limb_mask = create_mask(limb_radius, imagesize)
    thmap_data[limb_mask] = 8
    
    # Mask out the disk with quiet sun (value 7)
    qs_mask = create_mask(disk_radius, imagesize)
    thmap_data[qs_mask] = 7
    
    return thmap_data
