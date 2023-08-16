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
from cordmap.prompt import get_suvi_prompt_box

THMAP_SIZE = 256


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

class SUVIDataset(Dataset):
    def __init__(self, images, masks, processor):
        self.dataset = [{"image": img, "label": m} for img, m in zip(images, masks)]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

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

DIAM_SUN_ESTIMATE = 760 * 256/1280   # in an output cordmap of (256, 256) shape

# def get_solar_radius(image_set, channel=2, refine=False):
#         """
#         Gets the solar radius from the header of the specified channel
#         :param channel: channel to get radius from
#         :param refine: whether to refine the metadata radius to better approximate the edge
#         :return: solar radius specified in the header
#         """

#         try:
#             solar_radius = DIAM_SUN_ESTIMATE / 2
#             if refine:
#                 composite_img = image_set[channel]#self.images[channel].data
#                 # Determine image size
#                 image_size = np.shape(composite_img)[0]
#                 # Find center and radial mesh grid
#                 center = (image_size / 2) - 0.5
#                 xm, ym = np.meshgrid(np.linspace(0, image_size - 1, num=image_size),
#                                      np.linspace(0, image_size - 1, num=image_size))
#                 xm_c = xm - center
#                 ym_c = ym - center
#                 rads = np.sqrt(xm_c ** 2 + ym_c ** 2)
#                 # Iterate through radii within a range past the solar radius
#                 accuracy = 15
#                 rad_iterate = np.linspace(solar_radius, solar_radius + 50, num=accuracy)
#                 img_avgs = []
#                 for rad in rad_iterate:
#                     # Create a temporary solar image corresponding to the layer
#                     solar_layer = np.zeros((image_size, image_size))
#                     # Find indices in mask of the layer
#                     indx_layer = np.where(rad >= rads)
#                     # Set temporary image corresponding to indices to solar image values
#                     solar_layer[indx_layer] = composite_img[indx_layer]
#                     # Appends average to image averages
#                     img_avgs.append(np.mean(solar_layer))
#                 # Find "drop off" where mask causes average image brightness to drop
#                 diff_avgs = np.asarray(img_avgs[0:accuracy - 1]) - np.asarray(img_avgs[1:accuracy])
#                 # Return the radius that best represents the edge of the sun
#                 solar_radius = rad_iterate[np.where(np.amax(diff_avgs))[0] + 1]
#         except KeyError:
#             raise RuntimeError("Header does not include the solar diameter or radius")
#         else:
#             return solar_radius