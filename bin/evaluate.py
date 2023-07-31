from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from torch.optim import Adam
import monai
from transformers import SamModel
from torch.utils.data import DataLoader
from transformers import SamProcessor
import argparse
import numpy as np
from torch.utils.data import Dataset
import zarr
import sys
sys.path.append("/home/mhughes/repos/cordmap")
from cordmap.prompt import get_bounding_box
from transformers import SamProcessor
import matplotlib.pyplot as plt


if __name__ == "__main__":
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model = SamModel.from_pretrained("/home/mhughes/test_model/")
  model.to(device)


  images = zarr.open_array("/d0/mhughes/thmap_suvi_valid_x.zarr")
  masks = zarr.open_array("/d0/mhughes/thmap_suvi_valid_y.zarr")
  
  chosen_channels = np.array([2, 4, 5])
  images = np.array(images)[:, chosen_channels, :, :]
  masks = np.array(masks) == 3

  new_dataset = [{"image": img, "label": m} for img, m in zip(images, masks)] 
  
  idx = 100

  image = new_dataset[idx]["image"]
  print(image.shape, np.nanpercentile(image, 95))
  
  # get box prompt based on ground truth segmentation map
  ground_truth_mask = np.array(new_dataset[idx]["label"])
  prompt = get_bounding_box(ground_truth_mask)

  # prepare image + box prompt for the model
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
  inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
  model.eval()

  # forward pass
  with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)


  # apply sigmoid
  medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
  # convert soft mask to hard mask
  medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
  medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

  fig, axs = plt.subplots(ncols=3, figsize=(15, 7))
  axs[1].imshow(np.moveaxis(image, 0, -1))
  axs[1].set_title("Input")
  axs[0].imshow(ground_truth_mask)
  axs[0].set_title('"Ground truth"')
  axs[2].imshow(medsam_seg)
  axs[2].set_title("Prediction")

  for ax in axs:
      ax.set_axis_off()

  fig.savefig("/home/mhughes/repos/cordmap/example.png")
  