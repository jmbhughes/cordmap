import numpy as np
import zarr
from cordmap.model import CORDNN


if __name__ == "__main__":
    chosen_channels = ("Product.suvi_l2_ci171", 
                       "Product.suvi_l2_ci284", 
                       "Product.suvi_l2_ci304")
    chosen_channels = np.array([2, 4, 5])  # channels = 171, 284, 304
    
    images = zarr.open("/d0/mhughes/thmap_suvi_train_x.zarr")
    masks = zarr.open("/d0/mhughes/thmap_suvi_train_y.zarr")
    
    images = np.array(images)[:, chosen_channels, :, :]
    masks = np.array(masks)# == 3
    
    model = CORDNN()#themes_to_train={'bright_region': 3})
    model.train(images, masks, num_epochs=50)
    model.save("/home/mhughes/full_model_50/")

