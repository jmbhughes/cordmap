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


import sys
sys.path.append("/home/mhughes/repos/cordmap")


def get_bounding_box(ground_truth_map):
#   # get bounding box from mask
#     y_indices, x_indices = np.where(ground_truth_map > 0)
#     x_min, x_max = np.min(x_indices), np.max(x_indices)
#     y_min, y_max = np.min(y_indices), np.max(y_indices)
#     # add perturbation to bounding box coordinates
#     H, W = ground_truth_map.shape
#     x_min = max(0, x_min - np.random.randint(0, 20))
#     x_max = min(W, x_max + np.random.randint(0, 20))
#     y_min = max(0, y_min - np.random.randint(0, 20))
#     y_max = min(H, y_max + np.random.randint(0, 20))
#     bbox = [x_min, y_min, x_max, y_max]

    bbox = [24, 24, 1000, 1000]
    return bbox

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


def train(images, masks, num_epochs=3, model_name="facebook/sam-vit-base"):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    dataset = [{"image": img, "label": m} for img, m in zip(images, masks)]
    train_dataset = SAMDataset(dataset=dataset, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    model = SamModel.from_pretrained(model_name)

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
    
    return model

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset")
    # parser.add_argument("num_epochs")
    # parser.add_argument("batch_size")
    # parser.add_argument("learning_rate")
    from cordmap.data import SUVIImageDataset
    import cv2

    chosen_channels = ("Product.suvi_l2_ci171", "Product.suvi_l2_ci284", "Product.suvi_l2_ci304")
    d = SUVIImageDataset("/d0/mhughes/thmap_test/index.csv", "/d0/mhughes/thmap_test/", image_dim=(1024, 1024), channels=chosen_channels)
    print("num images", len(d))
    images = []
    masks = []
    for i in range(25):
        try:
            this_entry = d[i]
        except:
            pass
        else:
            images.append(np.transpose(this_entry[0], [1, 2, 0]))
            masks.append(cv2.resize(this_entry[1], dsize=(256, 256), interpolation=cv2.INTER_NEAREST) == 3)
            
    images = np.array(images)
    masks = np.array(masks)

    images = np.sign(images) * np.power(np.abs(images), 0.25)
    
    factors = dict()
    for i in range(images.shape[-1]):
        low, high = np.percentile(images[:, :, :, i], 3), np.percentile(images[:, :, :, i], 99.99)
        factors[i] = low, high
        
    for (c, (low, high)) in factors.items():
        images[:, :, :, c] = np.clip((images[:, :, :, c] - low) / (high - low) * 255, 0, 255)
        
    images = images.astype(np.uint8)
    
    model = train(images, masks, num_epochs=10)
    print(dir(model))
    model.save_pretrained("/home/mhughes/test_model/")
    
    print("going again!")
    second = train(images, masks, num_epochs=5, model_name="/home/mhughes/test_model/")
