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
from cordmap.data import SAMDataset


def train(images, masks, num_epochs=3, model_name="facebook/sam-vit-base"):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    dataset = [{"image": img, "label": m} for img, m in zip(images, masks)]
    train_dataset = SAMDataset(dataset=dataset, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=5, num_workers=5, shuffle=True)

    model = SamModel.from_pretrained(model_name)

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    chosen_channels = ("Product.suvi_l2_ci171", "Product.suvi_l2_ci284", "Product.suvi_l2_ci304")
    chosen_channels = np.array([2, 4, 5])
    
    images = zarr.open("/d0/mhughes/thmap_suvi_train_x.zarr")
    masks = zarr.open("/d0/mhughes/thmap_suvi_train_y.zarr")
    
    images = np.array(images)[:, chosen_channels, :, :]
    masks = np.array(masks) == 3
    
    print(images.shape, masks.shape)
    
    model = train(images, masks, num_epochs=10)
    model.save_pretrained("/home/mhughes/test_model/")

