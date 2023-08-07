import os
from tqdm import tqdm
from statistics import mean
import torch
from torch.optim import Adam
import monai
from transformers import SamModel
from torch.utils.data import DataLoader
from transformers import SamProcessor
import argparse
import numpy as np
import zarr
from glob import glob 
from cordmap.data import SUVIDataset, THMEMATIC_MAP_THEMES, create_thmap_template
from cordmap.prompt import get_suvi_prompt_box

THEMES_TO_TRAIN = {"filament": 4,
                   "prominence": 5,
                   "coronal_hole": 6,
                   "bright_region": 3,
                   "flare": 9}

class CORDNN:
    def __init__(self, themes_to_train=THEMES_TO_TRAIN):
        self.is_trained = False
        self.themes_to_train = themes_to_train
        self._models = dict()
    
    def train(self, images, masks, num_epochs=3, model_name="facebook/sam-vit-base"):
        for theme, theme_index in self.themes_to_train.items():
            self._models[theme] = CORDNN._train_single_theme(images, 
                                                             masks==theme_index, 
                                                             num_epochs=num_epochs,
                                                             model_name=model_name)
        self.is_trained = True
    
    def predict(self, image):
        predictions = {theme: self._predict_single_theme(image, theme) 
                       for theme in self._models}
        out = create_thmap_template()
        
        for theme, prediction in predictions.items():
            index = THEMES_TO_TRAIN[theme]
            out[prediction == 1] = index
        return out
        
    def save(self, directory: str):
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model")
        for theme, model in self._models.items():
            model.save_pretrained(os.path.join(directory, theme))
            
    @classmethod
    def load(cls, path):
        theme_paths = glob(os.path.join(path, "*"))
        submodels = {os.path.basename(theme_path): SamModel.from_pretrained(theme_path) for theme_path in theme_paths}
        out = cls()  # TODO: handle when you don't train all the themes
        out._models = submodels
        return out
        
    def _predict_single_theme(self, inputs, theme): # image, theme):
        prompt = get_suvi_prompt_box()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # prepare image + box prompt for the model
        #processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        #inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        self._models[theme].eval()

        # forward pass
        with torch.no_grad():
            outputs = self._models[theme](**inputs, multimask_output=False)

        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        return medsam_seg
    
    @staticmethod
    def _train_single_theme(images, masks, num_epochs=3, 
                            model_name="facebook/sam-vit-base"):
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        train_dataset = SUVIDataset(images, masks, processor=processor)

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=5, 
                                    num_workers=5, 
                                    shuffle=True)

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