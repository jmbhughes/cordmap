from __future__ import annotations
import os
from tqdm import tqdm
from statistics import mean
import torch
from torch.optim import Adam
import monai
from transformers import SamModel
from torch.utils.data import DataLoader
from transformers import SamProcessor
import numpy as np
from glob import glob 
from cordmap.data import SUVIDataset, create_thmap_template

# Theme names and their corresponding index in the the thematic map (according to NOAA)
THEMES_TO_TRAIN = {"filament": 4,
                   "prominence": 5,
                   "coronal_hole": 6,
                   "bright_region": 3,
                   "flare": 9}

class CORDNN:
    """Neural network for creating CORD maps"""
    def __init__(self):
        """Creates a CORDNN object, defaults to untrained with no models"""
        self._is_trained = False
        self._models = dict()
    
    def train(self, images: np.ndarray, masks: np.ndarray, 
              num_epochs: int = 3, model_name: str = "facebook/sam-vit-base"):
        """Trains a set of neural networks, one for each theme. 

        Args:
            images (np.ndarray): input 3 color images to train. shape=(N, 3, 1024, 1024)
            masks (np.ndarray): output masks to predict, shape=(N, 256, 256)
            num_epochs (int, optional): number of epochs to train for. Defaults to 3.
            model_name (str, optional): which SAM model to use as the base. 
                Defaults to "facebook/sam-vit-base".
        """
        for theme, theme_index in THEMES_TO_TRAIN.items():
            self._models[theme] = CORDNN._train_single_theme(images, 
                                                             masks==theme_index, 
                                                             num_epochs=num_epochs,
                                                             model_name=model_name)
        self._is_trained = True
    
    def predict(self, inputs) -> np.ndarray:
        """Predict a CORD map from a given input image 

        Args:
            inputs: a batch of one input, from a pytorch dataloader

        Returns:
            np.ndarray: CORD map! 
        """
        predictions = {theme: self._predict_single_theme(inputs, theme) 
                       for theme in self._models}
        out = create_thmap_template()
        
        for theme, prediction in predictions.items():
            index = THEMES_TO_TRAIN[theme]
            out[prediction == 1] = index
        return out
        
    def save(self, directory: str):
        """Saves a CORDNN to a directory

        Args:
            directory (str): where to save the result

        Raises:
            RuntimeError: if the model is not trained, it cannot save
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model")
        for theme, model in self._models.items():
            model.save_pretrained(os.path.join(directory, theme))
            
    @classmethod
    def load(cls, path: str) -> CORDNN:
        """Loads a CORDNN from file

        Args:
            path (str): the directory containing the model

        Returns:
            CORDNN: a trained model
        """
        theme_paths = glob(os.path.join(path, "*"))
        submodels = {os.path.basename(theme_path): SamModel.from_pretrained(theme_path) 
                     for theme_path in theme_paths}
        out = cls() 
        out._models = submodels
        return out
        
    def _predict_single_theme(self, inputs, theme: str) -> np.ndarray:
        """Predicts a binary mask for a given theme

        Args:
            inputs: a batch of one input, from a pytorch dataloader
            theme (str): which theme to predict

        Returns:
            np.ndarray: a binary mask where 1 is a True and 0 is a False
        """
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
                            model_name="facebook/sam-vit-base") -> SamModel:
        """Trains a model for a given theme

        Args:
            images (np.ndarray): input 3 color images to train. shape=(N, 3, 1024, 1024)
            masks (np.ndarray): output masks to predict, shape=(N, 256, 256)
            num_epochs (int, optional): number of epochs to train for. Defaults to 3.
            model_name (str, optional): which SAM base to use. 
                Defaults to "facebook/sam-vit-base".

        Returns:
            SamModel: PyTorch HuggingFace trained SAM model
        """
        processor = SamProcessor.from_pretrained(model_name)

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

        seg_loss = monai.losses.DiceCELoss(sigmoid=True, 
                                           squared_pred=True, 
                                           reduction='mean')

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