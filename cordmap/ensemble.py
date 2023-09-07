from cordmap.model import CORDNN
import torch
import os

class CORDEnsemble:
    def __init__(self, train_month_divisions, validation_month_divisions, random_seeds):
        self.train_month_divisions = train_month_divisions
        self.validation_month_divisions = validation_month_divisions
        self.random_seeds = random_seeds
        self.models = []
        self.is_trained = False
        
    def train(self, zarr_path,
                num_epochs: int = 3, 
                augmentations = None,
                model_name: str = "facebook/sam-vit-base"):
        for train_months, validation_months in zip(self.train_month_divisions, 
                                                   self.validation_month_divisions):
            for seed in self.random_seeds:
                torch.manual_seed(seed)
                this_model = CORDNN()
                this_model.train(zarr_path, train_months, validation_months, 
                                 num_epochs, augmentations, model_name)
                self.models.append(this_model)
        self.is_trained = True
        
    def save(self, path):
        for i, model in enumerate(self.models):
            model.save(os.path.join(path, f"{i:03d}"))
        with open(os.path.join(path, "model.txt"), 'w') as f:
            f.write(f"train_month_divisions: {self.train_month_divisions}\n")
            f.write(f"validation_month_divisions: {self.validation_month_divisions}\n")
            f.write(f"random_seeds: {self.random_seeds}\n")
    
    @classmethod
    def load(cls, path):
        model_dirs = [os.path.join(path, d) for d in sorted(os.listdir(path)) 
                      if os.path.isdir(os.path.join(path, d))]
        ensemble = cls(None, None, None)
        models = []
        for model_dir in model_dirs:
            models.append(CORDNN.load(model_dir))
        ensemble.models = models
        ensemble.is_trained = True
        return ensemble

        