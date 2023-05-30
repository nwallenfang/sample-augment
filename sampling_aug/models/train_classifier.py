# import os
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  

import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import CustomTensorDataset

from models.DenseNetClassifier import DenseNet201
from utils.paths import project_path
from pathlib import Path

from lightning.pytorch.strategies import DDPStrategy

def preprocess(dataset):
    # TODO if dtype == uint8
    data = dataset.tensors[0]
    data = data.float()
    data /= 255.0
    
    # might need to preprocess to required resolution
    preprocess = transforms.Compose([
        # transforms.Resize((224, 224)),
        # ImageNet normalization factors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # convert labels to expected Long dtype as well    
    dataset.tensors = (preprocess(data), dataset.tensors[1].long())   
    
    return dataset


def main():
    # dataset: TensorDataset = torch.load(project_path('data/interim/datasets/gc10_complete_tensor.pt'))

    # train_data, val_data, test_data = create_train_val_test_sets(dataset)
    train_data = CustomTensorDataset.load(Path(project_path('data/interim/gc10_train.pt')))
    val_data = CustomTensorDataset.load(Path(project_path('data/interim/gc10_val.pt')))
    # train_data is in uint8 format, since that's the expected format for StyleGAN training
    # convert to float format and rescale
    train_data = preprocess(train_data)
    val_data = preprocess(val_data)

    train_loader = DataLoader(train_data, num_workers=8)  # 1 worker should suffice since the data is in RAM already
    val_loader = DataLoader(val_data, num_workers=8)
    model = DenseNet201()

    conditional_args = {}
    
    if torch.cuda.is_available():
        # Set gloo as DDP backend. The default (NCCL) is unavailable on Windows.
        ddp = DDPStrategy(process_group_backend="gloo")
        conditional_args['strategy'] = ddp

    # debugging since the training doesn't seem to converge at all
    # try overfitting tiny set
    trainer = pl.Trainer(max_epochs=20, detect_anomaly=True, overfit_batches=0.01, **conditional_args)        
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(model.state_dict() )


if __name__ == '__main__':
    main()
