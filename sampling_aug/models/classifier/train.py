import lightning as pl
import torch
from torch import LongTensor
from torch.utils.data import DataLoader  # Gives easier dataset management
from torch.utils.data import TensorDataset

from data.train_test_split import stratified_split
from models.classifier.densenet201 import DenseNet201
from utils.paths import project_path


def main():
    dataset: TensorDataset = torch.load(project_path('data/interim/datasets/gc10_complete_tensor.pt'))
    train_val_data, test_data = stratified_split(dataset)
    # stratified split returns Subsets, turn train_val_data into a TensorDataset to split it again
    train_data, val_data = stratified_split(TensorDataset(dataset.tensors[0][train_val_data.indices],
                                                          dataset.tensors[1][train_val_data.indices].type(LongTensor)))
    train_loader = DataLoader(train_data, num_workers=2)
    val_loader = DataLoader(val_data, num_workers=1)
    model = DenseNet201()

    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
