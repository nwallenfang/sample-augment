import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_package.dataset import SamplingAugDataset

from DenseNetClassifier import DenseNet201
from utils.paths import project_path
from pathlib import Path

from lightning.pytorch.strategies import DDPStrategy


def preprocess(dataset):
    data = dataset.tensors[0]

    if data.dtype == torch.uint8:
        # convert to float32
        data = data.float()
        data /= 255.0

    # might need to preprocess to required resolution
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # ImageNet normalization factors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # convert labels to expected Long dtype as well    
    dataset.tensors = (transform(data), dataset.tensors[1].long())

    return dataset


DEBUG = False


def main():
    train_data = SamplingAugDataset.load_from_file(Path(project_path('data_package/interim/gc10_train.pt')))
    val_data = SamplingAugDataset.load_from_file(Path(project_path('data_package/interim/gc10_val.pt')))
    test_data = SamplingAugDataset.load_from_file(Path(project_path('data_package/interim/gc10_test.pt')))

    # train_data is in uint8 format, since that's the expected format for StyleGAN training
    # convert to float format and rescale
    train_data = preprocess(train_data)
    val_data = preprocess(val_data)
    test_data = preprocess(test_data)

    # 1 worker should suffice since the data_package is in RAM already
    train_loader = DataLoader(train_data, num_workers=1, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_data, num_workers=1, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, num_workers=1, pin_memory=True, persistent_workers=True)
    model = DenseNet201()

    conditional_args = {}

    if torch.cuda.is_available() and not DEBUG:
        # Set gloo as DDP backend. The default (NCCL) is unavailable on Windows.
        ddp = DDPStrategy(process_group_backend="gloo")
        conditional_args['strategy'] = ddp

    if DEBUG:
        # detect_anomaly=True, overfit_batches=0.01, log_every_n_steps=8
        conditional_args['detect_anomaly'] = False
        conditional_args['overfit_batches'] = 0.01
        conditional_args['log_every_n_steps'] = 8
        conditional_args['devices'] = 1

    # debugging since the training doesn't seem to converge at all
    # try overfitting tiny set
    # TODO configure output directory
    trainer = pl.Trainer(max_epochs=10, **conditional_args)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(ckpt_path='best', dataloaders=test_loader)


if __name__ == '__main__':
    main()
