import torch
from lightning import LightningModule
from torch import nn
from lightning.pytorch.utilities import grad_norm


class DenseNet201(LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super().__init__()  # use the model pretrained on imagenet
        self.lr = lr
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights='IMAGENET1K_V1')
        self.criterion = nn.CrossEntropyLoss()

        # Freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the classifier part of the model
        # TODO figure out why -> Michel
        self.model.classifier = nn.Sequential(
            nn.Linear(1920, 960),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(960, 240),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(240, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, num_classes))

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.model(images)
        val_loss = self.criterion(predictions, labels)
        self.log("val_loss", val_loss, sync_dist=True)


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # see https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html
        # TODO layer doesn't exist
        norms = grad_norm(self.model.layer, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
