from typing import Dict, Literal, Union

import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights, \
    efficientnet_v2_l, EfficientNet_V2_L_Weights


class DenseNet201(torchvision.models.DenseNet):
    num_classes: int

    def __init__(self, num_classes, load_pretrained=False):
        # Initialize with densenet201 configuration
        super().__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=1000)  # initially set num_classes to 1000 to match the pre-trained model
        if load_pretrained:
            # use the model pretrained on imagenet
            pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights='IMAGENET1K_V1')
            self.load_state_dict(pretrained.state_dict(), strict=False)
        # Freeze early layers
        for param in self.parameters():
            param.requires_grad = False

        # Modify the classifier part of the model
        self.classifier = nn.Sequential(
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

        self.num_classes = num_classes

    def get_kwargs(self) -> Dict:
        return {'num_classes': self.num_classes}


class ResNet50(torchvision.models.ResNet):
    def __init__(self, num_classes, load_pretrained=False):
        # initially set num_classes to 1000 to match the pre-trained model
        super(ResNet50, self).__init__(block=torchvision.models.resnet.Bottleneck,
                                       layers=[3, 4, 6, 3], num_classes=1000)

        # Freeze early layers
        if load_pretrained:
            # use the model pretrained on imagenet
            pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V2')
            self.load_state_dict(pretrained.state_dict(), strict=False)
        for param in self.parameters():
            param.requires_grad = False

        # Modify the classifier part of the model
        self.fc = nn.Sequential(
            nn.Linear(2048, 960),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(960, 240),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(240, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, num_classes))

        self.num_classes = num_classes

    def get_kwargs(self):
        return {'num_classes': self.num_classes}


class VisionTransformer(torchvision.models.VisionTransformer):
    def __init__(self, num_classes, load_pretrained=False):
        # use same settings as vit_b_16 to get the same architecture
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1000,
            representation_size=None)

        # Load the pretrained weights if requested
        if load_pretrained:
            # use the model pretrained on imagenet
            weights = torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
            # pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'vit_b_16', weights='IMAGENET1K_V1')
            self.load_state_dict(weights.get_state_dict(progress=True), strict=False)

        # Freeze the early layers
        for param in self.parameters():
            param.requires_grad = False

        # Create a custom classifier head
        self.heads = nn.Sequential(
            nn.Linear(768, 960),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(960, 240),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(240, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, num_classes)
        )
        self.num_classes = num_classes

        # Unfreeze the custom classifier head
        for param in self.heads.parameters():
            param.requires_grad = True

    def get_kwargs(self) -> Dict:
        return {'num_classes': self.num_classes}


class EfficientNetV2(nn.Module):
    num_classes: int
    size: Union[Literal['S'], Literal['L']]
    efficient_net: torchvision.models.EfficientNet

    def __init__(self, num_classes, size: Union[Literal['S'], Literal['L']], load_pretrained=False):
        super().__init__()
        self.efficient_net = efficientnet_v2_l()
        if size == 'S':
            # super().__init__(**efficientnet_v2_s().get_kwargs())
            if load_pretrained:
                self.efficient_net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            else:
                self.efficient_net = efficientnet_v2_s()
        elif size == 'L':
            # super().__init__(**efficientnet_v2_l().get_kwargs())
            if load_pretrained:
                self.efficient_net = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                self.efficient_net = efficientnet_v2_l()
        else:
            raise ValueError('invalid size param provided')

        self.size = size

        # Freeze early layers
        for param in self.parameters():
            param.requires_grad = False

        # Modify the classifier, get size of the last channel before classifier
        last_channel_size = self.efficient_net.classifier[1].in_features
        if size == 'S':
            self.efficient_net.classifier = nn.Sequential(
                nn.Linear(last_channel_size, last_channel_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(last_channel_size // 2, last_channel_size // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(last_channel_size // 4, 30),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(30, num_classes)
            )
        else:
            # TODO create a more complex classifier
            raise NotImplementedError()

        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return self.efficient_net.forward(x)

    def get_kwargs(self) -> Dict:
        return {
            'num_classes': self.num_classes,
            'size': self.size
        }
