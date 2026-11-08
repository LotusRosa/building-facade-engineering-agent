"""Source-extracted server-verified core; GPU validation still required for this adapter."""
from __future__ import annotations
import torch
from torch import nn
from torchvision.models import (ConvNeXt_Tiny_Weights, ResNet50_Weights,
                                Swin_T_Weights, convnext_tiny, resnet50, swin_t)

class FullImageClassifier(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        if backbone == "convnext_tiny":
            network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            feature_dim = network.classifier[2].in_features
            network.classifier[2] = nn.Identity()
        elif backbone == "resnet50":
            network = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            feature_dim = network.fc.in_features
            network.fc = nn.Identity()
        elif backbone == "swin_t":
            network = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
            feature_dim = network.head.in_features
            network.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.encoder = network
        self.head = nn.Linear(feature_dim, 3)
        self.feature_dim = feature_dim

    def forward(self, images):
        features = self.encoder(images)
        return self.head(features), features


def _pixels(images, device, hardware, training):
    images = images.to(device, non_blocking=True).to(torch.float32).div_(255.0)
    if training:
        batch = len(images)
        flip = torch.rand(batch, device=device) < .5
        images = torch.where(flip[:, None, None, None], torch.flip(images, (-1,)), images)
        brightness = 1 + (torch.rand(batch, 1, 1, 1, device=device) * 2 - 1) * .12
        contrast = 1 + (torch.rand(batch, 1, 1, 1, device=device) * 2 - 1) * .12
        saturation = 1 + (torch.rand(batch, 1, 1, 1, device=device) * 2 - 1) * .10
        images = images * brightness
        mean = images.mean(dim=(-3, -2, -1), keepdim=True)
        images = (images - mean) * contrast + mean
        gray = images[:, :1] * .2989 + images[:, 1:2] * .5870 + images[:, 2:3] * .1140
        images = ((images - gray) * saturation + gray).clamp_(0, 1)
    mean = images.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = images.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    images = (images - mean) / std
    if hardware["channels_last"]:
        images = images.contiguous(memory_format=torch.channels_last)
    return images
