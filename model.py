import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    backbone = models.mobilenet_v3_small(weights=weights)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone
