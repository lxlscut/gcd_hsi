import torch
import torch.nn as nn
from collections import OrderedDict


class GCD(nn.Module):
    def __init__(self, backbone, projector):
        super(GCD, self).__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x):
        h = self.backbone(x)
        x_proj, logits, residual_error = self.projector(h)
        return x_proj, logits, residual_error, h, h


