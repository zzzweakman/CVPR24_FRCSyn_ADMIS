import copy
import datetime
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.fft

import numpy as np
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel

# from torchjpeg import dct
from torch.nn import functional as F
from torchkit.backbone import get_model
from tasks.partialface.models import Decoder


def make_baseline_template_recon_model(backbone_name, input_size, backbone_ckpt=None):
    backbone_model = get_model(backbone_name)(input_size)
    if backbone_ckpt is not None:
        backbone_model.load_state_dict(torch.load(backbone_ckpt))

    recon_backbone_model = Decoder(512)
    backbone = BaselineTemplateReconModel(backbone=backbone_model, recon_backbone=recon_backbone_model)
    logging.info("{} Backbone with {} Reconstructor Generated".format(backbone_name, 'Decoder'))

    return backbone


class BaselineTemplateReconModel(nn.Module):
    def __init__(self, backbone, recon_backbone):
        super(BaselineTemplateReconModel, self).__init__()
        self.backbone = backbone
        self.recon_backbone = recon_backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.recon_backbone(x)
        return x
