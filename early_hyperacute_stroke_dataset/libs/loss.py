import copy
from typing import Dict, Callable

import torch
import segmentation_models_pytorch as smp


def init_loss(params: Dict) -> Callable:
    loss_type = params["type"]
    loss_params = copy.deepcopy(params)
    del loss_params["type"]

    if loss_type == "focal_loss":
        focal_loss = smp.losses.FocalLoss(**loss_params)

        def loss_fn(pred, gt):
            masks = torch.argmax(gt, dim=1)

            return focal_loss(pred, masks)

        return loss_fn
    elif loss_type == "dice_loss":
        dice_loss = smp.losses.DiceLoss(**loss_params)

        def loss_fn(pred, gt):
            masks = torch.argmax(gt, dim=1)

            return dice_loss(pred, masks)
        
        return loss_fn
    else:
        raise NotImplementedError()
