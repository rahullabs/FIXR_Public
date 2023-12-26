import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from backbone import MammothBackbone
import timm


def mammoth_efficientnet(nclasses: int, model_name: str, pretrained=True, train_bbone=False):
    """
    Instantiates a EfficientNet network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: EfficientNet network
    """
    # print(model_name)
    efp = timm.create_model('tf_efficientnet_b2_ns',pretrained=True)
    # print(efp) to check last later for in_features
    #let's update the pretarined model:
    if not train_bbone:
        for param in efp.parameters():
            param.requires_grad=False
    # print(efp)
    #for tf_efficientnet_b0_ns
    # efp.classifier = nn.Sequential(
    #     nn.Linear(in_features=1280, out_features=512), 
    #     nn.Dropout(p=0.5),
    #     nn.Linear(in_features=512, out_features=nclasses), 
    # )
    #for tf_efficientnet_b2_ns
    efp.classifier = nn.Sequential(
    nn.Linear(in_features=1408, out_features=512), 
    nn.Dropout(p=0.5),
    # nn.Linear(in_features=1408, out_features=1024), 
    # nn.Dropout(p=0.5),
    # nn.Linear(in_features=1024, out_features=512), 
    # nn.Dropout(p=0.5),
    nn.Linear(in_features=512, out_features=nclasses), 
    )
    # print(efp)
    param_list = list(efp.parameters())
    # for idx in range(len(param_list)):
    #         if param_list[idx].requires_grad:
    #             print(param_list[idx])
    # for param in efp.parameters():
    #     print(param.grad)
    return efp



