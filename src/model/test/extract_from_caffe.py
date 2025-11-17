import torch
import numpy as np

key_mapping = {
    "conv1_1.weight": "model.features.0.weight",
    "conv1_1.bias": "model.features.0.bias",
    "conv1_2.weight": "model.features.2.weight",
    "conv1_2.bias": "model.features.2.bias",
    "conv2_1.weight": "model.features.5.weight",
    "conv2_1.bias": "model.features.5.bias",
    "conv2_2.weight": "model.features.7.weight",
    "conv2_2.bias": "model.features.7.bias",
    "conv3_1.weight": "model.features.10.weight",
    "conv3_1.bias": "model.features.10.bias",
    "conv3_2.weight": "model.features.12.weight",
    "conv3_2.bias": "model.features.12.bias",
    "conv3_3.weight": "model.features.14.weight",
    "conv3_3.bias": "model.features.14.bias",
    "conv4_1.weight": "model.features.17.weight",
    "conv4_1.bias": "model.features.17.bias",
    "conv4_2.weight": "model.features.19.weight",
    "conv4_2.bias": "model.features.19.bias",
    "conv4_3.weight": "model.features.21.weight",
    "conv4_3.bias": "model.features.21.bias",
    "conv5_1.weight": "model.features.24.weight",
    "conv5_1.bias": "model.features.24.bias",
    "conv5_2.weight": "model.features.26.weight",
    "conv5_2.bias": "model.features.26.bias",
    "conv5_3.weight": "model.features.28.weight",
    "conv5_3.bias": "model.features.28.bias",
    "fc6.weight": "model.classifier.0.weight",
    "fc6.bias": "model.classifier.0.bias",
    "fc7.weight": "model.classifier.3.weight",
    "fc7.bias": "model.classifier.3.bias",
    "fc8-101.weight": "model.classifier.6.weight",
    "fc8-101.bias": "model.classifier.6.bias",
}


def get_state_dict(pth_file):
    state_dict = torch.load(pth_file)

    new_state_dict = {}

    # Rename state_dict keys
    for old_key, tensor in state_dict.items():
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
        else:
            new_key = old_key

        new_state_dict[new_key] = tensor.to("cuda")

    return new_state_dict
