import torch
import numpy as np
from torchvision import transforms


class Normalize(object):
    def __init__(self, model_type):
        assert model_type in ['resnet18', 'simplecnn']
        self.model_type = model_type

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.model_type == 'simplecnn':
            image = image / 255.0
        elif self.model_type == 'resnet18':
            image = image / 255.0
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image.type(torch.float32))
        else:
            print("Normalization type not supported")

        return {'image': image, 'label': label}


