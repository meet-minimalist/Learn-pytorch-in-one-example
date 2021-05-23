import torch
import numpy as np

class ToTensorOwn(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = image.transpose(2, 0, 1)
        return {'image' : torch.from_numpy(image).type(torch.long), 'label' : torch.tensor(label, dtype=torch.long)}