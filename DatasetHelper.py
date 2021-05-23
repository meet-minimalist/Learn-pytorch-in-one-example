
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import config
from utils.transforms.to_tensor import ToTensorOwn
from utils.transforms.normalize import Normalize
from utils.transforms.resize import PaddedResize
from utils.transforms.augmenter import Augmenter


class CatvsDogDataset(Dataset):
    def __init__(self, img_paths, num_classes, augment=False, basic_transforms=None, augment_transforms=None):
        
        self.img_paths = img_paths

        np.random.shuffle(self.img_paths)

        self.img_path_list = []
        for img_path in self.img_paths:
            if 'cat' in os.path.basename(img_path):
                self.img_path_list.append([img_path, 0])        # For cat label is 0
            elif 'dog' in os.path.basename(img_path):
                self.img_path_list.append([img_path, 1])        # For dog label is 1
            else:
                print("Class not found")
                exit()

        self.num_classes = num_classes
        self.basic_transforms = basic_transforms
        self.augment_transforms = augment_transforms
        self.augment = augment


    def __len__(self):
        return len(self.img_path_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.img_path_list[idx]
        # label can be 0 or 1 based on Cat or Dog respectively

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        sample = {'image' : img, 'label' : label}

        if self.augment:
            sample = self.augment_transforms(sample)
        
        # Mandatory transformations like resizing image to same size, normalizing the image and converting to tensor.
        sample = self.basic_transforms(sample)

        return sample



basic_transforms = transforms.Compose([
                        PaddedResize(size=config.input_size),
                        ToTensorOwn(),             # Custom ToTensor transform, converts to CHW from HWC only
                        Normalize(config.model_type),
                    ])

augment_transforms = Augmenter()


def get_train_loader():
    train_set = CatvsDogDataset(config.train_files, num_classes=config.num_classes, \
                                augment=True, basic_transforms=basic_transforms, augment_transforms=augment_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, \
                                            shuffle=True, num_workers=4, pin_memory=True, \
                                            drop_last=False, prefetch_factor=2, \
                                            # persistent_workers=True)
                                            persistent_workers=False)
    # persistent_workers and pin_memory both cant be set to true at the same time due to some bug.
    # Ref: https://github.com/pytorch/pytorch/issues/48370

    # For windows num_workers should be set to 0 due to some know issue. In ubuntu it works fine.
    # Ref: https://github.com/pytorch/pytorch/issues/4418#issuecomment-354614162
    return train_loader


def get_test_loader():
    test_set = CatvsDogDataset(config.test_files, num_classes=config.num_classes, \
                                augment=False, basic_transforms=basic_transforms)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, \
                                            shuffle=False, num_workers=0, pin_memory=True, \
                                            drop_last=False, prefetch_factor=2, \
                                            # persistent_workers=True)
                                            persistent_workers=False)
    # persistent_workers and pin_memory both cant be set to true at the same time due to some bug.
    # Ref: https://github.com/pytorch/pytorch/issues/48370

    # For windows num_workers should be set to 0 due to some know issue. In ubuntu it works fine.
    # Ref: https://github.com/pytorch/pytorch/issues/4418#issuecomment-354614162
    return test_loader


"""
# Below code is for debugging purpose only.
if __name__ == "__main__":
        
    iterator = iter(train_loader)
    
    for i in range(3):
        batch = next(iterator)

        train_img = batch['image']
        train_label = batch['label']
        print(type(train_img))
        print(type(train_label))
        print(train_img.shape)
        print(train_label.shape)
        
        for img, label in zip(train_img, train_label):
            img = np.transpose(img, (1, 2, 0))
            img = np.uint8(img * 255)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if label == 0:
                label = 'cat'
            else:
                label = 'dog'
            print("Label: ", label)
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        #exit()    
"""        