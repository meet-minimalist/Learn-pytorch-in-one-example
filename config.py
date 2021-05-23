# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:59:35 2020

@author: Meet
"""

import os
import glob
import torch
import numpy as np
import imgaug

# --------------- Setting Random Seeds ------------------ #
os.environ['PYTHONHASHSEED']=str(42)
os.environ["PL_GLOBAL_SEED"] = str(42) 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8'         
# Added above due to torch.set_deterministic(True) 
# Ref: https://github.com/pytorch/pytorch/issues/47672#issuecomment-725404192

np.random.seed(42)
imgaug.seed(42)         
# Although, imgaug seed and torch seed are set but internally when torch will be using multi threads and 
# We might not be having control over which thread will call imgaug augmenter with which img sequence.
# e.g. For exp-1, img-1, img-2, img-3 will be provided by thread-1, thread-3, thread-2 respectively.
#      In exp-2, img-1, img-2, img-3 might be provided by thread-3, thread-2, thread-1 respectively.
# And imgaug will provide augmentations to these img in same sequence.
# E.g. In exp-1, img-1, img-2, img-3 are provided to imgaug module in sequence 1, 3, 2, then
#      img-1 will face augmentation-1, img-3 will face augmentation-2 and img-2 --> augmentation-3
#      In exp-2, img-1, img-2, img-3 are provided to imgaug module in sequence 3, 2, 1, then
#      img-3 will face augmentation-1, img-2 will face augmentation-2 and img-1 --> augmentation-3
# So complete control over randomness is not achieved due to irregularities/randomness between imgaug and pytorch dataloader

torch.set_deterministic(True)       # This will set deterministic behaviour for cuda operations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)      # sets random seed for cuda for all gpus

torch.backends.cudnn.benchmark = False      # Cuda will not try to find the best possible algorithm implementations, performance might degrade due to this being set to False 
torch.backends.cudnn.deterministic=True     # cuda will use only deterministic implementations
# print("ERROR SEED NOT SET PROPERLY")

# pytorch reproducibility
# Ref: https://stackoverflow.com/q/56354461         
# Ref: https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/

# -------------------------------------------------------- #


# ----------------------- Dataset ------------------------ #
dataset_path    = "M:/Datasets/dogs-vs-cats/train/"
num_classes     = 2

cat_files = glob.glob(dataset_path + "/cat*.jpg")
dog_files = glob.glob(dataset_path + "/dog*.jpg")

np.random.shuffle(cat_files)
np.random.shuffle(dog_files)

train_files = []
test_files = []

train_files.extend(cat_files[:int(len(cat_files)*0.9)])
train_files.extend(dog_files[:int(len(dog_files)*0.9)])

test_files.extend(cat_files[int(len(cat_files)*0.9):])
test_files.extend(dog_files[int(len(dog_files)*0.9):])

np.random.shuffle(train_files)
np.random.shuffle(test_files)
# ------------------------------------------------------- #


# ------------------- Training Routine ------------------ #
use_amp 				= False				 # AMP will give reduced memory footprint and reduced computation time for GPU having Tensor Cores 
model_type              = 'resnet18'        # ['resnet18', 'simplecnn']
freeze_backend          = False				 # Only used when 'resnet18' pretrained model is used	
batch_size              = 256
epochs                  = 60
input_size              = [112, 112]            # [H x W]
l2_weight_decay         = 0.00005
weight_init_method      = 'msra'        # ['msra', 'xavier_normal']       # 'msra' also known as variance scaling initializer and Kaiming He (normal dist) initialization 

# He initialization works better for layers with ReLu activation.
# Xavier initialization works better for layers with sigmoid activation.
# Ref: https://stats.stackexchange.com/a/319849

exp_path = "./summaries/"
os.makedirs(exp_path, exist_ok=True)

train_steps = int(np.ceil(len(train_files) / batch_size))
test_steps  = int(np.ceil(len(test_files) / batch_size))
loss_logging_frequency  = (train_steps // 100) if 0 < (train_steps // 100) < 100 else 1 if (train_steps // 100) == 0 else 100           # At every 100 steps or num_training_steps/3 steps training loss will be printed and summary will be saved.
# ------------------------------------------------------- #


# --------------- Learning Rate --------------- #
warm_up         = True
warm_up_eps     = 2
init_lr         = 0.001
lr_scheduler    = 'exp_decay'       # ['exp_decay', 'cosine_annealing']
lr_exp_decay    = 0.94                     # Only for 'burn_in_decay'. Set this such that at the end of training (= after "epochs" number of iterations), the lr will be of scale 1e-6 or 1e-7.
steps_per_epoch = train_steps
burn_in_steps   = steps_per_epoch * warm_up_eps
# --------------------------------------------- #                    
