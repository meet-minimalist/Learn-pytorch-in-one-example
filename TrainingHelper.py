
import os
import time
import torch
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import config
from LRHelper import LRHelper
from DatasetHelper import get_train_loader, get_test_loader

from utils.Logger import Logger
from utils.CheckpointHandler import CheckpointHandler
from utils.SummaryHelper import SummaryHelper
from utils.misc import init_training, np_cpu, LossAverager

cuda = torch.device('cuda:0')
cpu = torch.device("cpu:0")


class TrainingHelper:
    def __init__(self):
        self.log, self.exp_path = init_training()
        self.lr_helper = LRHelper()

        ckpt_folder = self.exp_path + "/ckpt/"
        os.makedirs(ckpt_folder, exist_ok=True)
        
        ckpt_path = ckpt_folder + "model.pth"
        self.ckpt_handler = CheckpointHandler(ckpt_path, max_to_keep=3)


    def get_loss_and_accuracy(self, labels, logits, model, ce_loss_fn):
        # labels : [N] dims tensor
        # logits : [N x C] dims tensor

        loss_reg = torch.tensor(0, dtype=torch.float32, device=cuda, requires_grad=False)
        for layer in model.modules():
            if isinstance(layer,torch.nn.Conv2d):
                for p in layer.named_parameters():
                    if 'weight' in p[0]:
                        loss_reg += torch.sum((torch.square(p[1]) / 2))   

        loss_reg *= config.l2_weight_decay

        loss_cls = ce_loss_fn(logits, labels)  

        loss_total = loss_cls + loss_reg

        sm_outputs = F.softmax(logits.detach(), dim=-1)
        accuracy = (torch.argmax(sm_outputs, dim=1) == labels).sum() * 100 / labels.size(0)

        return loss_total, loss_cls, loss_reg, accuracy


    def get_model(self):
        if config.model_type == 'simplecnn':
            from models.SimpleCNN import ConvModel
            model = ConvModel(num_classes=config.num_classes).to(cuda, non_blocking=True)
        
        elif config.model_type == 'resnet18':
            from models.ResNetModel import ResNetModel
            model = ResNetModel(num_classes=config.num_classes, freeze_backend=config.freeze_backend).to(cuda, non_blocking=True)

        else:
            print("Unsupported model type.")
            exit()
        return model


    def train(self, resume=False, resume_ckpt=None, pretrained_ckpt=None):
        model = self.get_model()
        
        model_stats = summary(model, (3, config.input_size[0], config.input_size[1]))

        for line in str(model_stats).split('\n'):
            self.log(line)
        
        ce_loss_fn = nn.CrossEntropyLoss()
        # Why opt for nn.CrossEntropyLoss over nn.functional.cross_entropy
        # Ref : https://discuss.pytorch.org/t/f-cross-entropy-vs-torch-nn-cross-entropy-loss/25505/2

        opt = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)
        # Setting lr equal to 0.0 here so that it wont work as per this line.
        # But we will explicitly set lr for each weights dynamically, at every step.
        # Same is case for weight_decay, We will calculate L2_regularization_loss on our own separately.
        
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        
        if resume:
            checkpoint = torch.load(resume_ckpt)
            model.load_state_dict(checkpoint['model'])
            opt.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scalar'])
            resume_g_step = checkpoint['global_step']
            resume_eps = checkpoint['epoch']
            self.log("Resuming training from {} epochs.".format(resume_eps))
        elif pretrained_ckpt is not None and config.model_type == 'resnet18':
            self.log("Using pre-trained checkpoint from :".format(pretrained_ckpt))
            checkpoint = torch.load(pretrained_ckpt)
            
            filtered_checkpoint = {}
            self.log("\nFollowing variables will be restored:")
            for var_name, var_value in checkpoint.items():
                if var_name == 'fc.weight' or var_name == 'fc.bias':        
                    # As these layers change due to change in num classes
                    continue
                new_var_name = 'resnet_feat.' + var_name                
                # why this prefix? This comes as the model that we created contains a variable resnet_feat 
                # which is sequential group of layers containing resnet layers. So all the layers and parameters 
                # within it are prefixed with resnet_feat and for restoring resnet pretrained weights 
                # we need to update the statedict according to the model architectural definition.
                self.log(f"{new_var_name} : {list(var_value.size())}")
                filtered_checkpoint[new_var_name] = var_value

            self.log("\n\nFollowing variables will be initialized:")
            remaining_vars = model.load_state_dict(filtered_checkpoint, strict=False)
            for var_name in remaining_vars.missing_keys:
                self.log(var_name)
            
            resume_g_step = 0
            resume_eps = 0
        else:
            resume_g_step = 0
            resume_eps = 0

        train_writer = SummaryHelper(self.exp_path + "/train/")
        test_writer = SummaryHelper(self.exp_path + "/test/")

        input_x = torch.randn((1,3, config.input_size[0], config.input_size[1])).to(cuda, non_blocking=True)
        train_writer.add_graph(model, input_x)

        g_step = max(0, resume_g_step)
        for eps in range(resume_eps, config.epochs):
            # I hope you noticed one particular statement in the code, to which I assigned a comment “What is this?!?” — model.train().
            # In PyTorch, models have a train() method which, somewhat disappointingly, does NOT perform a training step. 
            # Its only purpose is to set the model to training mode. Why is this important? Some models may use mechanisms like Dropout, 
            # for instance, which have distinct behaviors in training and evaluation phases.
            # Ref: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
            model.train()

            train_loader = get_train_loader()
            train_iter = iter(train_loader)                     # This is creating issues sometimes. Check required.
            
            self.log("Epoch: {} Started".format(eps+1))
                
            for batch_num in tqdm(range(config.train_steps)):
                start = time.time()
                batch = next(train_iter)

                opt.zero_grad()                             # Zeroing out gradients before backprop
                                                            # We cab avoid to zero out if we want accumulate gradients for 
                                                            # Multiple forward pass and single backward pass.
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    logits = model(batch['image'].to(cuda, non_blocking=True))

                    loss_total, loss_cls, loss_reg, accuracy = self.get_loss_and_accuracy(batch['label'].to(cuda, non_blocking=True), logits, model, ce_loss_fn)

                #loss_total.backward()				# Used for normal training without AMP
                scaler.scale(loss_total).backward()		# Used when AMP is applied. The enabled flag will trigger normal FP32 behaviour or Mixed precision behaviour
                scaler.step(opt)
                scaler.update()

                lr = self.lr_helper.step(g_step, opt)
                opt.step()
                delta = (time.time() - start) * 1000        # in milliseconds
                print("Time: {:.2f} ms".format(delta))

                if (batch_num+1) % config.loss_logging_frequency == 0:
                    self.log("Epoch: {}/{}, Batch No.: {}/{}, Total Loss: {:.4f}, Loss Cls: {:.4f}, Loss Reg: {:.4f}, Accuracy: {:.2f}".format(\
                            eps+1, config.epochs, batch_num+1, config.train_steps, np_cpu(loss_total), \
                            np_cpu(loss_cls), np_cpu(loss_reg), np_cpu(accuracy)))
                    
                    train_writer.add_summary({'total_loss' : np_cpu(loss_total),
                                            'loss_cls' : np_cpu(loss_cls),
                                            'loss_reg' : np_cpu(loss_reg), 
                                            'accuracy' : np_cpu(accuracy),
                                            'lr' : lr}, g_step)
                
                g_step += 1
            
            model.eval()            # Putting model in eval mode so that batch normalization and dropout will work in inference mode.

            test_loader = get_test_loader()
            test_iter = iter(test_loader)
            test_losses = LossAverager(num_elements=4)

            with torch.no_grad():   # Disabling the gradient calculations will reduce the calculation overhead.

                for batch_num in tqdm(range(config.test_steps)):
                    batch = next(test_iter)
                    logits = model(batch['image'].to(cuda))

                    loss_total, loss_cls, loss_reg, accuracy = self.get_loss_and_accuracy(batch['label'].to(cuda, non_blocking=True), logits, model, ce_loss_fn)
                    test_losses([np_cpu(loss_total), np_cpu(loss_cls), np_cpu(loss_reg), np_cpu(accuracy)])
                    
                self.log("Epoch: {}/{} Completed, Test Total Loss: {:.4f}, Loss Cls: {:.4f}, Loss Reg: {:.4f}, Accuracy: {:.2f}".format(\
                            eps+1, config.epochs, test_losses.avg[0], test_losses.avg[1], test_losses.avg[2], test_losses.avg[3]))
                
                test_writer.add_summary({'total_loss' : test_losses.avg[0], 
                                        'loss_cls' : test_losses.avg[1], 
                                        'loss_reg' : test_losses.avg[2], 
                                        'accuracy' : test_losses.avg[3]}, g_step)

            checkpoint = {
                'epoch': eps + 1,
                'global_step': g_step,
                'test_loss': test_losses.avg[0],
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
				'scalar': scaler.state_dict()
            }
            # Above code taken from : https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
            self.ckpt_handler.save(checkpoint)
            self.log("Epoch {} completed. Checkpoint saved.".format(eps+1))

        print("Training Completed.")
        train_writer.close()
        test_writer.close()

