
import config

import torch
import torch.nn as nn
from torch.nn.functional import pad
import torchvision.models as models

def get_padded_tensor(x, k_size=(3, 3), stride=1, dilation=1, padding='same'):
    # Taken from : https://github.com/pytorch/pytorch/issues/3867#issuecomment-458423010
    
    if str(padding).upper() == 'SAME':
        input_rows, input_cols = [int(x) for x in x.shape[2:4]]     
        # x.shape returns pytorch tensor rather than python int list
        # And doing further computation based on that will grow the graph with such nodes
        # Which needs to be avoided when converting the model to onnx or torchscript.
        filter_rows, filter_cols = k_size
    
        out_rows = (input_rows + stride - 1) // stride
        out_cols = (input_cols + stride - 1) // stride
    
        padding_rows = max(0, (out_rows - 1) * stride +
                            (filter_rows - 1) * dilation + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
    
        padding_cols = max(0, (out_cols - 1) * stride +
                            (filter_cols - 1) * dilation + 1 - input_cols)
        cols_odd = (padding_rows % 2 != 0)
        
        x = pad(x, [padding_cols // 2, (padding_cols // 2) + int(cols_odd),
                    padding_rows // 2, (padding_rows // 2) + int(rows_odd)])        # This is only true for NCHW
                                                                                    # First 2 elements are for last dims
                                                                                    # Next 2 elements are for second last dims
        # Or alternatively we can do as below.
        #x = nn.ZeroPad2d((padding_cols // 2, (padding_cols // 2) + int(cols_odd),
        #            padding_rows // 2, (padding_rows // 2) + int(rows_odd)))(x)

        return x
    else:
        return x


def ConvLayer(in_channels, out_channels, conv_k_size=(3, 3), conv_stride=1, padding='same', bias=False):
    if bias:
        layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=conv_k_size, stride=conv_stride, padding=0, bias=bias),
                    nn.ReLU(),
                )
    else:
        layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=conv_k_size, stride=conv_stride, padding=0, bias=bias),
                    nn.BatchNorm2d(num_features=out_channels, eps=1e-6),
                    nn.ReLU()
                )
    return layer


def MaxPoolLayer(mx_k_size=(3, 3), mx_stride=2):
    mxpool = nn.MaxPool2d(kernel_size=mx_k_size, stride=mx_stride, padding=0)
    return mxpool
    

class ResNetModel(nn.Module):
    def __init__(self, num_classes=2, freeze_backend=False, inference=False):
        super(ResNetModel, self).__init__()

        resnet = models.resnet18(pretrained=False)      
        # Intentionally used pretrained=False, as the pretrained checkpoint will be fed from outside
        # before training.        

        self.resnet_feat = nn.Sequential()
        for layer_name, layer_module in list(resnet.named_children())[:-1]:
            # Used this approach so as to preserve the name of layers/parameters which 
            # can be used while restoring them before training separately
            self.resnet_feat.add_module(layer_name, layer_module)

        # Alternatively we can use below line, but name of parameters will not be in accordance with the 
        # resnet pretrained checkpoint file.
        #self.resnet_feat = nn.Sequential(*(list(resnet.children())[:-1])) 

        if freeze_backend:
            # Fix the parameters of the feature extractor: 
            for param in self.resnet_feat.parameters(): 
                param.requires_grad = False

        resnet_op_feat = resnet.layer4[1].conv2.out_channels

        self.fc = nn.Linear(in_features=resnet_op_feat, out_features=num_classes, bias=True)

        self.softmax = nn.Softmax(dim=1)

        self.inference = inference

        self.__init_weights()
    

    def __init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            
                if config.weight_init_method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                elif config.weight_init_method == 'msra':
                    nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                else:
                    print("Unsupported weight init method.")
                    exit()
            
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.running_var, 1)
                nn.init.constant_(module.running_mean, 0)
                
        
    def forward(self, x):
        # x : [B x 3 x 224 x 224]
        
        out = self.resnet_feat(x)
        # x : [B x 512 x 1 x 1]

        out = out.squeeze(dim=3).squeeze(dim=2)
        # x : [B x 512]
		
        out = self.fc(out)
        # x : [B x 2]
        
        if self.inference:
            out = self.softmax(out)
        
        return out
