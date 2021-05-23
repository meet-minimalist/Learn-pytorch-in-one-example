
import config

import torch
import torch.nn as nn
from torch.nn.functional import pad


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


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv_k_size=(3, 3), mx_k_size=(3, 3), conv_stride=1, mx_stride=2, padding='same', bias=False):
        super(ConvLayer, self).__init__()
        self.conv_k_size = conv_k_size
        self.mx_k_size = mx_k_size
        self.conv_stride = conv_stride
        self.mx_stride = mx_stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_k_size, stride=conv_stride, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-6)
        self.act = nn.ReLU()
        self.mxpool = nn.MaxPool2d(kernel_size=mx_k_size, stride=mx_stride, padding=0)


    def forward(self, x):
        x = get_padded_tensor(x, k_size=self.conv_k_size, stride=self.conv_stride, padding=self.padding)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = get_padded_tensor(x, k_size=self.mx_k_size, stride=self.mx_stride, padding=self.padding)
        x = self.mxpool(x)
        return x
    

class ConvModel(nn.Module):
    def __init__(self, num_classes=2, inference=False):
        super(ConvModel, self).__init__()

        self.layer1 = ConvLayer(3, 32)

        self.layer2 = ConvLayer(32, 64)
        
        self.layer3 = ConvLayer(64, 128)
        
        self.layer4 = ConvLayer(128, 256)

        self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)

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
                
        
    def forward(self, x):
        # x : [B x 3 x 224 x 224]

        out = self.layer1(x)
        # x : [B x 32 x 112 x 112]

        out = self.layer2(out)
        # x : [B x 64 x 56 x 56]

        out = self.layer3(out)
        # x : [B x 128 x 28 x 28]

        out = self.layer4(out)
        # x : [B x 256 x 14 x 14]
        
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        # x : [B x 256]

        out = self.fc(out)
        # x : [B x 2]
        
        if self.inference:
            out = self.softmax(out)
        return out

