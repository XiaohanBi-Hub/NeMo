import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import NonDynamicallyQuantizableLinear

import math

class NeuralMaskGeneratorConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(NeuralMaskGeneratorConv, self).__init__()
        kernel_size = 3
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_features, out_features if out_features < 3073 else out_features // 2, 
                      kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(True),
            nn.Conv1d(out_features if out_features < 3073 else out_features // 2, out_features, 1)  # 1x1 卷积用于变换特征维度
        )

        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)  # kaiming_normal_ is good
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 设置偏置为 0

    def forward(self, x):
        # x 的形状是 (batch_size, patch_size, in_features)
        # 卷积层需要 (batch_size, channels, length)，因此需要转置
        x = x.transpose(1, 2)  # 现在形状是 (batch_size, in_features, patch_size)

        x = self.conv_layers(x)

        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # 压缩最后一个维度


        x = torch.tanh(x)
        x = torch.relu(x)

        return x

class NeuralMaskGenerator(nn.Module):
    def __init__(self, in_features, out_features):
        super(NeuralMaskGenerator, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(in_features, out_features if out_features < 3073 else out_features // 2, bias=True),
            nn.ReLU(True),
            # nn.Linear(out_features if out_features < 1024 else out_features // 2,\
            #         out_features if out_features < 3073 else out_features // 2, bias=True),
            # nn.ReLU(True),
            nn.Linear(out_features if out_features < 3073 else out_features // 2, out_features)
        )
        for m in self.mask_generator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # kaiming_normal_ is good
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)  # 1.0 is good

    def forward(self, x): 
        # x = self.avg_pool(x)
        # 取第一个patch，该patch学习全局信息，用作生成mask
        # x = x[:, 0, :]
        # 对输入的x在patch维度做一个池化，相当于压缩了输入的数据
        # x = x.mean(dim=1)
        # 对features维度做池化，输出(batch_size, patch_size, 1)
        x = x.transpose(1,2)
        # x = F.adaptive_max_pool1d(x,1).squeeze(-1)
        x = F.adaptive_avg_pool1d(x,1).squeeze(-1)
        # 对patch维度做池化，输出(batch_size, in_features)
        mask = self.mask_generator(x)
        # 把mask沿着patch的维度求均值，相当于压缩了patch的信息
        # mask = mask.mean(dim=1)
        # (batch_size, patch_size, in_features)换成(batch_size,in_features,patch_size)
        # 按照patch做mask的平均池化，再去掉最后的维度
        # mask = mask.transpose(1, 2)
        # mask = F.adaptive_avg_pool1d(mask, 1).squeeze(-1)
        
        mask = torch.tanh(mask)
        mask = torch.relu(mask)

        return mask
    
class NeuralMaskLinear(nn.Module):
    # 带Neural Mask的线性层
    # 
    def __init__(self, in_features: int, out_features: int, bias: bool=True,
                 module_layer_masks:list=None, keep_generator: bool=True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NeuralMaskLinear,self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.keep_generator = keep_generator
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        if self.keep_generator:
            if module_layer_masks is None:  # for modular training phase
                self.mask_generator = NeuralMaskGenerator(in_features, out_features)
            else:  # for module reuse phase
                self.mask_generator = NeuralMaskGenerator(len(module_layer_masks[0]), len(module_layer_masks[1]),
                                                    module_layer_masks=module_layer_masks)
        self.masks = None

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.keep_generator:
            self.masks = self.mask_generator(x)
            # x = x[self.masks!=0]
            # x = x * self.masks[self.masks!=0].unsqueeze(1)
            # print("out.shape: ",out.shape)
            # print("self.masks.shape: ",self.masks.shape)
            # 把mask在patch所在的维度展开为1，方便广播
            out = out * (self.masks.unsqueeze(1))
            # out = out * self.masks
        return out
    # def forward(self, x):
    #     if self.keep_generator:
    #         self.masks = self.mask_generator(x)
    #         x = x * (self.masks.unsqueeze(1))
    #     out = F.linear(x, self.weight, self.bias)
    #     return out
    