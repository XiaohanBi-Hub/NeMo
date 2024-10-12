import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math

class MaskGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, module_layer_masks:list=None):
        super(MaskGenerator, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mask_generator = nn.Sequential(
            nn.Linear(in_channels, out_channels if out_channels < 128 else out_channels // 2, bias=False),
            nn.ReLU(True),
            nn.Linear(out_channels if out_channels < 128 else out_channels // 2, out_channels)
        )
        for m in self.mask_generator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # kaiming_normal_ is good
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)  # 1.0 is good

        self.is_module_reuse_phase = True if module_layer_masks is not None else False
        self.module_layer_masks = module_layer_masks

        if self.is_module_reuse_phase:
            self.retained_indices_cin = torch.nonzero(self.module_layer_masks[0], as_tuple=True)[0]
            self.retained_indices_out = torch.nonzero(self.module_layer_masks[1], as_tuple=True)[0]
            self.mask_before_dim = len(self.module_layer_masks[0])

    def forward(self, x):
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

        if self.is_module_reuse_phase:
            x_padding = torch.zeros((x.shape[0], self.mask_before_dim), dtype=x.dtype, device='cuda')
            x_padding[:, self.retained_indices_cin] = x
            x = x_padding

        mask = self.mask_generator(x)

        if self.is_module_reuse_phase:
            mask = mask[:, self.retained_indices_out]

        mask = torch.tanh(mask)
        mask = torch.relu(mask)

        return mask

class MaskConvBN(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int = 0, stride: int = 1,
                 groups: int = 1, dilation: int = 1 , bias=True, module_layer_masks:list=None,
                 keep_generator:bool=True):
        """
        :param module_layer_masks: for module reuse. [previous_layer_mask, current_layer_mask]
        """
        super(MaskConvBN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, bias=bias, padding=padding, groups=groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.keep_generator = keep_generator

        if keep_generator:
            if module_layer_masks is None:  # for modular training phase
                self.mask_generator = MaskGenerator(in_channels, out_channels)
            else:  # for module reuse phase
                self.mask_generator = MaskGenerator(len(module_layer_masks[0]), len(module_layer_masks[1]),
                                                    module_layer_masks=module_layer_masks)

        self.masks = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.keep_generator:
            self.masks = self.mask_generator(x)
            out = out * self.masks.unsqueeze(-1).unsqueeze(-1)
        return out

class MemoryBank(nn.Module):
    def __init__(self, memory_bank_size, feature_dim, momentum=0.999):
        super(MemoryBank, self).__init__()
        self.memory_bank_size = memory_bank_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.register_buffer('feature_bank', torch.randn(memory_bank_size, feature_dim))
        self.register_buffer('label_bank', torch.randn(memory_bank_size).long())
        self.ptr = 0

    def update(self, features, labels):
        batch_size = features.size(0)
        if self.ptr + batch_size <= self.memory_bank_size:
            self.feature_bank[self.ptr:self.ptr+batch_size] = features
            self.label_bank[self.ptr:self.ptr+batch_size] = labels
            self.ptr += batch_size
        else:
            remaining = self.memory_bank_size - self.ptr
            self.feature_bank[self.ptr:] = features[:remaining]
            self.label_bank[self.ptr:] = labels[:remaining]
            self.feature_bank[:batch_size-remaining] = features[remaining:]
            self.label_bank[:batch_size-remaining] = labels[remaining:]
            self.ptr = batch_size - remaining

class MaskGeneratorContra(nn.Module):
    def __init__(self, in_channels, out_channels, module_layer_masks:list=None):
        super(MaskGeneratorContra, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mask_generator = nn.Sequential(
            nn.Linear(in_channels, out_channels if out_channels < 128 else out_channels // 2, bias=False),
            nn.ReLU(True),
            nn.Linear(out_channels if out_channels < 128 else out_channels // 2, out_channels)
        )
        for m in self.mask_generator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # kaiming_normal_ is good
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)  # 1.0 is good

        self.is_module_reuse_phase = True if module_layer_masks is not None else False
        self.module_layer_masks = module_layer_masks

        if self.is_module_reuse_phase:
            self.retained_indices_cin = torch.nonzero(self.module_layer_masks[0], as_tuple=True)[0]
            self.retained_indices_out = torch.nonzero(self.module_layer_masks[1], as_tuple=True)[0]
            self.mask_before_dim = len(self.module_layer_masks[0])

    def forward(self, x):
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

        if self.is_module_reuse_phase:
            x_padding = torch.zeros((x.shape[0], self.mask_before_dim), dtype=x.dtype, device='cuda')
            x_padding[:, self.retained_indices_cin] = x
            x = x_padding

        mask = self.mask_generator(x)

        if self.is_module_reuse_phase:
            mask = mask[:, self.retained_indices_out]

        mask = torch.tanh(mask)
        mask = torch.relu(mask)

        return mask

class MaskConvBNContra(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int = 0, stride: int = 1,
                 groups: int = 1, dilation: int = 1 , bias=True, module_layer_masks:list=None,
                 keep_generator:bool=True, memory_bank_size:int=50000, momentum:float=0.99):
        """
        :param module_layer_masks: for module reuse. [previous_layer_mask, current_layer_mask]
        """
        super(MaskConvBNContra, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, bias=bias, padding=padding, groups=groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.keep_generator = keep_generator
        if keep_generator:
            if module_layer_masks is None:  # for modular training phase
                self.mask_generator = MaskGeneratorContra(in_channels, out_channels)
            else:  # for module reuse phase
                self.mask_generator = MaskGeneratorContra(len(module_layer_masks[0]), len(module_layer_masks[1]),
                                                    module_layer_masks=module_layer_masks)

        self.memory_bank_size = memory_bank_size
        self.momentum = momentum

        self.masks = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        self.layer_out = out
        if self.keep_generator:
            self.masks = self.mask_generator(x)
            pos_feat = out * self.masks.unsqueeze(-1).unsqueeze(-1)
        return pos_feat
    
class MaskConvBNFreeze(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int = 0, stride: int = 1,
                 groups: int = 1, dilation: int = 1 , bias=True, module_layer_masks:list=None,
                 keep_generator:bool=True, memory_bank_size:int=50000, momentum:float=0.99):
        """
        :param module_layer_masks: for module reuse. [previous_layer_mask, current_layer_mask]
        """
        super(MaskConvBNFreeze, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, bias=bias, padding=padding, groups=groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.keep_generator = keep_generator
        self.out_channels = out_channels
        self.masks = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.keep_generator:
            self.masks = torch.ones(x.shape[0],self.out_channels).to(out.device)
            out = out * self.masks.unsqueeze(-1).unsqueeze(-1)
        return out


class Binarization(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        bin_mask = (mask > 0).float()
        return bin_mask

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)