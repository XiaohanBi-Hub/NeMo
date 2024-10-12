from models.vgg_masked_contra import cifar10_vgg16_bn as vgg16
from models.utils_v2 import MaskConvBN, MaskConvBNContra, MaskConvBNFreeze
from models.utils_mask_neural import NeuralMaskLinear
from models.utils_v2 import Binarization
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random

def mean_list(input_list):
    return sum(input_list) / len(input_list)

def freeze_mask(model,num):
    # This could set mask to 1 and freeze them
    count = 0
    for name,child in model.named_children():
        if count >= num:
            break
        if (isinstance(child, MaskConvBN)\
            or isinstance(child, MaskConvBNContra)):
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size
            conv_freeze = MaskConvBNFreeze(in_channels,out_channels,kernel_size)
            conv_freeze.conv = child.conv
            conv_freeze.bn = child.bn
            setattr(model,name,conv_freeze)
            count += 1
        else:
            freeze_mask(child,num)
            
def replace_linear_vit(module, replace_attention):
    for name, child in module.named_children():
        if replace_attention is not True:
            if name in ['query', 'key', 'value']:
                # Do not replace attention
                continue
        if "0" in name and "10" not in name:
            # Do not replace the first encoder
            continue
        if "classifier" in name:
            # Do not replace classifier
            continue
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            neural_mask_linear = NeuralMaskLinear(in_features, out_features, bias)
            neural_mask_linear.weight = child.weight
            if bias:
                neural_mask_linear.bias= child.bias
            setattr(module, name, neural_mask_linear)
        else:
            replace_linear_vit(child, replace_attention)

def get_masks(model):
    masks = []
    for name,child in model.named_children():
        if (isinstance(child, NeuralMaskLinear) \
            or isinstance(child, MaskConvBN)\
            or isinstance(child, MaskConvBNContra))\
            and child.masks is not None:
            masks.append(child.masks)
        else:
            masks += get_masks(child)
    return masks

def get_layer_out(model):
    layers_out = []
    for name, child in model.named_children():
        if (isinstance(child, MaskConvBNContra))\
            and child.layer_out is not None:
            layers_out.append(child.layer_out)
        else:
            layers_out += get_layer_out(child)
    return layers_out

def get_contra_loss(model):
    contrast_loss_list = []
    for name,child in model.named_children():
        if isinstance(child, MaskConvBNContra) \
            and child.contrast_loss != None:
            contrast_loss_list.append(child.contrast_loss)
        else:
            contrast_loss_list += get_contra_loss(child)
    return contrast_loss_list

#######################################################################################
def replace_linear(model):
    count=2
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and count>0:
            count = count - 1
            new_linear = NeuralMaskLinear(module.in_features, module.out_features,\
                                        bias=module.bias is not None)
            new_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.clone()
            setattr(model, name, new_linear)
        else:
            replace_linear(module)
    return model
#######################################################################################

def loss_func_cos(masks, labels, DEVICE):
    loss_l1 = []
    loss_coupling, loss_cohesion = [], []
    kernel_rate = []
    
    tmp = labels.unsqueeze(0) - labels.unsqueeze(1)
    mask_sim_ground_truth = torch.ones_like(tmp, device=DEVICE)
    mask_sim_ground_truth[tmp != 0] = 0.0
    mask_sim_ground_truth = mask_sim_ground_truth[torch.triu(torch.ones_like(mask_sim_ground_truth, device=DEVICE), diagonal=1) == 1]
    
    for each_layer_mask in masks:
        norm_masks = F.normalize(each_layer_mask, p=2, dim=1)
        norm_masks_trans = norm_masks.T
        mask_sim = torch.mm(norm_masks, norm_masks_trans)
        mask_sim = mask_sim[torch.triu(torch.ones_like(mask_sim, device=DEVICE), diagonal=1) == 1]
        # the elements' value in masks range from 0 to 1, so the value of cosine similarity range from 0 to 1.
        loss_cohesion.append(1 - mask_sim[mask_sim_ground_truth == 1].mean())
        loss_coupling.append(mask_sim[mask_sim_ground_truth == 0].mean())
        loss_l1.append(each_layer_mask.mean())
        kernel_rate.append(torch.mean((each_layer_mask > 0).float()))

    loss_cohesion = mean_list(loss_cohesion)
    loss_coupling = mean_list(loss_coupling)
    loss_l1 = mean_list(loss_l1)
    kernel_rate = mean_list(kernel_rate)
    return loss_cohesion, loss_coupling, loss_l1, kernel_rate

def loss_func_contra(masks, labels, temperature, DEVICE):
    eps = 1e-8
    contrast_loss_list = []
    batch_size = len(labels)
    class_ids = torch.unique(labels)
    pos_idx = torch.zeros(batch_size, batch_size, dtype=torch.float).to(DEVICE)
    for class_id in class_ids:
        vec = (labels == class_id).float()
        pos_idx = pos_idx + torch.einsum('i,j->ij', [vec,vec])
    for each_layer_mask in masks:
        pos_vec = F.normalize(each_layer_mask, p=2, dim=1)
        sim_matrix = torch.matmul(pos_vec,pos_vec.T)
        pos_sim = torch.einsum("ij,ij->ij",[sim_matrix, pos_idx])
        neg_sim = torch.einsum("ij,ij->ij",[sim_matrix, (1-pos_idx)])

        pos_sum = torch.sum(torch.exp(pos_sim / temperature),dim=1)
        neg_sum = torch.sum(torch.exp(neg_sim / temperature),dim=1)
        _contrast_loss = - torch.log((pos_sum + eps) / (pos_sum + neg_sum + eps))
        contrast_loss_list.append(_contrast_loss.mean())
    contra_loss = mean_list(contrast_loss_list)
    return contra_loss

def kr_l1_cohe_loss(masks, labels, DEVICE):
    kernel_rate, loss_coupling, loss_cohesion, l1_reg_list= [],[],[],[]
    l1_reg = 0
    tmp = labels.unsqueeze(0) - labels.unsqueeze(1)
    mask_sim_ground_truth = torch.ones_like(tmp, device=DEVICE)
    mask_sim_ground_truth[tmp != 0] = 0.0
    mask_sim_ground_truth = mask_sim_ground_truth[torch.triu(torch.ones_like(mask_sim_ground_truth, device=DEVICE), diagonal=1) == 1]
    for each_layer_mask in masks:
        norm_masks = F.normalize(each_layer_mask, p=2, dim=1)
        norm_masks_trans = norm_masks.T
        mask_sim = torch.mm(norm_masks, norm_masks_trans)
        mask_sim = mask_sim[torch.triu(torch.ones_like(mask_sim, device=DEVICE), diagonal=1) == 1]
        loss_cohesion.append(1 - mask_sim[mask_sim_ground_truth == 1].mean())
        loss_coupling.append(mask_sim[mask_sim_ground_truth == 0].mean())

        bin_mask = Binarization.apply(each_layer_mask)
        kernel_rate.append(torch.mean(bin_mask))
        l1_reg_list.append(torch.mean(each_layer_mask))
    loss_cohesion = mean_list(loss_cohesion)
    loss_coupling = mean_list(loss_coupling)
    kernel_rate = mean_list(kernel_rate)
    l1_reg = mean_list(l1_reg_list)
    return loss_cohesion, loss_coupling, kernel_rate, l1_reg

def cal_modular_metrics(sample_masks, labels):
    tmp = labels.unsqueeze(0) - labels.unsqueeze(1)
    mask_sim_ground_truth = torch.ones_like(tmp, device='cuda')
    mask_sim_ground_truth[tmp != 0] = 0.0
    mask_sim_ground_truth = mask_sim_ground_truth[
        torch.triu(torch.ones_like(mask_sim_ground_truth, device='cuda'), diagonal=1) == 1]

    # intersection
    sample_masks_trans = sample_masks.T
    intersection_sum = torch.mm(sample_masks, sample_masks_trans)
    intersection_sum = intersection_sum[torch.triu(torch.ones_like(intersection_sum, device='cuda'), diagonal=1) == 1]

    # union
    sample_masks_copy_y = sample_masks.unsqueeze(0)
    sample_masks_copy_x = sample_masks.unsqueeze(1)
    union = sample_masks_copy_x + sample_masks_copy_y
    union = (union > 0).int()
    union_sum = torch.sum(union, dim=-1)
    union_sum = union_sum[torch.triu(torch.ones_like(union_sum, device='cuda'), diagonal=1) == 1]

    # Jaccard Index
    cohesion = intersection_sum / union_sum
    cohesion = cohesion[mask_sim_ground_truth == 1].mean()

    coupling = intersection_sum / union_sum
    coupling = coupling[[mask_sim_ground_truth == 0]].mean()

    # kernel retention rate
    krr = torch.mean(sample_masks.float())

    return cohesion, coupling, krr

def distillation_loss(outputs, teacher_logits, labels, alpha_distill=0.5, tau=1.0, target_classes: list = None):
    student_logits = outputs.cls_logits
    distill_logits = outputs.distillation_logits

    # For modularizer
    if target_classes is not None:
        teacher_logits = teacher_logits[:, target_classes]

    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / tau, dim=-1),
        F.softmax(teacher_logits / tau, dim=-1),
        reduction='batchmean'
    ) * (tau ** 2)
    
    distill_loss = F.kl_div(
        F.log_softmax(distill_logits / tau, dim=-1),
        F.softmax(teacher_logits / tau, dim=-1),
        reduction='batchmean'
    ) * (tau ** 2)
    
    return hard_loss * alpha_distill + (1 - alpha_distill) * 0.5 * (soft_loss + distill_loss)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
