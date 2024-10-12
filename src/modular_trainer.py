import argparse
from models.vgg_masked import cifar10_vgg16_bn as vgg16
from models.resnet_masked import ResNet18, ResNet34, ResNet50, ResNet101
from models_cnnsplitter.simcnn_masked import SimCNN
from models_cnnsplitter.rescnn_masked import ResCNN

from original_models.vgg import cifar10_vgg16_bn as original_vgg16

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from configs import Configs
from dataset_loader import load_cifar10, load_svhn, load_cifar100_sc, load_cifar100
from models.funcs import mean_list, loss_func_cos, cal_modular_metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vgg16', 'resnet18', 'resnet34', 'resnet50', 'simcnn', 'rescnn'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn', 'cifar100'], required=True)

    parser.add_argument('--lr_model', type=float, default=0.05)
    parser.add_argument('--lr_mask', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--cal_modular_metrics', action='store_true',
                        help='calculate the cohesion and coupling. '
                             'This is not necessary for modular training and '
                             'will slow down the training.')
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--cuda_device', type=str, default="0")
    parser.add_argument('--model_save_path', type=str, default=None)

    args = parser.parse_args()
    return args


def modular_training(model, train_loader, test_loader, alpha, beta, lr_mask, lr_model, num_epochs=200):
    model_param = [p for n, p in model.named_parameters() if 'mask_generator' not in n and p.requires_grad]
    mask_generator_param = [p for n, p in model.named_parameters() if 'mask_generator' in n and p.requires_grad]
    optim = torch.optim.SGD(
        [{'params': model_param},
         {'params': mask_generator_param, 'lr': lr_mask}],
        lr=lr_model, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)

    # begin modular training
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        print('-' * 50)

        for data_loader, phase in zip([train_loader, test_loader], ['train', 'test']):
            if not is_cal_modular_metrics:
                with torch.set_grad_enabled(phase == 'train'):
                    log_acc, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate = \
                        _modular_training(model, data_loader, alpha, beta, optim, phase)
            else:
                if phase == 'train':
                    with torch.set_grad_enabled(phase == 'train'):
                        log_acc, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate = \
                            _modular_training(model, data_loader, alpha, beta, optim, phase)
                else:
                    with torch.set_grad_enabled(phase == 'train'):
                        log_acc, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate, \
                        log_cohesion, log_coupling = _test_with_modular_metrics(model, data_loader)
                    writer.add_scalar(f'{phase}/Cohesion', mean_list(log_cohesion), epoch)
                    writer.add_scalar(f'{phase}/Coupling', mean_list(log_coupling), epoch)

            # draw log
            writer.add_scalar(f'{phase}/Accuracy', log_acc, epoch)
            writer.add_scalar(f'{phase}/Loss', mean_list(log_loss), epoch)
            writer.add_scalar(f'{phase}/Loss-CE', mean_list(log_loss_ce), epoch)
            writer.add_scalar(f'{phase}/Loss-Cohesion', mean_list(log_loss_cohesion), epoch)
            writer.add_scalar(f'{phase}/Loss-Coupling', mean_list(log_loss_coupling), epoch)
            writer.add_scalar(f'{phase}/Kernel-Rate', mean_list(log_kernel_rate), epoch)

        scheduler.step()
    return model


def _modular_training(model, data_loader, alpha, beta, optim, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1 = [], [], [], [], []
    n_correct, total_labels = 0, 0
    log_kernel_rate = []

    for inputs, labels in tqdm(data_loader, ncols=80):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss_ce = F.cross_entropy(outputs, labels)

        pred = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)

        masks = model.get_masks()
        loss_cohesion, loss_coupling, loss_l1, kernel_rate = loss_func_cos(masks, labels, DEVICE)
        loss = loss_ce + alpha * loss_cohesion + beta * loss_coupling

        if phase == 'train':
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_loss.append(loss.detach())
        log_loss_ce.append(loss_ce.detach())
        log_loss_cohesion.append(loss_cohesion.detach())
        log_loss_coupling.append(loss_coupling.detach())
        log_kernel_rate.append(kernel_rate.detach())
    return n_correct/total_labels, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate


@torch.no_grad()
def _test_with_modular_metrics(model, test_dataset):
    model.eval()

    log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1 = [], [], [], [], []
    n_correct, total_labels = 0, 0
    log_kernel_rate = []
    log_cohesion, log_coupling = [], []

    for inputs, labels in tqdm(test_dataset, ncols=80):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss_ce = F.cross_entropy(outputs, labels)

        pred = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)

        masks = model.get_masks()
        bin_masks = torch.cat(masks, dim=1)
        bin_masks = (bin_masks > 0).float()
        cohesion, coupling, _ = cal_modular_metrics(bin_masks, labels)
        log_cohesion.append(cohesion)
        log_coupling.append(coupling)

        loss_cohesion, loss_coupling, loss_l1, kernel_rate = loss_func_cos(masks, labels, DEVICE)
        loss = loss_ce + alpha * loss_cohesion + beta * loss_coupling

        log_loss.append(loss.detach())
        log_loss_ce.append(loss_ce.detach())
        log_loss_cohesion.append(loss_cohesion.detach())
        log_loss_coupling.append(loss_coupling.detach())
        log_kernel_rate.append(kernel_rate.detach())
    return n_correct / total_labels, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate, log_cohesion, log_coupling



def main():
    if dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers)
        class_num=10
    elif dataset_name == 'svhn':
        train_loader, test_loader = load_svhn(f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=num_workers)
        class_num=10
    elif dataset_name == 'cifar100':
        train_loader, test_loader = load_cifar100_sc(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, custom_defined=True)
        class_num=20
    else:
        raise ValueError

    if model_name == 'vgg16':
        model = vgg16(pretrained=False, num_classes=class_num).to(DEVICE)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=class_num).to(DEVICE)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes=class_num).to(DEVICE)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=class_num).to(DEVICE)
    elif model_name == 'simcnn':
        model = SimCNN().to(DEVICE)
    elif model_name == 'rescnn':
        model = ResCNN().to(DEVICE)
    else:
        raise ValueError


    
    print(model)

    model = modular_training(model, train_loader, test_loader,
                             alpha=alpha, beta=beta, lr_mask=lr_mask, lr_model=lr_model, num_epochs=n_epochs)
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    
    num_workers = 2
    batch_size = args.batch_size
    model_name = args.model
    dataset_name = args.dataset
    lr_model = args.lr_model
    lr_mask = args.lr_mask
    alpha = args.alpha
    beta = args.beta
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    is_cal_modular_metrics = args.cal_modular_metrics
    log_dir = args.log_dir
    cuda_device = args.cuda_device
    is_contra = True
    model_save_path = args.model_save_path
    DEVICE = torch.device(f'cuda:{cuda_device}')
    configs = Configs()

    writer = SummaryWriter(log_dir)

    main()