import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

from dataset_loader import load_cifar10, load_svhn, load_cifar100_sc, load_cifar100
from configs import Configs

from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher

from models.resnet import ResNet18
from models.funcs import get_masks, mean_list, loss_func_cos, replace_linear_vit, distillation_loss, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vit_s', 'vit', 'deit_s'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'svhn'], required=True)

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
    parser.add_argument('--replace_attention', action='store_true', default=False)
    parser.add_argument('--modular_superclass', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--cuda_device', type=str, default="0")
    parser.add_argument('--model_save_path', type=str, default=None)

    args = parser.parse_args()
    return args

def _modular_training(model, data_loader, alpha, beta, criterion, \
                      optim, phase, teacher_model=None):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    log_loss, log_loss_ce = [], []
    n_correct, total_labels = 0, 0
    log_loss_cohesion, log_loss_coupling, log_loss_l1 = [], [], []
    log_kernel_rate = []

    for inputs, labels in tqdm(data_loader, ncols=80):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        if model_name == "deit_s":
            teacher_model.eval()
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            loss_ce = distillation_loss(outputs, teacher_logits,labels)
        else:
            loss_ce = criterion(outputs.logits, labels)

        pred = torch.argmax(outputs.logits, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)
        loss = loss_ce
        masks = get_masks(model)
        loss_cohesion, loss_coupling, loss_l1, kernel_rate = loss_func_cos(masks, labels, DEVICE)
        loss = loss_ce + alpha * loss_cohesion + beta * loss_coupling

        if phase == 'train':
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_loss_ce.append(loss_ce.detach())
        log_loss.append(loss.detach())
        log_loss_cohesion.append(loss_cohesion.detach())
        log_loss_coupling.append(loss_coupling.detach())
        log_kernel_rate.append(kernel_rate.detach())
    return n_correct/total_labels, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate

def modular_training(model, train_loader, test_loader, alpha, beta, lr_mask, lr_model, num_epochs, teacher_model):
    model_param = [p for n, p in model.named_parameters() if 'mask_generator' not in n and p.requires_grad]
    mask_generator_param = [p for n, p in model.named_parameters() if 'mask_generator' in n and p.requires_grad]
    
    optim = torch.optim.SGD(
        [{'params': model_param},
        {'params': mask_generator_param, 'lr': lr_mask}],
        lr=lr_model, momentum=0.9, nesterov=True
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    # begin modular training
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        print('-' * 50)

        for data_loader, phase in zip([train_loader, test_loader], ['train', 'test']):
            if not is_cal_modular_metrics:
                with torch.set_grad_enabled(phase == 'train'):
                    log_acc, log_loss, log_loss_ce, log_loss_cohesion, log_loss_coupling, log_loss_l1, log_kernel_rate = \
                        _modular_training(model, data_loader, alpha, beta, \
                                          criterion, optim, phase, teacher_model)
            # draw log
            writer.add_scalar(f'{phase}/Accuracy', log_acc, epoch)
            writer.add_scalar(f'{phase}/Loss', mean_list(log_loss), epoch)
            writer.add_scalar(f'{phase}/Loss-CE', mean_list(log_loss_ce), epoch)
            writer.add_scalar(f'{phase}/Loss-Cohesion', mean_list(log_loss_cohesion), epoch)
            writer.add_scalar(f'{phase}/Loss-Coupling', mean_list(log_loss_coupling), epoch)
            writer.add_scalar(f'{phase}/Kernel-Rate', mean_list(log_kernel_rate), epoch)

        scheduler.step()
    return model

def main():
    set_seed(42)
    pic_size=32
    if args.dataset == 'cifar10':
        trainloader, testloader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, pic_size=pic_size)
        class_num=10
    elif args.dataset == 'cifar100':
        if modular_superclass:
            trainloader, testloader = load_cifar100_sc(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, pic_size=pic_size,
                                                        custom_defined=True)
            class_num=20
        else:
            trainloader, testloader = load_cifar100(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, pic_size=pic_size)
            class_num=100
    elif args.dataset == 'svhn':
        trainloader, testloader = load_svhn(f"{configs.dataset_dir}/svhn", batch_size=batch_size, num_workers=num_workers, pic_size=pic_size)
        class_num=10
    if model_name == "vit":
        config = ViTConfig(num_labels=10)
        _model = ViTForImageClassification._from_config(config).to(DEVICE)
        pic_size=224
    elif model_name == 'vit_s':
        pic_size=32
        config = ViTConfig(
            num_labels=class_num,
            hidden_size=384,
            intermediate_size=1536,
            image_size=pic_size,
            patch_size=4,
        )
        _model = ViTForImageClassification._from_config(config).to(DEVICE)
    elif model_name == 'deit_s':
        teacher_model = ResNet18(num_classes=class_num).to(DEVICE)
        teacher_model.load_state_dict(torch.load(f'/home/bixh/Documents/NeMo/data/data/modular_trained/resnet18_{dataset_name}/standard_model_lr_0.05_bz_128.pth'))
        teacher_model.eval()
        config = DeiTConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=pic_size,
            patch_size=4,
            num_channels=3,
            qkv_bias=True,
            encoder_stride=16,
            num_labels=class_num,
        )
        _model = DeiTForImageClassificationWithTeacher(config).to(DEVICE)

    model = copy.deepcopy(_model)
    replace_linear_vit(model,replace_attention)

    model = model.to(DEVICE) 
    print("MODEL: ",model)

    model = modular_training(model, trainloader, testloader,
                             alpha=alpha, beta=beta, lr_mask=lr_mask, 
                             lr_model=lr_model, num_epochs=n_epochs, teacher_model=teacher_model)

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":

    dataset_dir = '../data/data/dataset/'

    args = get_args()
    num_workers = 2
    model_name = args.model
    dataset_name = args.dataset
    lr_model = args.lr_model
    lr_mask = args.lr_mask
    alpha = args.alpha
    beta = args.beta
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    is_cal_modular_metrics = args.cal_modular_metrics
    modular_superclass = args.modular_superclass
    log_dir = args.log_dir
    replace_attention = args.replace_attention
    model_save_path = args.model_save_path
    cuda_device = args.cuda_device
    DEVICE = torch.device(f"cuda:{cuda_device}")

    writer = SummaryWriter(log_dir=log_dir)
    configs = Configs()
    main()





