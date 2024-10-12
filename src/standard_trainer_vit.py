import argparse
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification, ViTConfig
from models.funcs import mean_list
from configs import Configs
from dataset_loader import load_cifar10, load_svhn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vit_s', 'vit'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn'], required=True)

    parser.add_argument('--lr_model', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)

    args = parser.parse_args()
    return args


def training(model, train_loader, test_loader, num_epoch=200):
    optim = torch.optim.SGD(
        params=model.parameters(), lr=lr_model, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epoch)

    for epoch in range(num_epoch):
        print(f'Epoch {epoch}')
        print('-' * 50)

        for data_loader, phase in zip([train_loader, test_loader], ['train', 'test']):
            with torch.set_grad_enabled(phase == 'train'):
                acc, loss = _training(model, data_loader, optim, phase)
            writer.add_scalar(f'{phase}/Accuracy', acc, epoch)
            writer.add_scalar(f'{phase}/Loss', loss, epoch)
        scheduler.step()
    return model


def _training(model, data_loader, optim, phase):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    log_loss = []
    n_correct, total_labels = 0, 0

    for inputs, labels in tqdm(data_loader, ncols=80):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.logits, labels)

        pred = torch.argmax(outputs.logits, dim=1)
        n_correct += torch.sum((pred == labels).float())
        total_labels += len(labels)

        if phase == 'train':
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_loss.append(loss.detach())

    return n_correct/total_labels, mean_list(log_loss)

def main():
    if dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=num_workers, pic_size=pic_size)
    elif dataset_name == 'svhn':
        train_loader, test_loader = load_svhn(f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=num_workers, pic_size=pic_size)
    else:
        raise ValueError

    if model_name == 'vit':
        config = ViTConfig(num_labels=10)
        model = ViTForImageClassification._from_config(config).to(DEVICE)
    elif model_name == 'vit_s':
        config = ViTConfig(
            num_labels=10,
            hidden_size=384,
            intermediate_size=1536,
            image_size=pic_size,
            patch_size=4,
        )
        model = ViTForImageClassification._from_config(config).to(DEVICE)
    else:
        raise ValueError

    model = training(model, train_loader, test_loader, num_epoch=num_epochs)
    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    model_name = args.model
    dataset_name = args.dataset
    lr_model = args.lr_model
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    pic_size=32

    num_workers = 2
    configs = Configs()
    DEVICE = torch.device('cuda:1')
    log_path = f'/home/bixh/Documents/MwT_ext/src/ViT_logs/logs_st_{model_name}/standard_model_lr_{lr_model}_bs_{batch_size}_{model_name}'
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_path)

    model_save_path = f'{configs.data_dir}/{model_name}_{dataset_name}/' \
                      f'standard_model_lr_{lr_model}_bz_{batch_size}.pth'

    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    main()
