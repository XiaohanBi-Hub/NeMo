import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, SVHN, CIFAR100
from datasets.load_cifar100_superclass import load_cifar100_superclass

def load_cifar10(dataset_dir, batch_size, num_workers, pic_size=32):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform_train = transforms.Compose([transforms.Resize(pic_size),
                                    transforms.RandomCrop(pic_size, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    transform_test = transforms.Compose([
                                    transforms.Resize(pic_size),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR10(root=dataset_dir, train=False, download=True,
                           transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_cifar10_target_class(dataset_dir, batch_size, num_workers, target_classes):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform)
    train_targets = np.array(train_dataset.targets)
    idx = np.isin(train_targets, target_classes)
    target_label = train_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    train_dataset.targets = trans_label
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR10(root=dataset_dir, train=False,
                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_targets = np.array(test_dataset.targets)
    idx = np.isin(test_targets, target_classes)
    target_label = test_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    test_dataset.targets = trans_label
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def load_cifar100_sc(dataset_dir, batch_size, num_workers, pic_size=32, custom_defined=False):

    trainset = load_cifar100_superclass(dataset_dir=dataset_dir, is_train=True, pic_size=pic_size, custom_defined=custom_defined)
    testset = load_cifar100_superclass(dataset_dir=dataset_dir, is_train=False, pic_size=pic_size, custom_defined=custom_defined)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def load_cifar100(dataset_dir, batch_size, num_workers, pic_size=32):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.Resize(pic_size),
                                    transforms.RandomCrop(pic_size, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    trainset = CIFAR100(root=dataset_dir, train=True, download=False, transform=transform)
    transform_test = transforms.Compose([transforms.Resize(pic_size),
                                    transforms.ToTensor(),
                                    normalize])
    testset = CIFAR100(root=dataset_dir, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader




def load_svhn(dataset_dir, batch_size, num_workers, pic_size=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([transforms.Resize(pic_size),
                                        transforms.RandomCrop(pic_size, 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
    transform_test = transforms.Compose([
                                    transforms.Resize(pic_size),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_svhn_target_class(dataset_dir, batch_size, num_workers, target_classes):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform)
    train_labels = train_dataset.labels
    idx = np.isin(train_labels, target_classes)
    target_labels = train_labels[idx].tolist()
    trans_labels = np.array([target_classes.index(i) for i in target_labels])
    train_dataset.labels = trans_labels
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_labels = test_dataset.labels
    idx = np.isin(test_labels, target_classes)
    target_labels = test_labels[idx].tolist()
    trans_labels = np.array([target_classes.index(i) for i in target_labels])
    test_dataset.labels = trans_labels
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


if __name__ == '__main__':
    for tc in range(10):
        dataset = load_svhn_target_class(dataset_dir='../data/dataset/svhn', batch_size=128,
                                         num_workers=0, target_classes=[tc])
        print(f'tc_{tc} = {len(dataset[0])}')
