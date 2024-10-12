import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import numpy as np
import random


def load_cifar100_superclass(dataset_dir, is_train, pic_size=32, custom_defined=False):
    # The mean and std could be different in different developers;
    # however, this will not influence the test accuracy much.
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    if is_train:
        transform = transforms.Compose([transforms.Resize(pic_size),
                                        transforms.RandomCrop(pic_size, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
    else:
        transform = transforms.Compose([transforms.Resize(pic_size),
                                        transforms.ToTensor(),
                                        normalize])
    # dataset_dir = '/home/bixh/Documents/MwT_Neural/data/data/dataset'
    dataset =  datasets.CIFAR100(dataset_dir, train=is_train, transform=transform)
    if custom_defined:
        coarse_labels = np.array([12, 13, 11, 5, 5, 9, 2, 6, 11, 1, 
                                1, 11, 0, 10, 2, 19, 1, 0, 5, 7, 
                                9, 5, 1, 0, 6, 9, 6, 5, 1, 5, 
                                8, 5, 3, 7, 16, 11, 4, 7, 4, 17, 
                                9, 14, 16, 16, 5, 6, 11, 7, 14, 7, 
                                4, 5, 7, 12, 2, 5, 7, 12, 10, 7, 
                                7, 1, 2, 5, 15, 4, 15, 3, 7, 0, 
                                2, 0, 5, 8, 5, 5, 0, 3, 18, 6, 
                                5, 10, 2, 12, 9, 0, 17, 9, 16, 14, 
                                10, 13, 2, 3, 9, 8, 7, 15, 11, 18])
    else:
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    
    for i in range(len(dataset.targets)):
        dataset.targets[i] = coarse_labels[dataset.targets[i]]
    return dataset

if __name__ == '__main__':
    dataset_dir = '../data/data/dataset/'
    datasets = load_cifar100_superclass(dataset_dir,is_train=True)
    print(datasets.targets)