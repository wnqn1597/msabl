# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
# from logger import print_log

### Simple Image Dataset

class IMGDataset(Dataset):
    def __init__(self, img_dataset, choose_class=None, max_capacity=-1, only_clothes=False):
        self.choose_class = choose_class
        if choose_class is None:
            self.choose_class = list(range(10))
        if only_clothes:
            self.choose_class = [0, 2, 4, 6]

        self.images_list = []
        self.gt_list = []
        self.max_c = max_capacity

        self.src_img = img_dataset
        targets = img_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        assert isinstance(targets, list)

        for i in range(len(img_dataset)):
            if 0 < max_capacity <= len(self.images_list):
                break
            if targets[i] in self.choose_class:
                self.images_list.append(i)
                if only_clothes:
                    self.gt_list.append(targets[i] // 2)
                else:
                    self.gt_list.append(targets[i])

    def __getitem__(self, item):
        idx = self.images_list[item]
        return self.src_img[idx][0], self.gt_list[item]

    def __len__(self):
        return len(self.images_list)

### Instance-level Bag Image Dataset

class ABLDataset(Dataset):
    def __init__(self, img_dataset, fname, only_clothes):
        self.images_list = []
        self.gt_list = []
        self.label_list = []
        self.src_img = img_dataset

        with open(fname) as f:
            for line in tqdm.tqdm(f):
                equation = list(map(int, line.strip().split(" ")))
                operands, ans = equation[:-1], equation[-1]

                self.images_list.append(operands)
                if only_clothes:
                    self.gt_list.append(
                        torch.tensor([img_dataset.targets[x] // 2 for x in operands])
                    )
                else:
                    self.gt_list.append(
                        torch.tensor([img_dataset.targets[x] for x in operands])
                    )
                self.label_list.append(torch.tensor([ans]))

    def __getitem__(self, index):
        operands = self.images_list[index]
        img = torch.stack([self.src_img[x][0] for x in operands])
        label = self.label_list[index]
        gt = self.gt_list[index]
        return index, img, label, gt

    def __len__(self):
        return len(self.images_list)


### Phase

class ABLDatasetLevel(Dataset):
    def __init__(self, img_dataset, fname, only_clothes):
        self.images_list = []
        self.gt_list = []
        self.label_list = []
        self.src_img = img_dataset

        instances = defaultdict(list)
        with open(fname) as f:
            for line in tqdm.tqdm(f):
                equation = list(map(int, line.strip().split(" ")))
                operands, ans = equation[:-1], equation[-1]
                instances[ans].append(operands)

        self.per_target_num = [0] * len(instances.keys())

        for ans in sorted(instances.keys()):
            for operands in instances[ans]:
                self.images_list.append(operands)
                if only_clothes:
                    self.gt_list.append(
                        torch.tensor([img_dataset.targets[x] // 2 for x in operands])
                    )
                else:
                    self.gt_list.append(
                        torch.tensor([img_dataset.targets[x] for x in operands])
                    )
                self.label_list.append(torch.tensor([ans]))
            self.per_target_num[ans] += len(instances[ans])
        self.length = 0

    def set_level(self, lvl):
        if lvl == -1:
            lvl = len(self.per_target_num) - 1
        assert 0 <= lvl < len(self.per_target_num)
        self.length = sum(self.per_target_num[:lvl+1])
        print_log(f"Curriculum LVL={lvl}, num_samples={self.length}.", logger="log")

    def __getitem__(self, index):
        operands = self.images_list[index]
        img = torch.stack([self.src_img[x][0] for x in operands])
        label = self.label_list[index]
        gt = self.gt_list[index]
        return index, img, label, gt

    def __len__(self):
        if self.length == 0:
            self.set_level(-1)
        return self.length


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

train_trans_dict = {
    "mnist": transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    ),
    "fashion": transforms.ToTensor(),
    "cifar": train_transform
}

test_trans_dict = {
    "mnist": transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    ),
    "fashion": transforms.ToTensor(),
    "cifar": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
}

src_data_dict = {
    "mnist": MNIST,
    "fashion": FashionMNIST,
    "cifar": CIFAR10,
}


def getimgdataset(root, img_type, choose_class, max_cap=-1, only_clothes=False, train: bool=True):
    src_data_cls = src_data_dict[img_type]
    train_src_data = src_data_cls(root=root, train=train, transform=train_trans_dict[img_type])
    test_src_data = src_data_cls(root=root, train=train, transform=test_trans_dict[img_type])
    train_dataset = IMGDataset(train_src_data, choose_class, max_cap, only_clothes)
    test_dataset = IMGDataset(test_src_data, choose_class, max_cap, only_clothes)
    return train_dataset, test_dataset

def getabldataset(fname: str, img_type: str, train: bool=True):
    src_data_cls = src_data_dict[img_type]
    only_clothes = (img_type == "fashion") and (fname.find("max") != -1)
    # train_src_data = src_data_cls(root=".", train=train, transform=train_trans_dict[img_type])
    test_src_data = src_data_cls(root=".", train=train, transform=test_trans_dict[img_type])
    # train_dataset = ABLDataset(train_src_data, fname, only_clothes)
    test_dataset = ABLDataset(test_src_data, fname, only_clothes)
    return test_dataset

def getabldatasetlevel(fname: str, img_type: str, train: bool=True):
    src_data_cls = src_data_dict[img_type]
    only_clothes = (img_type == "fashion") and (fname.find("max") != -1)
    # train_src_data = src_data_cls(root=".", train=train, transform=train_trans_dict[img_type])
    test_src_data = src_data_cls(root=".", train=train, transform=test_trans_dict[img_type])
    # train_dataset = ABLDatasetLevel(train_src_data, fname, only_clothes)
    test_dataset = ABLDatasetLevel(test_src_data, fname, only_clothes)
    return test_dataset

def prepare_data(img_type, choose_class, pretrain_num_samples, fname, level=False):
    only_clothes = (img_type == "fashion") and (fname.find("max") != -1)
    pre_ds = getimgdataset(
        img_type, choose_class, pretrain_num_samples,
        only_clothes=only_clothes, train=True
    )
    if level:
        abl_ds = getabldatasetlevel(fname, img_type, train=True)
    else:
        abl_ds = getabldataset(fname, img_type, train=True)
    test_ds = getimgdataset(
        img_type, choose_class, only_clothes=only_clothes, train=False
    )
    return pre_ds, abl_ds, test_ds

def prepare_mnist_addition_data(level):
    img_type = "mnist"
    choose_class = list(range(10))
    pretrain_num_samples = 20
    fname = "dataset/mnist_add_2_train.txt"  # 500
    return prepare_data(img_type, choose_class, pretrain_num_samples, fname, level)

def prepare_fashion_addition_data(level):
    img_type = "fashion"
    choose_class = list(range(10))
    pretrain_num_samples = 40
    fname = "dataset/fashion_mnist_add_2_train.txt"  # 2000
    return prepare_data(img_type, choose_class, pretrain_num_samples, fname, level)

def prepare_cifar_addition_data(level):
    img_type = "cifar"
    choose_class = list(range(10))
    pretrain_num_samples = 100
    fname = "dataset/cifar_add_2_train.txt"  # 10000
    return prepare_data(img_type, choose_class, pretrain_num_samples, fname, level)



def prepare_mnist_max_data(level):
    img_type = "mnist"
    choose_class = list(range(4))
    pretrain_num_samples = 16
    fname = "dataset/mnist_max_4_5_train.txt"  # 200
    return prepare_data(img_type, choose_class, pretrain_num_samples, fname, level)

def prepare_fashion_max_data(level):
    img_type = "fashion"
    choose_class = [0, 2, 4, 6]
    pretrain_num_samples = 40
    fname = "dataset/fashion_mnist_max_4_5_train.txt"  # 1000
    return prepare_data(img_type, choose_class, pretrain_num_samples, fname, level)

def prepare_cifar_max_data(level):
    img_type = "cifar"
    choose_class = list(range(4))
    pretrain_num_samples = 100
    fname = "dataset/cifar_max_4_5_train.txt"  # 5000
    return prepare_data(img_type, choose_class, pretrain_num_samples, fname, level)
