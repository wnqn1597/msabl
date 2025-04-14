# -*- coding: utf-8 -*-
import os
from collections import defaultdict
import random
import numpy as np
import tqdm
from collections import Counter
# from torchvision.datasets import MNIST, FashionMNIST, CIFAR10


def gen_max_sample(num2idx: dict, y: int, arity: int):
    gt = np.random.randint(0, y + 1, arity)
    return gt.tolist()
    # if np.max(gt) != y:
    #     gt[np.random.randint(0, arity)] = y
    # idx = [random.choice(num2idx[i]) for i in gt]
    # return idx

def gen_max_sample_uni(num2idx: dict, y: int, arity: int):
    gt = np.random.poisson(y, arity).clip(max=y)
    if np.max(gt) != y:
        gt[np.random.randint(0, arity)] = y
    idx = [random.choice(num2idx[i]) for i in gt]
    print(gt)
    return idx

def gen_conjEq_sample(num2idx: dict):
    op1, op2 = np.random.randint(0, 2, (2,))
    gt = [op1, op2, op1 & op2]
    idx = [random.choice(num2idx[i]) for i in gt]
    return idx

def gen_add_sample(num2idx: dict, n_nums: int=2):
    cand = np.random.randint(0, 10, (n_nums,))
    idx = [random.choice(num2idx[i]) for i in cand]
    # print(cand, idx)
    return idx, sum(cand)


class MNISTDataGenerator:
    def __init__(self, root="."):
        train_dataset = MNIST(root=root, train=True)
        test_dataset = MNIST(root=root, train=False)
        train_num2idx = defaultdict(list)
        test_num2idx = defaultdict(list)
        l = len(train_dataset)
        for i in range(l):
            train_num2idx[train_dataset[i][1]].append(i)
        l = len(test_dataset)
        for i in range(l):
            test_num2idx[test_dataset[i][1]].append(i)
        self.num2idx = {
            "train": train_num2idx, "test": test_num2idx,
        }
        self.save_dir = "dataset"

    def gen_add_dataset(self, num_samples, n_nums, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)

        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"mnist_add_{n_nums}_{is_train}_large.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for _ in tqdm.tqdm(range(num_samples)):
            idx, ans = gen_add_sample(num2idx, n_nums)
            line = " ".join(map(str, idx + [ans])) + "\n"
            f.write(line)
        f.close()

    def gen_conjEq_dataset(self, num_samples, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)
        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"mnist_conjEq_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for _ in tqdm.tqdm(range(num_samples)):
            idx = gen_conjEq_sample(num2idx)
            line = " ".join(map(str, idx + [0])) + "\n"
            f.write(line)
        f.close()

    def gen_max_dataset(self, num_samples, n, arity, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)
        if isinstance(num_samples, int):
            num_samples = [num_samples] * n
        elif isinstance(num_samples, list):
            assert len(num_samples) == n
        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"mnist_max_{n}_{arity}_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for y in range(n):
            for _ in range(num_samples[y]):
                idx = gen_max_sample(num2idx, y, arity)
                line = " ".join(map(str, idx + [y])) + "\n"
                f.write(line)
        f.close()

class FashionMNISTDataGenerator:
    def __init__(self, root=".", only_up=True):
        train_dataset = FashionMNIST(root=root, train=True)
        test_dataset = FashionMNIST(root=root, train=False)
        train_num2idx = defaultdict(list)
        test_num2idx = defaultdict(list)

        if only_up:
            l = len(train_dataset)
            for i in range(l):
                if train_dataset[i][1] in [0, 2, 4, 6]:
                    train_num2idx[train_dataset[i][1] // 2].append(i)
            l = len(test_dataset)
            for i in range(l):
                if test_dataset[i][1] in [0, 2, 4, 6]:
                    test_num2idx[test_dataset[i][1] // 2].append(i)
        else:
            l = len(train_dataset)
            for i in range(l):
                train_num2idx[train_dataset[i][1]].append(i)
            l = len(test_dataset)
            for i in range(l):
                test_num2idx[test_dataset[i][1]].append(i)
        self.num2idx = {
            "train": train_num2idx, "test": test_num2idx,
        }
        self.save_dir = "dataset"

    def gen_max_dataset(self, num_samples, n, arity, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)
        if isinstance(num_samples, int):
            num_samples = [num_samples] * n
        elif isinstance(num_samples, list):
            assert len(num_samples) == n
        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"fashion_mnist_max_{n}_{arity}_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for y in range(n):
            for _ in range(num_samples[y]):
                idx = gen_max_sample(num2idx, y, arity)
                line = " ".join(map(str, idx + [y])) + "\n"
                f.write(line)
        f.close()

    def gen_add_dataset(self, num_samples, n_nums, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)

        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"fashion_mnist_add_{n_nums}_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for _ in tqdm.tqdm(range(num_samples)):
            idx, ans = gen_add_sample(num2idx, n_nums)
            line = " ".join(map(str, idx + [ans])) + "\n"
            f.write(line)
        f.close()

class CIFARDataGenerator:
    def __init__(self, root="."):
        train_dataset = CIFAR10(root=root, train=True)
        test_dataset = CIFAR10(root=root, train=False)
        train_num2idx = defaultdict(list)
        test_num2idx = defaultdict(list)
        l = len(train_dataset)
        for i in range(l):
            train_num2idx[train_dataset[i][1]].append(i)
        l = len(test_dataset)
        for i in range(l):
            test_num2idx[test_dataset[i][1]].append(i)
        self.num2idx = {
            "train": train_num2idx, "test": test_num2idx,
        }
        self.save_dir = "dataset"

    def gen_add_dataset(self, num_samples, n_nums, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)

        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"cifar_add_{n_nums}_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for _ in tqdm.tqdm(range(num_samples)):
            idx, ans = gen_add_sample(num2idx, n_nums)
            line = " ".join(map(str, idx + [ans])) + "\n"
            f.write(line)
        f.close()

    def gen_max_dataset(self, num_samples, n, arity, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)
        if isinstance(num_samples, int):
            num_samples = [num_samples] * n
        elif isinstance(num_samples, list):
            assert len(num_samples) == n
        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"cifar_max_{n}_{arity}_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for y in range(n):
            for _ in range(num_samples[y]):
                idx = gen_max_sample(num2idx, y, arity)
                line = " ".join(map(str, idx + [y])) + "\n"
                f.write(line)
        f.close()

    def gen_max_dataset_uni(self, num_samples, n, arity, seed, train=True):
        random.seed(seed)
        np.random.seed(seed)
        if isinstance(num_samples, int):
            num_samples = [num_samples] * n
        elif isinstance(num_samples, list):
            assert len(num_samples) == n
        is_train = 'train' if train else 'test'
        num2idx = self.num2idx[is_train]
        fname = f"cifar_max_{n}_{arity}_{is_train}.txt"
        f = open(os.path.join(self.save_dir, fname), "w")
        for y in range(n):
            for _ in range(num_samples[y]):
                idx = gen_max_sample_uni(num2idx, y, arity)
                line = " ".join(map(str, idx + [y])) + "\n"
                f.write(line)
        f.close()


def gen_max_dataset(num_samples, n, arity, seed, train=True):
    random.seed(seed)
    np.random.seed(seed)
    if isinstance(num_samples, int):
        num_samples = [num_samples] * n
    elif isinstance(num_samples, list):
        assert len(num_samples) == n

    c = Counter([])
    for y in range(n):
        for _ in range(num_samples[y]):
            idx = gen_max_sample(None, y, arity)
            c += Counter(idx)
    print(c)

if __name__ == '__main__':
    # arity = 5
    # num_samples = [20, 40, 60, 80]
    # generator = MNISTDataGenerator()
    # generator.gen_max_dataset(num_samples, 4, 5, 721, True)
    # generator.gen_conjEq_dataset(100, 1919)
    # generator.gen_add_dataset(5000, 2, 38)

    # generator = FashionMNISTDataGenerator(only_up=False)
    # num_samples = [100, 200, 300, 400]
    # generator.gen_add_dataset(2000, 2, 42)

    # generator = CIFARDataGenerator()
    num_samples = np.exp(0.11 * np.arange(10))*300
    num_samples = num_samples.astype(int)
    print(num_samples)
    # print()
    print(np.add.accumulate(num_samples / num_samples.sum()))
    # generator.gen_max_dataset_uni(num_samples, 10, 16, 42, True)

    # num_samples = np.exp(0.6 * np.arange(10))*15
    # num_samples = num_samples.astype(int)
    # num_samples = num_samples.tolist()
    # print(num_samples, sum(num_samples))
    # gen_max_dataset(num_samples, 10, 8, 42, True)