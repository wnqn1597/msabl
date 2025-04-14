import argparse
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from wmc_abducer import WMCAbducer
from nn import LeNet5, ResNet18Classifier
from abl_model import ABLModel
from kb import PrologKB
from reasoner import Reasoner
from bridge import Bridge
from metric import SymbolAccuracy
from dataset import (
    prepare_mnist_addition_data, prepare_mnist_max_data,
    prepare_fashion_addition_data, prepare_fashion_max_data,
    prepare_cifar_addition_data, prepare_cifar_max_data,
)
from logger import print_log, ABLLogger
import warnings
warnings.filterwarnings("ignore")

def get_method_name(use_ablsim, use_a3bl, use_level, use_nov, use_rej):
    if use_ablsim:
        return "ablsim"
    if use_a3bl:
        return "a3bl"
    if use_level:
        if use_nov:
            return "ablpn"
        if use_rej:
            return "ablpr"
        return "ablp"
    return "abl"

def main(args):
    use_cuda = not args.use_cpu
    print("use cuda" if use_cuda else "use cpu")
    task = args.task
    use_ablsim = args.use_ablsim
    use_a3bl = args.use_a3bl
    use_level = args.use_level
    use_nov = args.use_novelty
    use_rej = args.use_rejection
    if use_nov:
        assert use_level and not use_rej
    if use_rej:
        assert use_level and not use_nov and task.find("max") != -1

    method_name = get_method_name(use_ablsim, use_a3bl, use_level, use_nov, use_rej)
    if task.find("max") != -1:
        n_cls = n_tau = 4
        n_pos = 5
        kb_file = "prolog/max.pl"

        level_list = [0, 1, 2, 3]
        new_class_list = [[i] for i in range(4)]
        if task.find("mnist") != -1:
            loop_list = [1, 2, 2, 3]
            (pre_ds_train, pre_ds), abl_ds, (_, test_ds) = prepare_mnist_max_data(use_level)
        elif task.find("fashion") != -1:
            loop_list = [1, 3, 3, 4]
            (pre_ds_train, pre_ds), abl_ds, (_, test_ds) = prepare_fashion_max_data(use_level)
        elif task.find("cifar") != -1:
            loop_list = [1, 4, 5, 5]
            (pre_ds_train, pre_ds), abl_ds, (_, test_ds) = prepare_cifar_max_data(use_level)
        else:
            raise ValueError
        wmc = WMCAbducer(n_cls, n_pos, n_tau, "max") if use_a3bl else None
    elif task.find("add") != -1:
        n_cls, n_tau = 10, 19
        n_pos = 2
        kb_file = "prolog/add.pl"
        level_list = list(range(9)) + [-1]
        new_class_list = [[i] for i in range(10)]
        MSE = 0
        # level_list = level_list[:4]
        # new_class_list = new_class_list[:4]
        if task.find("mnist") != -1:
            loop_list = [1]*9 + [5]
            (pre_ds_train, pre_ds), abl_ds, (_, test_ds) = prepare_mnist_addition_data(use_level)
        elif task.find("fashion") != -1:
            loop_list = [1] + [2]*8 + [5]
            (pre_ds_train, pre_ds), abl_ds, (_, test_ds) = prepare_fashion_addition_data(use_level)
        elif task.find("cifar") != -1:
            loop_list = [1, 2, 2] + [3]*6 + [6]
            (pre_ds_train, pre_ds), abl_ds, (_, test_ds) = prepare_cifar_addition_data(use_level)
        else:
            raise ValueError
        wmc = WMCAbducer(n_cls, n_pos, n_tau, "add") if use_a3bl else None
    else:
        raise ValueError
    if args.loop_list is not None:
        assert len(args.loop_list) == len(level_list)
        loop_list = args.loop_list
    if use_a3bl:
        cls_idx = torch.tensor(pre_ds_train.gt_list)
        pre_ds_train.gt_list = torch.nn.functional.one_hot(cls_idx, n_cls).float()

    if task.find("cifar") != -1:
        cls = ResNet18Classifier(n_cls)
    else:
        cls = LeNet5(n_cls)

    if use_cuda:
        cls.cuda()
    pre_loss_fn = nn.CrossEntropyLoss()
    pre_optimizer = optim.Adam(cls.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-1)
    optimizer = optim.Adam(cls.parameters(), lr=args.lr)

    model = ABLModel(cls, loss_fn, optimizer, None)

    kb = PrologKB(list(range(n_cls)), kb_file)
    reasoner = Reasoner(
        kb,
        max_revision=args.max_revision,
        require_more_revision=args.require_more_revision,
        use_zoopt=False,
        wmc_abducer=wmc,
    )

    # metric_list = [SymbolAccuracy(), ReasoningMetric(kb=kb)]
    metric_list = [SymbolAccuracy(n_cls)]
    bridge = Bridge(
        model,
        reasoner,
        metric_list,
        use_cuda=use_cuda,
        use_novelty=use_nov,
        use_ablsim=use_ablsim,
        use_a3bl=use_a3bl,
        use_exp=use_rej,
        dataloader_num_workers=args.num_workers
    )

    path = osp.join("results", task)
    os.makedirs(path, exist_ok=True)
    local_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    log_file = osp.join(path, method_name + "_" + local_time + ".log")
    ABLLogger.get_instance(name="log", log_file=log_file)
    print_log(str(args), logger="log")

    bridge.pretrain(
        pre_ds_train,
        pre_loss_fn,
        pre_optimizer,
        test_ds,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.pretrain_epochs,
    )

    start = time.time()
    if use_level:
        assert len(loop_list) == len(level_list) == len(new_class_list)
        print_log(f"Level setting: {loop_list} {level_list} {new_class_list}", logger="log")
        bridge.curriculum_train(
            abl_data=abl_ds,
            pretrain_data=pre_ds,
            pretrain_data_train=pre_ds_train,
            test_data=test_ds,
            level_list=level_list,
            loop_list=loop_list,
            new_class_list=new_class_list,
            nbpt_list=args.nbpt_list,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_interval=args.eval_interval,
            epochs=args.epochs,
            first_level_epochs=args.first_level_epochs,
            # num_batch_per_train=args.num_batch_per_train
            reweight=args.reweight,
        )
    else:
        bridge.train(
            abl_data=abl_ds,
            pretrain_data=pre_ds,
            pretrain_data_train=pre_ds_train,
            test_data=test_ds,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_interval=args.eval_interval,
            loops=args.loops,
            epochs=args.epochs,
            num_batch_per_train=args.num_batch_per_train
        )

    end = time.time()
    print_log(f"Elapsed: {end-start:.2f} s.", logger="log")

def get_args():
    parser = argparse.ArgumentParser(description="ABL")
    parser.add_argument("--task", type=str, default="mnist-add")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--loop_list", type=int, nargs="+", default=[
        1,2,2,2,2,2,2,2,2,2
    ])
    parser.add_argument("--nbpt_list", type=int, nargs="+", default=[
        1,1,1,1,1,1,2,2,2,4
    ])
    # parser.add_argument("--loop_list", type=int, nargs="+", default=[
    #     1,4,4,4,3,3,3,3,3,10
    # ])
    # parser.add_argument("--nbpt_list", type=int, nargs="+", default=[
    #     1,1,1,1,1,1,1,1,1,2
    # ])
    parser.add_argument("--first_level_epochs", type=int, default=2)
    parser.add_argument("--pretrain_epochs", type=int, default=20)  # 20
    parser.add_argument("--epochs", type=int, default=10)  # 15
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=256)  # 256
    parser.add_argument("--eval_batch_size", type=int, default=256) # 256
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_batch_per_train", type=int, default=1)
    parser.add_argument("--max_revision", type=int, default=-1)
    # NOTE ABLSim calls for lots of abduction candidates, require more revision should be large.
    parser.add_argument("--require_more_revision", type=int, default=0)
    parser.add_argument("--use_ablsim", action="store_true")
    parser.add_argument("--use_a3bl", action="store_true")
    parser.add_argument("--use_level", action="store_true")
    parser.add_argument("--use_novelty", action="store_true")
    parser.add_argument("--use_rejection", action="store_true")
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.use_level = True
    args.use_cpu = True
    # args.use_rejection = True
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    arguments = get_args()
    set_seed(arguments.seed)
    main(arguments)
