# -*- coding: utf-8 -*-
import copy
from typing import Any, List, Optional, Dict, Union

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

import torch
import tqdm
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from dataset import IMGDataset, ABLDataset, ABLDatasetLevel
from abl_model import ABLModel
from reasoner import Reasoner
from logger import print_log


class Bridge:
    def __init__(
        self,
        model: ABLModel,
        reasoner: Reasoner,
        metric_list: List,
        use_novelty: bool=False,
        use_ablsim: bool=False,
        use_a3bl: bool=False,
        use_cuda: bool=True,
        use_exp: bool=False,
        dataloader_num_workers: int=0
    ) -> None:
        self.model = model
        self.reasoner = reasoner
        self.metric_list = metric_list
        self.nd_model = None
        self.use_novelty = use_novelty
        self.use_ablsim = use_ablsim
        self.use_a3bl = use_a3bl
        self.use_cuda = use_cuda

        self.exp = use_exp
        self.pretrain_epochs = -1
        self.num_workers = dataloader_num_workers

    def abduce_pseudo_label(
        self, output: Dict, Y: Tensor, novelty: ndarray, x: Tensor,
        labeled_prob, labeled_feat, labeled_y
    ) -> List[List[Any]]:
        pred_idx, pred_probs, X_feat = output["pred_idx"], output["pred_prob"], output["X_feat"]
        # B x inst, B x inst x 4, B
        pred_idx = [pi.detach().cpu().int().tolist() for pi in pred_idx]  # [List]
        pred_probs = [pp.detach().cpu().numpy() for pp in pred_probs]  # [ndarray]
        Y = Y.detach().cpu().squeeze(-1).int().tolist()  # [int]
        XF = [xf.detach().cpu() for xf in X_feat]  # [Tensor]
        novelty = [nov for nov in novelty] if novelty is not None else [None]*len(x)  # [ndarray]
        if self.use_ablsim:
            # x: Tensor(B x inst x img)
            abduced_pseudo_label = self.reasoner.batch_abduce_sim(
                pred_idx, pred_probs, Y, XF, novelty, -1,
                labeled_prob=labeled_prob,
                labeled_feat=labeled_feat,
                labeled_y=labeled_y,
            )
        elif self.use_a3bl:
            abduced_pseudo_label = self.reasoner.batch_abduce_a3bl(
                pred_idx, pred_probs, Y, XF, novelty
            )
        elif self.exp:
            abduced_pseudo_label = self.reasoner.batch_abduce_exp(
                pred_idx, pred_probs, Y, XF, novelty
            )
        else:
            abduced_pseudo_label = self.reasoner.batch_abduce(
                pred_idx, pred_probs, Y, XF, novelty
            )
        return abduced_pseudo_label

    def pretrain(
        self,
        pretrain_data_train: Dataset,
        loss_fn,
        optimizer,
        test_data: Optional[Dataset] = None,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        epochs: int = 1,
    ):
        print_log("Pretrain start: loop [0]", logger="log")
        self.pretrain_epochs = epochs
        if epochs > 0:
            print_log(f"Pretrain sample num: {len(pretrain_data_train)}", logger="log")
            pretrain_dl = self.get_dataloader(pretrain_data_train, train_batch_size, True)
            self.model.model.train()
            for _ in tqdm.tqdm(range(epochs)):
                for x, y in pretrain_dl:
                    if self.use_cuda:
                        x, y = x.cuda(), y.cuda()
                    out = self.model.model.forward_flatten(x)
                    loss = loss_fn(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        self.model.model.eval()
        test_dl = self.get_dataloader(test_data, eval_batch_size, False)
        msg = self.test_accuracy(test_dl)
        print_log(msg, logger="log")

    def get_labeled_lists(self, pret_dl):
        if not self.use_ablsim:
            return None, None, []
        if self.pretrain_epochs <= 0:
            return None, None, []
        labeled_prob, labeled_feat, labeled_y = [], [], []
        self.model.model.eval()
        with torch.no_grad():
            for bidx, batch in enumerate(pret_dl):
                x, gt = batch  # on gpu
                labeled_y.append(gt.numpy())
                if self.use_cuda:
                    x = x.cuda()
                out, feat = self.model.model.forward_flatten(x, True)
                labeled_prob.append(torch.softmax(out, -1).cpu().numpy())
                labeled_feat.append(feat.cpu().numpy())
        labeled_prob = np.concatenate(labeled_prob)
        labeled_feat = np.concatenate(labeled_feat)
        labeled_y = np.concatenate(labeled_y).tolist()
        return labeled_prob, labeled_feat, labeled_y

    def get_dataloader(self, dataset, bsz, shuffle):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=shuffle,
            num_workers=self.num_workers if self.use_cuda else 0,
            pin_memory=self.use_cuda,
        )

    def train(
        self,
        abl_data: Union[ABLDataset, ABLDatasetLevel],
        # abl_data_train: Dataset,
        pretrain_data: IMGDataset,
        pretrain_data_train: IMGDataset,
        val_data: Dataset=None,
        test_data: Dataset=None,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        loops: int = 1,
        epochs: int = 1,
        num_batch_per_train: int = 8,
        eval_interval: int = 1,
        test_interval: int = 1,
        reweight: bool = False,
    ):
        if val_data is None:
            val_data = abl_data

        # DataLoader
        abl_batch_size = eval_batch_size
        while abl_batch_size * 2 > len(abl_data):
            abl_batch_size //= 2
        abl_dl = self.get_dataloader(abl_data, abl_batch_size, True)
        val_dl = self.get_dataloader(val_data, eval_batch_size, False)
        test_dl = self.get_dataloader(test_data, eval_batch_size, False)
        pret_dl = self.get_dataloader(pretrain_data, eval_batch_size, False)

        if self.use_novelty and self.nd_model is not None:
            model_for_novelty = copy.deepcopy(self.model)
            model_for_novelty.model.eval()
        else:
            model_for_novelty = None

        labeled_prob, labeled_feat, labeled_y = self.get_labeled_lists(pret_dl)

        for loop in range(loops):
            print_log(f"Train: loop [{loop + 1}/{loops}]", logger="log")

            training_idx_list, training_ai_list, training_gt_list, training_nov_list = [], [], [], []
            self.model.model.eval()
            # TODO train scratch model with abduction
            iterator = tqdm.tqdm(enumerate(abl_dl), total=len(abl_dl), disable=True)
            for bidx, batch in iterator:
                ds_idx, x, y, gt = batch
                if self.use_cuda:
                    x = x.cuda()
                with torch.no_grad():
                    output = self.model.predict(x)  # B x inst, B x inst x 4
                    if self.use_novelty and self.nd_model is not None and loop == 0:
                        feat = model_for_novelty.predict(x)["X_feat"]
                        novelty = self.nd_predict(feat)
                        # novelty = self.nd_predict(output["X_feat"])
                    else:
                        novelty = None
                # self.exp = loop + 1 != loops
                abduced_idx = self.abduce_pseudo_label(
                    output, y, novelty, x,
                    labeled_prob, labeled_feat, labeled_y
                )  # B x inst (B x inst x cls  in a3bl)

                training_idx_list.append(ds_idx)
                if self.use_a3bl:
                    abduced_idx = np.stack(abduced_idx)  # B x inst x cls
                training_ai_list.append(torch.tensor(abduced_idx))
                training_gt_list.append(gt)
                if self.use_novelty and novelty is not None:
                    # print(novelty)
                    training_nov_list.append(novelty)

                if (bidx + 1) % num_batch_per_train == 0 or bidx + 1 == len(abl_dl):
                    training_idx = torch.cat(training_idx_list)  # N
                    training_ai = torch.cat(training_ai_list)  # N x n_inst (N x n_inst x cls  in a3bl)
                    training_gt = torch.cat(training_gt_list)  # N x n_inst
                    training_ai_flatten = training_ai.flatten(0, 1)  # (N x cls  in a3bl)
                    training_gt_flatten = training_gt.flatten(0, 1)

                    if self.use_novelty and len(training_nov_list) > 0:
                        training_nov_flatten = torch.tensor(np.concatenate(training_nov_list, 0).flatten())  # N x n_inst
                        assert len(self.reasoner.new_class) == 1
                        nc = self.reasoner.new_class[0]
                        where_nov = torch.where(training_nov_flatten == -1)[0]
                        where_nc = torch.where(training_gt_flatten == nc)[0]
                        nov_p = (training_gt_flatten[where_nov] == nc).float().mean().item()
                        nov_r = (training_nov_flatten[where_nc] == -1).float().mean().item()
                        print_log(f"Novelty P&R [bidx|loop][{bidx + 1}|{loop + 1}]: {nov_p:.4f} {nov_r:.4f}.", logger="log")

                    # Eval abduction
                    if not self.use_a3bl:
                        filt_idx = torch.where(training_ai_flatten != -1)[0]
                        acc = (training_ai_flatten[filt_idx] == training_gt_flatten[filt_idx]).float().mean().item()
                    else:
                        acc = (torch.argmax(training_ai_flatten, -1) == training_gt_flatten).float().mean().item()
                    print_log(f"Abduction acc [bidx|loop][{bidx + 1}|{loop + 1}]: {acc:.4f}", logger="log")
                    # Prepare dataset
                    train_data = copy.deepcopy(pretrain_data_train)
                    src_img_idx = []
                    for i in training_idx:
                        src_img_idx.extend(abl_data.images_list[i])
                    src_img_idx = torch.tensor(src_img_idx)

                    if not self.use_a3bl:
                        train_data.images_list = src_img_idx[filt_idx].tolist()
                        train_data.gt_list = training_ai_flatten[filt_idx].tolist()
                    else:
                        train_data.images_list = src_img_idx.tolist()
                        train_data.gt_list = training_ai_flatten.float()

                    if self.exp:
                        train_len = len(train_data.images_list)
                        seg_len = len(src_img_idx)
                        rej_ratio = (seg_len - train_len) / seg_len
                        print_log(f"Train sample num [bidx|loop][{bidx + 1}|{loop + 1}]: {train_len}/{seg_len}. Reject: {rej_ratio:.4f}.", logger="log")
                        if reweight:
                            # from collections import Counter
                            # print(Counter(train_data.gt_list))
                            nc_weight = [1, 1.5, 2.0, 3.0]
                            loss_weight = torch.ones(4).float()
                            nc = self.reasoner.new_class[0]
                            loss_weight[nc] *= nc_weight[nc]
                            # print("lossweight", loss_weight)
                            if self.use_cuda:
                                loss_weight = loss_weight.cuda()
                            self.model.loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)
                    if self.pretrain_epochs <= 0:
                        concat_data = train_data
                    else:
                        concat_data = ConcatDataset([pretrain_data_train, train_data])
                    train_dl = self.get_dataloader(concat_data, train_batch_size, True)

                    self.model.model.train()
                    for e in range(epochs):
                        for train_batch in train_dl:
                            x, abduced_idx = train_batch
                            if len(x) == 1:  # BatchNorm need more samples
                                continue
                            if self.use_cuda:
                                x, abduced_idx = x.cuda(), abduced_idx.cuda()
                            self.model.train(x, abduced_idx)
                    self.model.model.eval()
                    labeled_prob, labeled_feat, labeled_y = self.get_labeled_lists(pret_dl)
                    training_idx_list, training_ai_list, training_gt_list, training_nov_list = [], [], [], []

            if (loop + 1) % eval_interval == 0 or loop == loops - 1:
                msg = self._valid(val_dl)
                print_log(f"loop [{loop + 1}] " + msg, logger="log")

            if test_dl is not None and (loop + 1) % test_interval == 0 or loop == loops - 1:
                msg = self.test_accuracy(test_dl)
                print_log(f"loop [{loop + 1}] " + msg, logger="log")


    def curriculum_train(
        self,
        loop_list: List,
        new_class_list: List,
        level_list: List,
        nbpt_list: List,
        abl_data: ABLDatasetLevel,
        pretrain_data: IMGDataset,
        pretrain_data_train: IMGDataset,
        val_data: Dataset=None,
        test_data: Dataset=None,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        eval_interval: int = 1,
        epochs: int = 1,
        first_level_epochs: int = 1,
        reweight: bool=False,
    ):
        assert val_data is None
        assert hasattr(abl_data, "set_level")
        for i, (lvl, loops, nc, nbpt) in enumerate(zip(level_list, loop_list, new_class_list, nbpt_list)):
            if self.use_novelty and i != 0:
                self.nd_train(abl_data, eval_batch_size)
            self.reasoner.new_class = nc
            abl_data.set_level(lvl)
            self.train(
                abl_data=abl_data,
                # abl_data_train=abl_data_train,
                pretrain_data=pretrain_data,
                pretrain_data_train=pretrain_data_train,
                val_data=val_data,
                test_data=test_data,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                loops=loops,
                epochs=epochs if i != 0 else first_level_epochs,
                eval_interval=eval_interval,
                num_batch_per_train=nbpt,
                reweight=reweight,
            )

    def nd_train(self, abl_data: Dataset, eval_batch_size: int, nei: int=10):
        abl_dl = self.get_dataloader(abl_data, eval_batch_size, False)
        feats = []
        for bidx, batch in enumerate(abl_dl):
            _, x, y, gt = batch
            if self.use_cuda:
                x = x.cuda()
            outputs = self.model.predict(x)
            feat = outputs["X_feat"].flatten(0, 1)  # b x feat_dim
            feats.append(feat)
        feats = torch.cat(feats).detach().cpu().numpy()  # B x feat_dim
        lof = LocalOutlierFactor(
            n_neighbors=nei,
            novelty=True,
            metric="cosine",
        )
        lof.fit(feats)
        self.nd_model = lof


    def nd_predict(self, x: Tensor) -> ndarray:
        b, n = x.shape[0], x.shape[1]
        x = x.flatten(0, 1).detach().cpu().numpy()
        assert x.ndim == 2  # B x feat_dim
        pred = self.nd_model.predict(x)
        pred = pred.reshape(b, n)
        return pred

    def _valid(self, val_dl: DataLoader) -> str:
        with torch.no_grad():
            for bidx, batch in enumerate(val_dl):
                _, x, y, gt = batch  # on gpu
                if self.use_cuda:
                    x = x.cuda()
                pred_idx = self.model.eval_predict(x)  # B x n
                pred_pseudo_label = pred_idx.cpu().int().tolist()
                y = y.cpu().squeeze().int().tolist()
                gt = gt.cpu().squeeze().int().tolist()
                for metric in self.metric_list:
                    metric.process({
                        "pred": pred_pseudo_label,
                        "gt": gt,
                        "tgt": y,
                    })
        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Eval: "
        for k, v in res.items():
            msg += k + f": {v} "
        return msg

    # def test(self, test_data: Dataset, test_batch_size: int = 1,) -> None:
    #     print("Test start:")
    #     dl = self.get_dataloader(test_data, test_batch_size, False)
    #     self._valid(dl)

    def test_accuracy(self, test_dl: DataLoader) -> str:
        with torch.no_grad():
            for x, gt in test_dl:
                if self.use_cuda:
                    x = x.cuda()
                out = self.model.model.forward_flatten(x)
                pred_idx = torch.argmax(out, dim=-1)
                pred_pseudo_label = pred_idx.cpu().int().tolist()
                ground_truth = gt.cpu().squeeze().int().tolist()
                for metric in self.metric_list:
                    metric.process({
                        "pred": pred_pseudo_label,
                        "gt": ground_truth,
                        "tgt": None,
                    })

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        msg = "Test: "
        for k, v in res.items():
            msg += k + f": {v} "
        return msg

