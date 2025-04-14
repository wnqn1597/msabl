# -*- coding: utf-8 -*-
import logging
import pickle
from typing import Any, Optional, Callable, List, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class ABLModel:
    def __init__(
        self,
        model: Any,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Callable[..., Any]] = None,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_interval = save_interval
        self.save_dir = save_dir

    def predict(self, X: Tensor) -> Dict:
        assert X.ndim == 5  # B x n_inst x (img)
        logits, X_feat = self.model(X, return_feat=True)
        prob = torch.softmax(logits, dim=-1)  # B x n_inst x 4
        pred_idx = torch.argmax(prob, dim=-1)  # B x n_inst
        return {
            "pred_idx": pred_idx,
            "pred_prob": prob,
            "X_feat": X_feat
        }

    def eval_predict(self, X: Tensor) -> Tensor:
        assert X.ndim == 5  # B x n_inst x (img)
        logits = self.model(X)
        prob = torch.softmax(logits, dim=-1)  # B x n_inst x 4
        pred_idx = torch.argmax(prob, dim=-1)  # B x n_inst
        return pred_idx

    def train_epoch(self, X: Tensor, y: Tensor) -> float:
        out = self.model.forward_flatten(X)
        # TODO contrastive learning
        loss = self.loss_fn(out, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, X: Tensor, y: Tensor) -> float:
        assert len(X) == len(y)
        assert X.ndim == 4
        B = X.size(0)
        if B == 0:
            print("Skip train.")
            return 0.0
        # B' x n_inst x (img), B' x n_inst
        # y = torch.tensor(y).to(X.device)
        loss_value = self.train_epoch(X, y)
        if self.scheduler is not None:
            self.scheduler.step()
        # print_log(f"model loss: {loss_value:.5f}", logger="current")
        return loss_value


