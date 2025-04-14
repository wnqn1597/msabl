# -*- coding: utf-8 -*-

from typing import Optional, List, Any
from collections import defaultdict
import numpy as np

class SymbolAccuracy:
    def __init__(self, n_cls: int, prefix: Optional[str] = None) -> None:
        self.n_cls = n_cls
        self.default_prefix = ""
        self.prefix = prefix or self.default_prefix
        self.results = []
        self.cls_results = defaultdict(list)

    def process(self, data: dict) -> None:
        pred_pseudo_label, gt_label = data["pred"], data["gt"]
        pred = np.array(pred_pseudo_label).flatten()
        gt = np.array(gt_label).flatten()

        for i in range(self.n_cls):
            idx = np.where(gt == i)[0]
            if len(idx) == 0:
                continue
            is_eq = (pred[idx] == gt[idx]).astype(float)
            self.cls_results[i].append(is_eq)

        is_eq = (pred == gt).astype(float)
        self.results.append(is_eq)

    def compute_metrics(self) -> dict:
        results = self.results
        assert len(results) > 0
        metrics = dict()
        results = np.concatenate(results, axis=0)
        metrics["acc"] = f"{np.mean(results):.4f}"

        for k in self.cls_results.keys():
            acc = np.mean(np.concatenate(self.cls_results[k], axis=0))
            metrics[str(k)] = f"{acc:.4f}"
        return metrics

    def evaluate(self) -> dict:
        if len(self.results) == 0:
            assert False

        metrics = self.compute_metrics()
        # Add prefix to metrics names
        if self.prefix:
            metrics = {"/".join((self.prefix, k)): v for k, v in metrics.items()}

        # reset the results list
        self.results.clear()
        self.cls_results.clear()
        return metrics

class ReasoningMetric:
    def __init__(self, kb, prefix: Optional[str] = None) -> None:
        self.default_prefix = ""
        self.prefix = prefix or self.default_prefix
        self.kb = kb
        self.results = []
        self.hit = defaultdict(int)
        self.miss = defaultdict(int)

    def process(self, data: dict) -> None:
        pred_pseudo_label, Y = data["pred"], data["tgt"]
        for pred_pseudo_label, y in zip(pred_pseudo_label, Y):
            if self.kb._check_equal(
                self.kb.logic_forward(pred_pseudo_label), y
            ):
                self.results.append(1)
                self.hit[y] += 1
            else:
                self.results.append(0)
                self.miss[y] += 1

    def compute_metrics(self) -> dict:
        results = self.results
        assert len(results) > 0
        metrics = dict()
        metrics["ra"] = f"{sum(results) / len(results):.4f}"
        for i in self.hit.keys():
            tot = self.hit[i] + self.miss[i]
            if tot == 0:
                continue
            metrics[str(i)] = f"{self.hit[i]}/{tot} = {self.hit[i]/tot:.4f}"
        return metrics

    def evaluate(self) -> dict:
        if len(self.results) == 0:
            assert False

        metrics = self.compute_metrics()
        # Add prefix to metrics names
        if self.prefix:
            metrics = {"/".join((self.prefix, k)): v for k, v in metrics.items()}

        # reset the results list
        self.results.clear()
        self.hit.clear()
        self.miss.clear()
        return metrics