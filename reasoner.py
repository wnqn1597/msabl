# -*- coding: utf-8 -*-

from typing import Any, Callable, List, Union

import numpy as np
from zoopt import Dimension, Objective, Opt, Parameter, Solution
from itertools import product

from wmc_abducer import WMCAbducer
from kb import KBBase
# from similarity_calculator import nn_select_batch_abduced_result

def confidence_dist(
    pred_prob: np.ndarray, candidates_idxs: List[List[Any]]
) -> np.ndarray:
    pred_prob = np.clip(pred_prob, 1e-9, 1)
    cols = np.arange(len(candidates_idxs[0]))[None, :]
    return 1 - np.prod(pred_prob[cols, candidates_idxs], axis=1)


def avg_confidence_dist(
    pred_prob: np.ndarray, candidates_idxs: List[List[Any]]
) -> np.ndarray:
    cols = np.arange(len(candidates_idxs[0]))[None, :]
    return 1 - np.average(pred_prob[cols, candidates_idxs], axis=1)

class Reasoner:
    def __init__(
        self,
        kb: KBBase,
        dist_func: Union[str, Callable] = "confidence",
        max_revision: Union[int, float] = -1,
        require_more_revision: int = 0,
        use_zoopt: bool = False,
        wmc_abducer: WMCAbducer=None,
    ):
        self.kb = kb
        self.dist_func = dist_func
        self.use_zoopt = use_zoopt
        self.max_revision = max_revision
        self.require_more_revision = require_more_revision

        self.new_class = []
        self.wmc_abduer = wmc_abducer

    def _get_k_candidates(
        self,
        pred_prob: np.ndarray,
        candidates: List[List[Any]],
        k: int=1,
    ) -> List[Any]:
        if k <= 0:
            return candidates
        if len(candidates) == 0:
            return []
        cost_array = self._get_cost_list(pred_prob, candidates)
        if len(cost_array) < k:
            min_indices = range(len(cost_array))
        else:
            min_indices = np.argpartition(cost_array, k - 1)[:k]
        # candidate = candidates[np.argmin(cost_array)]
        k_candidates = [candidates[i] for i in min_indices]
        return k_candidates

    def _get_cost_list(
        self,
        pred_prob: np.ndarray,
        candidates: List[List[Any]],
    ) -> Union[List[Union[int, float]], np.ndarray]:
        if self.dist_func == "confidence":
            return confidence_dist(pred_prob, candidates)
        elif self.dist_func == "avg_confidence":
            return avg_confidence_dist(pred_prob, candidates)
        else:
            raise NotImplementedError

    def _zoopt_get_solution(
        self,
        pred_pseudo_label,
        pred_prob,
        Y,
        X,
        symbol_num: int,
        max_revision_num: int,
    ) -> Solution:
        # 维数=symbol_num，0或1，离散
        dimension = Dimension(
            size=symbol_num, regs=[[0, 1]] * symbol_num, tys=[False] * symbol_num
        )
        objective = Objective(
            lambda sol: self.zoopt_score(
                pred_pseudo_label, pred_prob, Y, X, symbol_num, sol
            ),
            dim=dimension,
            constraint=lambda sol: self._constrain_revision_num(
                sol, max_revision_num
            ),  # max - 1.T@sol >= 0
        )
        parameter = Parameter(
            budget=self.zoopt_budget(symbol_num),
            intermediate_result=False,
            autoset=True,
        )
        solution = Opt.min(objective, parameter)
        return solution

    def zoopt_score(
        self, pred_pseudo_label, pred_prob, Y, X, symbol_num: int, sol: Solution,
    ) -> int:
        revision_idx = np.where(sol.get_x() != 0)[0]
        candidates, reasoning_results = self.kb.revise_at_idx(
            pred_pseudo_label, Y, X, revision_idx
        )
        score = symbol_num
        if len(candidates) > 0:
            score = np.min(
                self._get_cost_list(pred_prob, candidates)
            )
        return score

    def zoopt_budget(self, symbol_num: int) -> int:
        return 10 * symbol_num

    def _constrain_revision_num(self, solution: Solution, max_revision_num: int) -> int:
        x = solution.get_x()
        return max_revision_num - x.sum()

    def _get_max_revision_num(
        self, max_revision: Union[int, float], symbol_num: int
    ) -> int:
        if not isinstance(max_revision, (int, float)):
            raise TypeError
        if max_revision == -1:
            return symbol_num
        if isinstance(max_revision, float):
            if not 0 <= max_revision <= 1:
                raise ValueError
            return round(symbol_num * max_revision)
        if max_revision < 0:
            raise ValueError
        return max_revision

    def abduce(
        self,
        pred_pseudo_label: List[Any],
        pred_prob: np.ndarray,
        Y: Any,
        X: Any,
        novelty: np.ndarray,
        k
    ) -> List[Any]:
        # TODO integrate NOVELTY
        symbol_num = len(pred_pseudo_label)
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)
        if self.use_zoopt:
            if novelty is not None:
                raise NotImplementedError
            solution = self._zoopt_get_solution(
                pred_pseudo_label, pred_prob, Y, X, symbol_num, max_revision_num
            )
            revision_idx = np.where(solution.get_x() != 0)[0]
            candidates, reasoning_results = self.kb.revise_at_idx(
                pseudo_label=pred_pseudo_label, y=Y, x=X, revision_idx=revision_idx,
            )
        else:
            if novelty is not None:
                force_revision_idx = np.where(novelty == -1)[0]
                if len(force_revision_idx) > 0:  # new class exists
                    new_class_candidates = []
                    abduce_c = product(self.new_class, repeat=len(force_revision_idx))
                    for c in abduce_c:
                        nc_pred_pseudo_label = pred_pseudo_label.copy()
                        for i, idx in enumerate(force_revision_idx):
                            nc_pred_pseudo_label[idx] = c[i]
                        candidates = self.kb.abduce_candidates(
                            pseudo_label=nc_pred_pseudo_label,
                            y=Y, x=X,
                            max_revision_num=max_revision_num,
                            require_more_revision=self.require_more_revision,
                            novelty=novelty
                        )
                        new_class_candidates.extend(candidates)
                    candidate = self._get_k_candidates(pred_prob, new_class_candidates, k)
                    if candidate:
                        return candidate
            # Normal abduction
            candidates = self.kb.abduce_candidates(
                pseudo_label=pred_pseudo_label,
                y=Y, x=X,
                max_revision_num=max_revision_num,
                require_more_revision=self.require_more_revision,
                novelty=None,
            )
        candidate = self._get_k_candidates(pred_prob, candidates, k)
        return candidate

    def batch_abduce(
        self,
        pred_idx: List[List[Any]],
        pred_prob: List[np.ndarray],
        Y: List[Any],
        X: List[Any],
        novelty: List[np.ndarray],
    ) -> List[List[Any]]:
        # print(pred_idx, pred_prob, Y, novelty)
        abduced_pseudo_label = []
        for pi, pp, y, x, nov in zip(pred_idx, pred_prob, Y, X, novelty):
            res = self.abduce(pi, pp, y, x, nov, 1)[0]
            assert res != []
            abduced_pseudo_label.append(res)
        return abduced_pseudo_label

    def batch_abduce_a3bl(
        self,
        pred_idx: List[List[Any]],
        pred_prob: List[np.ndarray],
        Y: List[Any],
        X: List[Any],
        novelty: List[np.ndarray],
    ) -> List[np.ndarray]:
        abduced_pseudo_label = []
        for pi, pp, y, x, nov in zip(pred_idx, pred_prob, Y, X, novelty):
            res = self.wmc_abduer.abduce(pp, y)
            abduced_pseudo_label.append(res)
        return abduced_pseudo_label

    def batch_abduce_exp(
        self,
        pred_idx: List[List[Any]],
        pred_prob: List[np.ndarray],
        Y: List[Any],
        X: List[Any],
        novelty: List[np.ndarray],
    ) -> List[List[Any]]:
        assert novelty[0] is None
        abduced_pseudo_label = []
        for pi, pp, y, x, nov in zip(pred_idx, pred_prob, Y, X, novelty):
            # Max
            # TODO: consider the case of all -1
            res = [y if p >= y else -1 for p in pi]
            # Add

            abduced_pseudo_label.append(res)
        return abduced_pseudo_label

    def __call__(self, *args, **kw) -> List[List[Any]]:
        return self.batch_abduce(*args, **kw)


    def batch_abduce_sim(
        self,
        pred_idx: List[List[Any]],
        pred_prob: List[np.ndarray],
        Y: List[Any],
        X: List[Any],
        novelty: List[np.ndarray],
        k: int=-1,
        beam_width: int=600,
        similar_coef :float=0.96,
        labeled_prob=None,
        labeled_feat=None,
        labeled_y=None,
    ) -> List[List[Any]]:
        batch_candidates = []
        for pi, pp, y, x, nov in zip(pred_idx, pred_prob, Y, X, novelty):
            candidates = self.abduce(pi, pp, y, x, nov, k)  # [List]
            candidates = [np.array(cand) for cand in candidates]
            batch_candidates.append(candidates)

        probs = pred_prob
        feats = [t.numpy() for t in X]
        abduced_pseudo_label = nn_select_batch_abduced_result(
            labeled_prob,
            labeled_feat,
            labeled_y,
            probs,  # List[ndarray(n_inst, 10)]
            feats,  # List[ndarray(n_inst, 2048)]
            batch_candidates,  # List[List[ndarray(n_inst)]]
            len(batch_candidates),  # 128
            ground_labels_list=None,
            beam_width=beam_width,
            similar_coef=similar_coef
        )
        abduced_pseudo_label = [apl.tolist() for apl in abduced_pseudo_label]
        return abduced_pseudo_label
    '''
    def score_func(self, sol, batch_candidates, dist_matrix, num_class):
        x = sol.get_x()
        gc = []
        for i, ci in enumerate(x):
            gc.extend(batch_candidates[i][int(ci)])
        gc = np.array(gc)

        cls_rows = np.zeros((num_class, len(gc)))
        for cls in range(num_class):
            cls_idx = np.where(gc == cls)[0]
            if len(cls_idx) == 0:
                continue
            num_same = len(cls_idx)
            num_diff = len(gc) - num_same
            # same class dist smaller, diff class dist bigger
            if num_diff == 0:
                cls_rows[cls, :] = 1 / num_same
            else:
                cls_rows[cls, :] = np.where(gc == cls, 1 / num_same, -1 / num_diff)
        cls_matrix = cls_rows[gc]
        measure = dist_matrix * cls_matrix
        score = np.mean(np.sum(measure, axis=1))
        return score
    '''
