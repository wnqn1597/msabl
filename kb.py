# -*- coding: utf-8 -*-
import os
import inspect
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple
from itertools import combinations, product
import pyswip
import numpy as np


class KBBase(ABC):
    def __init__(self, pseudo_label_list: List[Any], max_err: float = 1e-10):
        if not isinstance(pseudo_label_list, list):
            raise TypeError(
                f"pseudo_label_list should be list, got {type(pseudo_label_list)}"
            )
        self.pseudo_label_list = pseudo_label_list
        self.max_err = max_err
        argspec = inspect.getfullargspec(self.logic_forward)
        self._num_args = len(argspec.args) - 1

    @abstractmethod
    def logic_forward(
        self, pseudo_label: List[Any], x: Optional[List[Any]] = None
    ) -> Any:
        """ pass """

    def abduce_candidates(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        max_revision_num: int,
        require_more_revision: int,
        novelty: np.ndarray
    ) -> List:
        candidates = self._abduce_by_search(
            pseudo_label, y, x, max_revision_num, require_more_revision, novelty
        )
        return list(map(list, set(candidates)))

    def _check_equal(self, reasoning_result: Any, y: Any) -> bool:
        if reasoning_result is None:
            return False

        if isinstance(reasoning_result, (int, float)) and isinstance(y, (int, float)):
            return abs(reasoning_result - y) <= self.max_err
        else:
            return reasoning_result == y

    def revise_at_idx(
        self, pseudo_label: List[Any], y: Any, x: List[Any], revision_idx: List[int],
    ) -> List:
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            reasoning_result = self.logic_forward(
                candidate, *(x,) if self._num_args == 2 else ()
            )
            if self._check_equal(reasoning_result, y):
                candidates.append(candidate)
        return candidates

    def _revision(
        self, revision_num: int, pseudo_label: List[Any], y: Any, x: List[Any], novelty: np.ndarray
    ) -> List:
        new_candidates = []
        if novelty is not None:
            optional_revision_idx = np.where(novelty == 1)[0]
            force_revision_num = len(pseudo_label) - len(optional_revision_idx)
            if revision_num - force_revision_num < 0:
                return []
            revision_idx_list = combinations(optional_revision_idx, revision_num - force_revision_num)
        else:
            revision_idx_list = combinations(range(len(pseudo_label)), revision_num)
        for revision_idx in revision_idx_list:
            new_candidates.extend(self.revise_at_idx(pseudo_label, y, x, revision_idx))
        return new_candidates

    def _abduce_by_search(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        max_revision_num: int,
        require_more_revision: int,
        novelty: np.ndarray
    ) -> List:
        candidates = []
        min_revision_num = -1
        for revision_num in range(len(pseudo_label) + 1):
            candidates.extend(self._revision(revision_num, pseudo_label, y, x, novelty))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []
        for revision_num in range(
            min_revision_num + 1, min_revision_num + require_more_revision + 1
        ):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision(revision_num, pseudo_label, y, x, novelty))
        return candidates


class PrologKB(KBBase):
    def __init__(self, pseudo_label_list: List[Any], pl_file: str):
        super().__init__(pseudo_label_list)
        self.prolog = pyswip.Prolog()
        self.pl_file = pl_file
        if not os.path.exists(self.pl_file):
            raise FileNotFoundError(f"The Prolog file {self.pl_file} does not exist.")
        self.prolog.consult(self.pl_file)

    def logic_forward(self, pseudo_label: List[Any], x: Optional[List[Any]] = None) -> Any:
        result = list(self.prolog.query(f"logic_forward({pseudo_label}, Res)."))[0]["Res"]
        if result == "true":
            return True
        if result == "false":
            return False
        return result

    def revise_at_idx(
        self,
        pseudo_label: List[Any],
        y: Any,
        x: List[Any],
        revision_idx: List[int],
    ) -> List[Tuple]:
        candidates = []
        tmp = pseudo_label.copy()
        for i, ri in enumerate(revision_idx):
            tmp[ri] = f"P{i}"
        query_string = f"logic_forward({tmp}, {y}).".replace("'", "")
        abduce_c = [list(z.values()) for z in self.prolog.query(query_string)]
        for c in abduce_c:
            candidate = pseudo_label.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            candidates.append(tuple(candidate))
        return candidates