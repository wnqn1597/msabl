# -*- coding: utf-8 -*-
from itertools import product
import numpy as np
import tqdm


def get_add_candidates(tgt):
    return [[x, tgt-x] for x in range(10) if 0 <= tgt-x < 10]

def get_max_candidates(tgt, n_pos=5):
    ret = []
    for arr in product(range(4), repeat=n_pos):
        if max(arr) == tgt:
            ret.append(list(arr))
    return ret

def get_add4_candidates(tgt):
    ret = []
    for i in range(tgt+1):
        j = tgt - i
        if not (0 <= i <= 9999 and 0 <= j <= 9999):
            continue
        ret.append(list(map(int, list(str(i).rjust(4, "0")) + list(str(j).rjust(4, "0")))))
    return ret

get_cand_fct_dict = {
    "add": get_add_candidates,
    "max": get_max_candidates,
    "add4": get_add4_candidates,
}

class WMCAbducer:
    def __init__(self, n_cls, n_pos, tau, mode="add"):
        self.n_cls = n_cls
        self.n_pos = n_pos
        self.tau = tau
        self.candidates_dict = {}  # dict[tgt]
        self.where_cls_idx_dict = {}  # dict[tgt][pos][cls]
        get_cand_fct = get_cand_fct_dict[mode]
        for tgt in tqdm.tqdm(range(tau), disable=(mode!="add4")):
            candidates = np.array(get_cand_fct(tgt))
            self.candidates_dict[tgt] = candidates
            where_cls_idx_pre_pos = []
            for p in range(n_pos):
                where_cls_idx = []
                col = candidates[:, p]
                for k in range(n_cls):
                    idx = np.where(col == k)[0]
                    where_cls_idx.append(idx)
                where_cls_idx_pre_pos.append(where_cls_idx)
            self.where_cls_idx_dict[tgt] = where_cls_idx_pre_pos

    def abduce(self, output_probs: np.ndarray, tgt: int):
        # output_probs: n_pos x n_cls
        candidates = self.candidates_dict[tgt]
        idx0 = np.arange(self.n_pos)[None, :]
        y = output_probs[idx0, candidates]
        cand_prob = np.prod(y, axis=-1)
        cand_prob = cand_prob / np.sum(cand_prob)  # len(candidates)

        where_cls_idx = self.where_cls_idx_dict[tgt]
        abduced_prob = np.zeros((self.n_pos, self.n_cls))
        for p in range(self.n_pos):
            for k in range(self.n_cls):
                idx = where_cls_idx[p][k]
                abduced_prob[p][k] = np.sum(cand_prob[idx])
        # abduced_prob = self.abduced_prob_post_process(abduced_prob, 0.5)
        return abduced_prob

    def abduced_prob_post_process(self, abduced_prob: np.ndarray, thresh: float):
        if np.amax(abduced_prob) < thresh:
            return abduced_prob
        n_inst = abduced_prob.shape[0]
        ret = np.zeros_like(abduced_prob)
        idx = np.argmax(abduced_prob, axis=-1)
        ret[range(n_inst), idx] = 1.0
        return ret


    def abduce_batch(self, probs: np.ndarray, targets: np.ndarray, thresh):
        assert probs.ndim == 3 and targets.ndim == 1  # B x n_inst x n_cls, B
        abduced_probs = []
        for prob, tgt in zip(probs, targets):
            ap = self.abduce(prob, tgt)
            abduced_probs.append(ap)
        return abduced_probs
        # return np.stack(abduced_probs, axis=0)

    def select_candidates_batch(self, probs: np.ndarray, batch_candidates, k: int=3):
        res = []
        for prob, candidates in zip(probs, batch_candidates):
            candidates = np.stack(candidates)
            idx0 = np.arange(self.n_pos)[None, :]
            y = prob[idx0, candidates]
            cand_prob = np.prod(y, axis=-1)
            choose = np.argsort(-cand_prob)[:k]
            res.append([candidates[i] for i in choose])
        return res




if __name__ == '__main__':
    print(get_add4_candidates(10000))
    # wmca = WMCAbducer(10, 8, 19999, "add4")
    # prob = np.abs(np.random.randn(6, 8, 10))
    # prob = prob / np.sum(prob, axis=-1, keepdims=True)
    # targets = np.random.randint(0, 19998, (6,))
    # abduced = wmca.abduce_batch(prob, targets, 0.0)
    # print(abduced)


