from multiprocessing import reduction
from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from fairseq.modules import LayerNorm
from fairseq.utils import get_activation_fn

def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.0)
    return m

class LMSLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        langlist,
        rank=32,
        init_mean=0,
        init_sdev=0.01,
        linear=None,
        lms_type="pair",
        ):
        super().__init__()
        if linear is not None:
            self.linear = linear
        else:
            self.linear = Linear(input_size, output_size)
        self.lang2id = {langlist[i]: i for i in range(len(langlist))}
        self.lang2id["shared"] = -1
        self.rank = rank
        self.lms_type = lms_type
        self.w1 = torch.nn.Parameter(torch.zeros(len(langlist) + 1, output_size, rank).normal_(init_mean, init_sdev))
        self.w2 = torch.nn.Parameter(torch.zeros(len(langlist) + 1, rank, input_size))

    def forward(self, x, lang_id=None):
        if lang_id == None:
            x = F.linear(x, self.linear.weight, self.linear.bias)
        else:
            w = self.get_lms_function()(lang_id)
            x = F.linear(x, self.linear.weight + w, self.linear.bias)
        return x

    def get_lang_lms_matrix(self, lang_id):
        lang_id = self.lang2id[lang_id]
        w = self.w1[lang_id] @ self.w2[lang_id]
        return w

    def get_pair_lms_matrix(self, lang_id):
        src_id, tgt_id = lang_id.split("-")
        src_id = self.lang2id[src_id]
        tgt_id = self.lang2id[tgt_id]
        w = self.w1[src_id] @ self.w2[tgt_id]
        return w

    def get_lms_function(self):
        if "pair" in self.lms_type:
            return self.get_pair_lms_matrix
        else:
            return self.get_lang_lms_matrix

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias
       