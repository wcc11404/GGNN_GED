import torch
import numpy as np
from torch import nn
from torch.nn.modules.loss import _Loss

# 注意loss继承的父类和model不同，但不清楚有什么区别
class SLLoss(_Loss):
    def __init__(self, args):
        super(SLLoss, self).__init__()
        self.lm_cost_weight = args.lm_cost_weight
        self.lm_vocab_size = args.lm_vocab_size
        if self.lm_vocab_size == -1 or self.lm_vocab_size > args.word_vocabulary_size:
            self.lm_vocab_size = args.word_vocabulary_size

        if args.main_label_weight != 1:
            weight = nn.Parameter(torch.from_numpy(np.array([1, args.main_label_weight])).float())
            weight.requires_grad=False
            self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum", weight=weight)
        else:
            self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.forwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.bakwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, output, label, extra_label):
        out, (lm_fw_out, lm_bw_out) = output
        forwardlabel, bakwardlabel = extra_label
        loss = self.Loss(out.view(-1, 2), label.view(-1))
        loss += self.lm_cost_weight * self.forwardLoss(lm_fw_out.view(-1, self.lm_vocab_size), forwardlabel.view(-1))
        loss += self.lm_cost_weight * self.bakwardLoss(lm_bw_out.view(-1, self.lm_vocab_size), bakwardlabel.view(-1))
        return loss