from torch import nn
from torch.nn.modules.loss import _Loss

class LMLoss(_Loss):
    def __init__(self, args):
        super(LMLoss, self).__init__()
        self.lm_vocab_size = args.lm_vocab_size
        if self.lm_vocab_size == -1 or self.lm_vocab_size > args.word_vocabulary_size:
            self.lm_vocab_size = args.word_vocabulary_size

        self.forwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.bakwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, output, label, extra_label):
        out, (lm_fw_out, lm_bw_out) = output
        forwardlabel, bakwardlabel = extra_label
        loss = self.forwardLoss(lm_fw_out.view(-1, self.lm_vocab_size), forwardlabel.view(-1))
        loss += self.bakwardLoss(lm_bw_out.view(-1, self.lm_vocab_size), bakwardlabel.view(-1))
        return loss