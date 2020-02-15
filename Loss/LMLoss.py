from torch import nn

class LMLoss(nn.Module):
    def __init__(self, args):
        super(LMLoss, self).__init__()
        self.lm_vocab_size = args.lm_vocab_size

        self.forwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.bakwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, output, target, **kwargs):
        _, extra_label = target
        out, (lm_fw_out, lm_bw_out) = output
        forwardlabel, bakwardlabel = extra_label
        loss = self.forwardLoss(lm_fw_out.view(-1, self.lm_vocab_size), forwardlabel.view(-1))
        loss += self.bakwardLoss(lm_bw_out.view(-1, self.lm_vocab_size), bakwardlabel.view(-1))
        return loss