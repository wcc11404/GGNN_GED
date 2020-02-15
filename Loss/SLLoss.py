from torch import nn

class SLLoss(nn.Module):
    def __init__(self, args):
        super(SLLoss, self).__init__()
        self.lm_cost_weight = args.lm_cost_weight
        self.lm_vocab_size = args.lm_vocab_size

        self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")  # ,
            # weight=torch.from_numpy(np.array([1, 1.2])).float())
        self.forwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.bakwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, out, lm_fw_out, lm_bw_out, label, extra_label):
        # output, (label, extra_label) = inputs
        # out, (lm_fw_out, lm_bw_out) = output
        loss = self.Loss(out.view(-1, 2), label.view(-1))
        forwardlabel, bakwardlabel = extra_label
        loss += self.lm_cost_weight * self.forwardLoss(lm_fw_out.view(-1, self.lm_vocab_size), forwardlabel.view(-1))
        loss += self.lm_cost_weight * self.bakwardLoss(lm_bw_out.view(-1, self.lm_vocab_size), bakwardlabel.view(-1))
        return loss