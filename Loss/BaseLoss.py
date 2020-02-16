from torch import nn
from torch.nn.modules.loss import _Loss

class BaseLoss(_Loss):
    def __init__(self, args):
        super(BaseLoss, self).__init__()
        self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, output, label, extra_label):
        out, _ = output
        loss = self.Loss(out.view(-1, 2), label.view(-1))
        return loss