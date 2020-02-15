from torch import nn

class BaseLoss(nn.Module):
    def __init__(self, args):
        super(BaseLoss, self).__init__()
        self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def forward(self, output, target, **kwargs):
        label, _ = target
        out, _ = output
        loss = self.Loss(out.view(-1, 2), label.view(-1))
        return loss