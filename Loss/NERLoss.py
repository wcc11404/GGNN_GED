from Loss.LMLoss import LMLoss
from Loss.BaseLoss import BaseLoss
from Loss.SLLoss import SLLoss

def build_Loss(args):
    if args.criterion == "BaseLoss":
        loss = BaseLoss(args)
    elif args.criterion == "LMLoss":
        loss = LMLoss(args)
    elif args.criterion == "SLLoss":
        loss = SLLoss(args)
    else:
        raise ValueError("model criterion parameter illegal : " + args.criterion)
    return loss