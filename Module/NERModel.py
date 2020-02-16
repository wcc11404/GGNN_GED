from Module.BaseNER import BaseNER
from Module.SLNER import SLNER
from Module.GGNNNER import GGNNNER

def build_Model(args):
    if args.arch == "BaseNER":
        model = BaseNER(args)
    elif args.arch == "SLNER":
        model = SLNER(args)
    elif args.arch == "GGNNNER":
        model = GGNNNER(args)
    else:
        raise ValueError("model arch parameter illegal : " + args.arch)
    return model