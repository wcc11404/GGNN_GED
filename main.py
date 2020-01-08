import argparse
import numpy as np
import torch

from data.corpus import GedCorpus
from scripts.utils import train, test, load_args

from Module.BaseNER import BaseNER
from Module.SLNER import SLNER
from Module.GGNNNER import GGNNNER

def merage_args(user_args, load_args):
    load_args["mode"] = user_args["mode"]
    load_args["load_dir"] = user_args["load_dir"]
    load_args["use_cpu"] = user_args["use_cpu"]
    load_args["gpu_list"] = user_args["gpu_list"]
    return load_args

def main(args):
    if args.mode == "Test": # 如果是测试,直接读取超参数
        loadargs = load_args(args.save_dir + "/args.json")
        merageargs = merage_args(args.__dict__, loadargs)
        args.__dict__ = merageargs

    if args.random_seed is not None:
        setup_seed(args.random_seed)
    # if torch.cuda.is_available() and bool(args.use_gpu):
    #     torch.cuda.set_device(args.gpu_list)

    corpus = GedCorpus(args)

    if args.arch == "BaseNER":
        model = BaseNER(args)
    elif args.arch == "SLNER":
        model = SLNER(args)
    elif args.arch == "GGNNNER":
        model = GGNNNER(args)

    if torch.cuda.is_available() and not args.use_cpu:
        model.to("cuda")
        if args.use_fpp16:
            model.half()
    else:
        model.to("cpu")

    if args.mode == "Train":
        train(args, model, corpus)
    elif args.mode == "Test":
        test(args, model, corpus)

def setup_seed(seed):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu
    # torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    # torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速

def init_args(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cpu", action='store_true', default=False)
    parser.add_argument("--gpu-list", default="0")
    parser.add_argument("--mode", default="Train")
    parser.add_argument("--random-seed", type=int, default=44)
    parser.add_argument("--loginfor", action='store_true', default=True)
    parser.add_argument("--use-fpp16", action='store_true', default=False)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--arch", default="GGNNNER")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-lower", action='store_true', default=True)
    parser.add_argument("--word-embed-dim", type=int, default=300)
    parser.add_argument("--char-embed-dim", type=int, default=100)
    parser.add_argument("--embed-drop", type=float, default=0.3)
    parser.add_argument("--gnn-steps", type=int, default=1)

    parser.add_argument("--rnn-type", default="LSTM")
    parser.add_argument("--rnn-bidirectional", action='store_true', default=True)
    parser.add_argument("--rnn-drop", type=float, default=0.3)
    parser.add_argument("--gnn-drop", type=float, default=0.3)
    parser.add_argument("--linear-drop", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--lm-hidden-dim", type=int, default=50)
    parser.add_argument("--lm-vocab-size", type=int, default=-1)
    parser.add_argument("--lm-cost-weight", type=float, default=0.10)

    parser.add_argument("--save-dir", default="checkpoint/test") # 模型权重参数存储基地址
    parser.add_argument("--load-dir", default=None) # 模型加载地址
    parser.add_argument("--w2v-dir", default="data/process/glove.840B.300d.txt")# glove.840B.300d.txt w2v_300d.txt
    parser.add_argument("--preprocess-dir", default="data/preprocess.pkl")
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="adadelta")

    args = parser.parse_args()
    main(args)