import argparse
import numpy as np
import torch

from Module.BaseNER import baseNER
from Module.SLNER import SLNER
from Module.GGNNNER import GGNNNER

from data.corpus import GedCorpus
from scripts.utils import train, test

def main(args):
    if args.random_seed is not None:
        setup_seed(args.random_seed)
    corpus = GedCorpus("data", args)
    if args.arch == "BaseNER":
        model = baseNER(args)
    elif args.arch == "SLNER":
        model = SLNER(args)
    elif args.arch == "GGNNNER":
        model = GGNNNER(args)
    if bool(args.use_gpu):
        # if args.gpu_list is not None:
        #     os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_list)
        #     if bool(args.loginfor):
        #         print("use {} gpu".format(args.gpu_list))
        model.to("cuda")
    else:
        model.to("cpu")

    #Train
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-gpu", default="True")
    parser.add_argument("--gpu-list", default="0")
    parser.add_argument("--mode", default="Train")
    parser.add_argument("--use-lower", default="True")
    parser.add_argument("--random-seed", type=int, default=44)
    parser.add_argument("--arch", default="GGNNNER")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loginfor", default="True")

    # parser.add_argument("--vocabulary-size",type=int,default=32)
    parser.add_argument("--word-embed-dim", type=int, default=300)
    parser.add_argument("--char-embed-dim", type=int, default=100)
    parser.add_argument("--embed-drop", type=float, default=0.3)
    parser.add_argument("--gnn-steps", type=int, default=1)

    parser.add_argument("--rnn-type", default="LSTM")
    parser.add_argument("--rnn-bidirectional", default="True")
    parser.add_argument("--rnn-drop", type=float, default=0.3)
    parser.add_argument("--linear-drop", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--lm-hidden-dim", type=int, default=50)
    parser.add_argument("--lm-vocab-size", type=int, default=-1)
    parser.add_argument("--lm-cost-weight", type=float, default=0.15)

    parser.add_argument("--save-dir", default="checkpoint")
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--w2v-dir", default="data/process/w2v_300d.txt")
    parser.add_argument("--preprocess-dir", default="data/preprocess.pkl")
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="adadelta")

    args = parser.parse_args()
    main(args)