import argparse
import numpy as np
import torch

from data.corpus import GedCorpus
from myscripts.utils import train, test, load_args, load_checkpoint

from Module.BaseNER import BaseNER
from Module.SLNER import SLNER
from Module.GGNNNER import GGNNNER

def merage_args(user_args, loadargs):
    loadargs["mode"] = user_args["mode"]
    loadargs["load_dir"] = user_args["load_dir"]
    loadargs["use_cpu"] = user_args["use_cpu"]
    loadargs["gpu_list"] = user_args["gpu_list"]
    return loadargs

def main(args):
    if args.mode == "Test": # 如果是测试,直接读取超参数,并用部分覆盖
        loadargs = load_args(args.load_dir + "/args.json")
        merageargs = merage_args(args.__dict__, loadargs)
        args.__dict__ = merageargs

    if args.random_seed is not None:
        setup_seed(args.random_seed)

    # 初始化数据
    corpus = GedCorpus(args)

    # TODO
    # 初始化模型，后期会把optimizer初始化也放过来，解耦，而且load权重也会少加载优化器的部分权重！！！
    if args.arch == "BaseNER":
        model = BaseNER(args)
    elif args.arch == "SLNER":
        model = SLNER(args)
    elif args.arch == "GGNNNER":
        model = GGNNNER(args)
    else:
        raise KeyError("model arch parameter illegal : " + args.arch)

    # load 权重
    if args.load_dir is not None:
        load_checkpoint(model, args.load_dir)

    if torch.cuda.is_available() and not args.use_cpu:
        torch.cuda.set_device(args.gpu_list)
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
    parser.add_argument("--train-lm", action='store_true', default=False)
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
    parser.add_argument("--w2v-dir", default=None)# glove.840B.300d.txt w2v_300d.txt
    parser.add_argument("--data-dir", default="data/prepare/train.pkl")
    parser.add_argument("--vocab-dir", default="data/prepare")
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="adadelta")
    parser.add_argument("--evaluation", default="loss") # 评价指标 loss 和 f0.5 ;模型保存，earlystop等指标的依据

    args = parser.parse_args()
    main(args)