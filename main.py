import argparse
import numpy as np
import torch
import os
# from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from data.corpus import GedCorpus
from myscripts.utils import train, test, load_args, load_checkpoint, log_information
from myscripts.utils import run_demo, setup_ddp, clean_ddp
from myscripts.parallel import DataParallelModel, DataParallelCriterion

from Module.NERModel import build_Model
from Loss.NERLoss import build_Loss

def merage_args(user_args, loadargs):
    loadargs["mode"] = user_args["mode"]
    loadargs["load_dir"] = user_args["load_dir"]
    loadargs["use_cpu"] = user_args["use_cpu"]
    loadargs["gpu_ids"] = user_args["gpu_ids"]
    return loadargs

def check_args(args):
    if args.save_dir is not None and os.path.exists(args.save_dir):
        args.save_dir = os.path.abspath(args.save_dir)
    if args.load_dir is not None and os.path.exists(args.load_dir):
        args.load_dir = os.path.abspath(args.load_dir)

    if args.mode == "Test": # 如果是测试,直接读取超参数,并用部分覆盖
        loadargs = load_args(args.load_dir + "/args.json")
        merageargs = merage_args(args.__dict__, loadargs)
        args.__dict__ = merageargs

    if args.update_freq <= 0:
        raise ValueError("update_freq value error")

    if not args.use_cpu:
        if args.gpu_ids is None:
            raise ValueError("gpu ids value error")
        args.gpu_ids = [int(i) for i in args.gpu_ids]

    if not args.use_ddp and args.local_rank > 0:
        raise ValueError("Don't modification the local_rank parameter")

    if args.use_ddp == args.use_dp and args.use_ddp == True:
        raise ValueError() # ddp和dp模型不同同时为真
    if len(args.gpu_ids) > 1 and not args.use_ddp:
        args.use_dp = True
    return args

def ddp_main(rank, args):
    args.local_rank = rank
    main(args)

def main(args):
    # 一定要在前边=。=
    if not args.use_cpu and args.use_ddp:
        setup_ddp(args.local_rank, world_size=len(args.gpu_ids), backend=args.ddp_backend)

    # 设置随机种子
    if args.random_seed is not None:
        setup_seed(args.random_seed)

    # 初始化数据
    corpus = GedCorpus(args)
    corpus.build_Dataloader()

    # 初始化模型
    model = build_Model(args)

    # 初始化损失函数
    loss = build_Loss(args)

    # 初始化优化器
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html?highlight=dataparallel
    # 官方定义优化器在ddp设置之后，不确定是否必须
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=1e-08,
                                     weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 在cpu中加载权重
    # 同上，官方加载权重在所有都创建完毕后，不确定是否必须
    if args.load_dir is not None:
        best_dir = load_checkpoint(model, args.load_dir)
        log_information(args, "load checkpoint from " + best_dir)

    # 设置ddp, https://github.com/pytorch/fairseq/blob/e6422528dae0b899848469efe2dc404c1e639ce9/train.py#L44
    # 说设置ddp要在load数据之后，不确定是否必须
    if not args.use_cpu and args.use_ddp:
        device = torch.device('cuda', args.gpu_ids[args.local_rank])
        torch.cuda.set_device(device)
        model = model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        model = DDP(model) # apex
        # model = DDP(model, device_ids=[args.gpu_ids[args.local_rank]]) # torch
    elif not args.use_cpu and not args.use_ddp and torch.cuda.is_available():# 设置gpu和dp
        torch.cuda.set_device(args.gpu_ids[0])
        model.to("cuda")
        if args.use_fpp16:
            model.half()

        if len(args.gpu_ids) > 1:  # 设置DataParallel多卡并行参数
            model = DataParallelModel(model, device_ids=args.gpu_ids)
            loss = DataParallelCriterion(loss, device_ids=args.gpu_ids)
    else:
        model.to("cpu")

    if args.mode == "Train":
        train(args, model, loss, optimizer, corpus)
    elif args.mode == "Test":
        test(args, model, loss, corpus)

def setup_seed(seed):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu
    # torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cpu", action='store_true', default=False)
    parser.add_argument("--gpu-ids", nargs="+", default="0")
    parser.add_argument("--mode", default="Train")
    parser.add_argument("--random-seed", type=int, default=44)
    parser.add_argument("--loginfor", action='store_true', default=True)
    parser.add_argument("--use-fpp16", action='store_true', default=False)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--arch", default="GGNNNER")
    parser.add_argument("--criterion", default="BaseLoss")
    parser.add_argument("--train-lm", action='store_true', default=False)
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="adadelta")
    parser.add_argument("--evaluation", default="loss") # 评价指标 loss 和 f0.5 ;模型保存，earlystop等指标的依据

    parser.add_argument("--use-dp", action='store_true', default=False)

    # DDP
    parser.add_argument("--use-ddp", action='store_true', default=False)
    # python -m torch.distributed.launch 传参, 也可以是torch.multiprocessing传参，然后再复制给它
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ddp-backend", default="nccl")

    args = parser.parse_args()
    args = check_args(args)# 预处理程序参数

    if args.use_ddp and len(args.gpu_ids) > 1:
        run_demo(ddp_main, len(args.gpu_ids), args)
    else:
        main(args)