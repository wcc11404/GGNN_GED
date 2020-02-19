import torch
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
import apex as amp

def run_demo(demo_fn, world_size,args):
    mp.spawn(demo_fn, args=(args,), nprocs=world_size, join=True)

def setup_ddp(rank, world_size=1, backend="nccl"):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend=backend, init_method='tcp://localhost:23456', rank=rank, world_size=world_size)

def clean_ddp():
    dist.destroy_process_group()

def log_information(args, str, force=False):
    if force or args.loginfor == True:
        if force or args.local_rank == 0:
            print(str)

def update_best_checkpoint(save_dir, epoch):
    save_dir = os.path.abspath(save_dir)
    with open(save_dir+"/save.log",'w') as f:
        f.write(save_dir+"/checkpoint"+str(epoch)+".pt")

def save_checkpoint(model, dir):
    torch.save(model.state_dict(), dir)

def load_checkpoint(model, dir):
    if not os.path.exists(dir):
        raise KeyError("checkpoint is not exist")

    # TODO 输入可以是具体的权重文件而不是文件夹
    try:
        log = os.path.join(dir, "save.log")
        best_dir = open(log, "r").readline().strip()
        # 之前save会存储gpu信息，所以可能会导致load cuda error（之前gpu被占）
        load_checkpoint = torch.load(best_dir, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()  # 获得当前模型的参数字典
        load_dict = {k: v for k, v in load_checkpoint.items() if k in model_dict}  # 找名字一样的加载权重
        model.load_state_dict(load_dict)  # 加载权重
    except Exception:
        raise Exception

    return best_dir

def save_args(dic, dir):
    with open(dir, 'w') as f:
        json.dump(dic, f)

def load_args(dir):
    with open(dir, 'r') as f:
        dic = json.load(f)
    return dic

def train(args, model, loss, optimizer, Corpus):
    best_evaluation = 0 if args.evaluation == "f0.5" else 999999999
    max_index = 0
    early_stop = 0
    summary = []

    # 打印模型
    log_information(args, model)

    # 创建存储位置
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # 存储参数
    if args.local_rank == 0:
        save_args(args.__dict__, args.save_dir + "/args.json")

    for epoch in range(1, args.max_epoch + 1):
        log_information(args, "epoch {} training".format(epoch))

        batch_num = 1 # 更新到第n个batch，update_freq用
        # 训练
        model.train()
        #清理GPU碎片空间？？
        if not args.use_cpu:
            torch.cuda.empty_cache()
        if args.loginfor and args.local_rank == 0:
            trainer = tqdm(Corpus.traindataloader)
        else:
            trainer = Corpus.traindataloader
        for (train_x, train_y, train_length, extra_data, extra_label) in trainer:
            if not args.use_cpu:
                train_x = train_x.cuda(non_blocking=True)
                train_y = train_y.cuda(non_blocking=True)
                train_length = train_length.cuda(non_blocking=True)
                extra_data = [i.cuda(non_blocking=True) for i in extra_data]
                extra_label = [i.cuda(non_blocking=True) for i in extra_label]
                if args.use_fpp16:
                    extra_data = [i.half(non_blocking=True) if i.dtype == torch.float else i for i in extra_data]
                    extra_label = [i.half(non_blocking=True) if i.dtype == torch.float else i for i in extra_label]

            out = model(train_x, train_length, extra_data)
            loss_value = loss(out, train_y, extra_label)
            with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                scaled_loss.backward()
            if batch_num % args.update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_num += 1

        # 每个epoch评估
        # train_loss, train_p, train_r, train_f0_5 = evaluate(args, Corpus.traindataloader, model)
        if args.local_rank == 0:
            dev_loss, dev_p, dev_r, dev_f0_5 = evaluate(args, Corpus.devdataloader, model, loss)
            log = {"epoch": epoch,
                   "dev_loss": dev_loss,
                   "dev_p": dev_p,
                   "dev_r": dev_r,
                   "dev_f0.5": dev_f0_5
                   }
            log_information(args, "epoch {}  dev loss: {:.4f}  dev p: {:.4f}  dev r: {:.4f}  dev f0.5: {:.4f}"
                  .format(log["epoch"], log["dev_loss"], log["dev_p"], log["dev_r"], log["dev_f0.5"]))
            summary.append(log) # 日志，暂时没用

            # 存储策略
            if args.save_dir is not None:
                save_checkpoint(model, dir=args.save_dir + "/checkpoint" + str(epoch) + ".pt")

            if args.evaluation == "loss":
                if best_evaluation > dev_loss:
                    best_evaluation = dev_loss
                    max_index = epoch
                    early_stop = 0
                    update_best_checkpoint(args.save_dir, epoch)
                else:
                    early_stop += 1
                    if early_stop >= args.early_stop:
                        break
            elif args.evaluation == "f0.5":
                if best_evaluation < dev_f0_5:
                    best_evaluation = dev_f0_5
                    max_index = epoch
                    early_stop = 0
                    update_best_checkpoint(args.save_dir, epoch)  # 更新最好模型权重路径
                else:
                    early_stop += 1
                    if early_stop >= args.early_stop:
                        break
        if args.use_ddp:
            dist.barrier() # 其他线程等待

    # 打印最好结果
    log_information(args,
                    "epoch {} get the best ".format(max_index) + args.evaluation + " : {}".format(best_evaluation),
                    True if args.local_rank == 0 else False)

def test(args, model, loss, Corpus):
    #_, train_p, train_r, train_f = evaluate(args, Corpus.traindataloader, myscripts, Loss=None)
    if args.local_rank == 0:
        dev_loss, dev_p, dev_r, dev_f = evaluate(args, Corpus.devdataloader, model, loss)
        print("Dev Loss : {:.4f}\tDev Precision : {:.4f}\tDev Recall : {:.4f}\tDev F0.5 : {:.4f}"
              .format(dev_loss, dev_p, dev_r, dev_f))

        if Corpus.testdataloader is not None:
            test_loss, test_p, test_r, test_f = evaluate(args, Corpus.testdataloader, model, loss)
            print("Test Loss : {:.4f}\tTest Precision : {:.4f}\tTest Recall : {:.4f}\tTest F0.5 : {:.4f}"
                  .format(test_loss, test_p, test_r, test_f))

def evaluate(args, dataloader, model, loss, mode="average"):
    loss_value = 0
    length = 0
    predict = []
    groundtruth = []

    model.eval()
    for (train_x, train_y, train_length, extra_data, extra_label) in dataloader:
        if not args.use_cpu:
            train_x = train_x.cuda(non_blocking=True)
            train_y = train_y.cuda(non_blocking=True)
            train_length = train_length.cuda(non_blocking=True)
            extra_data = [i.cuda(non_blocking=True) for i in extra_data]
            extra_label = [i.cuda(non_blocking=True) for i in extra_label]
            if args.use_fpp16:
                extra_data = [i.half(non_blocking=True) if i.dtype == torch.float else i for i in extra_data]
                extra_label = [i.half(non_blocking=True) if i.dtype == torch.float else i for i in extra_label]

        out = model(train_x, train_length, extra_data)
        loss_value += loss(out, train_y, extra_label).item()
        if args.use_dp: # dp模式，out是多个gpu的结果，所以只取第一个结果，暂不确定对不对
            out = out[0]
        out = out[0].cpu().detach().numpy() # 只有out[0],模型的第一个输出，参与计算F值
        train_y = train_y.cpu().detach().numpy()
        train_length = train_length.cpu().detach().numpy()
        for o, y, l in zip(out, train_y, train_length):
            o = o[:l]
            predict.extend(o.argmax(axis=-1))
            groundtruth.extend(y[:l])
            length += l

    p, r, f = 0, 0, 0
    p, r, f, _ = precision_recall_fscore_support(groundtruth, predict, 0.5, average='binary')
    return loss_value / length if mode == "average" else loss_value, p, r, f