import torch
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

def update_best_checkpoint(save_dir, epoch):
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
        print("load checkpoint from " + best_dir)
        load_checkpoint = torch.load(best_dir)  # 找一下最好的
        model_dict = model.state_dict()  # 获得当前模型的参数字典
        load_dict = {k: v for k, v in load_checkpoint.items() if k in model_dict}  # 找名字一样的加载权重
        load_dict["shit"]=None ####
        print(model.wordembedding.wordembedding.weight)
        model.load_state_dict(load_dict)  # 加载权重
        print()
        print(model.wordembedding.wordembedding.weight)
        exit()
    except:
        print("failed to load")

def save_args(dic, dir):
    with open(dir, 'w') as f:
        json.dump(dic, f)

def load_args(dir):
    with open(dir, 'r') as f:
        dic = json.load(f)
    return dic

def train(args, model, Corpus):
    best_evaluation = 0 if args.evaluation == "f0.5" else 999999999
    max_index = 0
    early_stop = 0
    summary = []
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=1e-08,
                                     weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loginfor:
        print(model)
        print()
    # if args.load_dir is not None:
    #     print("load checkpoint from" + args.load_dir)
    #     print()
    #     load_checkpoint(model, args.load_dir)
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # 先存储参数
    save_args(args.__dict__, args.save_dir + "/args.json")

    for epoch in range(1, args.max_epoch + 1):
        print("epoch {} training".format(epoch))

        # 训练
        model.train()
        #清理GPU缓存？？
        if not args.use_cpu:
            torch.cuda.empty_cache()
        if args.loginfor:
            trainer = tqdm(Corpus.traindataloader)
        else:
            trainer = Corpus.traindataloader
        for (train_x, train_y, train_length, extra_data) in trainer:
            if not args.use_cpu:
                train_x = train_x.cuda(non_blocking=True)
                train_y = train_y.cuda(non_blocking=True)
                train_length = train_length.cuda(non_blocking=True)
                extra_data = [i.cuda(non_blocking=True) for i in extra_data]
                if args.use_fpp16:
                    extra_data = [i.half(non_blocking=True) if i.dtype == torch.float else i for i in extra_data]

            optimizer.zero_grad()
            out = model(train_x, train_length, extra_data)
            loss = model.getLoss(train_x, train_length, extra_data, out, train_y)
            loss.backward()
            optimizer.step()

        # 每个epoch评估
        model.eval()
        # train_loss, train_p, train_r, train_f0_5 = evaluate(args, Corpus.traindataloader, model)
        dev_loss, dev_p, dev_r, dev_f0_5 = evaluate(args, Corpus.devdataloader, model)
        log = {"epoch": epoch,
               #"train_loss": train_loss,
               #"train_p": train_p,
               #"train_r": train_r,
               #"train_f0.5": train_f0_5,
               "dev_loss": dev_loss,
               "dev_p": dev_p,
               "dev_r": dev_r,
               "dev_f0.5": dev_f0_5
               }
        # print("epoch {}  dev loss: {:.4f}  dev p: {:.4f}  dev r: {:.4f}  dev f0.5: {:.4f}  train f0.5: {:.4f}"
        #       .format(log["epoch"],log["dev_loss"],log["dev_p"],log["dev_r"],log["dev_f0.5"],log["train_f0.5"]))
        print("epoch {}  dev loss: {:.4f}  dev p: {:.4f}  dev r: {:.4f}  dev f0.5: {:.4f}"
              .format(log["epoch"], log["dev_loss"], log["dev_p"], log["dev_r"], log["dev_f0.5"]))
        summary.append(log)

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

    print("epoch {} get the best ".format(max_index)+args.evaluation+" : {}".format(best_evaluation))

def test(args, model, Corpus):
    # 如果load地址给定且合法则加载该地址，否则加载best地址
    # if args.load_dir is not None:
    #     load_checkpoint(model, args.load_dir)
    # elif args.save_dir is not None and os.path.exists(args.save_dir):
    #     d = open(args.save_dir + "/save.log", 'r').readline().strip()
    #     load_checkpoint(model, d)
    # else:
    #     raise KeyError("load_dir has an invaild value: None")

    model.eval()
    #_, train_p, train_r, train_f = evaluate(args, Corpus.traindataloader, myscripts, Loss=None)
    dev_loss, dev_p, dev_r, dev_f = evaluate(args, Corpus.devdataloader, model)
    print("Dev Loss : {:.4f}\tDev Precision : {:.4f}\tDev Recall : {:.4f}\tDev F0.5 : {:.4f}"
          .format(dev_loss, dev_p, dev_r, dev_f))

    if Corpus.testdataloader is not None:
        test_loss, test_p, test_r, test_f = evaluate(args, Corpus.testdataloader, model)
        print("Test Loss : {:.4f}\tTest Precision : {:.4f}\tTest Recall : {:.4f}\tTest F0.5 : {:.4f}"
              .format(test_loss, test_p, test_r, test_f))

def evaluate(args, dataloader, model, mode="average"):
    loss = 0
    length = 0
    predict = []
    groundtruth = []
    for (train_x, train_y, train_length, extra_data) in dataloader:
        if not args.use_cpu:
            train_x = train_x.cuda(non_blocking=True)
            train_y = train_y.cuda(non_blocking=True)
            train_length = train_length.cuda(non_blocking=True)
            extra_data = [i.cuda(non_blocking=True) for i in extra_data]
            if args.use_fpp16:
                extra_data = [i.half(non_blocking=True) if i.dtype == torch.float else i for i in extra_data ]

        out = model(train_x, train_length, extra_data)
        loss += model.getLoss(train_x, train_length, extra_data, out, train_y).item()
        out = out[0].cpu().detach().numpy() # 只有out[0]参与计算F值
        train_y = train_y.cpu().detach().numpy()
        train_length = train_length.cpu().detach().numpy()
        for o, y, l in zip(out, train_y, train_length):
            o = o[:l]
            predict.extend(o.argmax(axis=-1))
            groundtruth.extend(y[:l])
            length += l

    p, r, f = 0, 0, 0
    p, r, f, _ = precision_recall_fscore_support(groundtruth, predict, 0.5, average='binary')
    return loss / length if mode == "average" else loss, p, r, f