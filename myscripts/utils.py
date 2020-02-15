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
        # 之前save会存储gpu信息，所以可能会导致load cuda error（之前gpu被占）
        load_checkpoint = torch.load(best_dir, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()  # 获得当前模型的参数字典
        load_dict = {k: v for k, v in load_checkpoint.items() if k in model_dict}  # 找名字一样的加载权重
        model.load_state_dict(load_dict)  # 加载权重
    except:
        print("failed to load")

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

    if args.loginfor:
        print(model)
        print()

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

            optimizer.zero_grad()
            out = model(train_x, train_length, extra_data)
            loss_value = loss(out, train_y, extra_label)
            loss_value.backward()
            optimizer.zero_grad()

        # 每个epoch评估
        # train_loss, train_p, train_r, train_f0_5 = evaluate(args, Corpus.traindataloader, model)
        dev_loss, dev_p, dev_r, dev_f0_5 = evaluate(args, Corpus.devdataloader, model, loss)
        log = {"epoch": epoch,
               "dev_loss": dev_loss,
               "dev_p": dev_p,
               "dev_r": dev_r,
               "dev_f0.5": dev_f0_5
               }
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

def test(args, model, loss, Corpus):
    #_, train_p, train_r, train_f = evaluate(args, Corpus.traindataloader, myscripts, Loss=None)
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
        if len(args.gpu_ids) > 1:
            out = out[0]
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
    return loss_value / length if mode == "average" else loss_value, p, r, f