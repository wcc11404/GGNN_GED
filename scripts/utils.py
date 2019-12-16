import torch
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

def save_checkpoint(model, dir):
    torch.save(model.state_dict(), dir)

def load_checkpoint(model, dir):
    if not os.path.exists(dir):
        raise KeyError("checkpoint is not exist")
    model.load_state_dict(torch.load(dir))

def save_args(dic, dir):
    with open(dir, 'w') as f:
        json.dump(dic, f)

def load_args(dir):
    with open(dir, 'r') as f:
        dic = json.load(f)
    return dic

def train(args, model, Corpus):
    max_dev_f0_5 = 0
    max_index = 0
    early_stop=0
    summary = []
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=1e-08,
                                     weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.load_dir is not None:
        load_checkpoint(model, args.load_dir)
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # 先存储参数
    save_args(args.__dict__, args.save_dir + "/args.json")

    for epoch in range(1, args.max_epoch + 1):
        if bool(args.loginfor):
            print("epoch {} training".format(epoch))

        model.train()
        for (train_x, train_y, train_length, extra_data) in tqdm(Corpus.traindataloader):
            if bool(args.use_gpu):
                train_x = train_x.cuda()
                train_y = train_y.cuda()
                train_length = train_length.cuda()
                extra_data = [i.cuda() for i in extra_data]
                if bool(args.use_fpp16):
                    extra_data = [i.half() for i in extra_data if i.dtype == torch.float]

            optimizer.zero_grad()
            out = model(train_x, train_length, extra_data)
            loss = model.getLoss(train_x, train_length, extra_data, out, train_y)
            loss.backward()
            optimizer.step()

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
        if bool(args.loginfor):
            # print("epoch {}  dev loss: {:.4f}  dev p: {:.4f}  dev r: {:.4f}  dev f0.5: {:.4f}  train f0.5: {:.4f}"
            #       .format(log["epoch"],log["dev_loss"],log["dev_p"],log["dev_r"],log["dev_f0.5"],log["train_f0.5"]))
            print("epoch {}  dev loss: {:.4f}  dev p: {:.4f}  dev r: {:.4f}  dev f0.5: {:.4f}"
                  .format(log["epoch"],log["dev_loss"],log["dev_p"],log["dev_r"],log["dev_f0.5"]))
        summary.append(log)

        if args.save_dir is not None:
            save_checkpoint(model, dir=args.save_dir + "/checkpoint" + str(epoch) + ".pt")
        if max_dev_f0_5 < dev_f0_5:
            max_dev_f0_5 = dev_f0_5
            max_index = epoch
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= args.early_stop:
                break

    print("epoch {} get the max dev f0.5: {}".format(max_index, max_dev_f0_5))

def test(args, model, Corpus):
    if args.load_dir is None:
        raise KeyError("load_dir has an invaild value: None")

    load_checkpoint(model, args.load_dir)

    model.eval()
    #_, train_p, train_r, train_f = evaluate(args, Corpus.traindataloader, scripts, Loss=None)
    _, dev_p, dev_r, dev_f = evaluate(args, Corpus.devdataloader, model)
    print("Dev Precision : {:.4f}\tDev Recall : {:.4f}\tDev F0.5 : {:.4f}".format(dev_p, dev_r, dev_f))

    _, test_p, test_r, test_f = evaluate(args, Corpus.testdataloader, model)
    print("Test Precision : {:.4f}\tTest Recall : {:.4f}\tTest F0.5 : {:.4f}".format(test_p, test_r, test_f))

def evaluate(args, dataloader, model, mode="average"):
    loss = 0
    length = 0
    predict = []
    groundtruth = []
    for (train_x, train_y, train_length, extra_data) in dataloader:
        if bool(args.use_gpu):
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            train_length = train_length.cuda()
            extra_data = [i.cuda() for i in extra_data]

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