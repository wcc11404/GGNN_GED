from data.corpus import GedCorpus
from model.baseNER import baseNER
import os
import argparse
import torch
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from model.utils import savecheckpoint,loadcheckpoint
import numpy as np

def evaluate(args,dataloader,model,Loss=None,mode="average"):
    loss=0
    length=0
    predict=[]
    groundtruth=[]
    for (train_x, train_y, train_length) in dataloader:
        if bool(args.use_gpu):
            train_x=train_x.cuda()
            train_y=train_y.cuda()
            train_length=train_length.cuda()
        out=model(train_x,train_length)
        if Loss is not None:
            loss+=Loss(out.view(-1,2),train_y.view(-1)).item()
        out=out.cpu().detach().numpy()
        train_y=train_y.cpu().detach().numpy()
        train_length=train_length.cpu().detach().numpy()
        for o,y,l in zip(out,train_y,train_length):
            o=o[:l]
            predict.extend(o.argmax(axis=-1))
            groundtruth.extend(y[:l])
            length+=l

    p=0
    r=0
    f=0
    p, r, f, _ = precision_recall_fscore_support(groundtruth, predict, 0.5, average='binary')
    return loss / length if mode == "average" else loss, p, r, f

def train(args,model,Corpus):
    if args.load_dir is not None:
        loadcheckpoint(model, args.load_dir)

    max_dev_f0_5 = 0
    max_index = 0
    early_stop=0
    summary = []
    Loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=1e-08,weight_decay=0.0001)
    elif args.optimizer.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=0.0001)

    if args.load_dir is not None:
        loadcheckpoint(model, args.load_dir)
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        for (train_x, train_y, train_length) in Corpus.traindataloader:
            if bool(args.use_gpu):
                train_x=train_x.cuda()
                train_y=train_y.cuda()
                train_length=train_length.cuda()

            optimizer.zero_grad()
            out = model(train_x, train_length)
            loss = Loss(out.view(-1, 2), train_y.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss, train_p, train_r, train_f0_5 = evaluate(args, Corpus.traindataloader, model, Loss)
        dev_loss, dev_p, dev_r, dev_f0_5 = evaluate(args, Corpus.devdataloader, model, Loss)
        log = {"epoch": epoch,
               "train_loss": train_loss,
               "train_p": train_p,
               "train_r": train_r,
               "train_f0.5": train_f0_5,
               "dev_loss": dev_loss,
               "dev_p": dev_p,
               "dev_r": dev_r,
               "dev_f0.5": dev_f0_5
               }
        if bool(args.loginfor):
            print("epoch {}  dev loss: {:.4f}  dev p: {:.4f}  dev r: {:.4f}  dev f0.5: {:.4f}  train f0.5: {:.4f}"
                  .format(log["epoch"],log["dev_loss"],log["dev_p"],log["dev_r"],log["dev_f0.5"],log["train_f0.5"]))
        summary.append(log)

        if max_dev_f0_5 < dev_f0_5:
            max_dev_f0_5 = dev_f0_5
            max_index = epoch
            early_stop=0
            if args.save_dir is not None:
                savecheckpoint(model, dir=args.save_dir + "/checkpoint" + str(epoch) + ".pt")
        else:
            early_stop+=1
            if early_stop>= args.early_stop:
                break

    print("epoch {} get the max dev f0.5: {}".format(max_index,max_dev_f0_5))

def test(args,model,Corpus):
    if args.load_dir is None:
        raise KeyError("load_dir has an invaild value: None")
    loadcheckpoint(model,args.load_dir)
    model.eval()
    #_, train_p, train_r, train_f = evaluate(Corpus.traindataloader, model, Loss=None)
    #_, dev_p, dev_r, dev_f = evaluate(Corpus.devdataloader, model, Loss=None)
    _, test_p, test_r, test_f = evaluate(args, Corpus.testdataloader, model, Loss=None)
    print("Test Precision : {}\tTest Recall : {}\tTest F0.5 : {}".format(test_p,test_r,test_f))

def main(args):
    if args.random_seed is not None:
        setup_seed(args.random_seed)
    corpus=GedCorpus("data",args)
    if args.arch=="baseNER":
        model=baseNER(args)
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
    parser.add_argument("--use-gpu",default="True")
    parser.add_argument("--gpu-list",default="0")
    parser.add_argument("--mode",default="Train")
    parser.add_argument("--use-lower",default="True")
    parser.add_argument("--random-seed",type=int,default=44)
    parser.add_argument("--arch",default="baseNER")

    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--loginfor",default="True")

    # parser.add_argument("--vocabulary-size",type=int,default=32)
    parser.add_argument("--embed-dim",type=int,default=300)
    parser.add_argument("--embed-drop",type=float,default=0.5)

    parser.add_argument("--rnn-type", default="LSTM")
    parser.add_argument("--rnn-drop", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=50)

    parser.add_argument("--save-dir", default="checkpoint")
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--w2v-dir", default="data/w2v_300d.txt")
    parser.add_argument("--preprocess-dir", default="data/preprocess.pkl")
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--optimizer", default="adadelta")

    args=parser.parse_args()
    main(args)