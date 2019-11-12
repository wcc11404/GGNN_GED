import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import Counter

def collate_fn(train_data):
    def pad(data,max_length,paditem=0):
        re=[]
        for d in data:
            temp=d[:]
            temp.extend([paditem for _ in range(max_length-len(temp))])
            re.append(temp)
        return re

    def sort(*input): # [1,2,3,4] [4,4,1,2]
        temp = list(zip(*input)) #[(1, 4), (2, 4), (3, 1), (4, 2)]
        temp = sorted(temp, key=lambda x: x[2], reverse=True) #第三维是size
        return (list(i) for i in zip(*temp)) #([1, 2, 4, 3] [4, 4, 2, 1])

    train_x=[]
    train_y=[]
    train_length=[]
    for data in train_data:
        train_x.append(data[0])
        train_y.append(data[1])
        train_length.append(data[2])
    train_x, train_y, train_length = sort(train_x, train_y, train_length)
    train_x=pad(train_x,max(train_length),paditem=0)
    train_y=pad(train_y,max(train_length),paditem=-1)

    train_x=torch.from_numpy(np.array(train_x)).long()
    train_y=torch.from_numpy(np.array(train_y)).long()
    train_length=torch.from_numpy(np.array(train_length))

    return train_x, train_y, train_length

class GedCorpus:
    def __init__(self,fdir,args):
        self.trainx,self.trainy,self.trainsize=self.load(fdir+r"\fce-public.train.original.tsv")
        self.word2id,self.id2word=self.makeword2veclist([self.trainx])
        self.vocabularysize=len(self.id2word)
        args.vocabulary_size=self.vocabularysize
        args.id2word=self.id2word
        self.datasize=len(self.trainx)
        self.label2id={"c":0,"i":1}
        self.trainx,self.trainy=self.preprocess((self.trainx,self.trainy),ispad=False)
        if args.loginfor:
            print("dictionary size : "+str(self.vocabularysize))
            print("train size : " + str(self.datasize))
        self.traindataset=GedDataset(self.trainx,self.trainy,self.trainsize,args.batch_size)
        self.traindataloader=DataLoader(dataset=self.traindataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)

        #Dev
        self.devx,self.devy,self.devsize=self.load(fdir+r"\fce-public.dev.original.tsv")
        self.devx,self.devy=self.preprocess((self.devx,self.devy),ispad=False)
        self.devdataset=GedDataset(self.devx,self.devy,self.devsize,1)
        self.devdataloader=DataLoader(dataset=self.devdataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)

        #Test
        self.testx,self.testy,self.testsize=self.load(fdir+r"\fce-public.test.original.tsv")
        self.testx,self.testy=self.preprocess((self.testx,self.testy),ispad=False)
        self.testdataset=GedDataset(self.testx,self.testy,self.testsize,1)
        self.testdataloader=DataLoader(dataset=self.testdataset,batch_size=1,shuffle=False,collate_fn=collate_fn)

    def load(self,fpath):
        if not os.path.exists(fpath):
            raise FileNotFoundError("Can not find the file \""+fpath+"\"")
        templist=open(fpath).read().strip().split("\n\n")
        x=[]
        y=[]
        size=[]
        for sentence in templist:
            wordlist=[]
            labellist=[]
            sentence=sentence.split("\n")
            for wordpair in sentence:
                wordpair=wordpair.split("\t")
                wordlist.append(wordpair[0])
                labellist.append(wordpair[1])
            x.append(wordlist)
            y.append(labellist)
            size.append(len(wordlist))
        return x,y,size #['Dear', 'Sir', 'or', 'Madam', ',']  ['c', 'c', 'c', 'c', 'c'] 5

    def makeword2veclist(self,datasetlist):
        counter = Counter()
        for dataset in datasetlist:
            for instance in dataset:
                counter.update(instance)
        word2id={}
        id2word=[]
        word2id["<pad>"]=0
        word2id["<unk>"]=1
        id2word.append("<pad>")
        id2word.append("<unk>")
        num=len(id2word)
        for k,v in counter.most_common():
            word2id[k]=num
            id2word.append(k)
            num+=1
        return word2id,id2word

    def preprocess(self,dataset,ispad=False,padwordid=0,unkid=1,padlabel=-1,maxlength=150):
        x=[]
        for instance in dataset[0]:
            temp=[self.word2id[w] if w in self.word2id else unkid for w in instance]
            if ispad:
                temp=temp[:min(len(temp),maxlength)]
                temp.extend([padwordid for _ in range(maxlength-len(temp))])
            x.append(temp)

        y=[]
        for instance in dataset[1]:
            temp = [self.label2id[w] for w in instance]
            if ispad:
                temp = temp[:min(len(temp), maxlength)]
                temp.extend([padlabel for _ in range(maxlength - len(temp))])
            y.append(temp)

        return x,y

class GedDataset(Dataset):
    def __init__(self,x,y,size,batchsize):
        self.x=x
        self.y=y
        self.size=size
        #self.x,self.y,self.size=self.sort(self.x,self.y,self.size)
        #self.x=self.x[:-(len(x)%batchsize)]
        #self.y=self.y[:-(len(x)%batchsize)]
        #self.size=self.size[:-(len(x)%batchsize)]
        self.len=len(self.x)

    def sort(self,*input): # [1,2,3,4] [4,4,1,2]
        temp = list(zip(*input)) #[(1, 4), (2, 4), (3, 1), (4, 2)]
        temp = sorted(temp, key=lambda x: x[2], reverse=True) #第三维是size
        return (list(i) for i in zip(*temp)) #([1, 2, 4, 3] [4, 4, 2, 1])

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.size[index]

    def __len__(self):
        return self.len



