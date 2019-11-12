import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import Counter
import pickle

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
        self.args=args
        self.label2id = {"c": 0, "i": 1}
        if args.preprocess_dir is None or not os.path.exists(args.preprocess_dir):
            #os.makedirs(args.preprocess_dir)
            #Train
            self.trainx,self.trainy,self.trainsize=self.load(fdir+r"/fce-public.train.original.tsv")
            self.word2id,self.id2word=self.makeword2veclist([self.trainx])
            self.trainx, self.trainy = self.preprocess((self.trainx, self.trainy), ispad=False)

            #Dev
            self.devx, self.devy, self.devsize = self.load(fdir + r"/fce-public.dev.original.tsv")
            self.devx, self.devy = self.preprocess((self.devx, self.devy), ispad=False)

            #Test
            self.testx, self.testy, self.testsize = self.load(fdir + r"/fce-public.test.original.tsv")
            self.testx, self.testy = self.preprocess((self.testx, self.testy), ispad=False)

            self.save_preprocess(args.preprocess_dir)
        else:
            self.load_preprocess(args.preprocess_dir)

        self.vocabularysize=len(self.id2word)
        args.vocabulary_size=self.vocabularysize
        args.word2id=self.word2id
        self.datasize=len(self.trainx)

        if bool(args.loginfor):
            print("dictionary size : "+str(self.vocabularysize))
            print("train data size : " + str(self.datasize))
            print("dev data size : " + str(len(self.devx)))
            print("test data size : " + str(len(self.testx)))

        #Train
        self.traindataset=GedDataset(self.trainx,self.trainy,self.trainsize,args.batch_size)
        self.traindataloader=DataLoader(dataset=self.traindataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)

        #Dev
        self.devdataset=GedDataset(self.devx,self.devy,self.devsize,args.batch_size)
        self.devdataloader=DataLoader(dataset=self.devdataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)

        #Test
        self.testdataset=GedDataset(self.testx,self.testy,self.testsize,1)
        self.testdataloader=DataLoader(dataset=self.testdataset,batch_size=1,shuffle=False,collate_fn=collate_fn)

    def save_preprocess(self,dir):
        f=open(dir,'wb')
        pickle.dump(self.trainx,f)
        pickle.dump(self.trainy,f)
        pickle.dump(self.trainsize,f)
        pickle.dump(self.word2id,f)
        pickle.dump(self.id2word,f)
        pickle.dump(self.devx,f)
        pickle.dump(self.devy,f)
        pickle.dump(self.devsize,f)
        pickle.dump(self.testx,f)
        pickle.dump(self.testy,f)
        pickle.dump(self.testsize,f)
        f.close()

    def load_preprocess(self,dir):
        f = open(dir, 'rb')
        self.trainx = pickle.load(f)
        self.trainy = pickle.load(f)
        self.trainsize = pickle.load(f)
        self.word2id = pickle.load(f)
        self.id2word = pickle.load(f)
        self.devx = pickle.load(f)
        self.devy = pickle.load(f)
        self.devsize = pickle.load(f)
        self.testx = pickle.load(f)
        self.testy = pickle.load(f)
        self.testsize = pickle.load(f)
        f.close()

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
                if bool(self.args.use_lower):
                    wordpair[0]=wordpair[0].lower()
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



