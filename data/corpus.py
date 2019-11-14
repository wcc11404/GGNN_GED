import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import Counter
import pickle

def collate_fn(train_data):
    def pad(data, max_length, paditem=0):
        re=[]
        for d in data:
            temp=d[:]
            temp.extend([paditem for _ in range(max_length-len(temp))])
            re.append(temp)
        return re

    def padchar(data, max_seq, max_char, paditem=0):
        re=[]
        for s in data:
            stemp=[]
            for w in s:
                temp=w[:]
                temp.extend([paditem for _ in range(max_char-len(temp))])
                stemp.append(temp)
            for _ in range(max_seq-len(stemp)):
                stemp.append([paditem for _ in range(max_char)])
            re.append(stemp)
        return re

    def sort(*input): # [1,2,3,4] [4,4,1,2]
        temp = list(zip(*input)) #[(1, 4), (2, 4), (3, 1), (4, 2)]
        temp = sorted(temp, key=lambda x: x[2], reverse=True) #第三维是size
        return (list(i) for i in zip(*temp)) #([1, 2, 4, 3] [4, 4, 2, 1])

    def getmetrixmax(metrix):
        temp=[]
        for m in metrix:
            temp.append(max(m))
        return max(temp)

    train_x= []
    train_y = []
    train_length = []
    train_x_char = []
    train_length_char = []
    for data in train_data:
        train_x.append(data[0]) # B * S
        train_y.append(data[1]) # B
        train_length.append(data[2]) # B
        train_x_char.append(data[3]) # B * (S) * (W) ()代表需要pad
        train_length_char.append(data[4]) # B * (S)
    #train_x, train_y, train_length, train_x_char, train_length_char = sort(train_x, train_y, train_length,
    #                                                                           train_x_char, train_length_char)
    train_x = pad(train_x, max(train_length), paditem=0)  # B * S
    train_y = pad(train_y, max(train_length), paditem=-1)  # B * S
    maxchar = getmetrixmax(train_length_char)   # B
    train_x_char = padchar(train_x_char, max(train_length), maxchar, paditem=0) # B * S * W
    train_length_char = pad(train_length_char, max(train_length), paditem=1)   # B * S 必须pad1,长度不能为0

    train_x=torch.from_numpy(np.array(train_x)).long()
    train_y=torch.from_numpy(np.array(train_y)).long()
    train_length=torch.from_numpy(np.array(train_length))
    train_x_char=torch.from_numpy(np.array(train_x_char)).long()
    train_length_char=torch.from_numpy(np.array(train_length_char))

    return train_x, train_y, train_length, train_x_char, train_length_char

class GedCorpus:
    def __init__(self,fdir,args):
        self.args=args
        self.label2id = {"c": 0, "i": 1}
        if args.preprocess_dir is None or not os.path.exists(args.preprocess_dir):
            self.trainx, self.trainy, self.trainsize = self.load(fdir + r"/fce-public.train.original.tsv",bool(self.args.use_lower))
            self.devx, self.devy, self.devsize = self.load(fdir + r"/fce-public.dev.original.tsv",bool(self.args.use_lower))
            self.testx, self.testy, self.testsize = self.load(fdir + r"/fce-public.test.original.tsv",bool(self.args.use_lower))
            self.word2id, self.id2word = self.makeword2veclist([self.trainx, self.devx])
            self.char2id, self.id2char = self.makechar2veclist([self.trainx, self.devx])

            self.trainx_char, self.trainsize_char = self.preprocess_char(self.trainx, ispad=False)
            self.trainx, self.trainy = self.preprocess((self.trainx, self.trainy), ispad=False)
            self.devx_char, self.devsize_char = self.preprocess_char(self.devx, ispad=False)
            self.devx, self.devy = self.preprocess((self.devx, self.devy), ispad=False)
            self.testx_char, self.testsize_char = self.preprocess_char(self.testx, ispad=False)
            self.testx, self.testy = self.preprocess((self.testx, self.testy), ispad=False)

            self.save_preprocess(args.preprocess_dir)
        else:
            self.load_preprocess(args.preprocess_dir)

        self.wordvocabularysize = len(self.id2word)
        args.word_vocabulary_size = self.wordvocabularysize
        self.charvocabularysize = len(self.id2char)
        args.char_vocabulary_size = self.charvocabularysize
        args.word2id=self.word2id
        # args.char2id=self.char2id

        if bool(args.loginfor):
            print("word dictionary size : "+str(self.wordvocabularysize))
            print("char dictionary size : "+str(self.charvocabularysize))
            print("train data size : " + str(len(self.trainx)))
            print("dev data size : " + str(len(self.devx)))
            print("test data size : " + str(len(self.testx)))

        #Train
        self.traindataset = GedDataset(self.trainx, self.trainy, self.trainsize, self.trainx_char, self.trainsize_char)
        self.traindataloader = DataLoader(dataset=self.traindataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_fn)

        #Dev
        self.devdataset = GedDataset(self.devx, self.devy, self.devsize, self.devx_char, self.devsize_char)
        self.devdataloader = DataLoader(dataset=self.devdataset, batch_size=args.batch_size, shuffle=False,
                                        collate_fn=collate_fn)

        #Test
        self.testdataset = GedDataset(self.testx, self.testy, self.testsize, self.testx_char, self.testsize_char)
        self.testdataloader = DataLoader(dataset=self.testdataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    def save_preprocess(self,dir):
        f = open(dir, 'wb')

        pickle.dump(self.word2id, f)
        pickle.dump(self.id2word, f)
        pickle.dump(self.char2id, f)
        pickle.dump(self.id2char, f)

        pickle.dump(self.trainx,f)
        pickle.dump(self.trainy,f)
        pickle.dump(self.trainsize,f)
        pickle.dump(self.trainx_char,f)
        pickle.dump(self.trainsize_char,f)

        pickle.dump(self.devx,f)
        pickle.dump(self.devy,f)
        pickle.dump(self.devsize,f)
        pickle.dump(self.devx_char,f)
        pickle.dump(self.devsize_char,f)

        pickle.dump(self.testx,f)
        pickle.dump(self.testy,f)
        pickle.dump(self.testsize,f)
        pickle.dump(self.testx_char,f)
        pickle.dump(self.testsize_char,f)

        f.close()

    def load_preprocess(self,dir):
        f = open(dir, 'rb')

        self.word2id = pickle.load(f)
        self.id2word = pickle.load(f)
        self.char2id = pickle.load(f)
        self.id2char = pickle.load(f)

        self.trainx = pickle.load(f)
        self.trainy = pickle.load(f)
        self.trainsize = pickle.load(f)
        self.trainx_char = pickle.load(f)
        self.trainsize_char = pickle.load(f)

        self.devx = pickle.load(f)
        self.devy = pickle.load(f)
        self.devsize = pickle.load(f)
        self.devx_char = pickle.load(f)
        self.devsize_char = pickle.load(f)

        self.testx = pickle.load(f)
        self.testy = pickle.load(f)
        self.testsize = pickle.load(f)
        self.testx_char = pickle.load(f)
        self.testsize_char = pickle.load(f)

        f.close()

    def load(self, fpath, use_lower=False):
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
                if use_lower:
                    wordpair[0]=wordpair[0].lower()
                wordlist.append(wordpair[0])
                labellist.append(wordpair[1])
            x.append(wordlist)
            y.append(labellist)
            size.append(len(wordlist))
        return x,y,size #['Dear', 'Sir', 'or', 'Madam', ',']  ['c', 'c', 'c', 'c', 'c'] 5

    def makeword2veclist(self, datasetlist):
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

    def makechar2veclist(self, datasetlist):
        counter = Counter()
        for dataset in datasetlist:
            for instance in dataset:
                for word in instance:
                    counter.update(word)
        char2id={}
        id2char=[]
        char2id["<pad>"]=0
        char2id["<unk>"]=1
        id2char.append("<pad>")
        id2char.append("<unk>")
        num=len(id2char)
        for k,v in counter.most_common():
            char2id[k]=num
            id2char.append(k)
            num+=1
        return char2id,id2char

    def preprocess_char(self, dataset, ispad=False, padid=0, unkid=1, padlabel=-1, maxcharlength=-1):
        charx = []
        charsize = []
        for instance in dataset:
            ctemp = []
            sizetemp = []
            for w in instance:
                cc = [self.char2id[c] if c in self.char2id else unkid for c in w]
                ctemp.append(cc)
                sizetemp.append(len(cc))
            charx.append(ctemp)
            charsize.append(sizetemp)
        return charx,charsize

    def preprocess(self, dataset, ispad=False, padid=0, unkid=1, padlabel=-1, maxlength=150):
        x=[]
        for instance in dataset[0]:
            temp=[self.word2id[w] if w in self.word2id else unkid for w in instance]
            if ispad:
                temp=temp[:min(len(temp),maxlength)]
                temp.extend([padid for _ in range(maxlength-len(temp))])
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
    def __init__(self, x, y, size, x_char, size_char):
        self.x = x
        self.y = y
        self.size = size
        self.x_char = x_char
        self.size_char = size_char
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
        return self.x[index], self.y[index], self.size[index], self.x_char[index], self.size_char[index]

    def __len__(self):
        return self.len



