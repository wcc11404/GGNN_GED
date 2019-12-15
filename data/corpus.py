import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import Counter
import pickle

def collate_fn(train_data):
    def pad(data, max_length, paditem=0):
        re = []
        for d in data:
            temp = d[:]
            temp.extend([paditem for _ in range(max_length - len(temp))])
            re.append(temp)
        return re

    def padchar(data, max_seq, max_char, paditem=0):
        re = []
        for s in data:
            stemp = []
            for w in s:
                temp = w[:]
                temp.extend([paditem for _ in range(max_char - len(temp))])
                stemp.append(temp)
            for _ in range(max_seq-len(stemp)):
                stemp.append([paditem for _ in range(max_char)])
            re.append(stemp)
        return re

    def padgraph(data, max_seq, edge_num, paditem=0):
        re = []
        for sentence in data:
            stemp = [[[paditem for _ in range(edge_num)] for _ in range(max_seq)] for _ in range(max_seq)]
            for i, word in enumerate(sentence):
                for id, relation in word:
                    stemp[i][id - 1][relation] = 1
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
    train_graph_in = []
    train_graph_out = []

    tup = train_data[0][0]
    task, edge_num = tup

    for data in train_data:
        train_x.append(data[1]) # B * S
        train_y.append(data[2]) # B
        train_length.append(data[3]) # B
        train_x_char.append(data[4]) # B * (S) * (W) ()代表需要pad
        train_length_char.append(data[5]) # B * (S)
        train_graph_in.append(data[6])
        train_graph_out.append(data[7])

    #train_x, train_y, train_length, train_x_char, train_length_char = sort(train_x, train_y, train_length,
    #                                                                           train_x_char, train_length_char)
    train_x = pad(train_x, max(train_length), paditem=0)  # B * S
    train_y = pad(train_y, max(train_length), paditem=-1)  # B * S , y pad -1是为了计算loss时候忽略用
    maxchar = getmetrixmax(train_length_char)   # B
    train_x_char = padchar(train_x_char, max(train_length), maxchar, paditem=0) # B * S * W
    train_length_char = pad(train_length_char, max(train_length), paditem=1)   # B * S 必须pad1,长度不能为0
    train_graph_in = padgraph(train_graph_in, max(train_length), edge_num, paditem=0) # B * S * S * EN
    train_graph_out = padgraph(train_graph_out, max(train_length), edge_num, paditem=0) # B * S * S * EN

    train_x = torch.from_numpy(np.array(train_x)).long()
    train_y = torch.from_numpy(np.array(train_y)).long()
    train_length = torch.from_numpy(np.array(train_length))
    train_x_char = torch.from_numpy(np.array(train_x_char)).long()
    train_length_char = torch.from_numpy(np.array(train_length_char))
    train_graph_in = torch.from_numpy(np.array(train_graph_in)).float()
    train_graph_out = torch.from_numpy(np.array(train_graph_out)).float()

    if task == "GGNNNER":
        extra_data = (train_graph_in, train_graph_out)
    elif task == "BaseNER" or task == "SLNER":
        extra_data = (train_x_char, train_length_char)
    else:
        extra_data = ()

    return train_x, train_y, train_length, extra_data

class GedCorpus:
    def __init__(self, fdir, args):
        self.args = args
        # self.label2id = {"c": 0, "i": 1}
        # if args.preprocess_dir is None or not os.path.exists(args.preprocess_dir):
        #     self.trainx, self.trainy, self.trainsize = self.load(fdir + r"/process/fce-public.train.preprocess.tsv",
        #                                                          bool(self.args.use_lower))
        #     self.devx, self.devy, self.devsize = self.load(fdir + r"/process/fce-public.dev.preprocess.tsv",
        #                                                    bool(self.args.use_lower))
        #     self.testx, self.testy, self.testsize = self.load(fdir + r"/process/fce-public.test.preprocess.tsv",
        #                                                       bool(self.args.use_lower))
        #     self.train_graph = self.load_graph(fdir + r"/process/train_graph.txt")
        #
        #     self.dev_graph = self.load_graph(fdir + r"/process/dev_graph.txt")
        #     self.test_graph = self.load_graph(fdir + r"/process/test_graph.txt")
        #
        #     self.word2id, self.id2word = self.makeword2veclist([self.trainx, self.devx])
        #     self.char2id, self.id2char = self.makechar2veclist([self.trainx, self.devx])
        #     self.edge2id, self.id2edge = self.makeedge2veclist([self.train_graph, self.dev_graph])
        #
        #     self.trainx_char, self.trainsize_char = self.preprocess_char(self.trainx, ispad=False)
        #     self.trainx, self.trainy = self.preprocess((self.trainx, self.trainy), ispad=False)
        #     self.devx_char, self.devsize_char = self.preprocess_char(self.devx, ispad=False)
        #     self.devx, self.devy = self.preprocess((self.devx, self.devy), ispad=False)
        #     self.testx_char, self.testsize_char = self.preprocess_char(self.testx, ispad=False)
        #     self.testx, self.testy = self.preprocess((self.testx, self.testy), ispad=False)
        #     self.train_graph = self.preprocess_graph(self.train_graph)
        #
        #     self.dev_graph = self.preprocess_graph(self.dev_graph)
        #     self.test_graph = self.preprocess_graph(self.test_graph)
        #
        #     self.save_preprocess(args.preprocess_dir)
        # else:
        self.load_preprocess(args.preprocess_dir)

        self.wordvocabularysize = len(self.id2word)
        args.word_vocabulary_size = self.wordvocabularysize
        self.charvocabularysize = len(self.id2char)
        args.char_vocabulary_size = self.charvocabularysize
        self.edgevocabularysize = len(self.id2edge)
        args.edge_vocabulary_size = self.edgevocabularysize
        args.word2id = self.word2id

        if bool(args.loginfor):
            print("word dictionary size : " + str(self.wordvocabularysize))
            print("char dictionary size : " + str(self.charvocabularysize))
            print("edge dictionary size : " + str(self.edgevocabularysize))
            print("train data size : " + str(len(self.trainx)))
            print("max train data length : " + str(max(self.trainsize)))
            print("dev data size : " + str(len(self.devx)))
            print("max dev data length : " + str(max(self.devsize)))
            print("test data size : " + str(len(self.testx)))
            print("max test data length : " + str(max(self.testsize)))

        #Train
        self.traindataset = GedDataset((self.args.arch, self.edgevocabularysize), self.trainx, self.trainy,
                                       self.trainsize, self.trainx_char, self.trainsize_char, self.train_graph)
        self.traindataloader = DataLoader(dataset=self.traindataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_fn)

        #Dev
        self.devdataset = GedDataset((self.args.arch, self.edgevocabularysize), self.devx, self.devy, self.devsize,
                                     self.devx_char, self.devsize_char, self.dev_graph)
        self.devdataloader = DataLoader(dataset=self.devdataset, batch_size=args.batch_size, shuffle=False,
                                        collate_fn=collate_fn)

        #Test
        self.testdataset = GedDataset((self.args.arch, self.edgevocabularysize), self.testx, self.testy, self.testsize,
                                      self.testx_char, self.testsize_char, self.test_graph)
        self.testdataloader = DataLoader(dataset=self.testdataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    def save_preprocess(self,dir):
        f = open(dir, 'wb')

        pickle.dump(self.word2id, f)
        pickle.dump(self.id2word, f)
        pickle.dump(self.char2id, f)
        pickle.dump(self.id2char, f)
        pickle.dump(self.edge2id, f)
        pickle.dump(self.id2edge, f)

        pickle.dump(self.trainx, f)
        pickle.dump(self.trainy, f)
        pickle.dump(self.trainsize, f)
        pickle.dump(self.trainx_char, f)
        pickle.dump(self.trainsize_char, f)
        pickle.dump(self.train_graph, f)

        pickle.dump(self.devx, f)
        pickle.dump(self.devy, f)
        pickle.dump(self.devsize, f)
        pickle.dump(self.devx_char, f)
        pickle.dump(self.devsize_char, f)
        pickle.dump(self.dev_graph, f)

        pickle.dump(self.testx, f)
        pickle.dump(self.testy, f)
        pickle.dump(self.testsize, f)
        pickle.dump(self.testx_char, f)
        pickle.dump(self.testsize_char, f)
        pickle.dump(self.test_graph, f)

        f.close()

    def load_preprocess(self,dir):
        f = open(dir, 'rb')

        self.word2id = pickle.load(f)
        self.id2word = pickle.load(f)
        self.char2id = pickle.load(f)
        self.id2char = pickle.load(f)
        self.edge2id = pickle.load(f)
        self.id2edge = pickle.load(f)

        self.trainx = pickle.load(f)
        self.trainy = pickle.load(f)
        self.trainsize = pickle.load(f)
        self.trainx_char = pickle.load(f)
        self.trainsize_char = pickle.load(f)
        self.train_graph = pickle.load(f)

        self.devx = pickle.load(f)
        self.devy = pickle.load(f)
        self.devsize = pickle.load(f)
        self.devx_char = pickle.load(f)
        self.devsize_char = pickle.load(f)
        self.dev_graph = pickle.load(f)

        self.testx = pickle.load(f)
        self.testy = pickle.load(f)
        self.testsize = pickle.load(f)
        self.testx_char = pickle.load(f)
        self.testsize_char = pickle.load(f)
        self.test_graph = pickle.load(f)

        f.close()

    def load(self, fpath, use_lower=False):
        if not os.path.exists(fpath):
            raise FileNotFoundError("Can not find the file \""+fpath+"\"")
        templist = open(fpath).read().strip().split("\n\n")
        x = []
        y = []
        size = []
        for sentence in templist:
            wordlist = []
            labellist = []
            sentence = sentence.split("\n")
            for wordpair in sentence:
                wordpair = wordpair.split("\t")
                if use_lower:
                    wordpair[0] = wordpair[0].lower()
                wordlist.append(wordpair[0])
                labellist.append(wordpair[1])
            x.append(wordlist)
            y.append(labellist)
            size.append(len(wordlist))
        return x, y, size  # ['Dear', 'Sir', 'or', 'Madam', ',']  ['c', 'c', 'c', 'c', 'c'] 5

    def load_graph(self, fpath):
        f = open(fpath, 'r').read().strip().split("\n\n")
        graph_in = []
        graph_out = []
        for line in f:
            line = line.split("\n")
            graph_sen_in = []
            graph_sen_out = []
            for word in line:
                word = word.split(";")
                edge_in = word[1].split()
                edge_in = [[int(edge.split(',')[0]), edge.split(',')[1]] for edge in edge_in]
                edge_out = word[2].split()
                edge_out = [[int(edge.split(',')[0]), edge.split(',')[1]] for edge in edge_out]
                graph_sen_in.append(edge_in)
                graph_sen_out.append(edge_out)
            graph_in.append(graph_sen_in)
            graph_out.append(graph_sen_out)
        return [graph_in, graph_out] # in/out , sen , word , [id->relation]

    def makeword2veclist(self, datasetlist):
        counter = Counter()
        for dataset in datasetlist:
            for instance in dataset:
                counter.update(instance)
        word2id = {}
        id2word = []
        word2id["<pad>"] = 0
        word2id["<unk>"] = 1
        id2word.append("<pad>")
        id2word.append("<unk>")
        num = len(id2word)
        for k, v in counter.most_common():
            word2id[k] = num
            id2word.append(k)
            num += 1
        return word2id, id2word

    def makechar2veclist(self, datasetlist):
        counter = Counter()
        for dataset in datasetlist:
            for instance in dataset:
                for word in instance:
                    counter.update(word)
        char2id = {}
        id2char = []
        char2id["<pad>"] = 0
        char2id["<unk>"] = 1
        id2char.append("<pad>")
        id2char.append("<unk>")
        num = len(id2char)
        for k, v in counter.most_common():
            char2id[k] = num
            id2char.append(k)
            num += 1
        return char2id, id2char

    def makeedge2veclist(self, datasetlist):
        counter = Counter()
        for dataset in datasetlist:
            #其实只用统计入度边即可
            for instance in dataset[0]:
                    for word in instance:
                        for id_relation in word:
                            relation = id_relation[1]
                            counter.update([relation])
            for instance in dataset[1]:
                    for word in instance:
                        for id_relation in word:
                            relation = id_relation[1]
                            counter.update([relation])
        edge2id = {}
        id2edge = []
        edge2id["<pad>"] = 0 # 这个pad已经没用了
        edge2id["<unk>"] = 1
        id2edge.append("<pad>")
        id2edge.append("<unk>")
        num = len(id2edge)
        for k, v in counter.most_common():
            edge2id[k] = num
            id2edge.append(k)
            num += 1
        return edge2id, id2edge

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

    def preprocess_char(self, dataset, ispad=False, padid=0, unkid=1, maxcharlength=-1):
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
        return charx, charsize

    def preprocess_graph(self, dataset, ispad=False, padid=0, unkid=1):
        graph = []
        for in_out in dataset:
            graph_in_out = []
            for instance in in_out:
                graph_sen = []
                for word in instance:
                    graph_word = []
                    for id_relation in word:
                        if id_relation[1] in self.edge2id:
                            graph_word.append([id_relation[0], self.edge2id[id_relation[1]]])
                        else:
                            graph_word.append([id_relation[0], unkid])
                    graph_sen.append(graph_word)
                graph_in_out.append(graph_sen)
            graph.append(graph_in_out)
        return graph

class GedDataset(Dataset):
    def __init__(self, tup, x, y, size, x_char, size_char, graph):
        self.tup = tup
        self.x = x
        self.y = y
        self.size = size
        self.x_char = x_char
        self.size_char = size_char
        self.graph_in = graph[0]
        self.graph_out = graph[1]
        self.len = len(self.x)

    def sort(self,*input): # [1,2,3,4] [4,4,1,2]
        temp = list(zip(*input)) #[(1, 4), (2, 4), (3, 1), (4, 2)]
        temp = sorted(temp, key=lambda x: x[2], reverse=True) #第三维是size
        return (list(i) for i in zip(*temp)) #([1, 2, 4, 3] [4, 4, 2, 1])

    def __getitem__(self, index):
        return self.tup, self.x[index], self.y[index], self.size[index], self.x_char[index], self.size_char[index], self.graph_in[
            index], self.graph_out[index]

    def __len__(self):
        return self.len



