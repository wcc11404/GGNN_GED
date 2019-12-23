import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
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

    train_x = []
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
        extra_data = (train_x_char, train_length_char, train_graph_in, train_graph_out)
    elif task == "BaseNER" or task == "SLNER":
        extra_data = (train_x_char, train_length_char)
    else:
        extra_data = ()

    return train_x, train_y, train_length, extra_data

class GedCorpus:
    def __init__(self, args):
        self.args = args
        self.load_preprocess(args.preprocess_dir)

        self.wordvocabularysize = len(self.id2word)
        args.word_vocabulary_size = self.wordvocabularysize
        self.charvocabularysize = len(self.id2char)
        args.char_vocabulary_size = self.charvocabularysize
        self.edgevocabularysize = len(self.id2edge)
        args.edge_vocabulary_size = self.edgevocabularysize
        args.word2id = self.word2id

        if args.loginfor:
            print("word dictionary size : " + str(self.wordvocabularysize))
            print("char dictionary size : " + str(self.charvocabularysize))
            print("edge dictionary size : " + str(self.edgevocabularysize))
            print("train data size : " + str(len(self.trainx)))
            print("max train data length : " + str(max(self.trainsize)))
            print("dev data size : " + str(len(self.devx)))
            print("max dev data length : " + str(max(self.devsize)))
            print("test data size : " + str(len(self.testx)))
            print("max test data length : " + str(max(self.testsize)))
            print()

        #Train
        self.traindataset = GedDataset((self.args.arch, self.edgevocabularysize), self.trainx, self.trainy,
                                       self.trainsize, self.trainx_char, self.trainsize_char, self.train_graph)
        self.traindataloader = DataLoader(dataset=self.traindataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_fn, num_workers=args.num_workers)

        #Dev
        self.devdataset = GedDataset((self.args.arch, self.edgevocabularysize), self.devx, self.devy, self.devsize,
                                     self.devx_char, self.devsize_char, self.dev_graph)
        self.devdataloader = DataLoader(dataset=self.devdataset, batch_size=args.batch_size, shuffle=False,
                                        collate_fn=collate_fn, num_workers=args.num_workers)

        #Test
        self.testdataset = GedDataset((self.args.arch, self.edgevocabularysize), self.testx, self.testy, self.testsize,
                                      self.testx_char, self.testsize_char, self.test_graph)
        self.testdataloader = DataLoader(dataset=self.testdataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                         num_workers=args.num_workers)

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

    def sort(self, *input):  # [1,2,3,4] [4,4,1,2]
        temp = list(zip(*input)) #[(1, 4), (2, 4), (3, 1), (4, 2)]
        temp = sorted(temp, key=lambda x: x[2], reverse=True) #第三维是size
        return (list(i) for i in zip(*temp)) #([1, 2, 4, 3] [4, 4, 2, 1])

    def __getitem__(self, index):
        return self.tup, self.x[index], self.y[index], self.size[index], self.x_char[index], self.size_char[index],\
               self.graph_in[index], self.graph_out[index]

    def __len__(self):
        return self.len



