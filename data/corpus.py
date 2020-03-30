import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
from torch.utils.data.distributed import DistributedSampler
from myscripts.utils import log_information

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

    def forwardlabel(data, max_length, paditem=0):
        re = []
        for d in data:
            temp = d[1:]
            temp.append(0)
            temp.extend([paditem for _ in range(max_length - len(temp))])
            re.append(temp)
        return re

    def bakwardlabel(data, max_length, paditem=0):
        re = []
        for d in data:
            temp = [0]
            temp.extend(d[:-1])
            temp.extend([paditem for _ in range(max_length - len(temp))])
            re.append(temp)
        return re

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
    train_left_x = forwardlabel(train_x, max(train_length), paditem=-1)
    train_right_x = bakwardlabel(train_x, max(train_length), paditem=-1)
    train_y = pad(train_y, max(train_length), paditem=-1)  # B * S , y pad -1是为了计算loss时候忽略用
    if train_length_char[0] is not None:
        maxchar = getmetrixmax(train_length_char)   # B
        train_x_char = padchar(train_x_char, max(train_length), maxchar, paditem=0) # B * S * W
        train_length_char = pad(train_length_char, max(train_length), paditem=1)   # B * S 必须pad1,长度不能为0
    if train_graph_in[0] is not None:
            train_graph_in = padgraph(train_graph_in, max(train_length), edge_num, paditem=0) # B * S * S * EN
            train_graph_out = padgraph(train_graph_out, max(train_length), edge_num, paditem=0) # B * S * S * EN

    train_x = torch.from_numpy(np.array(train_x)).long()
    train_left_x = torch.from_numpy(np.array(train_left_x)).long()
    train_right_x = torch.from_numpy(np.array(train_right_x)).long()
    train_y = torch.from_numpy(np.array(train_y)).long()
    train_length = torch.from_numpy(np.array(train_length))
    if train_length_char[0] is not None:
        train_x_char = torch.from_numpy(np.array(train_x_char)).long()
        train_length_char = torch.from_numpy(np.array(train_length_char))
    if train_graph_in[0] is not None:
        train_graph_in = torch.from_numpy(np.array(train_graph_in)).float()
        train_graph_out = torch.from_numpy(np.array(train_graph_out)).float()

    if task=="GANNER":
        extra_data = (train_x_char, train_length_char)
        extra_label = (train_left_x, train_right_x)
    elif task == "GGNNNER":
        extra_data = (train_x_char, train_length_char, train_graph_in, train_graph_out)
        extra_label = (train_left_x, train_right_x)
    elif task == "SLNER":
        extra_data = (train_x_char, train_length_char)
        extra_label = (train_left_x, train_right_x)
    elif task == "BaseNER":
        extra_data = (train_x_char, train_length_char)
        extra_label = ()
    else:
        extra_data = ()
        extra_label = ()

    return train_x, train_y, train_length, extra_data, extra_label

class GedCorpus:
    def __init__(self, args):
        self.args = args
        self.load_preprocess(args)

        self.wordvocabularysize = len(self.id2word)
        args.word_vocabulary_size = self.wordvocabularysize
        if self.id2char is not None:
            self.charvocabularysize = len(self.id2char)
        else:
            self.charvocabularysize = 0
        args.char_vocabulary_size = self.charvocabularysize
        if self.id2edge is not None:
            self.edgevocabularysize = len(self.id2edge)
        else:
            self.edgevocabularysize = 0
        args.edge_vocabulary_size = self.edgevocabularysize
        args.word2id = self.word2id

        # 打印数据信息
        log_information(args, "word dictionary size : " + str(self.wordvocabularysize))
        if self.id2edge is not None:
            log_information(args, "char dictionary size : " + str(self.charvocabularysize))
        if self.id2edge is not None:
            log_information(args, "edge dictionary size : " + str(self.edgevocabularysize))
        log_information(args, "train data size : " + str(len(self.trainx)))
        log_information(args, "max train data length : " + str(max(self.trainsize)))
        log_information(args, "dev data size : " + str(len(self.devx)))
        log_information(args, "max dev data length : " + str(max(self.devsize)))
        if self.testx is not None:
            log_information(args, "test data size : " + str(len(self.testx)))
            log_information(args, "max test data length : " + str(max(self.testsize)))
            log_information(args, "")

        #Train
        self.traindataset = GedDataset((self.args.arch, self.edgevocabularysize), self.trainx, self.trainy,
                                       self.trainsize, self.trainx_char, self.trainsize_char, self.train_graph)

        #Dev
        self.devdataset = GedDataset((self.args.arch, self.edgevocabularysize), self.devx, self.devy, self.devsize,
                                     self.devx_char, self.devsize_char, self.dev_graph)

        #Test
        if self.testx is not None:
            self.testdataset = GedDataset((self.args.arch, self.edgevocabularysize), self.testx, self.testy, self.testsize,
                                          self.testx_char, self.testsize_char, self.test_graph)
        else:
            self.testdataset = None

    def load_preprocess(self, args):
        f = open(args.data_dir, 'rb')

        self.sign = pickle.load(f)
        assert self.sign[0] == 1 and self.sign[3] == 1
        item0 = self.sign[0] + self.sign[3] + self.sign[6]
        item1 = self.sign[1] + self.sign[4] + self.sign[7]
        item2 = self.sign[2] + self.sign[5] + self.sign[8]
        assert item0 == item1 and item1 == item2

        f1 = open(args.vocab_dir + "/wordvocab.pkl", "rb")
        self.word2id = pickle.load(f1)
        self.id2word = pickle.load(f1)
        f1.close()
        if self.sign[1] == 1:
            f1 = open(args.vocab_dir + "/charvocab.pkl", "rb")
            self.char2id = pickle.load(f1)
            self.id2char = pickle.load(f1)
            f1.close()
        else:
            self.char2id = None
            self.id2char = None
        if self.sign[2] == 1:
            f1 = open(args.vocab_dir + "/edgevocab.pkl", "rb")
            self.edge2id = pickle.load(f1)
            self.id2edge = pickle.load(f1)
            f1.close()
        else:
            self.edge2id = None
            self.id2edge = None

        self.trainx, self.trainy, self.trainsize = None, None, None
        self.trainx_char, self.trainsize_char, self.train_graph = None, None, None
        self.devx, self.devy, self.devsize = None, None, None
        self.devx_char, self.devsize_char, self.dev_graph = None, None, None
        self.testx, self.testy, self.testsize = None, None, None
        self.testx_char, self.testsize_char, self.test_graph = None, None, None

        if self.sign[0] == 1:
            self.trainx = pickle.load(f)
            self.trainy = pickle.load(f)
            self.trainsize = pickle.load(f)
        if self.sign[1] == 1:
            self.trainx_char = pickle.load(f)
            self.trainsize_char = pickle.load(f)
        if self.sign[2] == 1:
            self.train_graph = pickle.load(f)

        if self.sign[3] == 1:
            self.devx = pickle.load(f)
            self.devy = pickle.load(f)
            self.devsize = pickle.load(f)
        if self.sign[4] == 1:
            self.devx_char = pickle.load(f)
            self.devsize_char = pickle.load(f)
        if self.sign[5] == 1:
            self.dev_graph = pickle.load(f)

        if self.sign[6] == 1:
            self.testx = pickle.load(f)
            self.testy = pickle.load(f)
            self.testsize = pickle.load(f)
        if self.sign[7] == 1:
            self.testx_char = pickle.load(f)
            self.testsize_char = pickle.load(f)
        if self.sign[8] == 1:
            self.test_graph = pickle.load(f)

        f.close()

    def build_Dataloader(self):
        # Train loader
        if self.args.use_ddp:
            trainsampler = DistributedSampler(self.traindataset)
            self.traindataloader = DataLoader(dataset=self.traindataset, batch_size=self.args.batch_size,
                                              shuffle=False, collate_fn=collate_fn, num_workers=self.args.num_workers,
                                              sampler=trainsampler)
        elif self.args.use_dp:
            self.traindataloader = DataLoader(dataset=self.traindataset,
                                              batch_size=self.args.batch_size * len(self.args.gpu_ids),
                                              shuffle=True, collate_fn=collate_fn, num_workers=self.args.num_workers)
        else:
            self.traindataloader = DataLoader(dataset=self.traindataset, batch_size=self.args.batch_size,
                                              shuffle=True, collate_fn=collate_fn, num_workers=self.args.num_workers)

        # Dev loader
        if self.args.use_dp:
            self.devdataloader = DataLoader(dataset=self.devdataset,
                                            batch_size=1 * len(self.args.gpu_ids),
                                            shuffle=False, collate_fn=collate_fn, num_workers=self.args.num_workers)
        else:
            self.devdataloader = DataLoader(dataset=self.devdataset, batch_size=1,
                                            shuffle=False, collate_fn=collate_fn, num_workers=self.args.num_workers)

        # Test loader
        if self.testdataset is not None:
                self.testdataloader = DataLoader(dataset=self.testdataset, batch_size=1, shuffle=False,
                                                 collate_fn=collate_fn, num_workers=self.args.num_workers)
        else:
            self.testdataloader = None

class GedDataset(Dataset):
    def __init__(self, tup, x, y, size, x_char, size_char, graph):
        self.tup = tup
        self.x = x
        self.y = y
        self.size = size
        self.x_char = x_char
        self.size_char = size_char
        self.graph_in = graph[0] if graph is not None else None
        self.graph_out = graph[1] if graph is not None else None
        self.len = len(self.x)

    def sort(self, *input):  # [1,2,3,4] [4,4,1,2]
        temp = list(zip(*input)) #[(1, 4), (2, 4), (3, 1), (4, 2)]
        temp = sorted(temp, key=lambda x: x[2], reverse=True) #第三维是size
        return (list(i) for i in zip(*temp)) #([1, 2, 4, 3] [4, 4, 2, 1])

    def __getitem__(self, index):
        return self.tup, self.x[index], self.y[index], self.size[index], \
               self.x_char[index] if self.x_char is not None else None, \
               self.size_char[index] if self.size_char is not None else None,\
               self.graph_in[index] if self.graph_in is not None else None, \
               self.graph_out[index] if self.graph_out is not None else None

    def __len__(self):
        return self.len



