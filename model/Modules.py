import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os
import numpy as np

class EmbeddingTemplate(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_drop):
        super(EmbeddingTemplate, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.wordembedding = nn.Embedding(vocab_size, embed_dim)
        self.wordembeddingdropout = nn.Dropout(embed_drop)

    def forward(self, batchinput):# B * S
        embedout = self.wordembedding(batchinput)
        embedout = self.wordembeddingdropout(embedout)
        return embedout # B * S * E

    def load_from_w2v(self,word2id,padandunk=True,w2v_dir=None,loginfor=True):#加载w2v
        w2v={}
        if w2v_dir is None or not os.path.exists(w2v_dir):
            raise KeyError("w2v file is not exists")
        temp=self.wordembedding.weight.detach().numpy()

        num = 0
        with open(w2v_dir,"rb") as f:
            header=f.readline()
            vocab_size, layer1_size = map(int, header.split())  # 3000000 300
            assert self.embed_dim==layer1_size
            binary_len = np.dtype('float32').itemsize * layer1_size  # 1200
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in word2id:
                    temp[word2id[word]]=np.fromstring(f.read(binary_len), dtype='float32')
                    num+=1
                else:
                    f.read(binary_len)
                w2v[word]=np.fromstring(f.read(binary_len), dtype='float32')

        # num=0
        # temp=np.zeros(shape=[1,self.embed_dim],dtype=float)#pad
        # #temp.append([0 for _ in range(self.embed_dim)])
        # for i in range(1,self.vocab_size):
        #     if id2word[i] in w2v:
        #         temp=np.append(temp,[w2v[id2word[i]]],axis=0)
        #         num+=1
        #     else:
        #         temp=np.append(temp,np.random.normal(loc=0.0,scale=1.0,size=(1,self.embed_dim)),axis=0)#正态分布

        self.wordembedding.weight.data.copy_(torch.from_numpy(temp))

        if loginfor:
            print("load {} word embeddings".format(num))

class RnnTemplate(nn.Module):
    def __init__(self, rnn_type, batch_size, input_dim, hidden_dim, rnn_drop,
                 numLayers=1, bidirectional=True, initalizer_type="normal"):
        super(RnnTemplate, self).__init__()
        self.type=rnn_type
        self.batch_size=batch_size
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.rnndropout = nn.Dropout(rnn_drop)

        hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        if self.type=="LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=numLayers, bidirectional=bidirectional)
            if initalizer_type=="normal":
                self.hidden = (torch.normal(mean=torch.zeros(numLayers * 2 if bidirectional else numLayers, hidden_dim)).to("cuda"),
                          torch.normal(mean=torch.zeros(numLayers * 2 if bidirectional else numLayers, hidden_dim)).to("cuda"))
            elif initalizer_type=="xavier":
                # ToDo
                #self.hidden = (nn.init.xavier_normal_(),)
                raise KeyError("initalizer_type has an invaild value: " + initalizer_type)
            else:
                raise KeyError("initalizer_type has an invaild value: " + initalizer_type)
        elif self.type=="GRU":
            #ToDo
            raise KeyError("rnn_type has an invaild value: " + rnn_type)
        else:
            raise KeyError("rnn_type has an invaild value: " + rnn_type)

        self.init_weight()

    def init_weight(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, batchinput, batchlength): # B * S * E
        batchinput = batchinput.permute(1, 0, 2).contiguous() # S * B * E
        mask_input = pack_padded_sequence(batchinput, batchlength, batch_first=False)

        rnn_ouput, hidden = self.rnn(mask_input)#, self.hidden

        rnn_ouput, _ = pad_packed_sequence(rnn_ouput, batch_first=False)
        rnn_ouput = rnn_ouput.permute(1, 0, 2).contiguous() # B * S * E

        rnn_ouput=self.rnndropout(rnn_ouput)

        return rnn_ouput # B * S * E

class LinearTemplate(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super(LinearTemplate, self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)
        if activation=="sigmoid":
            self.activation=F.sigmoid
        elif activation=="softmax":
            self.activation=F.softmax
        elif activation=="tanh":
            self.activation=torch.tanh
        elif activation=="relu":
            self.activation=F.relu
        elif activation==None:
            self.activation=None
        else:
            raise KeyError("activation has an invaild value: " + activation)

        self.init_weight()

    def init_weight(self):
        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, batchinput):
        out=self.linear(batchinput)
        if self.activation is not None:
            if self.activation is F.softmax:
                out=self.activation(out,dim=-1)
            else:
                out=self.activation(out)
        return out
