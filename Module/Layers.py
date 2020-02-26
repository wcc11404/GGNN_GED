import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os
import numpy as np

class EmbeddingTemplate(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_drop=0.0, requires_grad=True):
        super(EmbeddingTemplate, self).__init__()
        self.requires_grad = requires_grad
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.wordembedding = nn.Embedding(vocab_size, embed_dim)
        self.wordembeddingdropout = nn.Dropout(embed_drop)

        self.init_weight()

    def init_weight(self):
        for name, param in self.wordembedding.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            param.requires_grad = self.requires_grad

    def forward(self, batchinput): # B * S
        embed_out = self.wordembedding(batchinput)
        embed_out = self.wordembeddingdropout(embed_out)
        return embed_out # B * S * E

    def load_from_w2v(self, word2id, padandunk=True, w2v_dir=None, lower=True, loginfor=True):  # 加载w2v
        if w2v_dir is None or not os.path.exists(w2v_dir):
            raise KeyError("w2v file is not exists")
        temp = self.wordembedding.weight.detach().numpy()
        temp[0] = np.zeros(shape=[1, self.embed_dim], dtype=float)

        s = set()
        num = 0
        with open(w2v_dir, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) <= 2:
                    continue
                w = line[0]
                if lower:
                    w = w.lower()
                if w in word2id and w not in s:
                    temp[word2id[w]] = np.array(line[1:])
                    s.add(w)
                    num += 1

        self.wordembedding.weight.data.copy_(torch.from_numpy(temp))

        return num

    def set_pad_zero(self):
        temp = self.wordembedding.weight.detach().numpy()
        temp[0] = np.zeros(shape=[1, self.embed_dim], dtype=float)
        self.wordembedding.weight.data.copy_(torch.from_numpy(temp))

class RnnTemplate(nn.Module):
    def __init__(self, rnn_type, batch_size, input_dim, hidden_dim, rnn_drop=0.0,
                 numLayers=1, bidirectional=True, initalizer_type="normal", residual=False,
                 layernorm=False, requires_grad=True):
        super(RnnTemplate, self).__init__()
        self.requires_grad = requires_grad
        self.type = rnn_type
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnndropout = nn.Dropout(rnn_drop)
        self.residual = residual
        self.use_layernorm = layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.hidden_dim)

        hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        if self.type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=numLayers, bidirectional=bidirectional,batch_first=False)
            # if initalizer_type=="normal":
            #     self.hidden = (torch.normal(mean=torch.zeros(numLayers * 2 if bidirectional else numLayers, hidden_dim)).to("cuda"),
            #               torch.normal(mean=torch.zeros(numLayers * 2 if bidirectional else numLayers, hidden_dim)).to("cuda"))
            # elif initalizer_type=="xavier":
            #     # ToDo
            #     #self.hidden = (nn.init.xavier_normal_(),)
            #     raise KeyError("initalizer_type has an invaild value: " + initalizer_type)
            # else:
            #     raise KeyError("initalizer_type has an invaild value: " + initalizer_type)
        elif self.type == "GRU":
            #ToDo
            raise KeyError("rnn_type has an invaild value: " + rnn_type)
        else:
            raise KeyError("rnn_type has an invaild value: " + rnn_type)

        self.init_weight()

    def init_weight(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            param.requires_grad = self.requires_grad

    def forward(self, batchinput, batchlength, ischar=False): # B * S * E
        residual = batchinput
        if ischar:
            assert len(batchinput.shape) == 4
            sl = batchinput.shape[1]
            wl = batchinput.shape[2]
            batchinput = batchinput.view(-1, wl, self.input_dim) # (B*S) * W * E
            batchlength = batchlength.view(-1) # (B*S)

        # rnn的pack_pad需要按照实际长度排序
        batchlength, itemIdx = batchlength.sort(0, descending=True)
        _, recoverItemIdx = itemIdx.sort(0, descending=False)
        batchinput = batchinput[itemIdx]

        batchinput = batchinput.permute(1, 0, 2).contiguous() # S * B * E
        # 多卡并行时，下方pad_packed函数需要在每个gpu中调用，调用时会分别计算最大长度，故而直接指定
        total_length = batchinput.size(0)
        mask_input = pack_padded_sequence(batchinput, batchlength, batch_first=False)

        rnn_ouput, hidden = self.rnn(mask_input)

        rnn_ouput, _ = pad_packed_sequence(rnn_ouput, batch_first=False, total_length=total_length)
        rnn_ouput = rnn_ouput.permute(1, 0, 2).contiguous() # B * S * E
        hidden = hidden[0].permute(1, 0, 2).contiguous()

        # 还原之前的排序
        rnn_ouput = rnn_ouput[recoverItemIdx]

        if ischar:
            rnn_ouput = rnn_ouput.view(-1, sl, wl, self.input_dim) # B * S * W * E
            hidden = hidden.view(-1, sl, 2, self.input_dim//2)  # B * S * 2 * E//2
            batchlength = batchlength.view(-1, sl)

        if self.residual:
            rnn_ouput = rnn_ouput + residual
        rnn_ouput = self.rnndropout(rnn_ouput)
        if self.use_layernorm:
            rnn_ouput = self.layernorm(rnn_ouput)

        # 注意: rnn_output为双向lstm，S个时间步骤输出的拼接
        #    c<-h<-a<-r<-
        #  ->c->h->a->r
        # S: 0  1  2  3
        # hidden为前向lstm的3和后向lstm的0的输出拼接
        return rnn_ouput, hidden # B * S * E , B * 2 * E//2

class LinearTemplate(nn.Module):
    def __init__(self, input_dim, output_dim, bn=False, activation=None, dropout=0.0,
                 residual=False, layernorm=False, requires_grad=True, use_bias=True):
        super(LinearTemplate, self).__init__()
        self.requires_grad = requires_grad
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "softmax":
            self.activation = F.softmax
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "None":
            self.activation = None
        elif activation is None:
            self.activation = None
        else:
            raise KeyError("activation has an invaild value: " + activation)
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            param.requires_grad = self.requires_grad

    def forward(self, batchinput):
        out = self.linear(batchinput)
        if self.activation is not None:
            if self.activation is F.softmax:
                out = self.activation(out, dim=-1)
            else:
                out = self.activation(out)
        out = self.dropout(out)
        return out

class AttentionTemplate(nn.Module):
    def __init__(self, input_dim):
        super(AttentionTemplate, self).__init__()
        self.evidence = LinearTemplate(input_dim * 2, input_dim, activation="tanh")
        self.weight = LinearTemplate(input_dim, input_dim, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        pass

    def forward(self, batchinput1, batchinput2):
        out = torch.cat((batchinput1, batchinput2), 2)
        out = self.evidence(out)
        weight = self.weight(out)
        out = batchinput1 * weight + batchinput2 * (1-weight)
        return out