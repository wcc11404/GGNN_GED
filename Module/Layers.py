import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os
import numpy as np

class EmbeddingTemplate(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_drop=0.0):
        super(EmbeddingTemplate, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.wordembedding = nn.Embedding(vocab_size, embed_dim)
        self.wordembeddingdropout = nn.Dropout(embed_drop)

        self.init_weight()

    def init_weight(self):
        for name, param in self.wordembedding.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

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
        with open(w2v_dir,'r') as f:
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

        if loginfor:
            print("load {} word embeddings".format(num))

    def set_pad_zero(self):
        temp = self.wordembedding.weight.detach().numpy()
        temp[0] = np.zeros(shape=[1, self.embed_dim], dtype=float)
        self.wordembedding.weight.data.copy_(torch.from_numpy(temp))

class RnnTemplate(nn.Module):
    def __init__(self, rnn_type, batch_size, input_dim, hidden_dim, rnn_drop=0.0,
                 numLayers=1, bidirectional=True, initalizer_type="normal"):
        super(RnnTemplate, self).__init__()
        self.type = rnn_type
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnndropout = nn.Dropout(rnn_drop)

        hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        if self.type=="LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=numLayers, bidirectional=bidirectional,batch_first=False)
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

    def forward(self, batchinput, batchlength, ischar=False): # B * S * E
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
        mask_input = pack_padded_sequence(batchinput, batchlength, batch_first=False)

        rnn_ouput, hidden = self.rnn(mask_input) #, self.hidden

        rnn_ouput, _ = pad_packed_sequence(rnn_ouput, batch_first=False)
        rnn_ouput = rnn_ouput.permute(1, 0, 2).contiguous() # B * S * E
        hidden = hidden[0].permute(1, 0, 2).contiguous()

        # 还原之前的排序
        rnn_ouput = rnn_ouput[recoverItemIdx]

        rnn_ouput = self.rnndropout(rnn_ouput)

        if ischar:
            rnn_ouput = rnn_ouput.view(-1, sl, wl, self.input_dim) # B * S * W * E
            hidden = hidden.view(-1, sl, 2, self.input_dim//2)  # B * S * 2 * E//2
            batchlength = batchlength.view(-1, sl)

        # 注意: rnn_output为双向lstm，S个时间步骤输出的拼接
        #    c<-h<-a<-r<-
        #  ->c->h->a->r
        # S: 0  1  2  3
        # hidden为前向lstm的3和后向lstm的0的输出拼接
        return rnn_ouput, hidden # B * S * E , B * 2 * E//2

class LinearTemplate(nn.Module):
    def __init__(self, input_dim, output_dim, bn=False, activation=None, dropout=0.0):
        super(LinearTemplate, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
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
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, batchinput):
        out = self.linear(batchinput)
        if self.activation is not None:
            if self.activation is F.softmax:
                out = self.activation(out, dim=-1)
            else:
                out = self.activation(out)
        out = self.dropout(out)
        return out

class GraphGateTemplate(nn.Module):
    def __init__(self, input_dim, n_edge_types, n_steps, dropout=0.0):
        super(GraphGateTemplate, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.n_edge_types = n_edge_types

        # self.edge_in = EmbeddingTemplate(self.n_edge_tpyes, self.input_dim)# * self.input_dim)
        # self.edge_out = EmbeddingTemplate(self.n_edge_tpyes, self.input_dim)# * self.input_dim)
        # self.temp = LinearTemplate(self.input_dim, 1)

        # self.edge_in = nn.ModuleList(
        #     [LinearTemplate(self.input_dim, self.input_dim) for _ in range(self.n_edge_types)])
        # self.edge_out = nn.ModuleList(
        #     [LinearTemplate(self.input_dim, self.input_dim) for _ in range(self.n_edge_types)])

        self.edge_in = LinearTemplate(self.n_edge_types * self.input_dim, self.n_edge_types * self.input_dim)
        self.edge_out = LinearTemplate(self.n_edge_types * self.input_dim, self.n_edge_types * self.input_dim)

        # GRUGate
        self.reset_gate = LinearTemplate(self.input_dim * 3, self.input_dim, activation="sigmoid")
        self.update_gate = LinearTemplate(self.input_dim * 3, self.input_dim, activation="sigmoid")
        self.transform = LinearTemplate(self.input_dim * 3, self.input_dim, activation="tanh")

        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        # self.edge_in.set_pad_zero()
        # self.edge_out.set_pad_zero()
        pass

    def GRUUpdater(self, nodein, nodeout, node):
        temp = torch.cat((nodein, nodeout, node), 2)  # B * S * 3E
        r = self.reset_gate(temp)
        z = self.update_gate(temp)
        joined_input = torch.cat((nodein, nodeout, r * node), 2)
        h_hat = self.transform(joined_input)
        output = (1 - z) * node + z * h_hat
        return output  # B * S * E

    def bk_forward(self, batchinput, batchgraphin, batchgraphout):
        sl = batchinput.shape[1]
        out = batchinput

        for step in range(self.n_steps):
            # Aggregater
            graph_in = self.edge_in(batchgraphin) # B * S * S * E^2
            graph_in = graph_in.view(-1, sl, self.input_dim, self.input_dim)  # BS * S * E * E
            graph_in = graph_in.view(-1, sl * self.input_dim, self.input_dim) # BS * SE * E
            graph_in = graph_in.permute(0, 2, 1).contiguous() # BS * E * SE

            graph_out = self.edge_out(batchgraphout)  # B * S * S * E^2
            graph_out = graph_out.view(-1, sl, self.input_dim, self.input_dim)  # BS * S * E * E
            graph_out = graph_out.view(-1, sl * self.input_dim, self.input_dim)  # BS * SE * E
            graph_out = graph_out.permute(0, 2, 1).contiguous()  # BS * E * SE

            temp_input = out.unsqueeze(1) # B * 1 * S * E
            temp_input = temp_input.repeat([1, sl, 1, 1])  # B * S * S * E
            temp_input = temp_input.view(-1, sl * self.input_dim) # BS * SE
            temp_input = temp_input.unsqueeze(2) # BS * SE * 1

            in_out = torch.bmm(graph_in, temp_input) # BS * E * 1
            in_out = in_out.view(-1, sl, self.input_dim) # B * S * E
            out_out = torch.bmm(graph_out, temp_input) # BS * E * 1
            out_out = out_out.view(-1, sl, self.input_dim) # B * S * E

            out = self.GRUUpdater(in_out, out_out, out)

        out = self.dropout(out)
        return out

    def bk2_forward(self, batchinput, batchgraphin, batchgraphout):
        sl = batchinput.shape[1]
        out = batchinput
        batchgraphin = batchgraphin.view(-1, sl, sl * self.n_edge_types)
        batchgraphout = batchgraphout.view(-1, sl, sl * self.n_edge_types)

        for step in range(self.n_steps):
            # Aggregater
            graph_in = []
            graph_out = []
            for i in range(self.n_edge_types):
                graph_in.append(self.edge_in[i](out))  # EN * B * S * E
                graph_out.append(self.edge_out[i](out))  # EN * B * S * E
            graph_in = torch.stack(graph_in).transpose(0, 1).contiguous() # B * EN * S * E
            graph_in = graph_in.view(-1, sl * self.n_edge_types, self.input_dim) # B * EN S * E
            graph_in = torch.bmm(batchgraphin, graph_in) # B * S * E
            graph_out = torch.stack(graph_out).transpose(0, 1).contiguous() # B * EN * S * E
            graph_out = graph_out.view(-1, sl * self.n_edge_types, self.input_dim) # B * EN S * E
            graph_out = torch.bmm(batchgraphout, graph_out) # B * S * E

            out = self.GRUUpdater(graph_in, graph_out, out)

        out = self.dropout(out)
        return out

    def forward(self, batchinput, batchgraphin, batchgraphout):
        sl = batchinput.shape[1]
        out = batchinput
        batchgraphin = batchgraphin.view(-1, sl, sl * self.n_edge_types)
        batchgraphout = batchgraphout.view(-1, sl, sl * self.n_edge_types)

        for step in range(self.n_steps):
            # Aggregater
            temp_out = out.unsqueeze(2)  # B * S * 1 * E
            temp_out = temp_out.expand(-1, -1, self.n_edge_types, -1)  # B * S * EN * E
            temp_out = temp_out.view(-1, sl, self.n_edge_types * self.input_dim) # B * S * EN E

            graph_in = self.edge_in(temp_out)  # B * S * EN E
            graph_in = graph_in.view(-1, sl, self.n_edge_types, self.input_dim)  # B * S * EN * E
            graph_in = graph_in.view(-1, sl * self.n_edge_types, self.input_dim)  # B * S EN * E
            graph_in = torch.bmm(batchgraphin, graph_in) # B * S * E

            graph_out = self.edge_out(temp_out)  # B * S * EN E
            graph_out = graph_out.view(-1, sl, self.n_edge_types, self.input_dim)  # B * S * EN * E
            graph_out = graph_out.view(-1, sl * self.n_edge_types, self.input_dim)  # B * S EN * E
            graph_out = torch.bmm(batchgraphout, graph_out) # B * S * E

            out = self.GRUUpdater(graph_in, graph_out, out) # B * S * E

        out = self.dropout(out)
        return out
