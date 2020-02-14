import torch
from torch import nn
from .Layers import EmbeddingTemplate, RnnTemplate, LinearTemplate, GraphGateTemplate, AttentionTemplate
import numpy as np

class GGNNNER(nn.Module):
    def __init__(self, args):
        super(GGNNNER, self).__init__()
        assert args.rnn_bidirectional and args.lm_cost_weight >= 0  # 暂时必须是双向lstm
        self.args = args
        self.lm_vocab_size = args.lm_vocab_size
        self.lm_cost_weight = args.lm_cost_weight
        # , requires_grad=False if not args.train_lm else True
        self.wordembedding = EmbeddingTemplate(args.word_vocabulary_size, args.word_embed_dim, args.embed_drop)
        self.gnn = GraphGateTemplate(args.word_embed_dim, args.edge_vocabulary_size, args.gnn_steps, args.gnn_drop,
                                     residual=False, layernorm=False)
        self.attention = AttentionTemplate(args.word_embed_dim)
        self.rnn = RnnTemplate(args.rnn_type, args.batch_size, args.word_embed_dim, args.word_embed_dim, args.rnn_drop,
                               bidirectional=args.rnn_bidirectional, residual=False, layernorm=False)

        if args.char_embed_dim is not None and args.char_embed_dim > 0:
            self.charembedding = EmbeddingTemplate(args.char_vocabulary_size, args.char_embed_dim, args.embed_drop)
            self.charrnn = RnnTemplate(args.rnn_type, args.batch_size, args.char_embed_dim, args.char_embed_dim,
                                       args.rnn_drop)
            self.hiddenlinear = LinearTemplate(args.word_embed_dim + args.char_embed_dim, args.hidden_dim,
                                               activation="tanh", dropout=args.linear_drop)
        else:
            self.charembedding = None
            self.hiddenlinear = LinearTemplate(args.word_embed_dim, args.hidden_dim, activation="tanh",
                                               dropout=args.linear_drop)

        self.classification = LinearTemplate(args.hidden_dim, 2, activation=None)

        ## LM
        self.fw_lm_hiddenlinear = LinearTemplate((args.word_embed_dim) // 2 + args.char_embed_dim, args.lm_hidden_dim,
                                                 activation="tanh", dropout=args.linear_drop)
        self.bw_lm_hiddenlinear = LinearTemplate((args.word_embed_dim) // 2 + args.char_embed_dim, args.lm_hidden_dim,
                                                 activation="tanh", dropout=args.linear_drop)
        if self.lm_vocab_size == -1 or self.lm_vocab_size > args.word_vocabulary_size:
            self.lm_vocab_size = args.word_vocabulary_size
        self.fw_lm_softmax = LinearTemplate(args.lm_hidden_dim, self.lm_vocab_size, activation=None)
        self.bw_lm_softmax = LinearTemplate(args.lm_hidden_dim, self.lm_vocab_size, activation=None)

        # 损失函数
        self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum",
                                        weight=torch.from_numpy(np.array([1, 1.5])).float())
        self.forwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.bakwardLoss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

        # 加载词表权重
        self.load_embedding(args)

    def load_embedding(self, args):
        if args.mode == "Train" and args.load_dir is None:
            if args.w2v_dir is not None:
                self.wordembedding.load_from_w2v(args.word2id, True, args.w2v_dir, args.use_lower, args.loginfor)
        del args.word2id

    def forward(self, batchinput, batchlength, batchextradata):
        batchinput_char, batchlength_char, graph_in, graph_out = batchextradata

        emb = self.wordembedding(batchinput)
        out = self.gnn(emb, graph_in, graph_out)
        out = self.attention(emb, out)
        out, _ = self.rnn(out, batchlength)  # B S E

        if self.charembedding is not None:
            charout = self.charembedding(batchinput_char)
            _, charout = self.charrnn(charout, batchlength_char, ischar=True) # B S 2 E//2
            charout = charout.view(charout.shape[0], charout.shape[1], -1)
            out = torch.cat((out, charout), 2)
            # out = self.mergelinear(out)
            # lm_fw_input = torch.cat((lm_fw_input, charout), 2)
            # lm_bw_input = torch.cat((lm_bw_input, charout), 2)

        lm_input = out.view(-1, out.shape[1], 2, out.shape[2] // 2).permute(2, 0, 1, 3).contiguous()  # 分成双向的
        lm_fw_input, lm_bw_input = lm_input[0], lm_input[1]

        out = self.hiddenlinear(out)
        out = self.classification(out)

        lm_fw_output = self.fw_lm_hiddenlinear(lm_fw_input)
        lm_bw_output = self.bw_lm_hiddenlinear(lm_bw_input)
        lm_fw_output = self.fw_lm_softmax(lm_fw_output)
        lm_bw_output = self.bw_lm_softmax(lm_bw_output)

        return out, (lm_fw_output, lm_bw_output)

    def getLoss(self, output, label, extra_label):
        if self.args.train_lm:
            return self.getLMLoss(output, extra_label)

        out, (lm_fw_out, lm_bw_out) = output
        loss = self.Loss(out.view(-1, 2), label.view(-1))
        loss += self.lm_cost_weight * self.getLMLoss(output, extra_label)

        return loss

    def getLMLoss(self, output, extra_label):
        loss = 0
        out, (lm_fw_out, lm_bw_out) = output
        forwardlabel, bakwardlabel = extra_label
        # fw_x = input[:, 1:]
        # fw_x = torch.cat((fw_x, torch.zeros(fw_x.shape[0], 1, dtype=torch.long, device=fw_x.device)), dim=-1)
        # bw_x = input[:, :-1]
        # bw_x = torch.cat((torch.zeros(bw_x.shape[0], 1, dtype=torch.long, device=bw_x.device), bw_x), dim=-1)
        loss += self.forwardLoss(lm_fw_out.view(-1, self.lm_vocab_size), forwardlabel.view(-1))
        loss += self.bakwardLoss(lm_bw_out.view(-1, self.lm_vocab_size), bakwardlabel.view(-1))
        return loss
