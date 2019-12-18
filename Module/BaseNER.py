import torch
from torch import nn
from .Layers import EmbeddingTemplate, RnnTemplate, LinearTemplate

class BaseNER(nn.Module):
    def __init__(self, args):
        super(BaseNER, self).__init__()
        # self.args = args
        self.wordembedding = EmbeddingTemplate(args.word_vocabulary_size, args.word_embed_dim, args.embed_drop)
        self.rnn = RnnTemplate(args.rnn_type, args.batch_size, args.word_embed_dim, args.word_embed_dim, args.rnn_drop,
                               bidirectional=args.rnn_bidirectional)

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
        # self.logsoftmax = nn.LogSoftmax(dim=2)
        self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

        self.load_embedding(args)

    def load_embedding(self, args):
        if args.mode == "Train" and args.load_dir is None:
            if args.w2v_dir is not None:
                self.wordembedding.load_from_w2v(args.word2id, True, args.w2v_dir, args.use_lower, args.loginfor)
                del args.word2id

    def forward(self, batchinput, batchlength, batchextradata):
        if self.charembedding is not None:
            batchinput_char, batchlength_char = batchextradata[0], batchextradata[1]

        out = self.wordembedding(batchinput)
        out, _ = self.rnn(out, batchlength)    # B S E

        if self.charembedding is not None:
            charout = self.charembedding(batchinput_char)
            _, charout = self.charrnn(charout, batchlength_char, ischar=True) # B S 2 E//2
            charout = charout.view(charout.shape[0], charout.shape[1], -1)
            out = torch.cat((out, charout), 2)

        out = self.hiddenlinear(out)
        out = self.classification(out)

        return out, ()

    def getLoss(self, input, length, extra_data, output, label):
        #x, xl, xc, xcl = input
        output, _ = output
        return self.Loss(output.view(-1, 2), label.view(-1))


