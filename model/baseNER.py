import torch
from torch import nn
from .Modules import EmbeddingTemplate, RnnTemplate, LinearTemplate

class baseNER(nn.Module):
    def __init__(self,args):
        super(baseNER, self).__init__()
        self.wordembedding = EmbeddingTemplate(args.word_vocabulary_size, args.word_embed_dim, args.embed_drop)
        self.rnn = RnnTemplate(args.rnn_type, args.batch_size, args.word_embed_dim, args.word_embed_dim, args.rnn_drop)

        if args.char_embed_dim is not None and args.char_embed_dim > 0:
            self.charembedding = EmbeddingTemplate(args.char_vocabulary_size, args.char_embed_dim, args.embed_drop)
            self.charrnn = RnnTemplate(args.rnn_type, args.batch_size, args.char_embed_dim, args.char_embed_dim,
                                       args.rnn_drop)
            self.index = torch.LongTensor([0]).cuda() if bool(args.use_gpu) else torch.LongTensor([0])
            self.hiddenlinear = LinearTemplate(args.word_embed_dim + args.char_embed_dim, args.hidden_dim,
                                               activation="tanh")
        else:
            self.charembedding = None
            self.hiddenlinear = LinearTemplate(args.word_embed_dim, args.hidden_dim, activation="tanh")

        self.classification = LinearTemplate(args.hidden_dim, 2, activation=None)
        #self.logsoftmax=nn.LogSoftmax(dim=2)
        self.Loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

        self.load_embedding(args)

    def load_embedding(self,args):
        if args.mode == "Train" and args.load_dir is None:
            if args.w2v_dir is not None:
                self.wordembedding.load_from_w2v(args.word2id, True, args.w2v_dir, bool(args.use_lower),
                                                 bool(args.loginfor))
                del args.word2id

    def forward(self, batchinput, batchlength, batchinput_char, batchlength_char):
        out = self.wordembedding(batchinput)
        out, _ = self.rnn(out, batchlength)    # B S E

        if self.charembedding is not None:
            charout = self.charembedding(batchinput_char)
            _, charout = self.charrnn(charout, batchlength_char, ischar=True) # B S W E
            #charout = charout.index_select(2, self.index) # B S 1 E
            #charout = charout.squeeze(2)    # B S E
            charout = charout.view(charout.shape[0], charout.shape[1], -1)
            out = torch.cat([out, charout], 2)

        out = self.hiddenlinear(out)
        out = self.classification(out)

        return out

    def getLoss(self, input, output, label):
        #x, xl, xc, xcl = input
        return self.Loss(output.view(-1, 2), label.view(-1))


