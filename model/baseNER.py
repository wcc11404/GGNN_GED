from torch import nn
from .Modules import EmbeddingTemplate, RnnTemplate, LinearTemplate

class baseNER(nn.Module):
    def __init__(self,args):
        super(baseNER, self).__init__()
        self.wordembedding = EmbeddingTemplate(args.vocabulary_size, args.embed_dim, args.embed_drop)
        self.rnn = RnnTemplate(args.rnn_type, args.batch_size, args.embed_dim, args.embed_dim, args.rnn_drop)
        self.hiddenlinear = LinearTemplate(args.embed_dim, args.hidden_dim, activation="tanh")
        self.linear = LinearTemplate(args.hidden_dim, 2, activation=None)
        #self.logsoftmax=nn.LogSoftmax(dim=2)

        self.load_embedding(args)

    def load_embedding(self,args):
        if args.w2v_dir is not None:
            self.wordembedding.load_from_w2v(args.word2id, True, args.w2v_dir, bool(args.use_lower), bool(args.loginfor))
            del args.word2id

    def forward(self, batchinput, batchlength):
        out = self.wordembedding(batchinput)
        out = self.rnn(out, batchlength)
        out = self.hiddenlinear(out)
        out = self.linear(out)

        return out

