import torch
from torch import nn
from Module.Layers import *

class GraphAttentionTemplate(nn.Module):
    def __init__(self, input_dim, n_head, n_steps, dropout=0.0, residual=False,
                 layernorm=False, requires_grad=True):
        super(GraphAttentionTemplate, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.n_head = n_head
        assert self.input_dim % self.n_head == 0

        self.weight_a = nn.Conv1d(self.input_dim, self.input_dim // self.n_head, 1, bias=False)
        self.weight_b = nn.Conv1d(self.input_dim // self.n_head, 1, 1)
        self.weight_c = nn.Conv1d(self.input_dim // self.n_head, 1, 1)
        self.weight_a = LinearTemplate(self.input_dim, self.input_dim, use_bias=False)
        self.weight_b = LinearTemplate(self.input_dim, self.input_dim)
        self.weight_c = LinearTemplate(self.input_dim, self.input_dim)
        self.bias = nn.Parameter(torch.FloatTensor(self.input_dim))

        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.use_layernorm = layernorm
        if self.use_layernorm:
            self.layernorm = nn.ModuleList([nn.LayerNorm(self.input_dim) for _ in range(n_steps)])
        self.init_weight()

    def init_weight(self):
        for name, param in self.weight_a.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.weight_b.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.weight_c.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.constant_(self.bias, 0)

    def head_attention(self, out, mask):
        # Aggregater

        out = out.permute(0, 2, 1).contiguous() # B * E * S
        out = self.weight_a(out) # conv = [E, E//n_head, 1] => B * (E//n_head) * S
        temp1 = self.weight_b(out) # conv = [E//n_head, 1, 1] => B * 1 * S
        temp2 = self.weight_c(out) # conv = [E//n_head, 1, 1] => B * 1 * S
        temp1 = temp1.permute(0, 2, 1).contiguous() + temp2 # B * S * 1 + B * 1 * S => B * S * S
        # temp1 = torch.bmm(temp1, temp2.permute(0, 2, 1).contiguous()) # B * S * 1 + B * 1 * S => B * S * S
        # coefs = nn.functional.softmax(nn.functional.leaky_relu(temp1, negative_slope=0.2), dim=-1)  # paper B * S * S
        temp1 = temp1 * (self.input_dim ** -0.5)
        temp1 = nn.functional.leaky_relu(temp1, negative_slope=0.2)
        # temp1 = torch.tanh(temp1)
        temp1 = temp1 + mask
        coefs = nn.functional.softmax(temp1, dim=-1)  # B * S * S

        out = out.permute(0, 2, 1).contiguous()  # B * S * (E//n_head)

        coefs = self.dropout(coefs)
        # input = self.dropout(input)

        # Updater
        out = torch.matmul(coefs, out) # B * S * (E//n_head)
        out = out + self.bias # B * S * (E//n_head)

        if self.residual:
            out = out + input

        out = nn.functional.elu(out)
        out = self.dropout(out)
        # out = torch.tanh(out)
        return out

    def genMask(self, len, maxlen, value=float('-inf')):
        a = torch.zeros(len).float()
        b = torch.ones(maxlen - len).float() * value
        c = torch.cat([a, b])
        return c

    def forward(self, batchinput, batchlength):
        out = batchinput
        mask = []
        for l in batchlength:
            mask.append(self.genMask(l, batchinput.size(1)))
        mask = torch.stack(mask, dim=0).unsqueeze(dim=1).cuda()

        for step in range(self.n_steps):
            head = []
            for _ in range(self.n_head):
                head.append(self.head_attention(out, mask))
            head = torch.stack(head, dim=0)
            out = torch.mean(head, 0)

            if self.use_layernorm:
                out = self.layernorm[step](out)
        return out