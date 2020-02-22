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
        self.bias = torch.zeros(self.input_dim // self.n_head, dtype=torch.float32, requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.use_layernorm = layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.input_dim)
        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.bias, 0.01)
        # for name, param in self.bias.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.01)
        #     elif 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def head_attention(self, input):
        # Aggregater
        out = self.dropout(input) # B * S * E

        out = out.permute(0, 2, 1).contiguous() # B * E * S
        print(out.shape)
        out = self.weight_a(out) # conv = [E, E//n_head, 1] => B * (E//n_head) * S
        temp1 = self.weight_b(out) # conv = [E//n_head, 1, 1] => B * 1 * S
        temp2 = self.weight_c(out) # conv = [E//n_head, 1, 1] => B * 1 * S
        temp1 = temp1.permute(0, 2, 1).contiguous() + temp2 # B * S * 1 + B * 1 * S => B * S * S
        coefs = nn.functional.softmax(nn.functional.leaky_relu(temp1, negative_slope=0.2))  # paper B * S * S
        out = out.permute(0, 2, 1).contiguous()  # B * S * (E//n_head)
        print(coefs.shape)
        print(out.shape)

        coefs = self.dropout(coefs)
        out = self.dropout(out)

        # Updater
        re = coefs * out # B * S * (E//n_head)
        re = re + self.bias # B * S * (E//n_head)

        if self.residual:
            re = re + out
        return re

    def forward(self, batchinput):
        out = batchinput
        for step in range(self.n_steps):
            head = []
            for _ in range(self.n_head):
                head.append(self.head_attention(out))
            head = torch.stack(head)
            out = torch.mean(head, 0)

        if self.use_layernorm:
            out = self.layernorm(out)
        return out