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
        # self.weight_a = LinearTemplate(self.input_dim, self.input_dim, use_bias=False)
        # self.weight_b = LinearTemplate(self.input_dim, self.input_dim)
        # self.weight_c = LinearTemplate(self.input_dim, self.input_dim)
        self.bias = nn.Parameter(torch.FloatTensor(self.input_dim // self.n_head))

        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.use_layernorm = layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.input_dim)
        self.init_weight()

    def init_weight(self):
        for name, param in self.weight_a.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out')
        for name, param in self.weight_b.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out')
        for name, param in self.weight_c.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out')
        nn.init.constant_(self.bias, 0)

    def head_attention(self, input, mask):
        # Aggregater
        out = self.dropout(input) # B * S * E

        out = out.permute(0, 2, 1).contiguous() # B * E * S
        out = self.weight_a(out) # conv = [E, E//n_head, 1] => B * (E//n_head) * S
        temp1 = self.weight_b(out) # conv = [E//n_head, 1, 1] => B * 1 * S
        temp2 = self.weight_c(out) # conv = [E//n_head, 1, 1] => B * 1 * S
        temp1 = temp1.permute(0, 2, 1).contiguous() + temp2 # B * S * 1 + B * 1 * S => B * S * S
        # temp1 = torch.bmm(temp1, temp2.permute(0, 2, 1).contiguous()) # B * S * 1 + B * 1 * S => B * S * S
        # coefs = nn.functional.softmax(nn.functional.leaky_relu(temp1, negative_slope=0.2), dim=-1)  # paper B * S * S
        temp1 = torch.tanh(temp1)
        temp1 = temp1 + mask
        # print(temp1[0])
        coefs = nn.functional.softmax(temp1, dim=-1)  # paper B * S * S
        # print("coefs")
        # print(coefs[0][0])
        # print(coefs[0][1])
        # print()

        out = out.permute(0, 2, 1).contiguous()  # B * S * (E//n_head)
        # print("out")
        # print(out)
        # print()
        coefs = self.dropout(coefs)
        out = self.dropout(out)

        # Updater
        re = torch.matmul(coefs, out) # B * S * (E//n_head)
        # print("re")
        # print(re)
        # print()
        re = re + self.bias # B * S * (E//n_head)
        # print("re")
        # print(re)
        # print()
        # exit()
        if self.residual:
            re = re + out

        re = torch.tanh(re)
        return re

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
        # print(mask.shape)
        # print(mask)
        for step in range(self.n_steps):
            head = []
            for _ in range(self.n_head):
                head.append(self.head_attention(out, mask))
            head = torch.stack(head, dim=0)
            out = torch.mean(head, 0)
            # print(out)
            # print()
        # exit()

        if self.use_layernorm:
            out = self.layernorm(out)
        return out