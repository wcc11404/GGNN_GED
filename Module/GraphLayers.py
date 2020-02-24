from Module.Layers import *

class GraphAttentionStep(nn.Module):
    def __init__(self, input_dim, n_head, dropout=0.0, residual=False, requires_grad=True):
        super(GraphAttentionStep, self).__init__()
        self.input_dim = input_dim
        self.n_head = n_head
        assert self.input_dim % self.n_head == 0

        # self.weight_a = nn.Conv1d(self.input_dim, self.input_dim // self.n_head, 1, bias=False)
        # self.weight_b = nn.Conv1d(self.input_dim // self.n_head, 1, 1)
        # self.weight_c = nn.Conv1d(self.input_dim // self.n_head, 1, 1)
        self.weight_a = LinearTemplate(self.input_dim, self.input_dim, use_bias=False)
        self.weight_b = LinearTemplate(self.input_dim, self.input_dim)
        self.weight_c = LinearTemplate(self.input_dim, self.input_dim)
        self.bias = nn.Parameter(torch.FloatTensor(self.input_dim))

        self.dropout = nn.Dropout(dropout)
        self.residual = residual
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

    def forward(self, input, mask):
        head = []
        for h in range(self.n_head):
            # Aggregater

            # input = input.permute(0, 2, 1).contiguous() # B * E * S
            input = self.weight_a(input) # conv = [E, E//n_head, 1] => B * (E//n_head) * S
            temp1 = self.weight_b(input) # conv = [E//n_head, 1, 1] => B * 1 * S
            temp2 = self.weight_c(input) # conv = [E//n_head, 1, 1] => B * 1 * S
            # temp1 = temp1.permute(0, 2, 1).contiguous() + temp2 # B * S * 1 + B * 1 * S => B * S * S
            temp1 = torch.bmm(temp1, temp2.permute(0, 2, 1).contiguous()) # B * S * 1 + B * 1 * S => B * S * S
            temp1 = temp1 * (self.input_dim ** -0.5)
            temp1 = nn.functional.leaky_relu(temp1, negative_slope=0.2)
            # temp1 = torch.tanh(temp1)
            temp1 = temp1 + mask
            coefs = nn.functional.softmax(temp1, dim=-1)  # B * S * S

            # input = input.permute(0, 2, 1).contiguous()  # B * S * (E//n_head)

            coefs = self.dropout(coefs)

            # Updater
            out = torch.matmul(coefs, input) # B * S * (E//n_head)
            out = out + self.bias # B * S * (E//n_head)

            if self.residual:
                out = out + input

            # out = torch.tanh(out)
            out = nn.functional.elu(out)
            out = self.dropout(out)

            head.append(out)
        return head

class GraphAttentionTemplate(nn.Module):
    def __init__(self, input_dim, n_head, n_steps, dropout=0.0, residual=False,
                 layernorm=False, requires_grad=True):
        super(GraphAttentionTemplate, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        # assert self.input_dim % self.n_head == 0

        self.gansteps = nn.ModuleList(
            [GraphAttentionStep(input_dim, n_head, dropout, residual, requires_grad) for _ in range(n_steps)])

        self.use_layernorm = layernorm
        if self.use_layernorm:
            self.layernorm = nn.ModuleList([nn.LayerNorm(self.input_dim) for _ in range(n_steps)])
        self.init_weight()

    def init_weight(self):
        pass

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
            out = self.gansteps[step](out, mask)
            out = torch.stack(out, dim=0)
            out = torch.mean(out, 0)

            if self.use_layernorm:
                out = self.layernorm[step](out)

        return out

class GraphGateTemplate(nn.Module):
    def __init__(self, input_dim, n_edge_types, n_steps, dropout=0.0, residual=False,
                 layernorm=False, requires_grad=True):
        super(GraphGateTemplate, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.n_edge_types = n_edge_types

        self.edge_in = nn.ModuleList(
            [LinearTemplate(self.input_dim, self.input_dim, requires_grad=requires_grad) for _ in range(self.n_edge_types)])
        self.edge_out = nn.ModuleList(
            [LinearTemplate(self.input_dim, self.input_dim, requires_grad=requires_grad) for _ in range(self.n_edge_types)])

        # self.edge_in = LinearTemplate(self.n_edge_types * self.input_dim, self.n_edge_types * self.input_dim)
        # self.edge_out = LinearTemplate(self.n_edge_types * self.input_dim, self.n_edge_types * self.input_dim)

        # GRUGate
        self.reset_gate = LinearTemplate(self.input_dim * 3, self.input_dim, activation="sigmoid", requires_grad=requires_grad)
        self.update_gate = LinearTemplate(self.input_dim * 3, self.input_dim, activation="sigmoid", requires_grad=requires_grad)
        self.transform = LinearTemplate(self.input_dim * 3, self.input_dim, activation="tanh", requires_grad=requires_grad)

        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.use_layernorm = layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(self.input_dim)
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

    def forward(self, batchinput, batchgraphin, batchgraphout):
        sl = batchinput.shape[1]
        out = batchinput
        residual = batchinput
        batchgraphin = batchgraphin.view(-1, sl, sl * self.n_edge_types)
        batchgraphout = batchgraphout.view(-1, sl, sl * self.n_edge_types)

        for step in range(self.n_steps):
            # Aggregater
            graph_in = []
            graph_out = []
            for i in range(self.n_edge_types):
                graph_in.append(self.edge_in[i](out))  # EN * B * S * E
                graph_out.append(self.edge_out[i](out))  # EN * B * S * E
            graph_in = torch.stack(graph_in).permute(1, 2, 0, 3).contiguous()  # B * S * EN * E
            graph_in = graph_in.view(-1, sl * self.n_edge_types, self.input_dim) # B * S EN * E
            graph_in = torch.bmm(batchgraphin, graph_in) # B * S * E
            graph_out = torch.stack(graph_out).permute(1, 2, 0, 3).contiguous()  # B * S * EN * E
            graph_out = graph_out.view(-1, sl * self.n_edge_types, self.input_dim) # B * S EN * E
            graph_out = torch.bmm(batchgraphout, graph_out) # B * S * E

            out = self.GRUUpdater(graph_in, graph_out, out)

        out = self.dropout(out)

        if self.residual:
            out = out + residual
        if self.use_layernorm:
            out = self.layernorm(out)
        return out

    def bk_forward(self, batchinput, batchgraphin, batchgraphout):
        sl = batchinput.shape[1]
        out = batchinput
        batchgraphin = batchgraphin.view(-1, sl, sl * self.n_edge_types)
        batchgraphout = batchgraphout.view(-1, sl, sl * self.n_edge_types)

        for step in range(self.n_steps):
            # Aggregater
            temp_out = out.unsqueeze(2)  # B * S * 1 * E
            temp_out = temp_out.repeat([1, 1, self.n_edge_types, 1])
            # temp_out = temp_out.expand(-1, -1, self.n_edge_types, -1).contiguous()  # B * S * EN * E
            temp_out = temp_out.view(-1, sl, self.n_edge_types * self.input_dim) # B * S * EN E

            graph_in = self.edge_in(temp_out)  # B * S * EN E
            # graph_in = graph_in.view(-1, sl, self.n_edge_types, self.input_dim)  # B * S * EN * E
            graph_in = graph_in.view(-1, sl * self.n_edge_types, self.input_dim)  # B * S EN * E
            graph_in = torch.bmm(batchgraphin, graph_in) # B * S * E

            graph_out = self.edge_out(temp_out)  # B * S * EN E
            # graph_out = graph_out.view(-1, sl, self.n_edge_types, self.input_dim)  # B * S * EN * E
            graph_out = graph_out.view(-1, sl * self.n_edge_types, self.input_dim)  # B * S EN * E
            graph_out = torch.bmm(batchgraphout, graph_out) # B * S * E

            out = self.GRUUpdater(graph_in, graph_out, out) # B * S * E

        out = self.dropout(out)
        return out
