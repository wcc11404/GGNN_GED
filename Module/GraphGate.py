import torch
from torch import nn
from Module.Layers import *

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
