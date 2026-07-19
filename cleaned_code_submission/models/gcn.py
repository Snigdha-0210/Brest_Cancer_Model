import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class GraphReasoningModule(nn.Module):

    def __init__(self, in_channels, num_nodes=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.phi = nn.Conv2d(in_channels, num_nodes, kernel_size=1)
        self.reduced_dim = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.reduced_dim, kernel_size=1)
        self.gcn1 = GraphConvolution(self.reduced_dim, self.reduced_dim)
        self.gcn2 = GraphConvolution(self.reduced_dim, self.reduced_dim)
        self.reproject = nn.Conv2d(self.reduced_dim, in_channels, kernel_size=1)
        nn.init.constant_(self.reproject.weight, 0)
        nn.init.constant_(self.reproject.bias, 0)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_matrix = self.phi(x).view(B, self.num_nodes, -1)
        proj_matrix = F.softmax(proj_matrix, dim=-1)
        x_reduced = self.theta(x).view(B, self.reduced_dim, -1)
        nodes = torch.bmm(proj_matrix, x_reduced.transpose(1, 2))
        adj = torch.bmm(nodes, nodes.transpose(1, 2))
        adj = F.softmax(adj, dim=-1)
        nodes_gcn = F.relu(self.gcn1(nodes, adj))
        nodes_gcn = F.relu(self.gcn2(nodes_gcn, adj))
        out = torch.bmm(nodes_gcn.transpose(1, 2), proj_matrix)
        out = out.view(B, self.reduced_dim, H, W)
        out = self.reproject(out)
        return x + out
