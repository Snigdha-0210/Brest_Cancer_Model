import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeSelfConvolution(nn.Module):
    """
    Node-Self Convolution (NSC) Layer.
    Computes: H^{(l+1)} = \sigma(A H^{(l)} W + H^{(l)} W_{self})
    This prevents over-smoothing and allows for deeper GCN networks
    by explicitly maintaining node identity.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Standard GCN weight for neighbors
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Node-self convolution weight for feature fusion
        self.weight_self = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_self)
        
    def forward(self, x, adj):
        # x: (Batch, Nodes, Features)
        # adj: (Batch, Nodes, Nodes)
        
        # 1. Standard neighbor message passing
        support = torch.matmul(x, self.weight)
        out_neighbors = torch.matmul(adj, support)
        
        # 2. Node-self convolution (feature fusion)
        out_self = torch.matmul(x, self.weight_self)
        
        # Combine
        return out_neighbors + out_self

class NSCGCNModule(nn.Module):
    """
    Node-Self Convolution Graph Convolutional Network (NSCGCN) Module.
    Projects spatial feature maps to graph nodes, applies NSC layers to capture 
    global context without over-smoothing, and projects back to spatial feature maps.
    """
    def __init__(self, in_channels, num_nodes=64):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Compute projection matrix (B x N x H*W)
        self.phi = nn.Conv2d(in_channels, num_nodes, kernel_size=1)
        
        # Reduce channel dimension for GCN operations
        self.reduced_dim = in_channels // 2
        self.theta = nn.Conv2d(in_channels, self.reduced_dim, kernel_size=1)
        
        # Node-Self Convolution Layers
        self.nsc1 = NodeSelfConvolution(self.reduced_dim, self.reduced_dim)
        self.nsc2 = NodeSelfConvolution(self.reduced_dim, self.reduced_dim)
        
        # Reproject to original channel dimension
        self.reproject = nn.Conv2d(self.reduced_dim, in_channels, kernel_size=1)
        
        # Initialization
        nn.init.constant_(self.reproject.weight, 0)
        nn.init.constant_(self.reproject.bias, 0)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # --- 1. Projection (Spatial -> Graph) ---
        # phi(x): (B, N, H*W)
        proj_matrix = self.phi(x).view(B, self.num_nodes, -1)
        # Normalize projection matrix
        proj_matrix = F.softmax(proj_matrix, dim=-1)
        
        # theta(x): (B, C_reduced, H*W)
        x_reduced = self.theta(x).view(B, self.reduced_dim, -1)
        
        # Node features: (B, N, C_reduced)
        nodes = torch.bmm(proj_matrix, x_reduced.transpose(1, 2))
        
        # --- 2. Graph Reasoning with NSCGCN ---
        # Learn dynamic adjacency matrix based on node similarity
        adj = torch.bmm(nodes, nodes.transpose(1, 2))
        adj = F.softmax(adj, dim=-1)
        
        # Apply Node-Self Convolution
        nodes_gcn = F.relu(self.nsc1(nodes, adj))
        nodes_gcn = F.relu(self.nsc2(nodes_gcn, adj))
        
        # --- 3. Reprojection (Graph -> Spatial) ---
        # Reverse projection: (B, C_reduced, H*W)
        out = torch.bmm(nodes_gcn.transpose(1, 2), proj_matrix)
        out = out.view(B, self.reduced_dim, H, W)
        
        # Map back to original channels
        out = self.reproject(out)
        
        # Residual connection
        return x + out
