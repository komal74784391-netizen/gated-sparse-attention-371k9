import torch

class GatedSparseAttention(torch.nn.Module):
    """Independent PyTorch implementation of Gated Sparse Attention."""

    def __init__(self, dim: int, heads: int, block_size: int = 64, top_k: int = 4):
        super().__init__()
        self.heads = heads
        self.block_size = block_size
        self.top_k = top_k
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.gate = torch.nn.Linear(dim, heads)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        gate_scores = torch.sigmoid(self.gate(x))
        return q, k, v, gate_scores
