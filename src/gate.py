import torch


def block_gate(query, block_keys, temperature: float = 1.0):
    scores = query @ block_keys.transpose(-1, -2) / temperature
    return torch.softmax(scores, dim=-1)
