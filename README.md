# Gated Sparse Attention — community reference implementation

An independent, production-oriented implementation of the method described in

> Gated Sparse Attention for Efficient Long-Context Transformers
> Yilin Zhao, Marcus Feld, Priya R. Chandra, Tomasz Bielik
> Meridian Institute of Applied Computing, 2027

This implementation is written and maintained outside the original research group. It focuses on a readable PyTorch port of the gating network and the block-routing kernel, together with reference training scripts.

Maintainers: Hanna Kovacs, Andre Silva — Garnet Works

## Contents

- `src/gsa.py` — GSA attention layer
- `src/gate.py` — per-head gating network
- `requirements.txt`
