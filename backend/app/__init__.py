"""Backend del Generative Models Visual Lab (FastAPI + PyTorch, solo MPS/CPU)."""

import os

# CLAUDE.md §0: nunca CUDA. En MPS, las operaciones no soportadas caen a CPU.
# Se fija lo antes posible, antes de que se ejecute cualquier op de torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
