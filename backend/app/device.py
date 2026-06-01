"""Detección de device: MPS (Apple Silicon) con fallback a CPU. Nunca CUDA."""

import torch


def get_device() -> torch.device:
    """Devuelve el mejor device disponible: MPS si lo hay, si no CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Nombre legible para la UI."""
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"
