"""Detección de device: MPS (Apple Silicon) con fallback a CPU. Nunca CUDA.

Incluye un cerrojo GLOBAL de proceso (`device_guard`) para serializar TODO el cómputo en el
device. Es imprescindible en MPS: el backend Metal de PyTorch **no es thread-safe**, y esta app
entrena en un hilo aparte (patrón hilo+cola para SSE) mientras uvicorn sigue atendiendo
peticiones de inferencia en otros hilos. Si el hilo de entrenamiento y un endpoint tocan Metal a
la vez, se corrompe el estado del pipeline de shaders y el proceso muere por SIGSEGV (se observó:
crash en `mps::MetalShaderLibrary::getLibraryPipelineState` durante un entrenamiento de GAN, que
tumbaba todo el backend → 502 en todas las pestañas). El cerrojo es reentrante (RLock): el mismo
hilo puede anidar guardas; lo que se impide es el solape ENTRE hilos.
"""

import threading
from contextlib import contextmanager
from typing import Iterator

import torch

# Cerrojo de proceso que serializa el acceso al device (ver docstring del módulo).
_device_lock = threading.RLock()


@contextmanager
def device_guard() -> Iterator[None]:
    """Serializa un bloque de cómputo en el device frente a otros hilos.

    Úsalo alrededor de CADA paso de entrenamiento y de CADA operación de inferencia que toque
    el modelo. En entrenamiento se adquiere/suelta por paso (o por epoch en la vista previa),
    de modo que las peticiones de inferencia pueden intercalarse entre pasos sin solaparse.
    """
    with _device_lock:
        yield


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
