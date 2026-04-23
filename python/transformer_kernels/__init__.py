try:
    from transformer_kernels.transformer_kernels_ext import (
        softmax, rmsnorm, rope, cpu_features,
    )
except ImportError as e:
    raise ImportError(
        "C++ extension not found – run: pip install -e ."
    ) from e

from transformer_kernels.inference import TransformerInference

__all__ = ["softmax", "rmsnorm", "rope", "cpu_features", "TransformerInference"]
__version__ = "0.1.0"
