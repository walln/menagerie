"""Common utility functions for sparse interpretability methods."""

import torch

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
