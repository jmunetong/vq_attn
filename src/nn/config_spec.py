import torch

class VQSpec:
    n_device: torch.Tensor
    n_block_per_update: torch.Tensor
    loss_mask: torch.Tensor

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in dataclasses.fields(VQSpec)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        return cls(**filtered)