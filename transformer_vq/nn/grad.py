import torch

def stop_gradient(vector: torch.Tensor) -> torch.Tensor:
    """
    Returns a new tensor from the input vector, detached from the current computation graph.
    
    Args:
        vector (torch.Tensor): A tensor for which gradients should be stopped.

    Returns:
        torch.Tensor: A new tensor that does not require gradients.
    """
    return vector.detach()