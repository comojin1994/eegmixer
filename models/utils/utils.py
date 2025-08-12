def torch2np(x_torch):
    """
    Converts a PyTorch tensor to a NumPy array.

    Parameters:
    x_torch (torch.Tensor): The input PyTorch tensor.

    Returns:
    numpy.ndarray: The converted NumPy array.
    """
    x_np = x_torch.detach().cpu().numpy()
    return x_np
