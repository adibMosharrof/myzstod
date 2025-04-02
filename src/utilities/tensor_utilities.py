import torch


def pad_to_len(tensor, max_len, pad_value=None):
    pad_value = pad_value
    pad_size = max_len - tensor.shape[1]
    if pad_size <= 0:
        return tensor  # No padding needed

    pad_shape = list(tensor.shape)
    pad_shape[1] = pad_size

    pad_tensor = torch.full(
        pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype
    )  # Uses same dtype & device
    return torch.cat([tensor, pad_tensor], dim=1)


def pad_tensor_batch(tensors, pad_value=None, max_len=None):
    max_len = max_len or max([a.shape[1] for a in tensors])
    padded_tensor = [pad_to_len(a, max_len, pad_value) for i, a in enumerate(tensors)]
    assert all([a.shape[1] == max_len for a in padded_tensor])
    return padded_tensor


def create_tensor(value, dtype=torch.int, device="cuda"):
    return torch.tensor(value, device=device, dtype=dtype)
