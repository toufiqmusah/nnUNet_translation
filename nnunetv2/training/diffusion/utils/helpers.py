import torch


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
    """
    Extract values from a 1-D tensor for a batch of indices.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    """Normalize image from [0, 1] to [-1, 1]"""
    return img * 2 - 1


def unnormalize_to_zero_to_one(img: torch.Tensor) -> torch.Tensor:
    """Unnormalize image from [-1, 1] to [0, 1]"""
    return (img + 1) * 0.5
