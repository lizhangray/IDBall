

import torch
import torch.nn as nn
import torch.nn.functional as F

def auto_pad_to_multiple(image, mod, padding_mode='reflect'):
    """
    Automatically pads the input image so that both its height and width are multiples of the specified number (mod).

    Args:
        image (torch.Tensor): Input image tensor, expected shape (C, H, W) or (B, C, H, W).
        mod (int): The specified multiple. The height and width of the image will be padded to multiples of mod.
        padding_mode (str, optional): Padding mode, defaults to 'reflect'.
                                      Other options include 'constant', 'replicate', 'circular', etc.
                                      Refer to the documentation of torch.nn.functional.pad for details.

    Returns:
        torch.Tensor: Padded image tensor.

    Raises:
        ValueError: If the input image is not a torch.Tensor or mod is not a positive integer.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input image must be a torch.Tensor type.")
    if not isinstance(mod, int) or mod <= 0:
        raise ValueError("mod must be a positive integer.")

    ndim = image.ndim
    if ndim not in [3, 4]:
        raise ValueError("The dimension of the input image tensor should be 3 (C, H, W) or 4 (B, C, H, W).")

    # Get the height and width dimension indices of the image
    height_dim_index = -2
    width_dim_index = -1

    # Get the height and width of the image
    height = image.shape[height_dim_index]
    width = image.shape[width_dim_index]

    # Calculate the padding size for the height direction
    if height % mod != 0:
        pad_height = (mod - height % mod)  # Total height to be padded
        pad_top = pad_height // 2          # Padding size at the top (distributed as evenly as possible)
        pad_bottom = pad_height - pad_top   # Padding size at the bottom

        # Apply padding in the height direction using the specified padding mode
        pad_layer_height = nn.ReflectionPad2d((0, 0, pad_top, pad_bottom)) if padding_mode == 'reflect' else \
                           nn.ConstantPad2d((0, 0, pad_top, pad_bottom), 0) if padding_mode == 'constant' else \
                           nn.ReplicationPad2d((0, 0, pad_top, pad_bottom)) if padding_mode == 'replicate' else \
                           nn.CircularPad2d((0, 0, pad_top, pad_bottom)) if padding_mode == 'circular' else \
                           nn.ReflectionPad2d((0, 0, pad_top, pad_bottom)) # Default to ReflectionPad2d

        image = pad_layer_height(image)
        height = image.shape[height_dim_index] # Update height because the image has been padded

    # Calculate the padding size for the width direction
    if width % mod != 0:
        pad_width = (mod - width % mod)     # Total width to be padded
        pad_left = pad_width // 2           # Padding size on the left (distributed as evenly as possible)
        pad_right = pad_width - pad_left    # Padding size on the right

        # Apply padding in the width direction using the specified padding mode
        pad_layer_width = nn.ReflectionPad2d((pad_left, pad_right, 0, 0)) if padding_mode == 'reflect' else \
                          nn.ConstantPad2d((pad_left, pad_right, 0, 0), 0) if padding_mode == 'constant' else \
                          nn.ReplicationPad2d((pad_left, pad_right, 0, 0)) if padding_mode == 'replicate' else \
                          nn.CircularPad2d((pad_left, pad_right, 0, 0)) if padding_mode == 'circular' else \
                          nn.ReflectionPad2d((pad_left, pad_right, 0, 0)) # Default to ReflectionPad2d

        image = pad_layer_width(image)


    return image


def pad2affine(image,mod):
    if image.shape[-2] % mod!=0:
        padnum=int(((int(image.shape[-2]/mod)+1)*mod-image.shape[-2])*0.5)
        pad=torch.nn.ReflectionPad2d((0,0,padnum,padnum))
        image=pad(image)
    if image.shape[-1] % mod!=0:
        padnum=int(((int(image.shape[-1]/mod)+1)*mod-image.shape[-1])*0.5)
        pad=torch.nn.ReflectionPad2d((padnum,padnum,0,0))
        image=pad(image)

    return image