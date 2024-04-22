import torch
import torch.nn as nn
import torch.nn.functional as F
from torchjpeg import dct


def bdct(x, sub_channels=None, size=8, stride=8, pad=0, dilation=1):
    x = x * 0.5 + 0.5  # x to [0, 1]

    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    x *= 255
    if x.shape[1] == 3:
        x = dct.to_ycbcr(x)
    x -= 128  # x to [-128, 127]
    bs, ch, h, w = x.shape
    block_num = h // stride
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad,
                 stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, size, size)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, size * size).permute(0, 1, 4, 2, 3)
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)

    return dct_block


def ibdct(x, size=8, stride=8, pad=0, dilation=1):
    bs, _, _, _ = x.shape
    sampling_rate = 8

    x = x.view(bs, 3, 64, 14 * sampling_rate, 14 * sampling_rate)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(bs, 3, 14 * 14 * sampling_rate * sampling_rate, 8, 8)
    x = dct.block_idct(x)
    x = x.view(bs * 3, 14 * 14 * sampling_rate * sampling_rate, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * sampling_rate, 112 * sampling_rate),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(bs, 3, 112 * sampling_rate, 112 * sampling_rate)
    x += 128
    x = dct.to_rgb(x)
    x /= 255
    x = F.interpolate(x, scale_factor=1 / sampling_rate, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x
