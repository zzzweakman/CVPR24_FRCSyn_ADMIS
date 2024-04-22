# ======================================================================================================================
# Based on: https://nn.labml.ai/diffusion/ddpm/unet.html
# @misc{labml,
# author = {Varuna Jayasiri, Nipun Wijerathne},
# title = {labml.ai Annotated Paper Implementations},
# year = {2020},
# url = {https://nn.labml.ai/},
# }
# ======================================================================================================================

import math
from functools import partial
from typing import Tuple, Union, List

import torch
import torch.nn as nn

from models.diffusion.util import (
    conv_nd,
    zero_module,
    TimestepEmbedSequential
)
from models.diffusion.nn import SpatialEmbeddingCrossAttentionBlock, SpatialSelfAttentionBlock
from utils.helpers import zero_module


class SinusoidalTimeEmbedding(torch.nn.Module):

    def __init__(self, n_channels: int, max_period: int = 10000):
        super().__init__()
        self.n_channels = n_channels
        self.max_period = max_period

        input_channels = n_channels // 4

        half = input_channels // 2
        self.frequencies = torch.exp(
            - math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )

        self.lin1 = torch.nn.Linear(input_channels, self.n_channels)
        self.act = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        frequencies = self.frequencies.to(t.device)
        args = t[:, None].float() * frequencies[None]

        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)

        return emb


class AdaptiveGroupNormalization(torch.nn.Module):

    def __init__(self, n_groups: int, n_channels: int):
        super().__init__()
        self.norm = torch.nn.GroupNorm(n_groups, n_channels)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        return (1 + scale) * self.norm(x) + shift


class ResidualBlock(torch.nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_channels: int,
                 context_channels: int,
                 n_groups: int = 32,
                 p_dropout: float = 0.0,
                 condition_type: str = 'AddPlusGN',
                 is_context_conditional: bool = False,
                 ):
        super().__init__()

        self.condition_type = condition_type
        self.is_context_conditional = is_context_conditional

        # group normalization and the first convolution layer
        self.norm1 = torch.nn.GroupNorm(n_groups, in_channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if condition_type.lower() in ['adagn', 'diffae', 'ca']:
            self.time_emb = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(time_channels, out_channels)
            )

            if self.is_context_conditional and condition_type.lower() != 'ca':
                self.context_emb = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(context_channels, out_channels)
                )

            self.scale_emb = torch.nn.Linear(out_channels, out_channels)
            self.shift_emb = torch.nn.Linear(out_channels, out_channels)

            self.norm2 = AdaptiveGroupNormalization(n_groups, out_channels)
        else:
            self.time_emb = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(time_channels, out_channels)
            )

            if self.is_context_conditional:
                self.context_emb = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(context_channels, out_channels)
                )

            self.norm2 = torch.nn.GroupNorm(n_groups, out_channels)

        self.act2 = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.conv2 = zero_module(torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))

        # add shortcut
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        h = self.conv1(self.act1(self.norm1(x)))

        # Addition + GroupNorm (Ho et al.)
        if self.condition_type.lower() == 'addplusgn':
            h += self.time_emb(t)[:, :, None, None]

            if self.is_context_conditional:
                h += self.context_emb(c)[:, :, None, None]

            h = self.norm2(h)

        # Adaptive Group Normalization (Dhariwal et al.)
        elif self.condition_type.lower() in ['adagn', 'diffae', 'ca']:

            emb = self.time_emb(t)

            if self.is_context_conditional and self.condition_type.lower() == 'adagn':
                emb += self.context_emb(c)

            scale = self.scale_emb(emb)[:, :, None, None]
            shift = self.shift_emb(emb)[:, :, None, None]

            h = self.norm2(h, scale=scale, shift=shift)

            # DiffAE uses 'AdaGN' for time condition and this scaling for the context (Preechakul et al.)
            if self.is_context_conditional and self.condition_type.lower() == 'diffae':
                h *= self.context_emb(c)[:, :, None, None]

        else:
            raise NotImplementedError

        h = self.conv2(self.dropout(self.act2(h)))

        return h + self.shortcut(x)


class DownBlock(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_channels: int,
                 context_channels: int,
                 condition_type: str,
                 is_context_conditional: bool,
                 has_attention: bool,
                 attention_heads: int,
                 attention_head_channels: int):
        super().__init__()

        self.has_attention = has_attention
        self.res = ResidualBlock(
            in_channels, out_channels,
            time_channels=time_channels, context_channels=context_channels,
            condition_type=condition_type, is_context_conditional=is_context_conditional
        )

        if has_attention:
            if condition_type.lower() == 'ca' and is_context_conditional:
                self.attn = SpatialEmbeddingCrossAttentionBlock(in_channels=out_channels, context_dim=context_channels,
                                                                n_heads=attention_heads, head_channels=attention_head_channels)
            else:
                self.attn = SpatialSelfAttentionBlock(in_channels=out_channels, n_heads=attention_heads, head_channels=attention_head_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        x = self.res(x, t, c)

        if self.has_attention:
            x = self.attn(x, c)


        return x


class UpBlock(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_channels: int,
                 context_channels: int,
                 condition_type: str,
                 is_context_conditional: bool,
                 has_attention: bool,
                 attention_heads: int,
                 attention_head_channels: int):
        super().__init__()

        self.has_attention = has_attention
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels,
            time_channels=time_channels, context_channels=context_channels,
            condition_type=condition_type, is_context_conditional=is_context_conditional
        )

        if has_attention:
            if condition_type.lower() == 'ca' and is_context_conditional:
                self.attn = SpatialEmbeddingCrossAttentionBlock(in_channels=out_channels, context_dim=context_channels,
                                                                n_heads=attention_heads, head_channels=attention_head_channels)
            else:
                self.attn = SpatialSelfAttentionBlock(in_channels=out_channels, n_heads=attention_heads, head_channels=attention_head_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        x = self.res(x, t, c)
        if self.has_attention:
            x = self.attn(x, c)
        return x


class MiddleBlock(torch.nn.Module):

    def __init__(self,
                 n_channels: int,
                 time_channels: int,
                 context_channels: int,
                 condition_type: str,
                 is_context_conditional: bool,
                 attention_heads: int,
                 attention_head_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels, n_channels,
            time_channels=time_channels, context_channels=context_channels,
            condition_type=condition_type, is_context_conditional=is_context_conditional
        )

        if condition_type.lower() == 'ca' and is_context_conditional:
            self.attn = SpatialEmbeddingCrossAttentionBlock(in_channels=n_channels, context_dim=context_channels,
                                                            n_heads=attention_heads, head_channels=attention_head_channels)
        else:
            self.attn = SpatialSelfAttentionBlock(in_channels=n_channels, n_heads=attention_heads, head_channels=attention_head_channels)

        self.res2 = ResidualBlock(
            n_channels, n_channels,
            time_channels=time_channels, context_channels=context_channels,
            condition_type=condition_type, is_context_conditional=is_context_conditional
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x = self.res1(x, t, c)
        x = self.attn(x, c)
        x = self.res2(x, t, c)
        return x


class Upsample(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        _, _ = t, c
        return self.conv(x)


class Downsample(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        _, _ = t, c
        return self.conv(x)

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size: int = 32,
            in_channels: int = 3,
            hint_channels: int = 3,
            model_channels: int = 96,
            channel_mult: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
            dims: int = 2, 
            input_channels: int = 3,
            initial_channels: int = 64,
            is_context_conditional: bool = True,
            channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2),
            is_attention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
            attention_heads: int = 4,
            attention_head_channels: int = 32,
            n_blocks_per_resolution: int = 2,
            condition_type: str = 'AddPlusGN',
            context_input_channels: int = 512,
            context_channels: int = 256,
            n_context_classes: int = 0,
            learn_empty_context: bool = False,
            context_dropout_probability: float = 0.0,
            use_fp16: bool = False,

    ):
        super().__init__()
    


        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.is_context_conditional = is_context_conditional

        self.channel_mult = channel_mult
        self.dtype = torch.float16 if use_fp16 else torch.float32


        # origin:
        # time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )

        # refined:
        time_channels = initial_channels * 4
        self.time_emb = SinusoidalTimeEmbedding(time_channels)

        ### copy from idiff unet structure ###
        if self.is_context_conditional:
            if context_dropout_probability > 0:
                self.context_dropout = torch.nn.Dropout(p=context_dropout_probability)
            else:
                self.context_dropout = torch.nn.Identity()

            if n_context_classes > 0:
                self.context_emb = torch.nn.Embedding(n_context_classes, embedding_dim=context_channels)
            else:
                self.context_emb = torch.nn.Linear(context_input_channels, context_channels)

            if learn_empty_context:
                # create learnable constant embedding for dropped contexts
                self.empty_context_embedding = torch.nn.Parameter(torch.empty(context_channels))
                torch.nn.init.normal_(self.empty_context_embedding)
            else:
                self.empty_context_embedding = torch.zeros(context_channels)
        down_sample_block = partial(DownBlock,
                                    context_channels=context_channels,
                                    time_channels=time_channels,
                                    attention_heads=attention_heads,
                                    attention_head_channels=attention_head_channels,
                                    condition_type=condition_type,
                                    is_context_conditional=is_context_conditional
                                    )

        middle_block = partial(MiddleBlock,
                               context_channels=context_channels,
                               time_channels=time_channels,
                               attention_heads=attention_heads,
                               attention_head_channels=attention_head_channels,
                               condition_type=condition_type,
                               is_context_conditional=is_context_conditional
                               )
        ### end ###



        # semantic hint
        self.input_hint_block = TimestepEmbedSequential(
                    conv_nd(dims, hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
                )
        

        # project image into feature map
        self.image_proj = torch.nn.Conv2d(input_channels, initial_channels, kernel_size=(3, 3), padding=(1, 1))
        # first zero_conv ~ image_proj
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        # from ControlNet
        ch = model_channels
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ### copy from idiff unet structure ###
        n_resolutions = len(channel_multipliers)
        # ======================================= DOWN SAMPLER =========================================================
        down = []
        # number of channels
        out_channels = in_channels = initial_channels
        # for each resolution
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]
            # for zero conv
            ch = channel_multipliers[i] * model_channels

            for _ in range(n_blocks_per_resolution):
                down.append(
                    down_sample_block(in_channels=in_channels, out_channels=out_channels, has_attention=is_attention[i]))
                in_channels = out_channels
                ch = out_channels
                # for zero conv
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
                

            # down sample at all resolutions except the last
            if i < n_resolutions - 1:
                ch = out_channels
                down.append(Downsample(in_channels))
                # for zero conv
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch

        # Combine the set of modules
        self.down = torch.nn.ModuleList(down)

        # ========================================== MIDDLE ============================================================
        self.middle = middle_block(n_channels=out_channels)
        # from ControlNet
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, dropout_mask = None,**kwargs):

        emb = self.time_emb(timesteps)
        # print("")

        # print("time_emb.shape:")
        # print(emb.shape)

        # print("image_proj_before.hape:")
        # print(x.shape)
        x = self.image_proj(x)
        # print("image_proj_after.shape:")
        # print(x.shape)

        ### for ControlNet: use c or context?? wait for ablation ###
        # use context only if the model is context_conditional
        if self.is_context_conditional:
            if context is None:
                c = self.empty_context_embedding.unsqueeze(0).repeat(len(x), 1).to(x.device)
            else:
                c = self.context_emb(context)

                # if entire samples is dropped out, use the empty context embedding instead
                if dropout_mask is not None:
                    c[dropout_mask] = self.empty_context_embedding.type(c.dtype).to(c.device)

                # maybe apply component dropout to counter context overfitting
                c = self.context_dropout(c)
        else:
            c = None
        # print("context.shape:")
        # print(context.shape)

        guided_hint = self.input_hint_block(hint, emb, context)
        # print("guided_hint.shape:")
        # print(guided_hint.shape)

        ### Here is different from ControlNet ### 
        # first zero_conv ~ image_proj
        x = self.zero_convs[0](x, emb, c)
        x = x  +  guided_hint
        outs = [x]

        h = x.type(self.dtype)
        # count = 0
        for module, zero_conv in zip(self.down, self.zero_convs[1:]):
            h = module(h, emb, c)
            # print("down_block_%d.shape:"%count)
            # print(h.shape)
            # count += 1
            outs.append(zero_conv(h, emb, c))

        h = self.middle(h, emb, c)
        # print("middle_block.shape:")
        # print(h.shape)
        output = self.middle_block_out(h, emb, c)
        outs.append(output)
        # print("output.shape:")
        # print(output.shape)
        return outs



class ConditionalUNet(torch.nn.Module):

    def __init__(self,
                 input_channels: int = 3,
                 initial_channels: int = 64,
                 channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 attention_heads: int = 4,
                 attention_head_channels: int = 32,
                 n_blocks_per_resolution: int = 2,
                 condition_type: str = 'AddPlusGN',
                 context_input_channels: int = 512,
                 context_channels: int = 512,
                 is_context_conditional: bool = False,
                 n_context_classes: int = 0,
                 learn_empty_context: bool = False,
                 context_dropout_probability: float = 0.0
                 ):

        super().__init__()

        n_resolutions = len(channel_multipliers)
        self.is_context_conditional = is_context_conditional

        # project image into feature map
        self.image_proj = torch.nn.Conv2d(input_channels, initial_channels, kernel_size=(3, 3), padding=(1, 1))

        # time embedding layer. Time embedding has `n_channels * 4` channels
        time_channels = initial_channels * 4
        self.time_emb = SinusoidalTimeEmbedding(time_channels)

        if self.is_context_conditional:

            if context_dropout_probability > 0:
                self.context_dropout = torch.nn.Dropout(p=context_dropout_probability)
            else:
                self.context_dropout = torch.nn.Identity()

            if n_context_classes > 0:
                self.context_emb = torch.nn.Embedding(n_context_classes, embedding_dim=context_channels)
            else:
                self.context_emb = torch.nn.Linear(context_input_channels, context_channels)

            if learn_empty_context:
                # create learnable constant embedding for dropped contexts
                self.empty_context_embedding = torch.nn.Parameter(torch.empty(context_channels))
                torch.nn.init.normal_(self.empty_context_embedding)
            else:
                self.empty_context_embedding = torch.zeros(context_channels)

        down_sample_block = partial(DownBlock,
                                    context_channels=context_channels,
                                    time_channels=time_channels,
                                    attention_heads=attention_heads,
                                    attention_head_channels=attention_head_channels,
                                    condition_type=condition_type,
                                    is_context_conditional=is_context_conditional
                                    )

        middle_block = partial(MiddleBlock,
                               context_channels=context_channels,
                               time_channels=time_channels,
                               attention_heads=attention_heads,
                               attention_head_channels=attention_head_channels,
                               condition_type=condition_type,
                               is_context_conditional=is_context_conditional
                               )

        up_sample_block = partial(UpBlock,
                                  context_channels=context_channels,
                                  time_channels=time_channels,
                                  attention_heads=attention_heads,
                                  attention_head_channels=attention_head_channels,
                                  condition_type=condition_type,
                                  is_context_conditional=is_context_conditional
                                  )

        # ======================================= DOWN SAMPLER =========================================================
        down = []
        # number of channels
        out_channels = in_channels = initial_channels
        # for each resolution
        count = 0
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]
            for _ in range(n_blocks_per_resolution):
                down.append(
                    down_sample_block(in_channels=in_channels, out_channels=out_channels, has_attention=is_attention[i]))
                in_channels = out_channels
                count += 1
                

            # down sample at all resolutions except the last
            if i < n_resolutions - 1:
               
                down.append(Downsample(in_channels))
        # print(count)
        # Combine the set of modules
        self.down = torch.nn.ModuleList(down)
        # ========================================== MIDDLE ============================================================
        self.middle = middle_block(n_channels=out_channels)

        # ======================================== UP SAMPLER ==========================================================
        up = []
        # number of channels
        in_channels = out_channels
        # for each resolution
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels

            for _ in range(n_blocks_per_resolution):
                up.append(up_sample_block(in_channels=in_channels, out_channels=out_channels, has_attention=is_attention[i]))

            # final block to reduce the number of channels
            out_channels = in_channels // channel_multipliers[i]
            up.append(up_sample_block(in_channels=in_channels, out_channels=out_channels, has_attention=is_attention[i]))

            in_channels = out_channels

            # up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # combine the set of modules
        self.up = torch.nn.ModuleList(up)

        # final normalization and convolution layer
        self.norm = torch.nn.GroupNorm(32, initial_channels)
        self.act = torch.nn.SiLU()
        self.final = zero_module(torch.nn.Conv2d(in_channels, input_channels, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor = None, dropout_mask: torch.Tensor = None):

        t = self.time_emb(t)
        # print("")

        # print("time_emb.shape:")
        # print(t.shape)

        # print("image_proj_before.hape:")
        # print(x.shape)
        x = self.image_proj(x)
        # print("image_proj_after.shape:")
        # print(x.shape)

        # use context only if the model is context_conditional
        if self.is_context_conditional:
            if context is None:
                c = self.empty_context_embedding.unsqueeze(0).repeat(len(x), 1).to(x.device)
            else:
                c = self.context_emb(context)

                # if entire samples is dropped out, use the empty context embedding instead
                if dropout_mask is not None:
                    c[dropout_mask] = self.empty_context_embedding.type(c.dtype).to(c.device)

                # maybe apply component dropout to counter context overfitting
                c = self.context_dropout(c)
        else:
            c = None
        # print("context.shape:")
        # print(c.shape)


        h = [x]

        # down_count = 0
        for m in self.down:
            x = m(x, t, c)
            # print("down_block_%d_output.shape:"%down_count)
            # print(x.shape)
            # down_count += 1
            h.append(x)
        
        x = self.middle(x, t, c)
        # print("middle_block_output.shape:")
        # print(x.shape)

        # up_count = 0
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t, c)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, c)
                # print("up_block_%d_output.shape:"%up_count)
                # print(x.shape)
                # up_count += 1

        noise_pred = self.final(self.act(self.norm(x)))
        # print("noise_pred.shape:")
        # print(noise_pred.shape)
        # exit()
        return noise_pred
        # return self.final(self.act(self.norm(x)))
