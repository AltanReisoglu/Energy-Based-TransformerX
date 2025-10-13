import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import pytorch_lightning as L
from typing import *
from modules.model_utils import init_whole_model_weights

import math
from collections import namedtuple
from dataclasses import dataclass
from functools import partial, wraps
from inspect import isfunction
from random import random
from typing import Callable, List, Optional, Tuple


from einops import pack, rearrange, reduce, repeat, unpack
from packaging import version
from torch import Tensor, einsum, nn
from modules.model_utils import RotaryEmbedding,MLP,RMSNorm



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        weight_initialization_gain: float
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        # hidden_dim = int(2 * hidden_dim / 3)
        # # custom dim factor multiplier
        # if ffn_dim_multiplier is not None:
        #     hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        hidden_dim = dim if ffn_dim_multiplier is None else int(dim*ffn_dim_multiplier)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w1, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(self.w2, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w3, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class SwigluFFN(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) ->None:
        super().__init__()
        self.w_q=MLP(in_features,2*hidden_features)
        init_whole_model_weights(self.w_q, "xavier", weight_initialization_gain=1.0)
        self.w_o=MLP (hidden_features,out_features)
        init_whole_model_weights(self.w_o, "xavier", weight_initialization_gain=1.0)
        self.sigmoid=nn.SiLU()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if(len(x.shape)<3):
            x=x.unsqueeze(1)
            B,T,C=x.shape
        else:
            B,T,C=x.shape
        x=self.w_q(x)
        x_1,x_2=torch.chunk(x,chunks=2,dim=-1)
        hidden = F.silu(x_1) * x_2
        out=self.w_o(hidden)
        out=out.squeeze()
        return out
    





class LIVConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.d_model = d_model

        self.input_linear = nn.Linear(d_model, d_model)

        self.B = nn.Parameter(torch.randn(d_model, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_model))

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2  
        )
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):

        h = self.input_linear(x)  

        hB = einsum('b n k, k j -> b n j', h, self.B)
        hC = einsum('b n k, k j -> b n j', h, self.C)
        gated = hB * hC      
        gated=rearrange(gated,"b n d -> b d n") 
        gated = gated.transpose(1, 2)

        conv_out = self.conv1d(gated)  
        conv_out = conv_out.transpose(1, 2)
        out = self.output_linear(conv_out)
        return out