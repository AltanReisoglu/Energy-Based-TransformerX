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


from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from moe import DeepseekV2MoE, MoeConfig

from einops import pack, rearrange, reduce, repeat, unpack
from packaging import version
from torch import Tensor, einsum, nn
from modules.model_utils import RotaryEmbedding, MLP, modulate,RMSNorm,EBTModelArgs,AdvancedDropout
from feed_forward import SwigluFFN
from core.make_them_det import set_all_seeds
from attention import Attend,Predict_Attend,AlibiPositionalBias,GRUGating
set_all_seeds(42)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
  
    return freqs_cis

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def compact(arr):
    return [*filter(exists, arr)]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def create_causal_mask(i, j, device):
    #return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
    
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_q: torch.Tensor,
    freqs_cis_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_q = reshape_for_broadcast(freqs_cis_q, xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis_k, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi Group Query Attention"""
    def __init__(self, args: EBTModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1 #NOTE this is hardcoded since we are using DDP
        self.n_local_heads = args.n_heads 
        self.n_local_kv_heads = args.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        value_dim_head = args.dim_head
        q_dim = args.dim_head * args.n_heads
        kv_heads = args.n_kv_heads
        self.heads = args.n_heads
        kv_heads = 1 if args.one_kv_head else kv_heads
        k_dim = args.dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * args.heads

        self.q_proj = nn.Linear(args.dim, q_dim, bias=False)
        init_whole_model_weights(self.q_proj, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.r_proj = nn.Linear(args.dim, v_dim, bias = False) if args.tensor_product else None
        self.k_proj = nn.Linear(args.dim, k_dim, bias=False)
        init_whole_model_weights(self.k_proj, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.v_proj = nn.Linear(args.dim, v_dim, bias=False)
        init_whole_model_weights(self.v_proj, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wo_gate = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo_gate, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        if args.head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, args.heads, 1, 1))




                # cosine sim MGQA
        self.qk_norm = args.qk_norm
        self.qk_norm_groups = args.qk_norm_groups
        self.qk_norm_scale = args.qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim MGQA when scale is equal to 1) - https://arxiv.org/abs/2302.05442
        self.qk_norm_dim_scale = args.qk_norm_dim_scale

        #self.qk_norm_q_scale = self.qk_norm_k_scale = 1
    
        self.qk_norm_q_scale = nn.Parameter(torch.empty(args.n_heads, 1, args.dim_head))
        self.qk_norm_k_scale = nn.Parameter(torch.empty(args.n_kv_heads, 1, args.dim_head))
        nn.init.constant_(self.qk_norm_q_scale, 1)


        self.attend=Attend(
            dim=args.dim,
            heads = args.heads,
            causal = args.causal,
            talking_heads = args.talking_heads,
            dropout = args.dropout,
            sparse_topk = args.sparse_topk,
            qk_norm = args.qk_norm,
            scale = args.qk_norm_scale,
            add_zero_kv = args.add_zero_kv,
            flash = False,
            onnxable = args.onnxable,
            linear_attention=False,
            window_size=16,
        )
        self.kv_heads=args.n_kv_heads
        self.pre_attend=Predict_Attend(
        dim=args.dim,
        heads = args.heads,
        causal = args.causal,
        talking_heads = args.talking_heads,
        dropout = args.dropout,
        qk_norm = args.qk_norm,
        scale = args.qk_norm_scale ,
        add_zero_kv = args.add_zero_kv,

        onnxable = args.onnxable,

        )

        self.head_scale=args.head_scale
        if args.gate_values:
            self.v_proj_gate = nn.Linear(args.dim, out_dim)
            nn.init.constant_(self.v_proj_gate.weight, 0)
            nn.init.constant_(self.v_proj_gate.bias, 1)

    def _parallelogram_mask(self, h, w, left=False):
            mask = torch.ones((h, w)).byte()
            m = min(h, w)
            mask[:m,:m] = torch.triu(mask[:m,:m])
            mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

            if left:
                return mask
            else:
                return mask.flip(0)
            
    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x
    
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        start_pos: int,
        freqs_cis_q: torch.Tensor,
        freqs_cis_k: torch.Tensor,
        mask: Optional[torch.Tensor],
        mem: Optional[torch.Tensor] = None,
        cache = None,
        rel_pos=None,prev_attn_prev=None,prev_attn_latter=None,mask2=None,pre_mem=None,**kwargs):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, full_seqlen, _ = x.shape # full_seqlen includes real embeds and pred embeds
        original_seqlen = full_seqlen//2 # length of original sequence without next pred
        context_length = original_seqlen + 1 # actual context length of model
        mem_len=mem.shape[1]
        if(cache is not None):
            cache_len=original_seqlen
        kv_input = context if exists(context) else x
        # NOTE the usage of S-1/S/S+1 is messed up and confusing here, I recommend checking the paper
        
        kv_h=self.kv_heads
        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(pre_mem):
            #heuristical method by getting intermediate values of linear polation process
            pre_mem=torch.transpose(pre_mem,2,1)
            pre_mem = F.interpolate(pre_mem, scale_factor=2, mode='linear', align_corners=True)
            pre_mem= torch.transpose(pre_mem,2,1)[:,1::2,:]
            mem=mem+pre_mem
        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

          
        r_input= r_input[:,:original_seqlen,:]

        xq, xk, xv,xr = self.q_proj(q_input), self.k_proj(k_input), self.v_proj(v_input),self.r_proj(r_input) 

        """xq= F.normalize(xq, dim=-1)
        xk= F.normalize(xk, dim=-1)"""

        

        xq = rearrange(xq, 'b n (h d) -> b h n d', h = self.heads)

        xk, xv, xr = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=kv_h) if t is not None else None, (xk, xv, xr))

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            xq, xk = map(qk_l2norm, (xq, xk))

            xq = xq * self.qk_norm_q_scale
            xk = xk * self.qk_norm_k_scale

        xq = rearrange(xq, 'b h n d -> b n h d')
        xk = rearrange(xk, 'b h n d -> b n h d')
        xv = rearrange(xv, 'b h n d -> b n h d')
        
             # _o is for original attention stuff
        xq_o = xq[:, :original_seqlen, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = xk[:, :original_seqlen+mem_len, :, :]
        xv_o = xv[:, :original_seqlen+mem_len, :, :]
        
        # _p is for predicted attention stuff
        xq_p = xq[:, original_seqlen:, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = xk[:, original_seqlen+mem_len:, :, :]
        xv_p = xv[:, original_seqlen+mem_len:, :, :]


        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis_q=freqs_cis_q[:original_seqlen],freqs_cis_k=freqs_cis_k[:original_seqlen+mem_len])

        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis_q=freqs_cis_q[original_seqlen:],freqs_cis_k=freqs_cis_k[original_seqlen+mem_len:]) 


        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk_o = xk_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        xv_o = xv_o.transpose(1, 2)
         # (bs, n_local_heads, seqlen, head_dim)
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk_p = xk_p.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        
        xv_p = xv_p.transpose(1, 2)

        


        if exists(cache):
            ck, cv = cache

            if exists(mem):
                
                mk , xk= xk_o[:,:,:mem_len,:],xk_o[:,:,mem_len:,:]
                mv , xv= xv_o[:,:,:mem_len,:],xv_o[:,:,mem_len:,:]
            
            xk_o = torch.cat((ck, xk), dim = -2)
            xv_o = torch.cat((cv, xv), dim = -2)

            new_cache=(xk,xv)
            if exists(mem):
                xk_o = torch.cat((mk, xk_o), dim = -2)
                xv_o = torch.cat((mv, xv_o), dim = -2)
        else:
            xk= xk_o[:,:,mem_len:,:]
            xv= xv_o[:,:,mem_len:,:]
            new_cache=(xk,xv)

        # use 1 since are the next preds and thus need to condition on a frame
        # I tested this compared to prepending row on S dimension and the tensors were the same

        i, j = map(lambda t: t.shape[-2], (xq_o, xk_o))
        
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)
        #original attn calc is more normal############################################

        output_o = self.attend(
            xq_o, xk_o, xv_o,
            mask = mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn_prev
        )
        
        if exists(xr):
            xr=repeat(xr, 'b kvh n d -> b (r kvh) n d', r = output_o.shape[1] // xr.shape[1])
            output_o = output_o * xr + output_o
        if self.head_scale:
            
            output_o = output_o * self.head_scale_params
        
        """scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S-1, S-1
        if mask is not None:
            #this mask needs to be seqlen, seqlen, was S, S
            o_mask = mask[:-1, :-1] #set to S-1, S-1 like 0 -inf -inf; 0 0 -inf, etc   
            scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1) # has B, S-1, D after"""
        
        #pred sequence attn calc is for energy-based transformer ########################################################################################
        if exists(cache):

            output_o=output_o[:,:,cache[0].shape[2]:,:]
      # seqlen here is S-1 which = original_seqlen
   # (bs, n_local_heads, seqlen, head_dim)
        output_p=self.pre_attend(xq_p,xk_p,xv_p, xk_o, xv_o,mask=mask2,prev_attn=prev_attn_latter,rel_pos=rel_pos,mem_packed_shape=mem_packed_shape,cache=cache,mem=mem)
         # B, 2(S-1), D

        if(output_o.shape != 3):
            output_o=rearrange(output_o,"b h n d -> b n (h d)")

        out = torch.cat((output_o, output_p), dim = 1) # B, 2(S-1), D
        

        if exists(self.v_proj_gate):
            gates = self.v_proj_gate(x)
            out = out * gates.sigmoid()
        return self.wo_gate(out),new_cache,mem,(output_o,output_p)

class AdaLNTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: EBTModelArgs):
        """
        Initialize a AdaLNTransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        config = MoeConfig()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        ""
        self.feed_forward = (
            DeepseekV2MoE(config)
            if config.n_routed_experts is not None
            else SwigluFFN(
                in_features=args.dim,
                hidden_features=args.ffn_dim_multiplier * args.dim if args.ffn_dim_multiplier is not None else 4 * args.dim,
                out_features=args.dim,
            )
        )
        self.rel_pos = AlibiPositionalBias(
                heads=args.alibi_num_heads,
                total_heads=args.heads
            )
        
        self.residual_fn = GRUGating(
            args.dim,
            scale_residual=args.scale_residual,
            scale_residual_constant=args.scale_residual_constant
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 6 * self.dim, bias=True)
        )
        if args.resi_dual:
            pre_norm = False
        self.pre_norm = pre_norm

        self.resi_dual = args.resi_dual
        assert 0 < args.resi_dual_scale <= 1., \
            'resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1.'
        self.resi_dual_scale = args.resi_dual_scale

        self.residual_attn = args.residual_attn
        self.cross_residual_attn = args.cross_residual_attn
        assert not (args.flash_attn and (args.residual_attn or args.cross_residual_attn)), \
            'flash MGQA is not compatible with residual MGQA'

        self.cross_attend = args.cross_attend
        self.cross_attn_tokens_dropout = args.cross_attn_tokens_dropout
        self.advanced_dropout=AdvancedDropout(self.dim)
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis_q: torch.Tensor,
        freqs_cis_k: torch.Tensor,
        mask: Optional[torch.Tensor],
        time_embeddings,
        mems: Optional[torch.Tensor] = None,
        past_cache: Optional[nn.ParameterDict] = None,
        prev_attn_prev=None,prev_attn_latter=None,mask2=None,pre_mem=None,
        **kwargs
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_embeddings).chunk(6, dim=1)
        attn,new_cache,mem,(output_o,output_p)=self.attention(
            modulate(self.attention_norm(x), shift_msa, scale_msa),None, start_pos, freqs_cis_q,freqs_cis_k,mask= mask,mem=mems,cache=past_cache,prev_attn_prev=prev_attn_prev,prev_attn_latter=prev_attn_latter,rel_pos=self.rel_pos,mask2=mask2,pre_mem=pre_mem
        )
        
        h = x + gate_msa.unsqueeze(1) * self.advanced_dropout(attn)
        x = self.residual_fn(h, x)
  
        out = h + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        out = self.residual_fn(out, x)
       
        return out,new_cache,mem,(output_o,output_p)
    
class FinalLayer(nn.Module):
    """
    The final layer of EBT when using adaLN.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias = False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

class EBTAdaLN(nn.Module):
    def __init__(self, params: EBTModelArgs, max_mcmc_steps):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.

        Attributes:
            params (EBTModelArgs): Model configuration parameters.
            n_layers (int): Number of layers in the model.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            block = AdaLNTransformerBlock(layer_id, params)
            if params.adaln_zero_init:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            self.layers.append(block) # confirmed all layers and final layer are initialized to 0

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.head_dim=params.head_dim
        
        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = FinalLayer(params.dim)
        if params.adaln_zero_init:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
        else:
            init_whole_model_weights(self.final_layer.linear, self.params.weight_initialization)
        self.shift_mem_down=params.shift_mem_down
        self.memory_tokens=nn.Parameter(torch.rand(params.num_memory_tokens, params.dim))
    def forward(self, embeddings: torch.Tensor, start_pos: int, mcmc_step = 0,past_cache_list=None):
        new_key_values = []
        _bsz, seqlen = embeddings.shape[:2]
        original_seqlen=seqlen//2
        mems = self.memory_tokens.expand(_bsz, -1, -1)
        self.freqs_cis_q = precompute_freqs_cis(self. head_dim, seqlen)
        self.freqs_cis_k= precompute_freqs_cis(self.head_dim,seqlen+mems.shape[1])
        seqlen = (seqlen+2) // 2 # do this since passed in seqlen is 2(S-1) so add 2 div 2 = S
        freqs_cis_q = self.freqs_cis_q.to(embeddings.device)
        freqs_cis_k = self.freqs_cis_k.to(embeddings.device)
        
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device = embeddings.device, dtype=torch.long)
        time_embeddings = self.time_embeddings(mcmc_step)

        mask = None
        
        if seqlen > 1:
            if(exists(past_cache_list)):
                eleman=seqlen+past_cache_list[0][0].shape[2]
            else:
                eleman=seqlen
            mask = torch.full(
                (eleman,eleman), float("-inf"), device=embeddings.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((eleman,mems.shape[1]), device=embeddings.device),
                mask
            ]).type_as(embeddings)
            # causal mask is like this by default 0, -inf, -inf
            #                         0, 0,    -inf
            # 
            # 
            
            mask2 = torch.full(
                (seqlen,seqlen), float("-inf"), device=embeddings.device
            )

            mask2 = torch.triu(mask2, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask2 = torch.hstack([
                torch.zeros((seqlen,0), device=embeddings.device),
                mask2
            ]).type_as(embeddings)
            # causal mask is like this by default 0, -inf, -inf
            #                         0, 0,    -inf
            #                         0, 0,    0                        
                
            if self.shift_mem_down and exists(mems):
                mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
                mems = [*mems_r, *mems_l]
            prev_attns=(None,None)
            pre_mem=None
            for i, layer in enumerate(self.layers):
                past = past_cache_list[i] if past_cache_list is not None else None
                embeddings,new_cache,pre_mem,prev_attns_out = layer(embeddings, start_pos, freqs_cis_q,freqs_cis_k, mask, time_embeddings,mems=mems,past_cache=past,prev_attn_prev=prev_attns[0],prev_attn_latter=prev_attns[1],mask2=mask2,pre_mem=pre_mem)
                
                new_key_values.append(new_cache)
                

            """new_mem = embeddings[:, :mems.shape[1], :]
            seq_out = embeddings[:, mems.shape[1]:, :]"""

            seq_out = self.norm(embeddings)
            energies = self.final_layer(seq_out, time_embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            
            return energies,new_key_values
        
if __name__ == "__main__":
    #mock trial
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rande=torch.randn(2,2,16,32,device=device)
    args = EBTModelArgs()
    past_cache_list=[(rande,rande),(rande,rande),(rande,rande),(rande,rande),(rande,rande),(rande,rande)]
    model = EBTAdaLN(args, max_mcmc_steps=10).to(device=device)
    x = torch.randn(2, 2, args.dim,device=device)
    out = model(x, start_pos=0, mcmc_step=0,past_cache_list=past_cache_list)
    print(out[0].shape)