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
from modules.model_utils import RotaryEmbedding, MLP, modulate,RMSNorm,EBTModelArgs 
from feed_forward import SwigluFFN
from core.make_them_det import set_all_seeds

set_all_seeds(42)



class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x)
        return l2norm(token_emb) if self.l2norm_embed else token_emb

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_MGQA_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_MGQA_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            nn.LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

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

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale
import math
def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim = -1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

    

DEFAULT_DIM_HEAD=32
@dataclass
class Intermediates:
    qk_similarities: Optional[torch.Tensor] = None
    pre_softmax_attn: Optional[torch.Tensor] = None
    post_softmax_attn: Optional[torch.Tensor] = None
    cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)
    

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

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

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

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
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
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attend(nn.Module):
    def __init__(
        self,
        dim,
        dropout = 0.1,
        causal = False,
        heads = None,
        talking_heads = False,
        sparse_topk = None,
        scale = None,
        qk_norm = False,
        flash = False,
        add_zero_kv = False,
        onnxable = False,
        linear_attention = False,*kwargs
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.attn_fn = partial(F.softmax, dtype = torch.float32) if not qk_norm else F.softmax

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and talking_heads), 'talking heads not compatible with flash MGQA'

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        # sparse topk

        assert not (flash and sparse_topk), 'sparse topk not compatible with flash MGQA'
        self.sparse_topk = sparse_topk

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/MGQA-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # flash MGQA

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash MGQA, you must be using pytorch 2.0 or above'

        # determine efficient MGQA configs for cuda and cpu
        self.cuda_config = namedtuple('FlashConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])(True, True, True)

        self.flash_attn = partial(F.scaled_dot_product_attention, dropout_p=0.0, is_causal=causal,softmax_in_fp32=not qk_norm) if flash else None
        #self.gru_cell=GRUGating(dim=heads*DEFAULT_DIM_HEAD)
        self.permission_linear_attention=linear_attention
        self.epsilon=1e-6
    def elu_feature_map(self, x):
        return F.elu(x) + 1

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device
        
        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query MGQA

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif kv_heads < heads:
            k, v = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads), (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = map(lambda t: F.pad(t, (0, 0, 1, 0), value = 0.), (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        
        if self.permission_linear_attention:
            # ELU feature map ile kernel tabanlı linear attention
            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
            q_prime = self.elu_feature_map(q)  
            k_prime = self.elu_feature_map(k)
            
            # Kümülatif toplam ile normalizasyon
            eps = getattr(self, "epsilon", 1e-6)

            # feature-map ile hazırlanmış q_prime, k_prime hazır gelmeli
            # (kodun önceki kısmında zaten q_prime, k_prime yapılıyor)
            # q_prime: [B, H, Nq, D]
            # k_prime: [B, H, Nk, D]
            # v:       [B, H, Nk, E]

            # 1) k_prime_sum zaten [B, H, 1, D] ise:
            k_prime_sum = k_prime.sum(dim=-2, keepdim=True) + eps  # [B, H, 1, D]

            # 2) kv = φ(K)^T @ V  -> [B, H, D, E]
            #    burada einsum: sum over Nk
            kv = einsum("bhjd,bhje->bhde", k_prime, v)  # [B, H, D, E]

            # 3) numerator = φ(Q) @ kv -> [B, H, Nq, E]
            numerator = einsum("bhid,bhde->bhie", q_prime, kv)  # [B,H,Nq,E]

            # qk_similarities benzerlik göstergesi olarak numerator'un clone'u olabilir
            qk_similarities = numerator.clone()

            # Eğer prev_attn uygun shape'teyse (örn [B,H,Nq,E]) ekle, değilse atla
            if exists(prev_attn) and prev_attn.shape == numerator.shape:
                numerator = numerator + prev_attn
                qk_similarities = numerator.clone()

            # (opsiyonel) talking heads öncesi işlem / attn_bias 
            # not: attn_bias genelde [B,H,Nq,Nk], burada Nk==1 ise kullanılabilir; bu örnekte atlıyoruz
            # if exists(attn_bias) and attn_bias.shape == (B,H,Nq,1): numerator = numerator + attn_bias.unsqueeze(-1)

            # 4) denominator = φ(Q) @ (φ(K)^T 1) -> [B, H, Nq, 1]
            #    k_prime_sum: [B,H,1,D], q_prime: [B,H,Nq,D]
            denominator = torch.einsum("bhid,bhjd->bhij", q_prime, k_prime_sum)  # [B, H, Nq, 1]

            # 5) attention (linear normalized) -> [B,H,Nq,1]
            attn = 1.0 / (denominator + eps)   # [B,H,Nq,1]
            attn = attn.type(numerator.dtype)

            # Klonla (intermediates için)
            post_softmax_attn = attn.clone()
            pre_softmax_attn = numerator.clone()  # linear'de pre-softmax olarak numerator kullanalım

            # Dropout (varsa)
            attn = self.attn_dropout(attn)

            # talking_heads post-softmax (Conv2d expects (B, C, H, W) = (B, heads, Nq, 1))
            if self.talking_heads:
                # post_softmax_talking_heads expects channels=heads, input is b h i 1 -> ok
                attn = self.post_softmax_talking_heads(attn)

            # 6) out = numerator * attn  -> broadcasting [B,H,Nq,1]
            out = numerator * attn
            print("linear attention executed")
            # return intermediates uyumlu şekilde oluştur
            intermediates = Intermediates(
                qk_similarities = qk_similarities,
                pre_softmax_attn = pre_softmax_attn,
                post_softmax_attn = post_softmax_attn
            )

            return out, intermediates
        


        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
        

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)
        if exists(attn_bias):
            dots = dots + attn_bias
        
        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max
        
        """if exists(self.sparse_topk) and self.sparse_topk < j:
            
            top_values, _ = dots.topk(self.sparse_topk, dim = -1)
            sparse_topk_mask = dots < top_values[..., -1:]
            mask = (mask & sparse_topk_mask) if exists(mask) else sparse_topk_mask"""
        if mask is not None:
            #this mask needs to be seqlen, seqlen, was S, S
            o_mask = mask[:-1, :-1] #set to S-1, S-1 like 0 -inf -inf; 0 0 -inf, etc   
            dots = dots+ o_mask  # (bs, n_local_heads, seqlen, seqlen)
        """if exists(mask):
            scores_o = scores_o.masked_fill(~mask, mask_value)"""
        
        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            dots= dots.masked_fill(causal_mask, mask_value)

        pre_softmax_attn = dots.clone()
        
        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)
        
        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        if self.talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        
        """intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )"""

        return out
    
class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)
    
class SigmoidGating(nn.Module):    
    def __init__(self, dim:int)->None:
        super().__init__()
        self.g_proj=nn.Linear(dim,dim)
        init_whole_model_weights(self.g_proj, "xavier", weight_initialization_gain=1.0)
        self.sigmoid=nn.Sigmoid()
        self.out_proj=nn.Linear(dim,dim)
        init_whole_model_weights(self.out_proj,"xavier", weight_initialization_gain=1.0)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        gate=self.sigmoid(self.g_proj(x))
        x=x*gate
        return self.out_proj(x) 
        
class MGQA(nn.Module):
    @classmethod
    def divisible_by(cls,num,den):
        return num %den==0
    def __init__(
        self,
        args:EBTModelArgs ,*kwargs
    ):
        super().__init__()
        self.scale = args.dim_head ** -0.5

        self.heads = args.heads
        self.causal = args.causal
        self.max_attend_past = args.max_attend_past


        assert not (exists(args.kv_heads) is not None and args.one_kv_head), 'either attn_one_kv_head is set to True (in which case kv_heads is set to 1), or attn_kv_heads is set, but not both'

        value_dim_head = args.dim_head
        kv_heads = args.heads

        kv_heads = 1 if args.one_kv_head else kv_heads
        assert kv_heads % args.heads == 0, 'key / value heads must be divisible by number of query heads'

        self.kv_heads = kv_heads
        
        
        q_dim = args.dim_head * args.heads
        k_dim = args.dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * args.heads

        self.q_proj = nn.Linear(args.dim, q_dim, bias = False)
        self.k_proj = nn.Linear(args.dim, k_dim, bias = False)

        # shared key / values, for further memory savings during inference
        assert not (args.shared_kv and value_dim_head != args.dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.v_proj = nn.Linear(args.dim, v_dim, bias = False) if not args.shared_kv else None

        # relations projection from tp-MGQA
        self.r_proj = nn.Linear(args.dim, v_dim, bias = False) if args.tensor_product else None

        # add GLU gating for aggregated values, from alphafold2
        self.v_proj_gate = None
        if args.gate_values:
            self.v_proj_gate = nn.Linear(args.dim, out_dim)
            nn.init.constant_(self.v_proj_gate.weight, 0)
            nn.init.constant_(self.v_proj_gate.bias, 1)

        # cosine sim MGQA
        self.qk_norm = args.qk_norm
        self.qk_norm_groups = args.qk_norm_groups
        self.qk_norm_scale = args.qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim MGQA when scale is equal to 1) - https://arxiv.org/abs/2302.05442
        self.qk_norm_dim_scale = args.qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if args.qk_norm and args.qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.empty(args.heads, 1, args.dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.empty(args.heads, 1, args.dim_head))
            nn.init.constant_(self.qk_norm_q_scale, 1)
            nn.init.constant_(self.qk_norm_k_scale, 1)#1 olmalı ones


        assert (not args.qk_norm) or MGQA.divisible_by(args.dim_head, args.qk_norm_groups), 'dimension per MGQA head must be divisible by the qk norm groups'
        assert not (args.qk_norm and (args.dim_head // args.qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # attend class - includes core MGQA algorithm + talking heads

        self.attend = Attend(
            dim=args.dim,
            heads = args.heads,
            causal = args.causal,
            talking_heads = args.talking_heads,
            dropout = args.dropout,
            sparse_topk = args.sparse_topk,
            qk_norm = args.qk_norm,
            scale = args.qk_norm_scale if args.qk_norm else self.scale,
            add_zero_kv = args.add_zero_kv,
            flash = args.flash,
            onnxable = args.onnxable,
            linear_attention=args.linear_attention,
        )
        # sigmoid gating
        if args.sigmoid_gating:
            self.sigmoid_gating=SigmoidGating(dim=out_dim) 

        # head scaling
        self.head_scale = args.head_scale
        if args.head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, args.heads, 1, 1))

        # explicit topk sparse MGQA
        self.sparse_topk = args.sparse_topk

        # add memory key / values
        self.num_mem_kv = args.num_mem_kv
        if args.num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(args.heads, args.num_mem_kv, args.dim_head))
            self.mem_v = nn.Parameter(torch.randn(args.heads, args.num_mem_kv, args.dim_head))

        # MGQA on MGQA
        self.attn_on_attn = args.on_attn
        self.to_out = nn.Sequential(nn.Linear(out_dim, args.dim * 2, bias = False), nn.GLU()) if args.on_attn else nn.Linear(out_dim, args.dim, bias = False)

        # whether to rotate positions into values, for absolute positions in addition to relative
        self.rotary_embed_values = args.rotary_embed_values

        # init output projection 0
        if args.zero_init_output:
            init_whole_model_weights(self.to_out,weight_initialization_method="zero")

        

    def forward(
        self,
        x,
        
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        mem = None,
        cache= None,
        return_intermediates = True,
        prev_attn:torch.Tensor = None,
        
    ):
        b, n, _, h, kv_h, head_scale, device, has_context = *x.shape, self.heads, self.kv_heads, self.head_scale, x.device, exists(context)
        print(has_context)
        original_seqlen=n//2
        kv_input = context if has_context else x

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x
        
        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        q = self.q_proj(q_input)
        k = self.k_proj(k_input)
        v = self.v_proj(v_input) if exists(self.v_proj) else k
        r = self.r_proj(r_input) if exists(self.r_proj) else None
        
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k, v, r = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=kv_h) if t is not None else None, (k, v, r))

        """xq_o = q[:, :,:original_seqlen, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = k[:, :,:original_seqlen, :]
        xv_o = v[:, :,:original_seqlen, :]
        xr_o = r[:, :,:original_seqlen, :] if r is not None else None
        
        # _p is for predicted attention stuff
        xq_p = q[:,:, original_seqlen:,  :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = k[:,:, original_seqlen:,  :]
        xv_p = v[:,:, original_seqlen:,  :]
        xr_p = r[:,:, original_seqlen:,  :] if r is not None else None"""

        if exists(cache) and not has_context:
            ck, cv = cache.cached_kv

            if exists(mem):
                mk, k = unpack(k, mem_packed_shape, 'b h * d')
                mv, v = unpack(v, mem_packed_shape, 'b h * d')

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            if exists(mem):
                k = torch.cat((mk, k), dim = -2)
                v = torch.cat((mv, v), dim = -2)
        print("cache çıktısı",k.shape)
        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale
        print("norm çıktısı",k.shape)
        if exists(rotary_pos_emb) and not has_context:
            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)
            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if self.rotary_embed_values:
                v = apply_rotary_pos_emb(v, freqs, k_xpos_scale)
        print("rotary",k.shape)
        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)
            #print("mem v shape",v.shape)
             # pad the mask
            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)
        print("mem_kv",k.shape)
        i, j = map(lambda t: t.shape[-2], (q, k))
        
        if return_intermediates:
            mem_len = mem.shape[-2] if exists(mem) else 0
            cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])
        # determine masking

        max_neg_value(q)
        """masks = []
        final_attn_mask = None"""

        """if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask.int())"""
        
        """if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'MGQA mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            masks.append(~attn_mask)"""
        
        """if exists(self.max_attend_past):
            range_q = torch.arange(j - i, j, device = device)
            range_k = torch.arange(j, device = device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            max_attend_past_mask = dist > self.max_attend_past
            masks.append(max_attend_past_mask)"""

        """if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)"""
        
        # prepare relative positional bias, if needed

        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)
        
        # MGQA is all we need
        out, intermediates = self.attend(
            q, k, v,
            
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )
        
        """v_interp = rearrange(out, 'b h n d -> b n (h d)')
        print(v_interp.shape)
        v_interp = F.interpolate(v_interp, size=q.shape[-2], mode="linear", align_corners=False)
        v_interp = rearrange(v_interp, 'b n (h d) -> b h n d', h=h)
        print(v_interp.shape)
        print(out.shape)

        
        out =torch.concat(out + v_interp,dim=-1)"""

        # https://arxiv.org/abs/2208.06061 proposes to add a residual for better gradients

        if exists(r):
            out = out * r + out

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # alphafold2 styled gating of the values

        if exists(self.v_proj_gate):
            gates = self.v_proj_gate(x)
            out = out * gates.sigmoid()

        # combine the heads

        out = self.to_out(out)

        

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates
    
class Predict_Attend(nn.Module):
    def __init__(self,dim,
        dropout = 0.1,
        causal = False,
        heads = None,
        talking_heads = False,

        scale = None,
        qk_norm = False,
        flash = False,
        add_zero_kv = False,
        onnxable = False,
        linear_attention = False,*kwargs ):
        super().__init__()
        
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.attn_fn = partial(F.softmax, dtype = torch.float32) if not qk_norm else F.softmax

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and talking_heads), 'talking heads not compatible with flash MGQA'

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        self.add_zero_kv = add_zero_kv

        # determine efficient MGQA configs for cuda and cpu
        self.cuda_config = namedtuple('FlashConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])(True, True, True)

        self.flash_attn = partial(F.scaled_dot_product_attention, dropout_p=0.0, is_causal=causal,softmax_in_fp32=not qk_norm) if flash else None
        #self.gru_cell=GRUGating(dim=heads*DEFAULT_DIM_HEAD)
        self.permission_linear_attention=linear_attention
        self.epsilon=1e-6
        
    def elu_feature_map(self, x):
        return F.elu(x) + 1
    
    def forward(
        self,
        q, k, v,keys_o,values_o,
        mask = None,
        rel_pos = None,
        prev_attn = None,mem_packed_shape=None,cache=None,mem=None
    ):
        if(exists(mem)):
            keys_o=keys_o[:,:,mem.shape[1]:,:]
            values_o=values_o[:,:,mem.shape[1]:,:]
            if (exists(cache)):
                keys_o=keys_o[:,:,cache[0].shape[-2]:,:]
                values_o=values_o[:,:,cache[0].shape[-2]:,:]

        bsz,_,original_seqlen,dim_head=q.shape
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device
        
        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query MGQA

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif kv_heads < heads:
            k, v,keys_o,values_o = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads), (k, v,keys_o,values_o))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v,keys_o,values_o = map(lambda t: F.pad(t, (0, 0, 1, 0), value = 0.), (k, v,keys_o,values_o))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, keys_o) * scale

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        i, j = map(lambda t: t.shape[-2], (q, k))
        
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)
            dots = dots + attn_bias
            

        temp_append = torch.zeros((dots.shape[0], dots.shape[1], dots.shape[2], 1), dtype=dots.dtype, device=dots.device) # B, N, S-1, 1; is used since context_length = original_length +1, superdiag needs this
        dots = torch.cat((dots, temp_append), dim = -1)
  
        insertion_superdiagonal = (q * k).sum(dim = 3) /math.sqrt(dim_head) # B, N, S-1; this is for next preds to attend to themselves
        insertion_superdiagonal = insertion_superdiagonal.to(dots.dtype) # for if using non 32 precision

        superdiag_rows = torch.arange(dots.shape[2]) #[0, ..., S-2] (len 15)
        superdiag_cols = torch.arange(1, dots.shape[3])


        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=dots.dtype, device=dots.device) # for zeroing out superdiag since dont want to include in matmul, do this in differentiable way
        diagonal_removal_mask = torch.ones_like(dots, dtype=dots.dtype, device=dots.device)

        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        dots = dots * diagonal_removal_mask  

        diagonal_addition_mask = torch.zeros_like(dots, dtype=dots.dtype, device=dots.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        dots = dots + diagonal_addition_mask   
        
        if exists(prev_attn):
                dots = dots + prev_attn
        if mask is not None:
            p_mask = mask[1:, :]  #S-1, S like 0 0 -inf -inf; 0 0 0, -inf, etc  
            dots = dots + p_mask

        dots = self.attn_fn(dots, dim = -1)
        #Q: why do I need to extract superdiagonal why cant i just do matmul after? A: its bc would need same subsequence in value matrix but dont have it, have original subsequence and then seperately all next preds
        scores_p_superdiagonal = dots.diagonal(offset=1, dim1=2, dim2=3).clone() # is B, N, S-1; basically how much each token on this superdiag should attent to itself; clone since dont want mask to change this
        
        dots = dots * diagonal_removal_mask # keeps scores_p as is except for superdiagonal which is next preds attention to selves, cant multiply these naively by values_p or values_o
        
        dots = dots[:, :, :, :-1] # B, N, S-1, S-1 now; next preds/scores_p_superdiagonal was why needed extra col earlier (temp_append)
        ##buradan aldım  
        if self.talking_heads:
            dots = self.post_softmax_talking_heads(dots)
            
        output_p = torch.matmul(dots, values_o) # B, N, S-1, H; is how next preds attend to all original previous tokens;

        #next_pred_self_attention is to get self attention based on extracted superdiagonal and the values matrix (for predictions)
        next_pred_self_attention = v * scores_p_superdiagonal.unsqueeze(dim = -1) # B, N, S-1, H this is for weighted sum of each next pred to its final embed rep.
        
        output_p = output_p + next_pred_self_attention 

        pre_softmax_attn = output_p.clone()

        

        output_p = self.attn_dropout(pre_softmax_attn)

        

        output_p = output_p.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
        
        return output_p  
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    attn = MGQA(
    dim=256,
    dim_head=32,
    heads=8,
    causal=True,
    flash=False,
    talking_heads=True,
    head_scale=True,
    sparse_topk=8,
    num_mem_kv=2,
    dropout=0.0,
    on_attn=False,
    gate_values=True,
    #zero_init_output=True,
    max_attend_past=True,
    qk_norm=True,
    qk_norm_groups=4,
    qk_norm_scale=True,
    qk_norm_dim_scale=True,
    
    kv_heads=2,
    shared_kv=True,
    value_dim_head=True,
    tensor_product=True,
    add_zero_kv=False,
    rotary_embed_values=True,
    onnxable=True,
    linear_attention=False,
    ).to(device)
    x=torch.rand(2,16,256).to(device)
    out=attn(x)
    print(out.shape)
    import torch
    total_params = sum(p.numel() for p in attn.parameters())
    print(f'Total parameters: {total_params}')
    fp_count = total_params  # her parametre 1 FP
    print("Floating point sayısı (parametre başına 1 FP):", fp_count)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(attn, (3, 224, 224), as_strings=True,
                                            print_per_layer_stat=False, verbose=False)
    print("MACs:", macs)
    print("Params:", params)

class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

    
def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)
class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)
    

