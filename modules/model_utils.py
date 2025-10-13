import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L
from typing import Optional,Any,List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from dataclasses import dataclass

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

@dataclass
class EBTModelArgs:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = 2
    kv_heads=2
    head_dim=32
    ffn_dim_multiplier= 4
    talking_heads = True
    dropout = True
    sparse_topk = 2
    qk_norm = False
    scale = True
    add_zero_kv = False
    flash = False
    onnxable = True
    norm_eps: float = 1e-5
    dyt_alpha_init: float = 0.5
    max_batch_size: int = 32
    max_seq_len: int = 32
    dim_head=32
    head_scale=True
    gate_values=True
    qk_norm_scale=10
    qk_norm_groups = 1
    tensor_product=True
    one_kv_head: bool = False
    weight_initialization: str = "xavier"
    adaln_zero_init: bool = True
    ebt_norm: str = "rms"
    ebt_act_func: str = "silu"
    weight_initialization_gain: float = 1.0
    head_dim: Optional[int] = 32
    #kv_head_dim:Optional[int] = 128
     # --- Attention ---
    heads: int = 8
    causal: bool = True
    cross_attend: bool = False
    only_cross: bool = False
    num_memory_tokens: Optional[int] = 16
    # --- Normalization ---
    use_scalenorm: bool = False
    use_rmsnorm: bool = False
    use_simple_rmsnorm: bool = False
    
    # --- Positional Biases ---
    alibi_pos_bias: bool = True
    alibi_num_heads: Optional[int] = n_heads
    rel_pos_bias: bool = False
    rel_pos_num_buckets: int = 32
    rel_pos_max_distance: int = 128
    dynamic_pos_bias: bool = False
    dynamic_pos_bias_log_distance: bool = False
    dynamic_pos_bias_mlp_depth: int = 2
    dynamic_pos_bias_norm: bool = False

    # --- Rotary Embeddings (RoPE) ---
    rotary_pos_emb: bool = True
    rotary_emb_dim: Optional[int] = dim//2
    rotary_xpos: bool = True
    rotary_interpolation_factor: float = 1.0
    rotary_xpos_scale_base: int = 256
    rotary_base_rescale_factor: float = 1.0

    # --- Layer Settings ---
    custom_layers: Optional[List[Any]] = None
    sandwich_coef: Optional[int] = None
    par_ratio: Optional[float] = None
    weight_tie_layers: bool = False   # ALBERT
    layers_execute_order: Optional[List[int]] = None
    residual_attn: bool = True
    cross_residual_attn: bool = False
    macaron: bool = False
    pre_norm: bool = True
    pre_norm_has_final_norm: bool = True
    gate_residual: bool = False
    scale_residual: bool = False
    scale_residual_constant: float = 1.0
    shift_tokens: int = 0
    sandwich_norm: bool = False
    resi_dual: bool = True
    resi_dual_scale: float = 1.0
    zero_init_branch_output: bool = False

    # --- Dropouts ---
    layer_dropout: float = 0.0
    cross_attn_tokens_dropout: float = 0.0
    shift_mem_down = 0
    # --- Outputs ---
    return_embeddings: bool = False
    return_logits_and_embeddings: bool = False
    return_intermediates: bool = True
    return_mems: bool = False
    return_attn: bool = False
    return_attn_z_loss: bool = False
    attn_z_loss_weight: float = 1e-4
    flash_attn: bool = False
    # --- Caching / Memory ---
    seq_start_pos: Optional[int] = None
    
    cache: Optional[Any] = None
    num_mem: int = 2  # default memory kv
    shift_mem_down: int = 0
    # --- Extra embeddings ---
    mask: Optional[Any] = None
    pos: Optional[Any] = None
    prepend_embeds: Optional[Any] = None
    sum_embeds: Optional[Any] = None
    max_mem_len: int = 32  # veya ihtiyacınıza göre 0, 16, 64
    emb_frac_gradient : float = 1.0  # 1.0 ise tam gradyan, 0.0 ise gradyans yok
    n_routed_experts: Optional[int] = 32

    qk_norm=True
    qk_norm_groups=1
    qk_norm_scale=10
    qk_norm_dim_scale=True

model_sizes = { # small -> xl same as mamba https://arxiv.org/pdf/2312.00752; all others estimated empirically. LRs based off mamba where applicable
    "4xs": { # LR 0.0024 recommended
        "num_transformer_blocks": 2,
        "multiheaded_attention_heads": 2,
        "embedding_dim": 128,
    },
    "3xs": { # LR 0.0018
        "num_transformer_blocks": 4,
        "multiheaded_attention_heads": 4,
        "embedding_dim": 256,
    },
    "xxs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "2xs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "xs": { # LR 0.0009
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "small": { # LR 0.0006
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "medium": { # 0.0003
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "large": { # 0.00025
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1536,
    },
    "xl": { # 0.0002
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 32,
        "embedding_dim": 2048,
    }
}


class ResidualBlock(L.LightningModule):
    def __init__(self,hidden_size:Optional[int],dropout_rate):
        super().__init__()
       
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.GELU("tanh")
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        residue=x
        x=self.linear(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=residue+x
        return x
    
def find_subsequences(input_tensor, sub_seq):
    sub_seq_len = len(sub_seq)
    batch_size, seq_len = input_tensor.shape
    sub_seq_tensor = torch.tensor(sub_seq, device=input_tensor.device)
    sub_seq_tensor = sub_seq_tensor.view(1, -1)
    windows = input_tensor.unfold(1, sub_seq_len, 1)
    matches = (windows == sub_seq_tensor).all(dim=2).long()
    
    if not matches.any(dim=1).all():
        raise ValueError("Sub-sequence not found in one or more sequences.")
    
    start_positions = matches.argmax(dim=1)
    return start_positions

def mask_q_tokens(input_tensor, tokenizer):
    '''
    input_tensor = [batch size, seq len]
    '''
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    answer_tag = tokenizer.encode("[[Answer]]:", add_special_tokens=True)
    
    answer_start_pos = find_subsequences(input_tensor, answer_tag)
    answer_start_pos += len(answer_tag)
    mask = torch.arange(seq_length, device=input_tensor.device).expand(batch_size, seq_length)
    mask = mask < answer_start_pos.unsqueeze(1)
    input_tensor = torch.where(mask, tokenizer.pad_token_id, input_tensor)
    
    return input_tensor
class AdvancedDropout(nn.Module):

    def __init__(self, num, init_mu=0, init_sigma=1.2, reduction=16):
        '''
        params:
        num (int): node number
        init_mu (float): intial mu
        init_sigma (float): initial sigma
        reduction (int, power of two): reduction of dimention of hidden states h
        '''
        super(AdvancedDropout, self).__init__()
        if init_sigma <= 0:
            raise ValueError("Sigma has to be larger than 0, but got init_sigma=" + str(init_sigma))
        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.weight_h = nn.Parameter(torch.rand([num // reduction, num]).mul(0.01))
        self.bias_h = nn.Parameter(torch.rand([1]).mul(0.01))

        self.weight_mu = nn.Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_mu = nn.Parameter(torch.Tensor([self.init_mu]))
        self.weight_sigma = nn.Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_sigma = nn.Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        if self.training:
            b ,c,n = input.size()
            # parameterized prior
            h = F.linear(input, self.weight_h, self.bias_h)
            mu = F.linear(h, self.weight_mu, self.bias_mu).mean()
            sigma = F.softplus(F.linear(h, self.weight_sigma, self.bias_sigma)).mean()
            # mask
            epsilon = mu + sigma * torch.randn([c, n]).to(device)
            mask = torch.sigmoid(epsilon)
      
            out = input.mul(mask).div(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.))).to(device)
        else:
            out = input

        return out

class RMSNorm(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
    
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)

import math
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

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


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

def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class MLP(nn.Module):
    def __init__(self, input_modality_size, final_size,dropout_rate=0.1, layer_norm=True, memory_size=32 ,num_hidden_layers=2):
        super().__init__() 
        input_modality_hidden_size = input_modality_size //2 
        self.add_residual_connections = True  # Residual connections are always on by default
        self.input_layers = nn.ModuleList()
        self.memory_size = memory_size
        self.input_layers.append(nn.Linear(input_modality_size,input_modality_hidden_size))
        if layer_norm:
                self.input_layers.append(nn.LayerNorm(input_modality_hidden_size))
        self.input_layers.append(nn.ReLU())
        self.input_layers.append(AdvancedDropout(input_modality_hidden_size))
        self.memory = nn.Embedding(1, memory_size)

        self.fusion_layers = nn.ModuleList()

        self.fusion_hidden_size = input_modality_hidden_size + memory_size

        for i in range(1,num_hidden_layers-1):
            add_residual = self.add_residual_connections and i % 2 == 0
            if add_residual:
                self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size,dropout_rate))
            else:
                self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, self.fusion_hidden_size, bias=False))
                self.fusion_layers.append(nn.GELU("tanh"))
        
        self.fusion_layers.append(AdvancedDropout(self.fusion_hidden_size))

        if final_size == self.fusion_hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
        else:
            self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, final_size, bias=False))

    def forward(self,x):
        B,T,C=x.shape
        for i ,layer in enumerate(self.input_layers):
            x=layer(x)
        

        memory_embedding = self.memory.weight[0]
        
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, self.memory_size)
        fused_x = torch.cat((x, memory_embedding), dim=-1) # B, S, H + M (where H is hidden size M is memory size)

        for idx,layer in enumerate(self.fusion_layers):
            fused_x=layer(fused_x)
                
        return fused_x


    
class Memory_Gating_MLP(nn.Module):
    def __init__(self, input_modality_size, embedding_dim, process_memory_type = "add", process_memory_linear_layer = False):
        super(Memory_Gating_MLP, self).__init__()
        self.vocab_to_embed = nn.Linear(input_modality_size, embedding_dim, bias = False)
        self.memory = nn.Embedding(1, embedding_dim)
        self.process_memory_linear_layer = process_memory_linear_layer
        if self.process_memory_linear_layer:
            self.memory_linear_layer = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.process_memory_type = process_memory_type

    def forward(self, x):
        vocab_embeds = self.vocab_to_embed(x)
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand_as(vocab_embeds)
        if self.process_memory_linear_layer:
            memory_embedding = self.memory_linear_layer(memory_embedding)
        
        if self.process_memory_type == "add":
            final_embeddings = vocab_embeds + memory_embedding
        elif self.process_memory_type == "gate":
            final_embeddings = vocab_embeds * memory_embedding
        elif self.process_memory_type == "residual_gate":
            final_embeddings = vocab_embeds + vocab_embeds * memory_embedding
        else:
            raise NotImplementedError(f"self.process_memory_type {self.process_memory_type} not yet implemented")

        return final_embeddings

if __name__ == "__main__":
    model = MLP(input_modality_size=512, final_size=256, dropout_rate=0.1, layer_norm=True, memory_size=32 ,num_hidden_layers=4)
    x = torch.randn(8, 10, 512)  # Example input tensor with shape (batch_size, sequence_length, input_modality_size)
    output = model(x)
    print(output.shape)  # Should print torch.Size([8, 10, 256])

def get_activation(name):
    activations=dict()
    def hook(model, input, output):
        activations[name] = output.detach()   # detach() ile gradyan zincirinden çıkarıyoruz
        print(f"{name} -> {output.shape}")
    return hook

def calc_out_of_bounds_loss(energy): # gives loss for < 0 or > 1
    lower_bound_loss = torch.abs(energy)
    upper_bound_loss = torch.abs(energy - 1)
    loss = torch.where(energy < 0, lower_bound_loss, 
                    torch.where(energy > 1, upper_bound_loss, torch.zeros_like(energy)))
    loss = torch.mean(loss)
    
    return loss

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def init_whole_model_weights(model, weight_initialization_method, nonlinearity='linear', weight_initialization_gain=1.0):


    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                valid_nonlinearities = ['linear', 'relu', 'leaky_relu', 'selu', 'tanh']
                if nonlinearity not in valid_nonlinearities:
                    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}. Must be one of {valid_nonlinearities}")
                
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "zero":
                nn.init.constant_(m.weight, 0.)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.) 
            else:
                raise ValueError(f"Unknown weight init method: {weight_initialization_method}")
    
    model.apply(init_weights)




def mask_q_tokens(input_tensor, tokenizer):
    '''
    input_tensor = [batch size, seq len]
    '''
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    answer_tag = tokenizer.encode("[[Answer]]:", add_special_tokens=True)
    
    answer_start_pos = find_subsequences(input_tensor, answer_tag)
    answer_start_pos += len(answer_tag)
    mask = torch.arange(seq_length, device=input_tensor.device).expand(batch_size, seq_length)
    mask = mask < answer_start_pos.unsqueeze(1)
    input_tensor = torch.where(mask, tokenizer.pad_token_id, input_tensor)
    
    return input_tensor

def analyse_tokens(input_tensor, tokenizer):
    '''for debugging only'''
    decode = tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    for i in range(input_tensor.shape[0]):
        print(input_tensor[i].tolist())
        print(decode[i])
        print('-'*60)

def setup_ebt(hparams): # specifically for EBT not for baseline transformer
    # to prevent circular import

    from main_transformer import EBTAdaLN
    max_seq_len = hparams.context_length+1 # for next pred in context 
    max_seq_len = max_seq_len + 1 if hparams.ebt_type == "time_embed" else max_seq_len # need +1 since cat time embed on sequence dim

    adaln_zero_init = True if hparams.ebt_type == "adaln_zero" else False
    transformer_args = EBTModelArgs()
    
    ebt = EBTAdaLN(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)

    return ebt

def setup_transformer(hparams): # specifically for baseline transformer
    from model.ar_transformer import Transformer, TransformerModelArgs
    transformer_args = TransformerModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=hparams.context_length, weight_initialization = hparams.weight_initialization_method, ffn_dim_multiplier=hparams.ffn_dim_multiplier, weight_initialization_gain=hparams.weight_initialization_gain)
    transformer = Transformer(params=transformer_args)
    return transformer

def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq):
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log="all", log_freq = wandb_watch_log_freq)
    
    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm) 
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        # print("\nNon-LayerNorm modules:")
        # for name, _ in non_layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        # print("\nLayerNorm modules:")
        # for name, _ in layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        wandb_logger.watch(non_layernorm_container, log="all", log_freq=wandb_watch_log_freq)
        wandb_logger.watch(layernorm_container, log="parameters", log_freq=wandb_watch_log_freq)




def setup_bidirectional_ebt(hparams):
    from model.bi_ebt_adaln import EBT
    assert hparams.image_dims[0] == hparams.image_dims[1], "need to use square image with current implementation"
    
    if hparams.image_task == "denoising":
        # For denoising task, use raw image dimensions (no VAE)
        input_size = hparams.image_dims[0]
        in_channels = 3  # RGB channels for raw images
    else:
        # For other tasks using VAE
        assert hparams.image_dims[0] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        input_size = hparams.image_dims[0] // 8
        in_channels = 4
    
    ebt = EBT(
        input_size=input_size,
        patch_size=hparams.patch_size,
        in_channels=in_channels,
        hidden_size=hparams.embedding_dim,
        depth=hparams.num_transformer_blocks,
        num_heads=hparams.multiheaded_attention_heads,
        mlp_ratio=hparams.ffn_dim_multiplier
    )
    
    return ebt


def get_text_embeddings(text_encoder, captions):
    with torch.no_grad():
        batch_outputs = text_encoder(**captions)
        caption_embeddings = batch_outputs.pooler_output
        return caption_embeddings

# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# note this is unused for text conditional instead of class conditional https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



# all pos enc functions from below are from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py, from DiT codebase
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out



class Memory_Augmented_MLP(nn.Module):
    def __init__(self, input_modality_size, memory_size, input_modality_hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(Memory_Augmented_MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.input_layers = nn.ModuleList()
        self.memory_size = memory_size

        # Initial layer
        self.input_layers.append(nn.Linear(input_modality_size, input_modality_hidden_size, bias=False))
        if layer_norm:
            self.input_layers.append(nn.LayerNorm(input_modality_hidden_size))
        self.input_layers.append(nn.ReLU())
        self.input_layers.append(nn.Dropout(dropout_rate))

        self.memory = nn.Embedding(1, memory_size)

        self.fusion_layers = nn.ModuleList()

        self.fusion_hidden_size = input_modality_hidden_size + memory_size

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
            else:
                self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, self.fusion_hidden_size, bias=False))
                self.fusion_layers.append(nn.ReLU())

            self.fusion_layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == self.fusion_hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
        else:
            self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, final_size, bias=False))

    def forward(self, x):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        for layer in self.input_layers:
            x = layer(x)
        
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_length, self.memory_size)
        fused_x = torch.cat((x, memory_embedding), dim=-1) # B, S, H + M (where H is hidden size M is memory size)
        
        for layer in self.fusion_layers:
            fused_x = layer(fused_x)
        
        return fused_x
    
class Memory_Gating_MLP(nn.Module):
    def __init__(self, input_modality_size, embedding_dim, process_memory_type = "add", process_memory_linear_layer = False):
        super(Memory_Gating_MLP, self).__init__()
        self.vocab_to_embed = nn.Linear(input_modality_size, embedding_dim, bias = False)
        self.memory = nn.Embedding(1, embedding_dim)
        self.process_memory_linear_layer = process_memory_linear_layer
        if self.process_memory_linear_layer:
            self.memory_linear_layer = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.process_memory_type = process_memory_type

    def forward(self, x):
        vocab_embeds = self.vocab_to_embed(x)
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand_as(vocab_embeds)
        if self.process_memory_linear_layer:
            memory_embedding = self.memory_linear_layer(memory_embedding)
        
        if self.process_memory_type == "add":
            final_embeddings = vocab_embeds + memory_embedding
        elif self.process_memory_type == "gate":
            final_embeddings = vocab_embeds * memory_embedding
        elif self.process_memory_type == "residual_gate":
            final_embeddings = vocab_embeds + vocab_embeds * memory_embedding
        else:
            raise NotImplementedError(f"self.process_memory_type {self.process_memory_type} not yet implemented")

        return final_embeddings