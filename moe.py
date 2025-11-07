
##A lot of inspiration was taken from https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek.py

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable
from  modules.model_utils import init_whole_model_weights
import pytorch_lightning as L
from feed_forward import *
from core.losses import AddAuxiliaryLoss
@dataclass
class MoeConfig:
    vocab_size = 102400
    hidden_size = 256  
    intermediate_size = 4 * 256 
    moe_intermediate_size = 1407
    num_hidden_layers = 12
    num_attention_heads = 8
    num_key_value_heads = 4
    n_shared_experts = None
    n_routed_experts = 32
    ep_size = 1
    routed_scaling_factor = 1.0
    kv_lora_rank = 512
    q_lora_rank = 1536
    qk_rope_head_dim = 64
    v_head_dim = 256
    qk_nope_head_dim = 128
    topk_method = 'greedy'
    n_group = None
    topk_group = None
    num_experts_per_tok = 1
    moe_layer_freq = 1
    first_k_dense_replace = 0
    norm_topk_prob = False
    scoring_func = 'softmax'
    aux_loss_alpha = 0.001
    seq_aux = True
    hidden_act = "silu"
    max_position_embeddings = 2048
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    pad_token_id = None
    bos_token_id = 100000
    eos_token_id = 100001
    pretraining_tp = 1
    tie_word_embeddings = False
    rope_theta = 10000.0
    rope_scaling = None
    attention_bias = False
    attention_dropout = 0.0
    use_mla = True




class MoEGate(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = 0  # Varsayılan rank
            
            self.experts = nn.ModuleList(
                [
                    (
                        SwigluFFN(
                            in_features=config.hidden_size,
                            hidden_features=config.moe_intermediate_size,
                            out_features=config.hidden_size,
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            
            self.experts = nn.ModuleList(
                [
                    SwigluFFN(
                        in_features=config.hidden_size,
                        hidden_features=config.moe_intermediate_size,
                        out_features=config.hidden_size,
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        
        self.gate = MoEGate(config)
        
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = SwigluFFN(
                in_features=config.hidden_size,
                hidden_features=config.intermediate_size,
                out_features=config.hidden_size,
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                if expert is not None:
                    y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()

            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = torch.zeros(shape=(gathered_tokens.shape[0],), dtype=torch.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s: s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            if expert is not None:
                tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
                expert_out = expert(tokens_for_this_expert)
                outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = MoeConfig()
    moe = DeepseekV2MoE(config).to(device)
    
    #mock kod bebek gibi
    x = torch.randn(1,128,256,device=device)
    
    print("Input shape:", x.shape)
    output = moe(x)
    print("Output shape:", output.shape)