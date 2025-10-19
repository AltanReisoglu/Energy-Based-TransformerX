import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import pytorch_lightning as L
from typing import *
from collections import defaultdict
from modules.model_utils import init_whole_model_weights
from main_transformer import *
import math
from collections import namedtuple
from dataclasses import dataclass
from functools import partial, wraps
from inspect import isfunction
from random import random
from typing import Callable, List, Optional, Tuple

import numpy as np
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

class ModelAnalyzer:
    """
    PyTorch modellerini analiz eden kapsamlı araç.
    Parametre sayısı, bellek kullanımı, layer detayları ve daha fazlasını gösterir.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.param_stats = self._calculate_parameters()
        
    def _calculate_parameters(self) -> Dict:
        """Tüm parametreleri hesapla"""
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        layer_params = []
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params
                
            layer_params.append({
                'name': name,
                'shape': tuple(param.shape),
                'params': num_params,
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            })
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params,
            'layers': layer_params
        }
    
    def _format_number(self, num: int) -> str:
        """Sayıyı okunabilir formata çevir"""
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    def print_summary(self, detailed: bool = True):
        """Model özetini yazdır"""
        print("\n" + "="*80)
        print(f"{'MODEL PARAMETER SUMMARY':^80}")
        print("="*80)
        
        # Genel istatistikler
        stats = self.param_stats
        print(f"\n{'Total Parameters:':<30} {stats['total']:>20,} ({self._format_number(stats['total'])})")
        print(f"{'Trainable Parameters:':<30} {stats['trainable']:>20,} ({self._format_number(stats['trainable'])})")
        print(f"{'Non-trainable Parameters:':<30} {stats['non_trainable']:>20,} ({self._format_number(stats['non_trainable'])})")
        
        # Bellek hesaplamaları
        print(f"\n{'='*80}")
        print(f"{'MEMORY ESTIMATES':^80}")
        print(f"{'='*80}")
        
        memory_fp32 = (stats['total'] * 4) / (1024**2)  # MB
        memory_fp16 = memory_fp32 / 2
        memory_int8 = memory_fp32 / 4
        
        print(f"{'FP32 (32-bit):':<30} {memory_fp32:>15.2f} MB")
        print(f"{'FP16 (16-bit):':<30} {memory_fp16:>15.2f} MB")
        print(f"{'INT8 (8-bit):':<30} {memory_int8:>15.2f} MB")
        
        # Detaylı layer bilgisi
        if detailed and stats['layers']:
            print(f"\n{'='*80}")
            print(f"{'LAYER-BY-LAYER BREAKDOWN':^80}")
            print(f"{'='*80}")
            print(f"{'Layer Name':<45} {'Shape':<20} {'Parameters':>14}")
            print("-"*80)
            
            for layer in stats['layers']:
                trainable_mark = "✓" if layer['trainable'] else "✗"
                shape_str = str(layer['shape'])
                print(f"{layer['name']:<45} {shape_str:<20} {layer['params']:>13,} {trainable_mark}")
        
        print("="*80 + "\n")
    
    def get_layer_groups(self) -> Dict[str, int]:
        """Layer'ları gruplara ayır ve parametre sayılarını topla"""
        groups = defaultdict(int)
        
        for layer in self.param_stats['layers']:
            # Layer isminin ilk kısmını al (ör: "encoder.layer.0" -> "encoder")
            layer_type = layer['name'].split('.')[0]
            groups[layer_type] += layer['params']
        
        return dict(groups)
    
    def print_layer_groups(self):
        """Gruplandırılmış layer istatistiklerini yazdır"""
        groups = self.get_layer_groups()
        
        print("\n" + "="*60)
        print(f"{'PARAMETERS BY LAYER GROUP':^60}")
        print("="*60)
        print(f"{'Group':<30} {'Parameters':>15} {'Percentage':>12}")
        print("-"*60)
        
        total = self.param_stats['total']
        sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
        
        for group, params in sorted_groups:
            percentage = (params / total) * 100
            print(f"{group:<30} {params:>15,} {percentage:>11.1f}%")
        
        print("="*60 + "\n")
    
    def compare_with_input(self, input_shape: Tuple):
        """Input boyutu ile karşılaştırma yap"""
        input_size = np.prod(input_shape)
        param_count = self.param_stats['total']
        ratio = param_count / input_size
        
        print("\n" + "="*60)
        print(f"{'INPUT vs PARAMETERS':^60}")
        print("="*60)
        print(f"{'Input shape:':<30} {str(input_shape)}")
        print(f"{'Input elements:':<30} {input_size:>15,}")
        print(f"{'Model parameters:':<30} {param_count:>15,}")
        print(f"{'Param/Input ratio:':<30} {ratio:>15.2f}x")
        print("="*60 + "\n")
    
    def estimate_training_memory(self, batch_size: int, sequence_length: int = None):
        """Training sırasında gereken belleği tahmin et"""
        params = self.param_stats['total']
        
        # Parametreler (FP32)
        param_memory = (params * 4) / (1024**2)
        
        # Gradients (FP32)
        grad_memory = param_memory
        
        # Optimizer states (Adam için 2x: momentum + variance)
        optimizer_memory = param_memory * 2
        
        # Activations (tahmini - model ve batch size'a göre değişir)
        if sequence_length:
            # Transformer için yaklaşık hesap
            activation_memory = (batch_size * sequence_length * 512 * 4) / (1024**2)  # Tahmini
        else:
            activation_memory = param_memory * 0.5  # Genel tahmin
        
        total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
        
        print("\n" + "="*70)
        print(f"{'TRAINING MEMORY ESTIMATE (MB)':^70}")
        print("="*70)
        print(f"{'Parameters (FP32):':<40} {param_memory:>15.2f} MB")
        print(f"{'Gradients (FP32):':<40} {grad_memory:>15.2f} MB")
        print(f"{'Optimizer States (Adam):':<40} {optimizer_memory:>15.2f} MB")
        print(f"{'Activations (estimated):':<40} {activation_memory:>15.2f} MB")
        print("-"*70)
        print(f"{'Total Estimated:':<40} {total_memory:>15.2f} MB")
        print(f"{'Total Estimated:':<40} {total_memory/1024:>15.2f} GB")
        print("="*70 + "\n")


# Demo: Örnek model oluştur ve analiz et
if __name__ == "__main__":
    import time
 
    #mock trial
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    rande=torch.randn(2,2,16,32,device=device)
    args = EBTModelArgs()
    past_cache_list=[(rande,rande),(rande,rande),(rande,rande),(rande,rande),(rande,rande),(rande,rande)]
    model = EBTAdaLN(args, max_mcmc_steps=10).to(device=device)
    x = torch.randn(2, 32, args.dim,device=device)
    out = model(x, start_pos=0, mcmc_step=0)
    print(out[0].shape)
    

    
    # Analyzer oluştur
    analyzer = ModelAnalyzer(model)
    
    # Analizleri yap
    analyzer.print_summary(detailed=True)
    analyzer.print_layer_groups()
    #analyzer.compare_with_input(input_shape=(8, 512))  # batch_size=8, seq_len=512
    #analyzer.estimate_training_memory(batch_size=8, sequence_length=512)
    
    # Özet istatistikler
    print("\n📊 QUICK STATS:")
    print(f"   Total Parameters: {analyzer._format_number(analyzer.param_stats['total'])}")
    print(f"   FP16 Memory: {(analyzer.param_stats['total'] * 2) / (1024**2):.2f} MB")
    print(f"   Number of Layers: {len(analyzer.param_stats['layers'])}")