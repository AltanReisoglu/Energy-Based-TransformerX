import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import math
import random

# ============================================================================
# YENİ EKLEME 1: Adaptive Step Size Controller
# ======================================== == == == == == == == == == == == == == == == == == ==
class AdaptiveStepSizeController(nn.Module):
    """
    Her MCMC adımında step size'ı dinamik ayarlar.
    - Enerji düşüş hızına göre alpha'yı otomatik ayarla
    - Gradient magnitude'a göre adaptation
    - Momentum tabanlı smooth update
    """
    def __init__(self, initial_alpha=0.01, min_alpha=0.0001, max_alpha=0.1):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Momentum için exponential moving average
        self.ema_energy_decrease = None
        self.ema_beta = 0.9
        
        # Her sequence position için ayrı alpha öğrenilebilir
        self.position_weights = nn.Parameter(torch.ones(1))
        
    def forward(self, current_energy, previous_energy, gradient_norm, step_idx, total_steps):
        """
        Adaptive step size hesapla
        
        Args:
            current_energy: Mevcut enerji değeri
            previous_energy: Önceki enerji değeri
            gradient_norm: Gradient'in normu
            step_idx: Mevcut adım indeksi
            total_steps: Toplam adım sayısı
        """
        batch_size = current_energy.shape[0]
        
        # 1. Enerji düşüş oranına göre ayarlama
        energy_decrease = previous_energy - current_energy
        if self.ema_energy_decrease is None:
            self.ema_energy_decrease = energy_decrease.mean()
        else:
            self.ema_energy_decrease = (self.ema_beta * self.ema_energy_decrease + 
                                       (1 - self.ema_beta) * energy_decrease.mean())
        
        # 2. Eğer enerji yeterince düşmüyorsa, alpha'yı artır
        # Eğer çok hızlı düşüyorsa, alpha'yı azalt (stability için)
        energy_ratio = energy_decrease / (torch.abs(previous_energy) + 1e-8)
        
        # 3. Gradient magnitude'a göre adaptation
        # Büyük gradient -> küçük step (overshoot önleme)
        # Küçük gradient -> büyük step (daha hızlı convergence)
        grad_factor = 1.0 / (1.0 + gradient_norm)
        
        # 4. Annealing: Başlangıçta büyük, sonra küçül
        annealing_factor = 1.0 - (step_idx / total_steps) * 0.5
        
        # 5. Tüm faktörleri birleştir
        adaptive_alpha = self.initial_alpha * grad_factor * annealing_factor
        
        # Energy decrease çok düşükse artır, çok yüksekse azalt
        energy_adjustment = torch.clamp(1.0 + energy_ratio * 0.5, 0.5, 2.0)
        adaptive_alpha = adaptive_alpha * energy_adjustment
        
        # Position-specific weighting
        adaptive_alpha = adaptive_alpha * torch.sigmoid(self.position_weights)
        
        # Min/max bounds
        adaptive_alpha = torch.clamp(adaptive_alpha, self.min_alpha, self.max_alpha)
        
        return adaptive_alpha


# ============================================================================
# YENİ EKLEME 2: Dynamic Temperature Scaling
# ============================================================================
class DynamicTemperatureScheduler(nn.Module):
    """
    Softmax temperature'ı dinamik ayarlar.
    - Başlangıçta yüksek temperature (exploration)
    - Sonra düşük temperature (exploitation)
    - Uncertainty'e göre adaptasyon
    """
    def __init__(self, initial_temp=1.0, min_temp=0.5, max_temp=2.0):
        super().__init__()
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        
        # Learnable temperature parameters
        self.temp_scale = nn.Parameter(torch.tensor(1.0))
        self.temp_bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, logits, step_idx, total_steps, entropy=None):
        """
        Adaptive temperature hesapla
        
        Args:
            logits: Model çıktısı [B, S, V]
            step_idx: Mevcut MCMC adımı
            total_steps: Toplam MCMC adım sayısı
            entropy: Mevcut distribution'ın entropisi (opsiyonel)
        """
        # 1. Annealing schedule: başlangıçta exploration, sonra exploitation
        progress = step_idx / total_steps
        annealing_temp = self.max_temp - (self.max_temp - self.min_temp) * progress
        
        # 2. Entropy-based adaptation
        if entropy is not None:
            # Düşük entropy -> model emin -> temperature düşür
            # Yüksek entropy -> model belirsiz -> temperature yükselt
            entropy_normalized = entropy / math.log(logits.shape[-1])  # Normalize [0,1]
            entropy_factor = 0.5 + entropy_normalized * 0.5
        else:
            entropy_factor = 1.0
        
        # 3. Learnable component
        learned_temp = self.temp_scale * annealing_temp + self.temp_bias
        
        # 4. Combine
        final_temp = torch.clamp(learned_temp * entropy_factor, self.min_temp, self.max_temp)
        
        return final_temp


# ============================================================================
# YENİ EKLEME 3: Uncertainty-Aware Gradient Clipping
# ============================================================================
class UncertaintyAwareGradientClipper(nn.Module):
    """
    Gradient clipping'i model uncertainty'sine göre ayarlar.
    - Model emin olduğunda: Daha büyük gradientler (hızlı convergence)
    - Model belirsiz olduğunda: Daha küçük gradientler (stability)
    """
    def __init__(self, base_clip_value=1.0):
        super().__init__()
        self.base_clip_value = base_clip_value
        self.uncertainty_history = []
        self.max_history = 100
        
    def compute_uncertainty(self, logits):
        """
        Distribution uncertainty'sini hesapla
        - Entropy kullanarak
        - Confidence score ile
        """
        probs = F.softmax(logits, dim=-1)
        
        # 1. Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = math.log(logits.shape[-1])
        normalized_entropy = entropy / max_entropy
        
        # 2. Confidence (max probability)
        max_probs, _ = probs.max(dim=-1)
        uncertainty_from_conf = 1.0 - max_probs
        
        # Combine
        uncertainty = (normalized_entropy + uncertainty_from_conf) / 2.0
        
        return uncertainty.mean()
    
    def forward(self, gradients, logits):
        """
        Uncertainty-aware gradient clipping uygula
        
        Args:
            gradients: Clip edilecek gradientler
            logits: Model çıktısı (uncertainty hesaplamak için)
        """
        uncertainty = self.compute_uncertainty(logits)
        
        # Uncertainty'i history'e ekle
        self.uncertainty_history.append(uncertainty.item())
        if len(self.uncertainty_history) > self.max_history:
            self.uncertainty_history.pop(0)
        
        # Adaptive clip value
        # Yüksek uncertainty -> küçük clip (dikkatli ol)
        # Düşük uncertainty -> büyük clip (hızlı git)
        uncertainty_factor = 1.0 - uncertainty * 0.5
        adaptive_clip_value = self.base_clip_value * uncertainty_factor
        
        # Clip
        clipped_gradients = torch.clamp(
            gradients, 
            min=-adaptive_clip_value, 
            max=adaptive_clip_value
        )
        
        return clipped_gradients, uncertainty


# ============================================================================
# YENİ EKLEME 4: Multi-Scale MCMC
# ============================================================================
class MultiScaleMCMC(nn.Module):
    """
    Farklı resolution'larda MCMC yap:
    - Coarse scale: Hızlı exploration (büyük adımlar)
    - Fine scale: Detaylı optimization (küçük adımlar)
    - Hierarchical refinement
    """
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        
        # Her scale için farklı alpha
        self.scale_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1 / (2**i))) for i in range(num_scales)
        ])
        
    def forward(self, initial_state, energy_fn, steps_per_scale=5):
        """
        Multi-scale MCMC optimization
        
        Args:
            initial_state: Başlangıç durumu
            energy_fn: Enerji hesaplama fonksiyonu
            steps_per_scale: Her scale'de kaç adım
        """
        current_state = initial_state
        all_states = []
        
        # Coarse-to-fine: Büyük adımlardan küçük adımlara
        for scale_idx in range(self.num_scales):
            alpha = self.scale_alphas[scale_idx]
            
            for step in range(steps_per_scale):
                current_state = current_state.detach().requires_grad_()
                energy = energy_fn(current_state)
                
                grad = torch.autograd.grad(energy.sum(), current_state, create_graph=True)[0]
                
                # Scale-specific noise
                noise_scale = alpha * 0.1
                noise = torch.randn_like(current_state) * noise_scale
                
                current_state = current_state - alpha * grad + noise
                all_states.append(current_state.detach())
        
        return current_state, all_states


# ============================================================================
# YENİ EKLEME 5: Attention-Weighted Energy Aggregation
# ============================================================================
class AttentionWeightedEnergy(nn.Module):
    """
    Farklı token pozisyonlarının enerjilerini attention ile ağırlıklandır.
    - Önemli pozisyonlara daha fazla ağırlık
    - Context-aware energy computation
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, embeddings, energies):
        """
        Args:
            embeddings: [B, S, D] token embeddings
            energies: [B, S] token energies
        """
        # Attention weights hesapla
        attention_logits = self.attention_net(embeddings)  # [B, S, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, S, 1]
        
        # Weighted energy
        weighted_energy = (energies.unsqueeze(-1) * attention_weights).sum(dim=1)
        
        return weighted_energy, attention_weights


# ============================================================================
# YENİ EKLEME 6: Early Stopping Based on Energy Convergence
# ============================================================================
class EnergyConvergenceMonitor:
    """
    Enerji convergence'ını izle ve erken dur.
    - Hesaplama tasarrufu
    - Gereksiz iterasyonları önle
    """
    def __init__(self, patience=3, threshold=0.001):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.energy_history = []
        
    def should_stop(self, current_energy):
        """
        Erken durma kararı ver
        
        Returns:
            bool: Durmalı mı?
        """
        self.energy_history.append(current_energy.mean().item())
        
        # Yeterli history yok
        if len(self.energy_history) < self.patience + 1:
            return False
        
        # Son N adımda enerji değişimi
        recent_energies = self.energy_history[-self.patience-1:]
        energy_changes = [abs(recent_energies[i] - recent_energies[i+1]) 
                         for i in range(len(recent_energies)-1)]
        
        # Tüm değişimler threshold'dan küçükse dur
        if all(change < self.threshold for change in energy_changes):
            return True
        
        return False
    
    def reset(self):
        self.energy_history = []


# ============================================================================
# YENİ EKLEME 7: Curriculum Learning for MCMC Steps
# ============================================================================
class MCMCCurriculumScheduler:
    """
    Eğitim ilerledikçe MCMC adım sayısını artır.
    - Başlangıç: Az adım (hızlı eğitim)
    - İleri aşama: Çok adım (kaliteli optimization)
    """
    def __init__(self, min_steps=1, max_steps=10, warmup_epochs=5):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.warmup_epochs = warmup_epochs
        
    def get_num_steps(self, current_epoch):
        """
        Mevcut epoch için kaç MCMC adımı kullanılacak
        """
        if current_epoch < self.warmup_epochs:
            # Linear increase
            progress = current_epoch / self.warmup_epochs
            num_steps = int(self.min_steps + (self.max_steps - self.min_steps) * progress)
        else:
            num_steps = self.max_steps
        
        return num_steps


# ============================================================================
# KULLANIM ÖRNEĞİ: Enhanced EBT Forward Pass
# ============================================================================
class EnhancedEBTForward:
    """
    Tüm adaptive özellikleri entegre eden forward pass örneği
    """
    def __init__(self, model, hparams):
        self.model = model
        self.hparams = hparams
        
        # Initialize adaptive components
        self.step_controller = AdaptiveStepSizeController(
            initial_alpha=hparams.mcmc_step_size
        )
        self.temp_scheduler = DynamicTemperatureScheduler()
        self.grad_clipper = UncertaintyAwareGradientClipper()
        self.convergence_monitor = EnergyConvergenceMonitor()
        self.curriculum = MCMCCurriculumScheduler()
        
    def forward_with_adaptations(self, x, current_epoch):
        """
        Tüm adaptive özellikleri kullanarak forward pass
        """
        # 1. Curriculum learning ile adım sayısı belirle
        num_steps = self.curriculum.get_num_steps(current_epoch)
        
        # 2. İlk durum
        embeddings = self.model.embeddings(x)
        predicted_tokens = self.model.corrupt_embeddings(embeddings)
        
        previous_energy = torch.tensor(float('inf'))
        
        for step in range(num_steps):
            # 3. Adaptive temperature ile softmax
            temp = self.temp_scheduler(predicted_tokens, step, num_steps)
            predicted_tokens_normalized = F.softmax(predicted_tokens / temp, dim=-1)
            
            # 4. Energy hesapla
            predicted_embeddings = self.model.vocab_to_embed(predicted_tokens_normalized)
            all_embeddings = torch.cat([embeddings, predicted_embeddings], dim=1)
            energy = self.model.transformer(all_embeddings)
            
            # 5. Gradient hesapla
            predicted_tokens = predicted_tokens.detach().requires_grad_()
            grad = torch.autograd.grad(energy.sum(), predicted_tokens, create_graph=True)[0]
            
            # 6. Uncertainty-aware gradient clipping
            grad_norm = torch.norm(grad, p=2, dim=-1).mean()
            clipped_grad, uncertainty = self.grad_clipper(grad, predicted_tokens)
            
            # 7. Adaptive step size
            adaptive_alpha = self.step_controller(
                energy, previous_energy, grad_norm, step, num_steps
            )
            
            # 8. Update
            predicted_tokens = predicted_tokens - adaptive_alpha * clipped_grad
            
            # 9. Convergence check
            if self.convergence_monitor.should_stop(energy):
                print(f"Early stopping at step {step}")
                break
            
            previous_energy = energy.detach()
        
        return predicted_tokens, energy


# ============================================================================
# KULLANIM NOTU
# ============================================================================
"""
Bu adaptive özellikler sayesinde model:

1. **Daha Hızlı Converge Olur**: 
   - Adaptive step size ile optimal hızda ilerler
   - Early stopping ile gereksiz hesaplama yapılmaz

2. **Daha Stabil**: 
   - Uncertainty-aware clipping patlamayı önler
   - Dynamic temperature exploration/exploitation dengesini sağlar

3. **Daha Akıllı Öğrenir**:
   - Curriculum learning ile kolay->zor progression
   - Multi-scale MCMC ile global + local optimization

4. **Daha Az Hyperparameter Tuning**:
   - Çoğu parametre otomatik ayarlanır
   - Manuel tuning ihtiyacı azalır

ENTEGRASYON:
Bu modülleri mevcut EBT_NLP sınıfına ekleyerek kullanabilirsiniz.
Her modül bağımsız çalışır, istediğinizi seçip kullanabilirsiniz.
"""