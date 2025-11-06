# Energy-Based Transformer (EBT)

Bu proje, yeni nesil bir Energy-Based Transformer modelinin implementasyonudur. Özel yapay sinir ağı mimarisi, Lyapunov kararlılık teorisi ve adaptif öğrenme mekanizmalarını birleştirerek daha güçlü ve kararlı bir dil modeli sunmaktadır.

## Özellikler

- **AdaLN Transformer**: Adaptive Layer Normalization ile geliştirilmiş transformer mimarisi
- **Mixture of Experts (MoE)**: DeepSeek-V2 tarzı uzman ağ sistemi
- **Energy-Based Training**: Lyapunov kararlılık teorisine dayalı enerji-bazlı eğitim
- **Türkçe Dil Desteği**: Özel Türkçe tokenizer ve Türkçe veri seti desteği
- **Sliding Window Attention**: Bellek-verimli dikkat mekanizması
- **Multi-GPU Eğitim**: Dağıtık eğitim desteği (Accelerate kütüphanesi ile)

## Kurulum

### Gereksinimler

```bash
python >= 3.10
CUDA >= 11.7 (GPU desteği için)
```

### Docker ile Kurulum

```bash
# Docker image'ı oluştur
docker-compose build

# Modeli çalıştır
docker-compose up energic-model

# Jupyter notebook için (opsiyonel)
docker-compose up jupyter
```

### Manuel Kurulum

```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Bağımlılıkları yükle
pip install poetry
poetry install
```

## Kullanım

### Eğitim

```powershell
# PowerShell script ile
./run.ps1 train

# Veya direkt Python ile
python nlp/train_ebt.py
```

### Model Parametreleri

```python
hparams = dict(
    lr=1e-3,                        # Öğrenme oranı
    batch_size_per_device=2,        # Batch size
    embedding_dim=256,              # Gömme boyutu
    num_transformer_blocks=6,        # Transformer katman sayısı
    multiheaded_attention_heads=6,   # Dikkat başı sayısı
    context_length=256,             # Maksimum sekans uzunluğu
    model_max_length=256            # Model maksimum uzunluğu
)
```

## Model Mimarisi

Model üç ana bileşenden oluşur:

1. **Encoder**:
   - AdaLN Transformer blokları
   - Sliding Window Attention
   - Rotary Positional Encoding

2. **Energy-Based Training**:
   - Lyapunov kararlılık teorisi
   - Adaptif adım boyutu kontrolü
   - MCMC sampling

3. **Mixture of Experts**:
   - DeepSeek-V2 MoE katmanı
   - Router ve expert ağları
   - Load balancing

## Veri Seti

Proje, Bilkent Turkish Writings veri setini kullanmaktadır:
- Özel Türkçe tokenizer
- Yapılandırılmış veri önişleme
- Bellek-verimli veri yükleme

## Optimizasyonlar

- Activation checkpointing
- Mixed-precision training (FP16)
- Gradient accumulation
- EMA (Exponential Moving Average)
- Adaptive step size control

## Katkıda Bulunma

1. Fork'layın
2. Feature branch oluşturun
3. Değişikliklerinizi commit'leyin
4. Branch'inizi push'layın
5. Pull Request açın

## Lisans

MIT License

## Referanslar

- [Energy-Based Models](https://arxiv.org/abs/2010.03507)
- [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL2)
- [Mixture of Experts](https://arxiv.org/abs/2202.08906)
