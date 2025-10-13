from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, load_from_disk
import torch
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json
from turkish_tokenizer import HFTurkishTokenizer

"""
class RedPajamaDataset(Dataset):
    def __init__(self, hparams): # dont use tokenizer is in collator
        if hparams.execution_mode != "pretrain":
            raise ValueError("Bilkent NLP is a pretrain dataset, no other execution modes supported.")
            
        #NOTE there is only 1 split (train) so every other split does the same here
        self.max_length = hparams.context_length+1
        #hf_home = os.getenv('HF_HOME')
        dataset_dir = hparams.dataset_dir 
        self.tokenizer =  HFTurkishTokenizer(
            bos_token="<s>",
            eos_token="</s>",
            sep_token="<sep>",
            cls_token="<cls>",
            mask_token="<mask>",
            pad_token="<pad>",
            unk_token="<unk>",
            model_max_length=hparams.model_max_length
        )
        self.tokenizer_pad_token = tokenizer.eos_token

        if hparams.pretokenize_dataset:
            save_path = os.path.join(dataset_dir, hparams.dataset_name + '_preprocessed', hparams.tokenizer.replace('/', '_'), "max_length_" + str(self.max_length))
            print("pretokenized dataset save_path", save_path)

            if os.path.exists(save_path): # load dataset it exists
                print(f"loading {hparams.dataset_name} dataset")
                self.dataset = load_from_disk(save_path)
            else: # need to create dataset
                print(f"no pre-tokenized {hparams.dataset_name} dataset with correct settings, loading and saving")
                self.dataset = load_dataset("selimfirat/bilkent-turkish-writings-dataset", split = "train", cache_dir=dataset_dir, trust_remote_code=True, keep_in_memory = False)

                num_proc = hparams.num_workers * hparams.num_gpus
                print("num_proc using for dataset map", num_proc) # found that if have 192 cpus then cannot use 96 (it freezes), so 48 was good. make sure to test this with your own hardware and adjust num workers accordingly
                # NOTE this code may freeze and takes a very long time to run, make sure to test what values for num_proc and num_workers are best
                self.dataset = self.dataset.map(self.tokenization) # batched=True, batch_size=hparams.batch_size_per_device,
                print("done preprocessing dataset")
                self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
                print("done formatting dataset")
                self.dataset.save_to_disk(save_path)
        else:
            self.dataset = load_dataset("selimfirat/bilkent-turkish-writings-dataset", "sample-100B", split = "train", cache_dir=dataset_dir, trust_remote_code=True, keep_in_memory = False)

        self.hparams = hparams

    def tokenization(self, example):
        
        return self.tokenizer(example['text'], padding=True, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.dataset[:10])
    
    def __getitem__(self, idx):
        if self.hparams.pretokenize_dataset:
            return self.dataset[idx]
        else:
            return self.dataset[idx]['text']"""

class RedPajamaDataset(Dataset):
    def __init__(self, hparams): 
        if hparams.execution_mode != "pretrain":
            raise ValueError("Bilkent NLP is a pretrain dataset, no other execution modes supported.")
            
        self.max_length = hparams.context_length + 1
        self.block_size = 128
        self.stride = 128 if hasattr(hparams, 'block_stride') else self.block_size  # Default: overlap yok
        dataset_dir = hparams.dataset_dir
        self.hparams = hparams
        self.tokenizer = HFTurkishTokenizer(
            bos_token="<s>",
            eos_token="</s>",
            sep_token="<sep>",
            cls_token="<cls>",
            mask_token="<mask>",
            pad_token="<pad>",
            unk_token="<unk>",
            model_max_length=128
        )
        self.tokenizer_pad_token = self.tokenizer.eos_token
        
        # Sequential reading için değişkenler
        self.text_blocks = []
        self.block_indices = []
        self.current_epoch = 0
        self.current_start_idx = 0  # Her epoch'ta nereden başlayacağımız
        
        if hparams.pretokenize_dataset:
            save_path = os.path.join(
                dataset_dir,
                hparams.dataset_name + '_preprocessed',
                "max_length_" + str(self.max_length)
            )
            print("pretokenized dataset save_path", save_path)

            if os.path.exists(save_path):
                print(f"loading {hparams.dataset_name} dataset")
                self.dataset = load_from_disk(save_path)
                self.dataset = self.dataset.select(range(10))
                self._prepare_all_text_blocks()
            else:
                print(f"no pre-tokenized {hparams.dataset_name} dataset with correct settings, loading and saving")
                self.dataset = load_dataset(
                    "selimfirat/bilkent-turkish-writings-dataset",
                    split="train",
                    cache_dir=dataset_dir,
                    trust_remote_code=True,
                    keep_in_memory=False
                )
                self.dataset = self.dataset.select(range(10))
                num_proc = hparams.num_workers * hparams.num_gpus
                print("num_proc using for dataset map", num_proc)
                
                self.dataset = self.dataset.map(self.tokenization)
                print("done preprocessing dataset")

                self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
                print("done formatting dataset")

                self.dataset.save_to_disk(save_path)
                self._prepare_all_text_blocks()
        else:
            self.dataset = load_dataset(
                "selimfirat/bilkent-turkish-writings-dataset",
                split="train",
                cache_dir=dataset_dir,
                trust_remote_code=True,
                keep_in_memory=False
            )
            self.dataset = self.dataset.select(range(10))
            self._prepare_all_text_blocks()

        
        print(f"Total blocks created: {len(self.block_indices)}")
        print(f"Block size: {self.block_size}, Stride: {self.stride}")

    def _prepare_all_text_blocks(self):
        """Tüm olası block'ları hazırlar"""
        self.all_text_blocks = []
        self.all_block_indices = []
        
        for doc_idx in range(len(self.dataset)):
            if self.hparams.pretokenize_dataset:
                item = self.dataset[doc_idx]
                input_ids = item['input_ids']
                
                # Stride ile block'lar oluştur
                for start_idx in range(0, len(input_ids) - self.block_size + 1, self.stride):
                    end_idx = start_idx + self.block_size
                    block = input_ids[start_idx:end_idx]
                    self.all_text_blocks.append(block)
                    self.all_block_indices.append((doc_idx, start_idx, end_idx))
            else:
                item = self.dataset[doc_idx]
                text = item['text']
                
                if '\n' in text:
                    text = text.split('\n', 1)[1]
                
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                
                for start_idx in range(0, len(tokens) - self.block_size + 1, self.stride):
                    end_idx = start_idx + self.block_size
                    block = tokens[start_idx:end_idx]
                    self.all_text_blocks.append(block)
                    self.all_block_indices.append((doc_idx, start_idx, end_idx))
        
        # Başlangıç için tüm block'ları kullanıma aç
        self.text_blocks = self.all_text_blocks.copy()
        self.block_indices = self.all_block_indices.copy()

    def set_epoch(self, epoch):
        """Her epoch'ta başlangıç pozisyonunu ayarlar"""
        self.current_epoch = epoch
        # Her epoch'ta farklı bir başlangıç noktasından başla
        self.current_start_idx = (epoch * self.stride) % len(self.all_text_blocks)
        
        # Mevcut epoch için kullanılacak block'ları belirle
        start = self.current_start_idx
        self.text_blocks = self.all_text_blocks[start:] + self.all_text_blocks[:start]
        self.block_indices = self.all_block_indices[start:] + self.all_block_indices[:start]
        
        print(f"Epoch {epoch}: Starting from block {start}, Total blocks: {len(self.text_blocks)}")

    def tokenization(self, example):
        return self.tokenizer(example['text'], padding=True, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.text_blocks)

    def __getitem__(self, idx):
        # Sequential reading: idx -> current_start_idx + idx
        actual_idx = (self.current_start_idx + idx) % len(self.all_text_blocks)
        
        if self.hparams.pretokenize_dataset:
            block_tokens = self.all_text_blocks[actual_idx]
            
            # Block'lar zaten doğru uzunlukta olduğu için padding gerekmez
            return {
                'input_ids': block_tokens,
                'attention_mask': torch.ones_like(block_tokens)
            }
        else:
            block_tokens = self.all_text_blocks[actual_idx]
            
            return {
                'input_ids': torch.tensor(block_tokens),
                'attention_mask': torch.ones(self.block_size)
            }
        
if __name__=="__main__":
    from types import SimpleNamespace
    hparams = dict(
        # optimisation
        lr=1e-3,
        batch_size_per_device=2,
        num_workers_per_gpu=12,
        max_steps=100000,

        # data
        dataset_dir="",
        dataset_name="selimfirat/bilkent-turkish-writings-dataset",
        context_length=8,
        pretokenize_dataset=True,
        

        # model choice
        model_name="ebt",  # "baseline_transformer" or "ebt"

        # model size
        embedding_dim=256,
        num_transformer_blocks=6,
        multiheaded_attention_heads=6,
        ffn_dim_multiplier=4,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,
        model_max_length=256,
        # misc
        execution_mode="pretrain",
        debug_unused_parameters=False,
        mcmc_step_size=500.0,
        num_workers=4,
        num_gpus=1
    )

    ebt_params = dict(
        model_max_length=128,
        mcmc_step_size=500.0,
        model_name="ebt",
        mcmc_step_size_lr_multiplier=1500.0,
        mcmc_num_steps=3,
        ebt_type="default",
        normalize_initial_condition=True,
        denoising_initial_condition="random_noise",
        mcmc_step_size_learnable=True,
        no_mcmc_detach=False,
        ebt_norm="rms",
        ebt_act_func="silu",
        dyt_alpha_init=0.5,
        mcmc_replay_buffer=False,
        gaussian_random_noise_scaling=1.0,
        normalize_initial_condition_only_first_step=False,
        randomize_mcmc_step_size_scale=1.0,
        randomize_mcmc_num_steps=0,
        randomize_mcmc_num_steps_min=0,
        randomize_mcmc_num_steps_final_landscape=False,
        langevin_dynamics_noise=0.0,
        langevin_dynamics_noise_learnable=False,
        vocab_to_embed_uses_prob_dist=False,
        num_modality_processing_mlp_layers=1,
        truncate_mcmc=False,
        clamp_futures_grad=False,
        clamp_futures_grad_max_change=9.0,
        absolute_clamp=0.0,
        clamp_max_after_warm_up=0.0,
        sharpen_predicted_distribution=0.0,
        reconstruction_coeff=1.0,
        contrastive_loss=False,
        contrastive_loss_coeff=0.0005,
        soften_target_prob_dist=0.0,
        adaptive=False,
        infer_ebt_override_alpha=0.,
        infer_generated_samples=1,
        infer_steps_final_landscape=False,
        infer_energy_sampling_technique="min",
        infer_alpha_final_landscape=False,
        infer_langevin_first_step=True,
        infer_accept_lower_energies=True,
        infer_ebt_num_steps=1,
        
    )
    hparams.update(ebt_params)

    hparams = SimpleNamespace(**hparams)
    dataset=RedPajamaDataset(hparams=hparams)

    # dataset şunda nasıl çalışıyor deneyeceğim
    print("\n--- Dataset Kontrolü ---")
    print(f"Blok Uzunluğu (context_length+1): {dataset.block_size}")
    print(f"Atlama Adımı (stride): {dataset.stride}")
    print(f"Toplam Olası Blok Sayısı (Tüm Epochlar): {len(dataset.all_text_blocks)}")
    print(f"İlk Belge Token ID'leri (Kısmi): {dataset.dataset[0]['input_ids'][:20] if hparams.pretokenize_dataset else 'Ham metin olduğu için bu alandan direkt okunmaz.'}")
    
    # 3 farklı epoch'u simüle edelim
    num_test_epochs = 3
    for epoch in range(num_test_epochs):
        print(f"\n--- Epoch {epoch} Testi ---")
        
        # set_epoch çağrısı: Veri kümesini epoch'a göre döndürür
        dataset.set_epoch(epoch)
        
        current_len = len(dataset)
        print(f"Epoch {epoch} için Aktif Blok Sayısı: {current_len}")
        
        if current_len == 0:
            print("Uyarı: Hiç blok oluşturulamadı. (Muhtemelen veri çok kısa veya context_length çok büyük)")
            continue
            
        # İlk bloğu al ve içeriğini kontrol et
        first_item = dataset[0]
        print(f"Epoch {epoch} - İlk Bloğun İndisi: {dataset.block_indices[0],dataset.block_indices}")
        print(f"Epoch {epoch} - İlk Bloğun input_ids uzunluğu: {len(first_item['input_ids'])}")
        print(f"Epoch {epoch} - İlk Bloğun input_ids (ilk 5 token): {first_item['input_ids'][:5]}")
        
        # İkinci bloğu al
        if current_len > 1:
            second_item = dataset[1]
            print(f"Epoch {epoch} - İkinci Bloğun İndisi: {dataset.block_indices[1]}")
            # Ardışık okuma ve set_epoch'un çalışıp çalışmadığını kontrol et
            
        # Son bloğu al
        last_item = dataset[current_len - 1]
        print(f"Epoch {epoch} - Son Bloğun İndisi: {dataset.block_indices[-1]}")
        print(f"Epoch {epoch} - Son Bloğun input_ids (ilk 5 token): {last_item['input_ids'][:5]}")


    print("\n--- Çapraz Kontrol ---")
    # Epoch 0'ın ilk bloğu ile Epoch 1'in ilk bloğunun farklı olup olmadığını kontrol edelim.
    # Stride=4 olduğu için blokların farklı olması beklenir.

    # Epoch 0
    dataset.set_epoch(0)
    e0_first_block = dataset[0]['input_ids']

    # Epoch 1
    dataset.set_epoch(1)
    e1_first_block = dataset[0]['input_ids']
    
    # Epoch 2
    dataset.set_epoch(2)
    e2_first_block = dataset[0]['input_ids']

    print(f"Epoch 0 (İlk Bloğun Başlangıç Tokeni): {e0_first_block[0]}")
    print(f"Epoch 1 (İlk Bloğun Başlangıç Tokeni): {e1_first_block[0]}")
    print(f"Epoch 2 (İlk Bloğun Başlangıç Tokeni): {e2_first_block[0]}")

    if torch.equal(e0_first_block, e1_first_block):
        print("UYARI: set_epoch doğru çalışmıyor, bloklar aynı.")
    else:
        print("BAŞARILI: set_epoch doğru çalışıyor, bloklar farklı.")