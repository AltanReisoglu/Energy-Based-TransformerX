import os
import sys
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm   # ilerleme çubuğu

# Parent klasörü ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import RedPajamaDataset
from collator import NLP_HF_Collator
from ema import ExponentialMovingAverage
from ebt import EBT_NLP
from types import SimpleNamespace

import os, sys
from types import SimpleNamespace
from pathlib import Path
import pytorch_lightning as L

from torch.optim import AdamW
from tqdm import tqdm

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed

from torch.cuda.amp import autocast, GradScaler

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def maybe_enable_activation_checkpointing(model):
    # If your model supports a method like model.gradient_checkpointing_enable(), call it.
    # Otherwise, consider wrapping heavy submodules with torch.utils.checkpoint.checkpoint
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Enabled model.gradient_checkpointing_enable()")
    else:
        print("Model has no gradient_checkpointing_enable() method; consider wrapping big blocks with torch.utils.checkpoint")


def build_optimizer(model, hparams):
    params = [p for p in model.parameters() if p.requires_grad]
    if hparams.use_bnb_optimizer and BNB_AVAILABLE:
        # AdamW8bit from bitsandbytes
        optim = bnb.optim.AdamW8bit(params, lr=hparams.lr)
        
    else:
        optim = AdamW(params, lr=hparams.lr)
    return optim

def main():
 
    hparams = dict(
        # optimisation
        lr=1e-3,
        batch_size_per_device=2,
        num_workers_per_gpu=12,
        max_steps=100000,

        # data
        dataset_dir="",
        dataset_name="selimfirat/bilkent-turkish-writings-dataset",
        context_length=256,
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
        gradient_accumulation_steps=4,
        use_amp=False,
        use_activation_checkpointing=True,
        use_torch_compile=True,
        use_bnb_optimizer=False
    )
    hparams.update(ebt_params)

    hparams = SimpleNamespace(**hparams)

    # =========================
    # Model & Dataset
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = {
        
        "ebt": EBT_NLP,
    }[hparams.model_name]
    

    model = model_cls(hparams).to(device)
    dataset = RedPajamaDataset(hparams)
    collate_fn = NLP_HF_Collator(hparams)

    workers = torch.cuda.device_count() * hparams.num_workers_per_gpu
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    #optimizer = AdamW(model.parameters(), lr=hparams.lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    ema=ExponentialMovingAverage(trainable_params)
    # =========================
    # Eğitim Döngüsü
    # =========================
    model.train()
    global_step = 0
    max_steps = hparams.max_steps
    cache=None
    epochs=100

    accelerator = Accelerator(
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        mixed_precision="fp16" if hparams.use_amp else "no",
    )
    device = accelerator.device

    if hparams.use_activation_checkpointing:
        maybe_enable_activation_checkpointing(model)
    
        if hparams.use_torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("Wrapped model with torch.compile()")
            except Exception as e:
                print("torch.compile failed:", e)

    dataset = RedPajamaDataset(hparams)

    optimizer = build_optimizer(model, hparams)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    scaler = GradScaler(enabled=(accelerator.mixed_precision == "fp16"))

    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # batch is already moved by accelerator.prepare (but ensure tensors on device)
            # use accelerator.accumulate to handle gradient sync/accum correctly
            with accelerator.accumulate(model):
                with autocast(enabled=(accelerator.mixed_precision == "fp16")):
                    metrics, cache, energy = model.forward_loss_wrapper(batch, "train")
                    loss = metrics["loss"]

                # scaler + backward via accelerator
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    # optional gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                # update EMA (if desired, ensure EMA works with distributed — you may need to sync)
                try:
                    ema.update(model.parameters())
                except Exception:
                    pass

                # logging and step accounting
                if accelerator.is_main_process:
                    print(f"[Main] Step {step} | loss={loss.item():.4f}")

                global_step += 1


    torch.save(model.state_dict(), "Altan_Ebt.pt")
    # Save final model (only on main process)
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "final_model.pt")
        print("Saved final_model.pt")

if __name__ == "__main__":
    main()





if __name__ == "__main__":
    main()
