"""Simple inference script for EBT_NLP.

Usage examples:

# single prompt inference (CPU/GPU auto)
python nlp/infer_ebt.py --ckpt checkpoint_epoch_0.pt --prompt "Merhaba dünya" --max_len 64

# from file with multiple lines (each line = example)
python nlp/infer_ebt.py --ckpt final_model.pt --input_file examples.txt --batch_size 4

This script:
- constructs model with sensible defaults from `nlp/train_ebt.py`
- loads checkpoint (state_dict or plain dict)
- tokenizes input using the project's `HFTurkishTokenizer`
- runs `ebt_advanced_inference` and decodes outputs
"""

import argparse
import torch
from types import SimpleNamespace
from ebt import EBT_NLP
from turkish_tokenizer import HFTurkishTokenizer
import os


def default_hparams():
    # Keep these consistent with nlp/train_ebt.py defaults used in training
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
        execution_mode="inference",
        debug_unused_parameters=False,
        mcmc_step_size=500.0,
        num_workers=4,
        num_gpus=1
    )

    ebt_params = dict(
        model_max_length=64,
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
        use_bnb_optimizer=False,
        infer_langevin_dynamics_noise=0
    )
    hparams.update(ebt_params)

    hparams = SimpleNamespace(**hparams)
    return hparams

def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    try:
        model.load_state_dict(state)
    except Exception:
        # try non-strict (useful if checkpoint saved with Accelerator or has extra prefix)
        model.load_state_dict(state, strict=False)


def prepare_inputs(prompts, tokenizer, max_len, device):
    # prompts: list[str]
    enc = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    input_ids = enc['input_ids'] if 'input_ids' in enc else enc['input_ids']
    return input_ids.to(device)


def decode_tokens(tokenizer, token_ids):
    # token_ids: Tensor[B, S]
    outs = []
    for ids in token_ids.cpu().numpy().tolist():
        try:
            txt = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            # fallback: join token texts
            toks = [tokenizer.convert_ids_to_tokens(i) for i in ids]
            txt = tokenizer.convert_tokens_to_string(toks)
        outs.append(txt)
    return outs


def main():

    essentially=dict(
        ckpt=r"C:\Users\bahaa\OneDrive\Masaüstü\energic_model\checkpoint_epoch_1.pt",
        prompt=" Okul güzel bir yer ama bana göre değil Okul güz",
        batch_size=1,
        max_len=64
        


    )
    essentially = SimpleNamespace(**essentially)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # create hparams and model
    hparams = default_hparams()
    config_dict = dict(H_cycles=1, L_cycles=1, H_layers=1, L_layers=1)
    model = EBT_NLP(hparams, config_dict)
    model.to(device)
    model.eval()

    # load checkpoint
    if not os.path.exists(essentially.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {essentially.ckpt}")
    load_checkpoint_to_model(model, essentially.ckpt, device)

    # tokenizer (must match model's tokenizer settings)
    tokenizer = HFTurkishTokenizer(
        bos_token="<s>",
        eos_token="</s>",
        sep_token="<sep>",
        cls_token="<cls>",
        mask_token="<mask>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=hparams.model_max_length-1
    )

    prompts = []
    if essentially.prompt:
        prompts.append(essentially.prompt)
   
    else:
        print("Provide --prompt or --input_file")
        return

    carry = None
    with torch.no_grad():
        

        
        input_ids = prepare_inputs(essentially.prompt, tokenizer,essentially.max_len , device).unsqueeze(0)
        input_ids=torch.cat((input_ids,input_ids),dim=0)
        if carry is None:
            carry = model.initial_carry(input_ids.size(0),  2*input_ids.size(1)-2)

        print("Input IDs shape:", input_ids.shape)
        # model.ebt_advanced_inference expects token ids (B, S)
        # if tokenizer returns shape (B, 1, S) or similar, squeeze
        

        # run inference
        final_output, energies_list_accum, predicted_distributions_accum, new_cache,carry = model.ebt_advanced_inference(input_ids, start_pos=0, learning=False, past_cache=None,carry=carry)
        # final_output: logits (B, S, V)    
        probs = torch.softmax(final_output, dim=-1)
        token_ids = torch.argmax(probs, dim=-1)
        texts = decode_tokens(tokenizer, token_ids)
        

    
    print(final_output)



if __name__ == '__main__':
    main()
