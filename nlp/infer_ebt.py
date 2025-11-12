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


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

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
        ckpt=r"C:\Users\bahaa\OneDrive\Masaüstü\energic_model\checkpoint_epoch_2.pt",
        prompt="Okul güzel bir yer ama bana göre değil Okul güz",
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
    
    
    
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    echo = hparams.infer_echo
    # ppl = model.forward_loss_wrapper(questions, phase="test")['perplexity'].item() # just in case want to debug model PPL

    prompt_tokens = [] #NOTE this was to fix a bug where this generation code was not working for bs > 1 due to pad_token_id being same as eos_token_id and min_prompt_len being wrong
    
    
    params = model.transformer.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    total_len = min(hparams.context_length, max_gen_len + max_prompt_len)
    cache=[]
    with torch.no_grad():
        cache=None
        for cur_pos in range(min_prompt_len, total_len):

            top_p=hparams.top_p
            input_ids = prepare_inputs(essentially.prompt, tokenizer,essentially.max_len , device).unsqueeze(0)
    
            input_ids=torch.cat((input_ids,input_ids))
            if carry is None:
                carry = model.initial_carry(input_ids.size(0),  2*input_ids.size(1))

            print("Input IDs shape:", input_ids.shape)
            # model.ebt_advanced_inference expects token ids (B, S)
            # if tokenizer returns shape (B, 1, S) or similar, squeeze
            
            
            # run inference
            final_output, energies_list_accum, predicted_distributions_accum, cache,carry = model.ebt_advanced_inference(input_ids, start_pos=0, learning=False, past_cache=cache,carry=carry)
            # final_output: logits (B, S, V)    
            probs = torch.softmax(final_output[:,-1]/temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
            
           
            next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

     
    past_cache = None
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # mark finished if EOS already present in prompt
    if eos_id is not None:
        finished = finished | (buffer_ids == eos_id).any(dim=1).to(device)

    for step in range(max_steps):
        # compute insertion positions (first pad position)
        cur_pos = (buffer_ids != pad_id).sum(dim=1)

        # if all finished or no space left, stop
        if torch.all(finished) or torch.all(cur_pos >= S):
            break

        # run model; returns logits for all positions plus cache and carry
        # ebt_advanced_inference returns: final_output, energies_list_accum, predicted_distributions_accum, new_cache, new_carry
        final_output, energies, pred_dists, new_cache, new_carry = model.ebt_advanced_inference(buffer_ids, start_pos=0, learning=False, past_cache=past_cache, carry=carry)
        # final_output: (B, S, V)

        b_idx = torch.arange(B, device=device)
        pos_clamped = torch.clamp(cur_pos, max=S - 1)
        logits = final_output[b_idx, pos_clamped, :]

        # softmax + temperature
        probs = torch.softmax(logits / max(1e-8, float(temperature)), dim=-1)

        if top_p is None or top_p >= 1.0:
            next_tokens = torch.argmax(probs, dim=-1)
        else:
            next_tokens = sample_top_p(probs, top_p)

        # for finished examples or those with no room, set pad
        invalid = (cur_pos >= S) | finished
        next_tokens = next_tokens.to(device)
        next_tokens[invalid] = pad_id

        # write tokens
        buffer_ids[b_idx, pos_clamped] = next_tokens

        # mark finished where eos produced
        if eos_id is not None:
            finished = finished | (next_tokens == eos_id)

        # update cache and carry for next step
        past_cache = new_cache
        carry = new_carry

    return buffer_ids   
    print(probs.shape)



if __name__ == '__main__':
    main()
