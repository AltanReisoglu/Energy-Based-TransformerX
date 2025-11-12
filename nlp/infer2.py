import torch
import argparse
from types import SimpleNamespace
from ebt import EBT_NLP
from turkish_tokenizer import HFTurkishTokenizer


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.
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
    """Default hyperparameters matching train_ebt.py"""
    hparams = dict(
        vocab_size=32000,
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        hidden_dim=2048,
        n_experts=8,
        n_expert_per_token=2,
        max_seq_len=2048,
        dropout=0.1,
        bias=True,
        norm_eps=1e-5,
        rope_theta=10000.0,
        use_cache=True,
        causal=True,
        qk_norm=False,
        flash=False,
        add_zero_kv=False,
        talking_heads=False,
        sparse_topk=None,
        onnxable=False,
        linear_attention=False,
        sliding_window=0,
        sliding_window_causal=True,
        max_mem_len=0,
        emb_frac_gradient=1.0,
    )
    return SimpleNamespace(**hparams)


def load_checkpoint_to_model(model, ckpt_path, device):
    """Load checkpoint into model, handling both state_dict and full ckpt formats"""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    
    try:
        model.load_state_dict(state, strict=True)
        print("✓ Checkpoint loaded (strict)")
    except RuntimeError as e:
        print(f"⚠ Strict load failed, trying non-strict: {e}")
        model.load_state_dict(state, strict=False)
        print("✓ Checkpoint loaded (non-strict)")
    
    return model


def prepare_inputs(prompts, tokenizer, max_len, device):
    """Tokenize prompts and prepare input tensors"""
    enc = tokenizer(prompts, padding='max_length', truncation=True, 
                    max_length=max_len, return_tensors='pt')
    input_ids = enc['input_ids']
    return input_ids.to(device)


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs back to text"""
    outs = []
    for ids in token_ids.cpu().numpy().tolist():
        try:
            txt = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            toks = [tokenizer.convert_ids_to_tokens(i) for i in ids]
            txt = ''.join(toks)
        outs.append(txt)
    return outs


def _move_obj_to_device(obj, device):
    """Recursively move tensors in nested object to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_obj_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_obj_to_device(item, device) for item in obj)
    elif hasattr(obj, '__dict__'):
        new_obj = object.__new__(type(obj))
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, _move_obj_to_device(v, device))
        return new_obj
    else:
        return obj


def generate(
    model,
    input_ids,
    tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    device='cpu',
    eos_token_id=None
):
    """
    Generate tokens using cache and carry, stopping at EOS token.
    
    Args:
        model: EBT_NLP model
        input_ids: (batch_size, prompt_len) tensor
        tokenizer: HFTurkishTokenizer instance
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        device: torch device
        eos_token_id: token ID to stop generation (default: tokenizer.eos_token_id)
    
    Returns:
        generated_ids: (batch_size, prompt_len + num_generated) tensor
    """
    batch_size, prompt_len = input_ids.shape
    
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    # Initialize carry and cache
    model.eval()
    with torch.no_grad():
        carry = model.initial_carry(batch_size, prompt_len)
        carry = _move_obj_to_device(carry, device)
        past_cache = None
        
        # Process prompt
        logits, energies, pred_dists, past_cache, carry = model.ebt_advanced_inference(
            input_ids, carry=carry, past_cache=past_cache
        )
        
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for step in range(max_new_tokens):
            # Get last logits (batch_size, vocab_size)
            next_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            
            # Get probabilities
            next_probs = torch.softmax(next_logits, dim=-1)
            
            # Sample or greedy
            if top_p < 1.0:
                # Top-p sampling per example
                next_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                for i in range(batch_size):
                    if not finished[i]:
                        next_token[i] = sample_top_p(next_probs[i:i+1], top_p)
                    else:
                        next_token[i] = eos_token_id
            else:
                # Greedy (argmax)
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            for i in range(batch_size):
                if next_token[i, 0].item() == eos_token_id:
                    finished[i] = True
            
            # Stop if all finished
            if finished.all():
                break
            
            # Forward pass for next step (only for unfinished examples)
            logits, energies, pred_dists, past_cache, carry = model.ebt_advanced_inference(
                next_token, carry=carry, past_cache=past_cache
            )
    
    return generated


def main():
    parser = argparse.ArgumentParser(description='EBT_NLP Inference')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default=None, help='Input prompt')
    parser.add_argument('--input_file', type=str, default=None, help='File with prompts (one per line)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_len', type=int, default=64, help='Max input length')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling threshold')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--out_file', type=str, default=None, help='Output file for results')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    hparams = default_hparams()
    model = EBT_NLP(hparams).to(device)
    load_checkpoint_to_model(model, args.ckpt, device)
    
    tokenizer = HFTurkishTokenizer()
    
    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide --prompt or --input_file")
    
    # Process in batches
    results = []
    for batch_idx in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + args.batch_size]
        
        print(f"\n[Batch {batch_idx // args.batch_size + 1}]")
        input_ids = prepare_inputs(batch_prompts, tokenizer, args.max_len, device)
        
        # Generate
        generated_ids = generate(
            model, input_ids, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode and print
        texts = decode_tokens(tokenizer, generated_ids)
        for prompt, text in zip(batch_prompts, texts):
            print(f"Prompt: {prompt}")
            print(f"Generated: {text}\n")
            results.append({'prompt': prompt, 'generated': text})
    
    # Save results if requested
    if args.out_file:
        import json
        with open(args.out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.out_file}")


if __name__ == '__main__':
    main()