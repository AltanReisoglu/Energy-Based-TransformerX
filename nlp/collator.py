from transformers import AutoTokenizer, DataCollatorWithPadding
from turkish_tokenizer import HFTurkishTokenizer
import torch
class NLP_HF_Collator:
    def __init__(self, hparams):
        self.hparams = hparams
        self.max_length = hparams.context_length + 1
        self.tokenizer = None

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }