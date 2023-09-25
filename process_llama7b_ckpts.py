from transformers import AutoModelForCausalLM
import torch
from pathlib import Path
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
import os

ckpts = list(Path('/work/scratch/hj36wegi/data/goemotions/llama2_sft_template1/lightning_logs').glob('llama7b*/*/*.ckpt'))
base_dir = "/home/hj36wegi/scratch/data/goemotions/llama2_sft_template1/adapters/"

def process_checkpoint(ckpt):
    payload = torch.load(ckpt,map_location='cpu')

    with init_empty_weights():
        base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

    peft_config = LoraConfig(
        r=payload['hyper_parameters']['lora_r'],
        lora_alpha=payload['hyper_parameters']['lora_alpha'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, peft_config)

    state_dict = {}

    for k,v in payload['state_dict'].items():
        new_k = k[6:]
        state_dict[new_k]=v
    
    base_model.load_state_dict(state_dict=state_dict,strict=False)
    base_model.to('cpu')
    return base_model

for ckpt in ckpts:
    config = ckpt.parts[-3]
    epoch = ckpt.parts[-1][:-5]
    save_dir = base_dir+'-'.join([config,epoch])
    if os.path.exists(save_dir):
        print(f"{save_dir} exists. Skipping checkpoint")
        continue
    print(f"saving in {save_dir}")
    model = process_checkpoint(ckpt)
    model.save_pretrained(save_dir)