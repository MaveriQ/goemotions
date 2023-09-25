from pathlib import Path
from supervised_finetuning import GoEmotionsLightningModule
from tqdm import tqdm
import torch
import os

path_to_ckpts = '/work/scratch/hj36wegi/data/goemotions/llama2_sft_template1/lightning_logs'

ckpts = list(Path(path_to_ckpts).rglob('*.ckpt'))
output_base_dir = "/home/hj36wegi/scratch/data/goemotions/llama2_sft_template1/adapters/"

for ckpt in tqdm(ckpts,total=len(ckpts)):
    config = ckpt.parts[-3] # e.g. llama13b_epoch3_bs8_lr1e-5
    if "llama7b" in config:
        print('Skipping llama7b checkpoints for now')
        continue
    epoch = ckpt.parts[-1][:-5] # e.g. epoch=1-step=1118
    save_dir = output_base_dir+'-'.join([config,epoch])

    if os.path.exists(save_dir):
        print(f"{save_dir} exists. Skipping checkpoint")
        continue
    print(f"saving in {save_dir}")

    model = GoEmotionsLightningModule.load_from_checkpoint(ckpt)
    model.model.save_pretrained(save_dir) # saves only the adapters 
    del model
    torch.cuda.empty_cache()