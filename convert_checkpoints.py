from pathlib import Path
from lit_sft import GoEmotionsLightningModule
from tqdm import tqdm
import torch
import os

ckpts = list(Path('/work/scratch/hj36wegi/data/goemotions/llama2_sft_template1/lightning_logs').rglob('*.ckpt'))
base_dir = "/home/hj36wegi/scratch/data/goemotions/llama2_sft_template1/adapters/"

for ckpt in tqdm(ckpts,total=len(ckpts)):
    config = ckpt.parts[-3]
    if "llama7b" in config:
        print('Skipping llama7b checkpoints for now')
        continue
    epoch = ckpt.parts[-1][:-5]
    save_dir = base_dir+'-'.join([config,epoch])
    if os.path.exists(save_dir):
        print(f"{save_dir} exists. Skipping checkpoint")
        continue
    print(f"saving in {save_dir}")

    model = GoEmotionsLightningModule.load_from_checkpoint(ckpt)
    model.model.save_pretrained(save_dir)
    del model
    torch.cuda.empty_cache()