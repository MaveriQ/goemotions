from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from accelerate import Accelerator
from datasets import load_from_disk
from templates import template1
from functools import partial
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import os

flatten = lambda sublist : [itm for lst in sublist for itm in lst]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--peft_model_id') # The adapters saved from convert_checkpoints.py
    parser.add_argument('--eval_base_model',default=None) # If we want to evaluate Llama2 pretrained models without any finetuning
    parser.add_argument('--max_length',type=int,default=60) # max length for tokenizer. The value 60 was empirically found 
    parser.add_argument('--max_new_tokens',type=int,default=10) # we want to generate only one or two class labels. 
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--load_in_8bit',type=bool,default=False)
    parser.add_argument('--load_in_4bit',type=bool,default=True)

    tmp_args = "--peft_model_id llama13b_epoch3_bs8_lr1e-5-epoch=1-step=1118".split()
    args = parser.parse_args()

    return args

def get_model_tokenizer(args):
    
    quantization_config = BitsAndBytesConfig(
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    # bnb_8bit_compute_dtype=torch.float16
                )
    if args.eval_base_model is not None:
        model = AutoModelForCausalLM.from_pretrained(args.eval_base_model,
                                                    quantization_config=quantization_config,
                                                    torch_dtype = torch.bfloat16,
                                                    device_map = {"": Accelerator().local_process_index})
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.eval_base_model,padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        config = PeftConfig.from_pretrained(args.model_loc)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                    quantization_config=quantization_config,
                                                    torch_dtype = torch.bfloat16,
                                                    device_map = {"": Accelerator().local_process_index})
        model = PeftModel.from_pretrained(model, args.model_loc)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer

def main():

    args = parse_args()
    adapter_dir = Path('/work/scratch/hj36wegi/data/goemotions/llama2_sft_template1/adapters')

    # Get Model and Tokenizer
    if args.eval_base_model is not None:
        print(f"Processing {args.eval_base_model}\n")
        model,tokenizer = get_model_tokenizer(args)
        output_file = adapter_dir/f'predictions_{args.eval_base_model.split("/")[1]}.pkl'
    else:
        print(f"Processing {args.peft_model_id}\n")
        args.model_loc = adapter_dir/args.peft_model_id
        assert os.path.exists(args.model_loc), f"{args.model_loc} does not exist"
        model,tokenizer = get_model_tokenizer(args)
        output_file = adapter_dir/f'predictions_{args.peft_model_id}.pkl'

    assert not os.path.exists(output_file), f"{output_file} already exists"

    # prepare tokenizer and template with default values
    tokenize = partial(tokenizer,max_length=args.max_length,padding='max_length',truncation=False)
    template = partial(template1,eval=True)

    # Prepare Dataset
    dataset = load_from_disk('goemotion_subset')
    dataset = dataset['validation']
    
    prompted = dataset.map(template,remove_columns=dataset.column_names) # convert to prompts using template
    tokenized = prompted.map(lambda e: tokenize(e['prompt']),remove_columns=['prompt'])

    tokenized.set_format('pt')

    loader = DataLoader(tokenized,batch_size=args.batch_size)
    predictions = []

    # Run inference
    for batch in tqdm(loader,total=len(loader)):
        with torch.no_grad():
            output = model.generate(input_ids=batch['input_ids'].to(model.device),
                                    attention_mask=batch['attention_mask'].to(model.device),
                                    max_new_tokens=args.max_new_tokens)
        new_token_ids = output[:,args.max_length:].detach().cpu().numpy()
        predicted_labels = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        predictions.append(predicted_labels)

    pred = flatten(predictions)

    pickle.dump(pred,open(output_file,'wb'))

if __name__=="__main__":
    main()