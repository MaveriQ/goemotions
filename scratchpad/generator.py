from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, default_data_collator
from transformers.data.data_collator import _torch_collate_batch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from functools import partial
from templates import template1

max_length = 50
max_new_tokens = 10
batch_size = 4

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2',padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = load_from_disk('goemotion_subset')
dataset = dataset['validation']

template = partial(template1,eval=True)
prompted = dataset.map(template,remove_columns=dataset.column_names)

tokenize = partial(tokenizer,max_length=max_length,padding='max_length',truncation=False)
tokenized = prompted.map(lambda e: tokenize(e['prompt']),remove_columns=['prompt'])
tokenized.set_format('pt')

for idx in range(len(tokenized)//batch_size):
    batch=tokenized[idx:idx+batch_size]
    output = model.generate(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            max_new_tokens=max_new_tokens)
    new_tokens = output[:,max_length:].detach().cpu().numpy()
    print(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))