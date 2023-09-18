from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM

model_name = 'gpt2'

# Setup DataLoader
bookcorpus = load_dataset('bookcorpus',split='train').select(range(1000))
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenize = lambda example: tokenizer(text=example['text'],truncation=True,
                                     max_length=tokenizer.max_len_single_sentence,
                                     padding='max_length'
                                     )

dataset = bookcorpus.map(tokenize,remove_columns=bookcorpus.column_names)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

loader = DataLoader(dataset,collate_fn=collator,batch_size=4)

# Setup Model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
                r=64,
                lora_alpha=16,
                bias="none",
                task_type="CAUSAL_LM",
            )
model = get_peft_model(base_model, peft_config)
print(model.print_trainable_parameters())
# automodel = AutoPeftModelForCausalLM(base_model)

# Forward Pass
batch = next(iter(loader))
peft_output = model(**batch)
base_output = base_model(**batch)
print(peft_output.loss)
print(base_output.loss)