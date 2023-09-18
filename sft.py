## Modified from https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py

from trl import SFTTrainer
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, Trainer
from templates import template1
from utils import compute_metrics, DataCollator
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from peft import LoraConfig

SYSTEM_MESSAGE = "Find the emotions from the sentence given below. The options are 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'. The sentence can be one or more emotions from this list"

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="prompt", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="sft_output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
id2label = {v:k for k,v in label2id.items()}

prompt_template = {
    "with_label": \
    "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] The emotions in this sentence are {labels} </s>",

    "without_label": \
    "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] </s>"
}

def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_from_disk('goemotion_subset')
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

    collator = DataCollator(tokenizer=tokenizer)

    tokenize = lambda text,special_tokens : tokenizer.encode(text,truncation=True,padding=False,max_length=512,add_special_tokens=special_tokens)

    def generate_prompt(example, prompt_template=prompt_template, tokenize=tokenize):
        text = example['text']
        labels_int = example['labels']    
        labels_str = ", ".join([id2label[x] for x in labels_int])
        
        with_label = prompt_template["with_label"].format(
                text=text,labels=labels_str,instr=SYSTEM_MESSAGE.strip())
        
        without_label = prompt_template["without_label"].format(
                text=text,instr=SYSTEM_MESSAGE.strip())
        
        tokenized_with_label = tokenize(with_label,False)
        tokenized_wo_label = tokenize(without_label,False)
        prompt_len = len(tokenized_wo_label)-2 # For [/INST] </s>
        mask = [-100]*prompt_len
        labels = tokenized_with_label.copy()
        labels[:prompt_len]=mask
        
        enc = {}
        enc['input_ids']=tokenized_with_label
        enc['label']=labels

        return enc
    
    prompted = dataset.map(generate_prompt,remove_columns=dataset['train'].column_names)

    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        metric_for_best_model='f1',
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    trainer = Trainer(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=prompted['train'],
        eval_dataset=prompted['validation'],
        dataset_text_field=script_args.dataset_text_field,
        peft_config=peft_config,
        data_collator=collator
        # compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(script_args.output_dir)

if __name__=="__main__":
    main()