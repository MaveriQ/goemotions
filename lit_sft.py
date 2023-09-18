from typing import Any
from lightning import LightningDataModule, LightningModule
from datasets import load_from_disk
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import DataCollator

class GoEmotionsDataModule(LightningDataModule):

    label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
    id2label = {v:k for k,v in label2id.items()}

    SYSTEM_MESSAGE = "Find the emotions from the sentence given below. The options are 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'. The sentence can have one or more emotions from this list."

    prompt_template = {
        "with_label": \
        "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] The emotions in this sentence are {labels} </s>",

        "without_label": \
        "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] </s>"
    }

    def __init__(self,
                 batch_size,
                 model_name,
                ):
        super().__init__()

        self.save_hyperparameters()

        raw_dataset = load_from_disk('goemotion_subset')
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenize = lambda text,special_tokens : tokenizer.encode(text,
                                                                      truncation=True,
                                                                      padding=False,
                                                                      max_length=512,
                                                                      add_special_tokens=special_tokens)

        self.dataset = raw_dataset.map(self.generate_prompt,remove_columns=raw_dataset['train'].column_names)
        self.dataset = self.dataset.rename_columns({'label':'labels'})
        self.collator = DataCollator(tokenizer=tokenizer)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset['train'],collate_fn=self.collator,batch_size=self.hparams.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset['validation'],collate_fn=self.collator,batch_size=self.hparams.batch_size)

    def generate_prompt(self,example, prompt_template=prompt_template):
        text = example['text']
        labels_int = example['labels']    
        labels_str = ", ".join([self.id2label[x] for x in labels_int])
        
        with_label = prompt_template["with_label"].format(
                text=text,labels=labels_str,instr=self.SYSTEM_MESSAGE.strip())
        
        without_label = prompt_template["without_label"].format(
                text=text,instr=self.SYSTEM_MESSAGE.strip())
        
        tokenized_with_label = self.tokenize(with_label,False)
        tokenized_wo_label = self.tokenize(without_label,False)
        prompt_len = len(tokenized_wo_label)-2 # For [/INST] and </s>
        mask = [-100]*prompt_len
        labels = tokenized_with_label.copy()
        labels[:prompt_len]=mask
        
        enc = {}
        enc['input_ids']=tokenized_with_label
        enc['label']=labels

        return enc
    
class GoEmotionsLightningModule(LightningModule):

    def __init__(self,
                 model_name,
                 load_in_8bit,
                 load_in_4bit,
                 use_peft,
                 lora_r,
                 lora_alpha
                ):
        super().__init__()

        self.save_hyperparameters()

        if self.hparams.load_in_8bit and self.hparams.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif self.hparams.load_in_8bit or self.hparams.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.hparams.load_in_8bit, load_in_4bit=self.hparams.load_in_4bit
            )
            # Copy the model to each device
            device_map = {"": Accelerator().local_process_index}
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None

        base_model = AutoModelForCausalLM.from_pretrained(
            self.hparams.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        if self.hparams.use_peft:
            peft_config = LoraConfig(
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, peft_config)
        else:
            self.model = base_model


    def forward(self, **batch: Any) -> Any:
        return self.model(**batch)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        output = self(batch)
        loss = output.loss
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:

        output = self(batch)
        loss = output.loss
        return loss
    

if __name__=="__main__":

    model = GoEmotionsLightningModule(model_name='gpt2',
                                      load_in_8bit=False,
                                      load_in_4bit=False,
                                      use_peft=True,
                                      lora_r=64,
                                      lora_alpha=16)
    
    dm = GoEmotionsDataModule(4,'gpt2')

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    output = model(**batch)

    print(output.loss)