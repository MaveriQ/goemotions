from typing import Any
from lightning import LightningDataModule, LightningModule, seed_everything
from datasets import load_from_disk
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
import torch
from accelerate import Accelerator
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import DataCollator
import warnings
warnings.filterwarnings("ignore", ".*command is available on your*")

class GoEmotionsDataModule(LightningDataModule):

    label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
    id2label = {v:k for k,v in label2id.items()}

    SYSTEM_MESSAGE = "Find the emotions from the sentence given below. The options are 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'. The sentence can have one or more emotions from this list."

    prompt_template_1 = {
        "with_label": \
        "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] The emotions in this sentence are {labels}.</s>",

        "without_label": \
        "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] </s>"
    }

    prompt_template_2 = {
        "with_label": \
        "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] {labels}.</s>",

        "without_label": \
        "<s>[INST] <<SYS>>\n{instr}\n<</SYS>>\n\n {text} [/INST] </s>"
    }

    def __init__(self,
                 model_name_or_path="meta-llama/Llama-2-7b-hf",
                 batch_size=4,
                ):
        super().__init__()

        self.save_hyperparameters()

        raw_dataset = load_from_disk('goemotion_subset')
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenize = lambda text,special_tokens : tokenizer.encode(text,
                                                                      truncation=True,
                                                                      padding=False,
                                                                      max_length=4096,
                                                                      add_special_tokens=special_tokens)

        self.dataset = raw_dataset.map(self.generate_prompt,remove_columns=raw_dataset['train'].column_names)
        self.dataset = self.dataset.rename_columns({'label':'labels'})
        self.collator = DataCollator(tokenizer=tokenizer)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset['train'],collate_fn=self.collator,
                          batch_size=self.hparams.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset['validation'],collate_fn=self.collator,
                          batch_size=self.hparams.batch_size)

    def generate_prompt(self,example, prompt_template=prompt_template_1):
        text = example['text']
        labels_int = example['labels']
        if len(labels_int)>1:
            labels_str = " and ".join([self.id2label[x] for x in labels_int])
        else:
            labels_str = self.id2label[labels_int[0]]
        
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
                 model_name_or_path = "meta-llama/Llama-2-13b-hf",
                 load_in_8bit = True,
                 load_in_4bit = False,
                 use_peft = True,
                 lora_r = 64,
                 lora_alpha = 16,
                 lr: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                ):
        super().__init__()

        self.save_hyperparameters()

        if self.hparams.load_in_8bit and self.hparams.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif self.hparams.load_in_8bit or self.hparams.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.hparams.load_in_8bit,
                load_in_4bit=self.hparams.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_8bit_compute_dtype=torch.float16
            )
            # Copy the model to each device
            device_map = {"": Accelerator().local_process_index}
            torch_dtype = torch.float16
        else:
            device_map = None
            quantization_config = None
            torch_dtype = None

        base_model = AutoModelForCausalLM.from_pretrained(
            self.hparams.model_name_or_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_cache=False
        )

        prepare_model_for_kbit_training(base_model)

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

        output = self(**batch)
        loss = output.loss
        self.log('train/loss',loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:

        output = self(**batch)
        loss = output.loss
        self.log('val/loss',loss)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_lightning_class_args(TensorBoardLogger, "tb_logger")
        # parser.set_defaults({"tb_logger.save_dir": "./", "my_early_stopping.patience": 5})
        parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
        # parser.link_arguments("trainer.logger.init_args.version", "trainer.callbacks.init_args.filename")
        # parser.link_arguments("trainer.logger.init_args.name", "trainer.log_dir")
        # parser.link_arguments("data.eval_splits", "model.eval_splits", apply_on="instantiate")
        # parser.link_arguments("model.task_name", "trainer.logger.init_args.version")

def main():
    seed_everything(42)
    cli = MyLightningCLI(model_class=GoEmotionsLightningModule,
                       datamodule_class=GoEmotionsDataModule)    

if __name__=="__main__":
    main()

def debug():
    model = GoEmotionsLightningModule(model_name='gpt2',
                                      load_in_8bit=False,
                                      load_in_4bit=False,
                                      use_peft=True,
                                      lora_r=64,
                                      lora_alpha=16)
    
    dm = GoEmotionsDataModule(batch_size=4,
                              model_name='gpt2')

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    output = model(**batch)

    print(output.loss)