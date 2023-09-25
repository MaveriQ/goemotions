This repository contains code for data preperation, training and evaluation of various classifiers on Go-Emotions Dataset.

### goemotion_6class_subset.ipynb
Use this file to see how the original 28-class problem was mapped to 6-class problem.

### seq_classification.py
This file implements sequence classification pipeline from HuggingFace. Additionally, there is a parameter search implementation using Ray library and HuggingFace Trainer module.

### supervised_finetuning.py
This file has Lightning modules for data preperation and training to be used with Lightning Framework. I have used 4-bit quantization from BitsAndBytes and peft library from HuggingFace, to train a 13 billion parameter model on a 16GB GPU. 

### config.yaml
This file needs to be used with previous file to run supervised finetuning under various parameter settings. 

### convert_checkpoints.py
I extract the adapters from the checkpoints created from supervised_finetuning.py. This saves space because the base Llama2-13B model is fixed during supervised finetuning, and therefore doesn't need to be saved.

### evaluate_peft_model.py
Here I load the adapters saved in the previous step with the Llama2-13B model and generate text from validation split of Go Emotions dataset. The module has been designed to take inputs from commandline for various checkpoints.

### zeroshot-openai.py
This is a short scipt to run OpenAI text completion API to generate labels from the validation split, in a zero-shot fashion. 

### visualize_peft.ipynb
A notebook to visualize a pytorch model with adapter modules.

### scratchpad
A folder with various debugging scripts and notebooks.

### cmds
A file with the commands I used for supervised finetuning.