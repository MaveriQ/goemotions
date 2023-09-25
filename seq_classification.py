from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, GPT2Model
from argparse import ArgumentParser
from utils import compute_metrics
from ray import tune
import pickle

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name',default='roberta-base')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--num_epochs',type=int,default=3)
    parser.add_argument('--lr',default=5e-5,type=float)
    parser.add_argument('--output_dir',default='/home/hj36wegi/scratch/work/flek')

    tmp = "--model_name bert-base-uncased".split()
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataset = load_from_disk('goemotion_subset')
    subset_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    label2id = {emotion:idx for idx,emotion in enumerate(subset_emotions)}
    id2label = {v:k for k,v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_data(example):
        # take a line of text
        text = example["text"]
        # encode it
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
        # add labels
        labels = [1.0 if i in example['labels'] else 0.0 for i in range(6)]
        
        encoding["label"] = labels
    
        return encoding

    encoded_dataset = dataset.map(preprocess_data,remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format('pt')

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, 
                                                                problem_type="multi_label_classification", 
                                                                num_labels=len(subset_emotions),
                                                                id2label=id2label,
                                                                label2id=label2id,
                                                                trust_remote_code=True)
        if model.config.pad_token_id is None:
            print(f'Setting model.config.pad_token_id = {tokenizer.eos_token_id}')
            model.config.pad_token_id = tokenizer.eos_token_id
        return model

    trg_args = TrainingArguments(
        output_dir=args.output_dir+f"/{args.model_name.replace('/','-')}-finetuned-goemotions",
        evaluation_strategy = "epoch",
        # eval_steps=500, 
        save_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        #push_to_hub=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=trg_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
     
    def ray_hp_space(trial):
        return {
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "per_device_train_batch_size": tune.choice([8, 16, 32, 64]),
        }
    
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 3e-5, 5e-5]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        }

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=9,
        # compute_objective=compute_objective,
    )
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.save_model(args.output_dir)
    pickle.dump(best_run,open('best_run.pkl','wb'))

if __name__=="__main__":
    main()