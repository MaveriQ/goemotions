import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import numpy as np

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

# source : https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

class DataCollator:
    
    def __init__(self,tokenizer,pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self,batch):
        
        batch = {k:[example[k] for example in batch] for k in batch[0].keys()}
        for k,v in batch.items():
            batch[k]=self.collate(v)
            
        return batch
        
    def collate(self,examples):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (self.pad_to_multiple_of is None or length_of_first % self.pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if self.pad_to_multiple_of is not None and (max_length % self.pad_to_multiple_of != 0):
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if self.tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result
    