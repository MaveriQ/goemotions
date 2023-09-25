
label2id = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
id2label = {v:k for k,v in label2id.items()}

def template1(example,eval=False):
    text = example['text']
    labels_int = example['labels']
    labels_str = [id2label[x] for x in labels_int]
    if len(labels_int)>1:
        labels = " and ".join(labels_str)
        
    else:
        labels = labels_str[0]
        # prompt = f'The label for the sentence : "{text}" is {labels}'
    if eval==True:
        prompt = f'The emotions in the sentence : "{text}" are'
        return {"prompt": prompt,
                "labels": labels}
    else:
        prompt = f'The emotions in the sentence : "{text}" are {labels}'
        return {"prompt": prompt}

def template2(example):
    text = example['text']
    labels_int = example['labels']
    labels_str = [id2label[x] for x in labels_int]
    
    if len(labels_int)>1:
        labels = " and ".join(labels_str)
        prompt = f'The emotions in the sentence : "{text}" are {labels}'
    else:
        labels = labels_str[0]
        prompt = f'The label for the sentence : "{text}" is {labels}'
    
    return {"prompt": prompt}