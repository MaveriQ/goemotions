import openai
import os
from datasets import load_from_disk
from templates import template1
from functools import partial
import pickle
from tqdm import tqdm

dataset = load_from_disk('goemotion_subset')
dataset = dataset['validation']

openai.api_key = os.environ['OPENAI']

template = partial(template1,eval=True)
prompted = dataset.map(template,remove_columns=dataset.column_names)
prompted = prompted.select(range(100))
print(prompted)

instruction = "Select one or more emotions in the following sentence from this list of emotions : 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'."

results = []
responses = []
for sample in tqdm(prompted):
    prompt = instruction + sample['prompt']
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct",prompt = prompt, max_tokens = 16)

    pred = response.choices[0].text
    result = (sample['labels'],pred)
    results.append(result)
    responses.append(response)

pickle.dump(results,open('gpt3.5-turbo-instruct-results.pkl','wb'))
pickle.dump(responses,open('gpt3.5-turbo-instruct-responses.pkl','wb'))