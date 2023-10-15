import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

home = '/home/npgpu7'

pearl_base = os.path.join(home, '../../../../../../ellt/dkyo/base_edit')


os.chdir(pearl_base)

bert_model = AutoModelForSequenceClassification.from_pretrained("output_2/checkpoint-17197")
bert_tokenizer = AutoTokenizer.from_pretrained("output_2/checkpoint-17197")

t5_base = os.path.join(home, '../outputs')
os.chdir(t5_base)

t5_model = AutoModelForSequenceClassification.from_pretrained("2e-4")
t5_tokenizer = AutoTokenizer.from_pretrained("2e-4")

bert_ratio = 0.5
t5_ratio = 0.5

threshold = 0.5

sentiments = ['joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']

import pdb;pdb.set_trace()

with open('ensemble_output.jsonl', 'w', encoding='utf8') as writer:
    with open('../../../data/nikluge-ea-2023-test-modified.jsonl', 'r', encoding='utf8') as file:

        import pdb; pdb.set_trace()

        for line in file:
            entry = json.loads(line)

            bert_input = bert_tokenizer(entry['input']['form'], return_tensors="pt")
            t5_input = t5_tokenizer(entry['input']['form'], return_tensors="pt")

            with torch.no_grad():
                bert_output = bert_model(**bert_input).logits
                t5_output = t5_model(**t5_input).logits

            bert_probs = F.softmax(bert_output, dim=-1)
            t5_probs = F.softmax(t5_output, dim=-1)

            ensemble_probs = (bert_ratio * bert_probs) + (t5_ratio * t5_probs)
            ensemble_preds = (ensemble_probs > threshold).numpy().astype(int)

            output = {sentiment: str(bool(pred)) for sentiment, pred in zip(sentiments, ensemble_preds.flatten())}
            writer.write(json.dumps({"id": entry['id'], "input": entry['input'], "output": output}) + '\n') # 잘 돌아가는지 확인하면 형식 맞춰서
