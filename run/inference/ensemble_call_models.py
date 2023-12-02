import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

home = os.path.dirname(os.path.abspath(__file__))
os.chdir(home)

kcelectra_path = "kcelectra"
t5_path = "t5"
twhin_bert_path = "twhinbert/model_weights.pth"

kcelectra_model = AutoModelForSequenceClassification.from_pretrained(kcelectra_path, local_files_only=True)

kcelectra_tokenizer = AutoTokenizer.from_pretrained(kcelectra_path, local_files_only=True)

twhin_bert_model = AutoModelForSequenceClassification.from_pretrained("Twitter/twhin-bert-base", num_labels=8)
twhin_bert_model.load_state_dict(torch.load(twhin_bert_path, map_location=torch.device('cpu')))
twhin_bert_model.eval()

# twhin_bert_model = AutoModelForSequenceClassification.from_pretrained(twhin_bert_path, local_files_only=True)

t5_model = AutoModelForSequenceClassification.from_pretrained(t5_path, local_files_only=True)

t5_tokenizer = AutoTokenizer.from_pretrained(t5_path, local_files_only=True)

twhin_bert_tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base", local_files_only=False)

kcelectra_ratio = 0.4
t5_ratio = 0.2
twhin_ratio = 0.4

threshold = 0.5

sentiments = ['joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']

cnt = 0

with open('ensemble_output_442.jsonl', 'w', encoding='utf-8') as writer:
    with open('data/nikluge-ea-2023-test-modified.jsonl', 'r', encoding='utf-8') as file:

        for line in file:
            entry = json.loads(line)

            kcelectra_input = kcelectra_tokenizer(entry['input']['form'], return_tensors="pt")
            t5_input = t5_tokenizer(entry['input']['form'], return_tensors="pt")
            twhin_input = twhin_bert_tokenizer(entry['input']['form'], return_tensors="pt")

            with torch.no_grad():
                kcelectra_output = kcelectra_model(**kcelectra_input).logits
                t5_output = t5_model(**t5_input).logits
                twhin_output = twhin_bert_model(**twhin_input).logits

            kcelectra_probs = F.softmax(kcelectra_output, dim=-1)
            t5_probs = F.softmax(t5_output, dim=-1)
            twhin_probs = F.softmax(twhin_output, dim=-1)

            ensemble_probs = (kcelectra_ratio * kcelectra_probs) + (t5_ratio * t5_probs) + (twhin_ratio * twhin_probs)
            ensemble_preds = (ensemble_probs > threshold).numpy().astype(int)

            output = {sentiment: str(bool(pred)) for sentiment, pred in zip(sentiments, ensemble_preds.flatten())}
            writer.write(json.dumps({"id": entry['id'], "input": entry['input'], "output": output}, ensure_ascii=False) + '\n') # 잘 돌아가는지 확인하면 형식 맞춰서

            cnt += 1
            print(f'{cnt}/4748')

print('===== Done =====')
