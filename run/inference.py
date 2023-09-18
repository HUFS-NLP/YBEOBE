
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(prog="train", description="Inference Table to Text with BART")

parser.add_argument("--model-ckpt-path", type=str, help="model path")
parser.add_argument("--output-path", type=str, required=True, help="output tsv file path")
parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
parser.add_argument("--max-seq-len", type=int, default=218, help="summary max sequence length")
parser.add_argument("--threshold", type=float, default=0.5, help="inferrence threshold")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


def main(args):
    logger = logging.getLogger("inference")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f'[+] Load Tokenizer from "{args.model_ckpt_path}"')
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt_path)

    logger.info(f'[+] Load Dataset')
    test_ds = Dataset.from_json("/home/nlpgpu7/ellt/dkyo/base_edit/resource/data/nikluge-2023-ea-test.jsonl")
    with open(os.path.join(args.model_ckpt_path, "..", "label2id.json")) as f:
        label2id = json.load(f)
    labels = list(label2id.keys())
    id2label = {}
    for k, v in label2id.items():
        id2label[v] = k

    def preprocess_data(examples):
        # take a batch of texts
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=args.max_seq_len)

        return encoding

    encoded_tds = test_ds.map(preprocess_data, remove_columns=test_ds.column_names).with_format("torch")
    data_loader = DataLoader(encoded_tds, batch_size=args.batch_size)

    logger.info(f'[+] Load Model from "{args.model_ckpt_path}"')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt_path,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    logger.info("[+] Start Inference")
    
    # 변경된 부분
    # sigmoid = torch.nn.Sigmoid()
    # outputs = []
    # for batch in tqdm(data_loader):
    #     oup = model(
    #         batch["input_ids"].to(device),
    #         token_type_ids=batch["token_type_ids"].to(device),
    #         attention_mask=batch["attention_mask"].to(device)
    #     )
    #     oup.logits
        
    #     probs = sigmoid(oup.logits).cpu().detach().numpy()
    #     # next, use threshold to turn them into integer predictions
    #     y_pred = np.zeros(probs.shape)
    #     y_pred[np.where(probs >= args.threshold)] = 1

    #     outputs.extend(y_pred.tolist())
  

    sigmoid = torch.nn.Sigmoid()
    outputs = []

    for batch in tqdm(data_loader):
        oup = model(
            batch["input_ids"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )
        
        probs = sigmoid(oup.logits).cpu().detach().numpy()
        y_pred = np.zeros(probs.shape)
        
        # Check if any probability exceeds the threshold
        has_threshold_exceed = (probs >= args.threshold).any(axis=1)
        
        for i, exceed in enumerate(has_threshold_exceed):
            if not exceed:
                # If all probabilities for this item in the batch are below the threshold, set only the highest one to 1
                max_prob_index = np.argmax(probs[i])
                y_pred[i, max_prob_index] = 1
            else:
                # If some probabilities exceed the threshold, use the original logic for this item in the batch
                y_pred[i, np.where(probs[i] >= args.threshold)] = 1

        outputs.extend(y_pred.tolist())




        

    def jsonlload(fname):
        with open(fname, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            j_list = [json.loads(line) for line in lines]

        return j_list

    def jsonldump(j_list, fname):
        with open(fname, "w", encoding='utf-8') as f:
            for json_data in j_list:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

    j_list = jsonlload("/home/nlpgpu7/ellt/dkyo/base_edit/resource/data/nikluge-2023-ea-test.jsonl")
    for idx, oup in enumerate(outputs):
        j_list[idx]["output"] = {}
        for jdx, v in enumerate(oup):
            if v:
                j_list[idx]["output"][id2label[jdx]] = "True"
            else:
                j_list[idx]["output"][id2label[jdx]] = "False"

    jsonldump(j_list, args.output_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
