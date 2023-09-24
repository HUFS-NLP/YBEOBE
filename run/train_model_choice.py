import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    AutoConfig,
    AutoModel,
    TrainerCallback
    )


from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from run.LSTM_attention import LSTM_attention


parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, default="/home/nlpgpu9/ellt/eojin/EA/", help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="beomi/KcELECTRA-base-v2022", help="model file path")
g.add_argument("--tokenizer", type=str, default="beomi/KcELECTRA-base-v2022", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=128, help="max sequence length")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=8, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=30, help="the numnber of training epochs")
g.add_argument("--learning-rate", type=float, default=4e-5, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--pre_threshold", type=float, default=0.5, help="threshold")
g.add_argument("--model-choice", type=str, default=AutoModelForSequenceClassification, help="or LSTM_attention")


def main(args):
    # device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("train")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Dataset')
    train_ds = Dataset.from_json("/home/nlpgpu9/ellt/eojin/EA/nikluge-ea-2023-train_수정_중복제거.jsonl")
    valid_ds = Dataset.from_json("/home/nlpgpu9/ellt/eojin/EA/nikluge-ea-2023-dev_수정.jsonl")
    test_ds = Dataset.from_json("/home/nlpgpu9/ellt/eojin/EA/nikluge-ea-2023-test_수정.jsonl")

    labels = list(train_ds["output"][0].keys())
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    def preprocess_data(examples):
        # take a batch of texts
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        # target_begin = examples["input"]["target"].get("begin")
        # target_end = examples["input"]["target"].get("end")

        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=args.max_seq_len)
        # add labels
        if examples["output"] != "":
            encoding["labels"] = [0.0] * len(labels)
            for key, idx in label2id.items():
                if examples["output"][key] == 'True':
                    encoding["labels"][idx] = 1.0
        
        return encoding

    encoded_tds = train_ds.map(preprocess_data, remove_columns=train_ds.column_names)
    encoded_vds = valid_ds.map(preprocess_data, remove_columns=valid_ds.column_names)
    encoded_test_ds = test_ds.map(preprocess_data, remove_columns=train_ds.column_names)

    logger.info(f'[+] Load Model from "{args.model_path}"')


    # config = AutoConfig.from_pretrained(args.model_path)
    # config.output_hidden_states = True
    # config.problem_type = "multi_label_classification"
    # config.num_labels = len(labels)
    # config.id2label = id2label
    # config.label2id = label2id
   
        
    model_choices = {
                    "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
                    "LSTM_attention": LSTM_attention
                }

    common_params = {
        'model_path': args.model_path,
        'problem_type': "multi_label_classification",
        'num_labels': len(labels),
        'id2label': id2label,
        'label2id': label2id
    }

    ModelClass = model_choices.get(args.model_choice)

    if ModelClass is None:
        raise ValueError("Invalid model choice")

    if args.model_choice == "LSTM_attention":
        common_params['output_hidden_states'] = True


    model = ModelClass(**common_params)


    targs = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model= "f1",
    )

    def multi_label_metrics(predictions, labels, threshold=args.threshold):
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

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        with open(f"{args.output_dir}log.txt", "a") as f:
            f.write(json.dumps(result) + '\n')
        return result
        

    def jsonlload(fname):
        with open(fname, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            j_list = [json.loads(line) for line in lines]
        return j_list

    def jsonldump(j_list, fname):
        with open(fname, "w", encoding='utf-8') as f:
            for json_data in j_list:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

    class TestInferenceCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            logger.info("Epoch ended. Running inference on test set...")
            test_dataset = encoded_test_ds
        
            trainer = Trainer(
                model,
                targs,
                compute_metrics=compute_metrics
            )
        
            predictions, label_ids, _ = trainer.predict(test_dataset)
            sigmoid = torch.nn.Sigmoid()
            threshold_values = sigmoid(torch.Tensor(predictions))
            outputs = (threshold_values >= 0.5).tolist()
        
            j_list = jsonlload("/home/nlpgpu9/ellt/eojin/EA/nikluge-ea-2023-test_수정.jsonl")
            
            for idx, oup in enumerate(outputs):
                j_list[idx]["output"] = {}

                if not any(oup):
                    max_index = threshold_values[idx].argmax().item()
                    oup[max_index] = True

                for jdx, v in enumerate(oup):
                    if v:
                        j_list[idx]["output"][id2label[jdx]] = "True"
                    else:
                        j_list[idx]["output"][id2label[jdx]] = "False"
        
            jsonldump(j_list, os.path.join(args.output_dir, f"test_predictions_epoch_{state.epoch}.jsonl"))

    

    trainer = Trainer(
        model,
        targs,
        train_dataset=encoded_tds,
        eval_dataset=encoded_vds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[TestInferenceCallback()]
    )
    trainer.train()


if __name__ == "__main__":
    exit(main(parser.parse_args()))
