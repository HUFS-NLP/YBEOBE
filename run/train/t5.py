import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
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

def train_test_t5():
    with open('model_settings.jsonl', 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_config = config["test_T5"]

    output_dir = model_config["output_dir"]
    model_path = model_config["model_path"]
    tokenizer_path = model_config["tokenizer_path"]
    max_seq_length = model_config["max_seq_length"]
    batch_size = model_config["batch_size"]
    valid_batch_size = model_config["valid_batch_size"]
    accumulate_grad_batches = model_config["accumulate_grad_batches"]
    epochs = model_config["epochs"]
    learning_rate = model_config["learning_rate"]
    weight_decay = model_config["weight_decay"]
    seed = model_config["seed"]
    model_max_length = model_config["model_max_length"]
    train_ds_path = model_config["train_ds_path"]
    valid_ds_path = model_config["valid_ds_path"]
    test_ds_path = model_config["test_ds_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_ds = Dataset.from_json(train_ds_path)
    valid_ds = Dataset.from_json(valid_ds_path)
    test_ds = Dataset.from_json(test_ds_path)

    # train_ds = Dataset.from_json("train_dataset_sample.jsonl")
    # valid_ds = Dataset.from_json("valid_dataset_sample.jsonl")
    # test_ds = Dataset.from_json("test_dataset_sample.jsonl")

    labels = list(train_ds["output"][0].keys())
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    def preprocess_data(examples):
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        target_begin = examples["input"]["target"].get("begin")
        target_end = examples["input"]["target"].get("end")

        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_length)
        # add labels
        if examples["output"] != "":
            encoding["labels"] = [0.0] * len(labels)
            for key, idx in label2id.items():
                if examples["output"][key] == 'True':
                    encoding["labels"][idx] = 1.0


        # 타겟 찾기 (attention, multitask 위해)
        encoding["target_positions"] = [0] * len(encoding['input_ids'])  # 문장 길이만큼 0으로 초기화

        if text2 != None:
            encoded_target = tokenizer(text2, add_special_tokens=False)["input_ids"]
            encoded_text = tokenizer(text1, add_special_tokens=False)["input_ids"]

            for i in range(len(encoded_text) - len(encoded_target) + 1):
                if encoded_text[i:i+len(encoded_target)] == encoded_target:
                    target_begin = i + 1  # [CLS] 떄문에 + 1
                    target_end = i + len(encoded_target) + 1  # 나중에 리스트 슬라이싱 때문에 + 1
                    break

        # Mark the target positions with 1
            for i in range(target_begin, target_end):
                encoding["target_positions"][i] = 1  # 타겟이면 1, 타겟이 아니면 0

        return encoding

    encoded_tds = train_ds.map(preprocess_data, remove_columns=train_ds.column_names)
    encoded_vds = valid_ds.map(preprocess_data, remove_columns=valid_ds.column_names)
    encoded_test_ds = test_ds.map(preprocess_data, remove_columns=train_ds.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        problem_type = "multi_label_classification",
        num_labels=len(labels),
        id2label = id2label,
        label2id = label2id,
        )

    targs = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=valid_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        )

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

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        with open(f"{output_dir}log.txt", "a") as f:
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
            test_dataset = encoded_test_ds

            trainer = Trainer(
                model,
                targs,
                compute_metrics=compute_metrics
            )


            # 각 클래스별 임계값 설정
            threshold_dict = {
                0: 0.5, #joy
                1: 0.5, #anticipation
                2: 0.5, #trust
                3: 0.5, #surprise
                4: 0.5, #disgust
                5: 0.5, #fear
                6: 0.5, #anger
                7: 0.5 #sadness
            }
            # 예측값에 시그모이드 함수 적용
            predictions, label_ids, _ = trainer.predict(test_dataset)
            softmax = torch.nn.Softmax(dim=-1)
            
            threshold_values = softmax(torch.Tensor(predictions[0]))

            # 각 클래스별로 임계값 적용하여 outputs 생성
            outputs = []
            for thresholded in threshold_values:
                output = []
                for jdx, value in enumerate(thresholded):
                    output.append(float(value) >= threshold_dict.get(jdx, 0.5))
                outputs.append(output)

            # outputs는 이제 각 예측에 대해 8개의 클래스별 boolean 값을 포함한 리스트가 됩니다.

            j_list = jsonlload("test_dataset.jsonl")

            for idx, oup in enumerate(outputs):
                j_list[idx]["output"] = {}

                # oup에서 True 또는 1인 값의 개수를 확인
                true_count = sum(oup)

                if true_count >= 4:
                    # threshold_values의 상위 3개 인덱스를 찾음
                    top_three_indices = np.argsort(threshold_values[idx])[-3:]
                    # oup를 모두 False로 초기화
                    oup = [False] * len(oup)
                    # 상위 3개 인덱스만 True로 설정
                    for top_idx in top_three_indices:
                        oup[top_idx] = True
                            
                elif not any(oup):
                    max_index = threshold_values[idx].argmax().item()
                    oup[max_index] = True

                for jdx, v in enumerate(oup):
                    j_list[idx]["output"][id2label[jdx]] = ["True" if v else "False", threshold_values[idx][jdx].item()]

            jsonldump(j_list, os.path.join(output_dir, f"test_predictions_epoch_{state.epoch}.jsonl"))

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
