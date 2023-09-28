# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 08:00:00 2023

This module is for training model.

Example:

    $ python train.py \
        --output-dir outputs \
        --model-path klue/roberta-base \
        --tokenizer klue/roberta-base \
        --max-seq-len 218 \
        --batch-size 32 \
        --valid-batch-size 32 \
        --accumulate-grad-batches 1 \
        --epochs 10 \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --gpus 0 \
        --seed 42 \
        --model-type AutoModelForSequenceClassification \
        --train-data-path nikluge-ea-2023-train-modified.jsonl \
        --valid-data-path nikluge-ea-2023-dev-modified.jsonl \
        --test-data-path nikluge-ea-2023-test-modified.jsonl

Todo:
    - init of classes
    - docstrings
"""

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
from LSTM_attention import LSTM_attention, LSTM_multitask, loss_function    # need to change later

home = os.path.dirname(os.path.abspath(__file__))

class ArgumentParserHandler:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog="train", description="Train model")

        self.parser.add_argument(   # output-dir
            "--output-dir",
            type=str,
            required=True,
            help="output directory path to save artifacts",
        )

        self.parser.add_argument(   # model-path
            "--model-path",
            type=str,
            default="beomi/KcELECTRA-base-v2022",
            help="model file path",
        )

        self.parser.add_argument(   # tokenizer
            "--tokenizer",
            type=str,
            default="beomi/KcELECTRA-base-v2022",
            help="tokenizer file path",
        )

        self.parser.add_argument(   # max-seq-len
            "--max-seq-len",
            type=int,
            default=218,
            help="maximum sequence length",
        )

        self.parser.add_argument(   # batch-size
            "--batch-size",
            type=int,
            default=32, # or 64
            help="batch size",
        )

        self.parser.add_argument(   # valid-batch-size
            "--valid-batch-size",
            type=int,
            default=32, # or 64
            help="validation batch size",
        )

        self.parser.add_argument(   # accumulate-grad-batches
            "--accumulate-grad-batches",
            type=int,
            default=8,
            help="the number of gradient accumulation steps",
        )

        self.parser.add_argument(   # epochs
            "--epochs",
            type=int,
            default=30,
            help="epochs",
        )

        self.parser.add_argument(   # learning-rate
            "--learning-rate",
            type=float,
            default=4e-5,
            help="max learning rate",
        )

        self.parser.add_argument(   # weight-decay
            "--weight-decay",
            type=float,
            default=0.01,
            help="weight decay",
        )

        self.parser.add_argument(   # gpus
            "--gpus",
            type=int,
            default=0,
            help="the number of gpus",
        )

        self.parser.add_argument(   # seed
            "--seed",
            type=int,
            default=42,
            help="random seed",
        )

        self.parser.add_argument(   # model-type
            "--model-type",
            type=str,
            default="AutoModelForSequenceClassification",
            help="or LSTM_attention or LSTM_multitask or loss_function",
        )

        self.parser.add_argument(   # ensemble
            "--ensemble",
            type=bool,
            default=False,
            help="whether you do ensemble",
        )

        self.parser.add_argument(   # ensemble-models
            "--ensemble-models",
            type=str,   # need to split with ',' later
            default="klue/roberta-base,klue/roberta-large",
            help="ensemble models",
        )

        self.parser.add_argument(   # train-data-path
            "--train-data-path",
            type=str,
            default="nikluge-ea-2023-train-modified.jsonl",
            help="train data path",
        )

        self.parser.add_argument(   # valid-data-path
            "--valid-data-path",
            type=str,
            default="nikluge-ea-2023-dev-modified.jsonl",
            help="valid data path",
        )

        self.parser.add_argument(   # test-data-path
            "--test-data-path",
            type=str,
            default="nikluge-ea-2023-test-modified.jsonl",
            help="test data path",
        )

        self.parser.add_argument(   # show-test-inference
            "--show-test-inference",
            type=bool,
            default=True,
            help="whether it shows test inference callback",
        )

        self.args = self.parser.parse_args()

        self.output_dir = os.path.join(home, "../outputs-model", self.args.output_dir)

class LoggerHandler:
    def __init__(self):
        self.logger = logging.getLogger("train")
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            self.handler = logging.StreamHandler(sys.stdout)
            self.handler.setFormatter(logging.Formatter("[%(asctime)s %(message)s]"))
            self.logger.addHandler(self.handler)

    def inform_output_dir(self, output_dir):
        self.logger.info("[+] Save output to %s", output_dir)

    def save_log(self, output_dir):
        log_file = os.path.join(output_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        self.logger.addHandler(file_handler)

        sys.stdout = open(log_file, 'a')
        sys.stderr = sys.stdout

    def inform_arguments(self, arguments):
        self.logger.info(" ====== Arguments ====== ")
        for key, value in vars(arguments).items():
            self.logger.info("%-25s: %s", key, value)

    def inform_seed(self, arguments):
        self.logger.info("[+] Set random seed to %d", arguments.seed)

    def inform_tokenizer(self, arguments):
        self.logger.info("[+] Load tokenizer from %s", arguments.tokenizer)

    def inform_dataset(self, arguments):
        self.logger.info("[+] Load datasets from")
        self.logger.info("    - train: %s", arguments.train_data_path)
        self.logger.info("    - dev: %s", arguments.valid_data_path)
        self.logger.info("    - test: %s", arguments.test_data_path)

    def inform_labels(self):
        self.logger.info("[+] Get labels")

    def inform_model(self, arguments):
        self.logger.info("[+] Load model from %s", arguments.model_path)

    def inform_training_arguments(self):
        self.logger.info("[+] Set training arguments")

    def inform_trainer(self):
        self.logger.info("[+] Set trainer")

    def inform_train_start(self):
        self.logger.info("[+] Train model")

    def inform_train_complete(self):
        self.logger.info("[+] Training complete")

    def inform_best_model_path(self, model_info):
        self.logger.info(f"Best model saved at: {model_info.best_model_path}")

    def inform_inference_start(self):
        self.logger.info("Epoch ended. Running inference on test set...")        

class DataHandler:
    def __init__(self):
        pass

    def get_dataset(self, arguments, lh):
        lh.inform_dataset(arguments)

        self.train_data_full_path = os.path.join(home, '../../../data', arguments.train_data_path)
        self.valid_data_full_path = os.path.join(home, '../../../data/', arguments.valid_data_path)
        self.test_data_full_path = os.path.join(home, '../../../data/', arguments.test_data_path)

        self.train_dataset = Dataset.from_json(self.train_data_full_path)
        self.valid_dataset = Dataset.from_json(self.valid_data_full_path)
        self.test_dataset = Dataset.from_json(self.test_data_full_path)

    def get_labels(self):
        self.labels = list(self.train_dataset["output"][0].keys())
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

    def save_id2label(self, output_dir):
        with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf8") as file:
            json.dump(self.label2id, file)

    def preprocess_data(self, examples):
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        target_begin = examples["input"]["target"].get("begin")
        target_end = examples["input"]["target"].get("end")

        encoding = self.mh.tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.arguments.max_seq_len,
        )

        # add labels
        if examples["output"] != "":
            encoding["labels"] = [0.0] * len(self.labels)
            for key, idx in self.label2id.items():
                if examples["output"][key] == 'True':
                    encoding["labels"][idx] = 1.0

        # 타겟 찾기 (attention, multitask 위해)
        encoding["target_positions"] = [0] * len(encoding['input_ids'])  # 문장 길이만큼 0으로 초기화

        if text2 != None:
            encoded_target = self.mh.tokenizer(text2, add_special_tokens=False)["input_ids"]
            encoded_text = self.mh.tokenizer(text1, add_special_tokens=False)["input_ids"]

            for i in range(len(encoded_text) - len(encoded_target) + 1):
                if encoded_text[i:i+len(encoded_target)] == encoded_target:
                    target_begin = i + 1  # [CLS] 떄문에 + 1
                    target_end = i + len(encoded_target) + 1  # 나중에 리스트 슬라이싱 때문에 + 1
                    break

        # Mark the target positions with 1
            for i in range(target_begin, target_end):
                encoding["target_positions"][i] = 1  # 타겟이면 1, 타겟이 아니면 0

        return encoding

    def encode_datasets(self, mh, arguments):
        # it's better to use partial from functools instead of using them as below since mh and arguments are not truly properties of this object, but rather are just needed for the preprocess_data function only
        self.mh = mh
        self.arguments = arguments

        self.encoded_tds = self.train_dataset.map(self.preprocess_data, remove_columns=self.train_dataset.column_names)
        self.encoded_vds = self.valid_dataset.map(self.preprocess_data, remove_columns=self.valid_dataset.column_names)
        self.encoded_test_ds = self.test_dataset.map(self.preprocess_data, remove_columns=self.test_dataset.column_names)

class ModelHandler:
    def __init__(self):
        pass

    def set_random_seed(self, arguments, lh):
        lh.inform_seed(arguments)
        np.random.seed(arguments.seed)
        os.environ["PYTHONHASHSEED"] = str(arguments.seed)
        torch.manual_seed(arguments.seed)
        torch.cuda.manual_seed(arguments.seed)   # type: ignore

    def get_tokenizer(self, arguments, lh):
        lh.inform_tokenizer(arguments)
        self.tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)

    def set_config(self, arguments, dh):    # not using
        self.config = AutoConfig.from_pretrained(arguments.model_path)
        self.config.output_hidden_states = True
        self.config.problem_type = "multi_label_classification"
        self.config.num_labels = len(dh.labels)
        self.config.id2label = dh.id2label
        self.config.label2id = dh.label2id

    def get_model_type(self):
        self.model_type = {
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
            "LSTM_attention": LSTM_attention,
            "LSTM_multitask": LSTM_multitask,
            "loss_function": loss_function,
        }

    def get_common_params(self, arguments, data):
        self.common_params = {
            'model_path': arguments.model_path,
            'problem_type': "multi_label_classification",
            'num_labels': len(data.labels),
            'id2label': data.id2label,
            'label2id': data.label2id,
        }

    def get_model_class(self, arguments):
        self.ModelClass = self.model_type.get(arguments.model_type)

        if not self.ModelClass:
            raise ValueError("Model type not found")

        if arguments.model_type in ["LSTM_attention", "LSTM_multitask", "loss_function"]:
            self.common_params["output_hidden_states"] = True

    def load_model(self, arguments, data, lh):
        lh.inform_model(arguments)
        self.get_model_type()
        self.get_common_params(arguments, data)
        self.get_model_class(arguments)
        self.model = self.ModelClass(**self.common_params)

    def custom_optimizer(self): # not using
        return Adam([
            {"params": self.model.parameters(), "lr": 1e-5},    # pretrained LM
            {"params": self.model.bi_lstm.parameters(), "lr": 1e-3},  # bi-LSTM
            {"params": self.model.parameters(), "lr": 1e-3}     # linear layer
        ])

    def set_training_args(self, output_dir, arguments, lh):
        lh.inform_training_arguments()
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=arguments.learning_rate,
            per_device_train_batch_size=arguments.batch_size,
            per_device_eval_batch_size=arguments.valid_batch_size,
            num_train_epochs=arguments.epochs,
            weight_decay=arguments.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
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

    def compute_metrics(self, eval_pred: EvalPrediction):
        output_dir = self.training_args.output_dir
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        result = self.multi_label_metrics(predictions=preds, labels=eval_pred.label_ids)
        with open(f"{output_dir}log.txt", "a") as f:
            f.write(json.dumps(result) + '\n')
        return result

    def set_trainer(self, arguments, data, lh, cfh):
        lh.inform_trainer()
        
        if arguments.show_test_inference:
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=data.encoded_tds,
                eval_dataset=data.encoded_vds,
                compute_metrics=self.compute_metrics,
                callbacks=[TestInferenceCallback(lh, data, self, cfh)],
            )

        else:
            self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=data.encoded_tds,
                eval_dataset=data.encoded_vds,
                compute_metrics=self.compute_metrics,
            )

    def train_model(self, lh):
        lh.inform_train_start()
        self.trainer.train()
        lh.inform_train_complete()

    def save_best_model(self, output_dir, lh, mh):
        self.best_model_path = os.path.join(output_dir, "best_model")
        self.trainer.save_model(self.best_model_path)
        lh.inform_best_model_path(mh)

class CustomFileHandler:
    def __init__(self):
        pass

    def make_output_dir(self, output_dir, lh):
        os.makedirs(output_dir, exist_ok=True)
        lh.inform_output_dir(output_dir)

    def jsonlload(self, fname):
        with open(fname, encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            j_list = [json.loads(line) for line in lines]
        return j_list

    def jsonldump(self, j_list, fname):
        with open(fname, "w", encoding="utf-8") as f:
            for json_data in j_list:
                f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

class TestInferenceCallback(TrainerCallback):
    def __init__(self, lh, dh, mh, cfh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lh = lh
        self.dh = dh
        self.mh = mh
        self.cfh = cfh

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.lh.inform_inference_start()
        test_dataset = self.dh.encoded_test_ds

        trainer = Trainer(
            self.mh.model,
            self.mh.targs,
            compute_metrics=self.mh.compute_metrics,
        )

        predictions, label_ids, _ = trainer.predict(test_dataset)
        sigmoid = torch.nn.Sigmoid()
        threshold_values = sigmoid(torch.Tensor(predictions))
        outputs = (threshold_values >= 0.5).tolist()

        j_list = self.cfh.jsonlload(self.dh.test_data_full_path)

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

            for jdx, value in enumerate(oup):
                if value:
                    j_list[idx]["output"][self.dh.id2label[jdx]] = "True"
                else:
                    j_list[idx]["output"][self.dh.id2label[jdx]] = "False"

        self.cfh.jsonldump(j_list, os.path.join(self.lh.output_dir, f"test_predictions_epoch_{state.epoch}.jsonl"))

def main():
    aph = ArgumentParserHandler()
    lh = LoggerHandler()
    dh = DataHandler()
    mh = ModelHandler()
    cfh = CustomFileHandler()

    # make output directory
    cfh.make_output_dir(aph.output_dir, lh)

    # save log
    lh.save_log(aph.output_dir)

    # inform arguments
    lh.inform_arguments(aph.args)

    # set seed
    mh.set_random_seed(aph.args, lh)

    # load tokenizer
    mh.get_tokenizer(aph.args, lh)

    # load datasets
    dh.get_dataset(aph.args, lh)

    # get & save labels
    dh.get_labels()
    dh.save_id2label(aph.output_dir)

    # encode datasets
    dh.encode_datasets(mh, aph.args)

    # set config - not used
    # mh.set_config(aph.args, dh)

    # load model
    mh.load_model(aph.args, dh, lh)

    # custom optimize - not used
    # mh.custom_optimizer()

    # set training arguments
    mh.set_training_args(aph.output_dir, aph.args, lh)

    # set trainer
    mh.set_trainer(aph.args, dh, lh, cfh)

    # train model
    mh.train_model(lh)

    # save best model
    mh.save_best_model(aph.output_dir, lh, mh)

if __name__ == "__main__":
    sys.exit(main())
