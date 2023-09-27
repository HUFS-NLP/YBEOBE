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
from run.LSTM_attention import LSTM_attention, LSTM_multitask, loss_function    # need to change later

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
            default="klue/roberta-base",
            help="model file path",
        )

        self.parser.add_argument(   # tokenizer
            "--tokenizer",
            type=str,
            default="klue/roberta-base",
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
            default=1,
            help="the number of gradient accumulation steps",
        )

        self.parser.add_argument(   # epochs
            "--epochs",
            type=int,
            default=10,
            help="epochs",
        )

        self.parser.add_argument(   # learning-rate
            "--learning-rate",
            type=float,
            default=2e-5,
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

class DataHandler:
    def __init__(self):
        pass

    def get_dataset(self, arguments, lh):
        lh.inform_dataset(arguments)
        self.train_dataset = Dataset.from_json(os.path.join(home, '../../../data', arguments.train_data_path))
        self.valid_dataset = Dataset.from_json(os.path.join(home, '../../../data/', arguments.valid_data_path))
        self.test_dataset = Dataset.from_json(os.path.join(home, '../../../data/', arguments.test_data_path))

    def get_labels(self):
        self.labels = list(self.train_dataset["output"][0].keys())
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

    def save_id2label(self, output_dir):
        with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf8") as file:
            json.dump(self.label2id, file)

    def preprocess_data(self, examples):
        pass

    def encode_datasets(self):
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
        self.tokenizer = AutoTokenizer.from_pretrined(arguments.tokenizer)

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

        if arguments.model_type == "LSTM_attention" or "LSTM_multitask" or "loss_function":
            self.common_params["output_hidden_states"] = True

    def load_model(self, arguments, data, lh):
        lh.inform_model(arguments)
        self.get_model_type()
        self.get_common_params(arguments, data)
        self.get_model_class(arguments)
        self.model = self.ModelClass(**self.common_params)

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
        pass

    def compute_metrics(self, eval_pred: EvalPrediction):
        pass

    def set_trainer(self, data, lh):
        lh.inform_trainer()
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
    dh.encode_datasets()

    # load model
    mh.load_model(aph.args, dh, lh)

    # set training arguments
    mh.set_training_args(aph.output_dir, aph.args, lh)

    # set trainer
    mh.set_trainer(dh, lh)

    # train model
    mh.train_model(lh)

    # save best model
    mh.save_best_model(aph.output_dir, lh, mh)

if __name__ == "__main__":
    sys.exit(main())
