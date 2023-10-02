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
from run.LSTM_attention import LSTM_attention, LSTM_multitask, loss_function


parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, default="/home/nlpgpu7/ellt/eojin/EA/", help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="beomi/KcELECTRA-base-v2022", help="model file path")
g.add_argument("--tokenizer", type=str, default="beomi/KcELECTRA-base-v2022", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=218, help="max sequence length")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=8, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=30, help="the numnber of training epochs")
g.add_argument("--learning-rate", type=float, default=4e-5, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--model-choice", type=str, default="AutoModelForSequenceClassification", help="or LSTM_attention or LSTM_multitask or loss_function")


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
    train_ds = Dataset.from_json("/home/nlpgpu7/ellt/dkyo/base_edit/resource/data/nikluge-2023-ea-train.jsonl")
    valid_ds = Dataset.from_json("/home/nlpgpu7/ellt/dkyo/base_edit/resource/data/nikluge-2023-ea-dev.jsonl")
    test_ds = Dataset.from_json("/home/nlpgpu7/ellt/dkyo/base_edit/resource/data/nikluge-2023-ea-test.jsonl")

    labels = list(train_ds["output"][0].keys())
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    def preprocess_data(examples):
        # take a batch of texts
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        target_begin = examples["input"]["target"].get("begin")
        target_end = examples["input"]["target"].get("end")

        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=args.max_seq_len)
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

    logger.info(f'[+] Load Model from "{args.model_path}"')


    # config = AutoConfig.from_pretrained(args.model_path)
    # config.output_hidden_states = True
    # config.problem_type = "multi_label_classification"
    # config.num_labels = len(labels)
    # config.id2label = id2label
    # config.label2id = label2id
   
        
    model_choices = {
                    "LSTM_attention": LSTM_attention,
                    "LSTM_multitask": LSTM_multitask,
                    "loss_function": loss_function
                    }

    common_params = {
        'model_path': args.model_path,
        'problem_type': "multi_label_classification",
        'num_labels': len(labels),
        'id2label': id2label,
        'label2id': label2id
    }

    ModelClass = model_choices.get(args.model_choice)


    if args.model_choice in model_choices:
        common_params['output_hidden_states'] = True
        model = ModelClass(**common_params)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        # config=config
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
        )


    # def custom_optimizer(model):  # 나중에 모델 별로 학습률 다르게 할 때 추가
    #     return Adam([
    #         {'params': model.model.parameters(), 'lr': 1e-5},  # 사전학습된 모델
    #         {'params': model.bi_lstm.parameters(), 'lr': 1e-3},  # 양방향 LSTM
    #         {'params': model.linear.parameters(), 'lr': 1e-3}  # 선형 레이어
    #     ])
    

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
        
            # 각 클래스별 임계값 설정
            threshold_dict = {
                0: 0.5, #joy
                1: 0.5, #anticipation
                2: 0.5, #trust
                3: 0.5, #surprise
                4: 0.5, #disgust
                5: 0.4, #fear
                6: 0.43, #anger
                7: 0.5 #sadness
            }
            # 예측값에 시그모이드 함수 적용
            predictions, label_ids, _ = trainer.predict(test_dataset)
            sigmoid = torch.nn.Sigmoid()
            threshold_values = sigmoid(torch.Tensor(predictions))

            # 각 클래스별로 임계값 적용하여 outputs 생성
            outputs = []
            for thresholded in threshold_values:
                output = []
                for jdx, value in enumerate(thresholded):
                    output.append(float(value) >= threshold_dict.get(jdx, 0.5))
                outputs.append(output)

            # outputs는 이제 각 예측에 대해 8개의 클래스별 boolean 값을 포함한 리스트가 됩니다.

            j_list = jsonlload("/home/nlpgpu7/ellt/dkyo/base_edit/resource/data/nikluge-2023-ea-test.jsonl")
            
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
