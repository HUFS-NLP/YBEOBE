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
#from run.LSTM_attention import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, Trainer


class LSTM_attention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(LSTM_attention, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification", 
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)                                                                          
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # AutoModelForSequenceClassification의 차원 768
        self.linear = nn.Linear(256, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
        lstm_out, (h_n, _) = self.bi_lstm(outputs)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        target_indices_row, target_indices_col = (target_positions == 1).nonzero(as_tuple=True)  # token_type_ids == 1인 것이 타겟. [0]은 배치 기준, [1]은 문장 기준
        query = lstm_out[target_indices_row, target_indices_col, :]  # [배치 크기, 타겟 길이, 임베딩 차원]

        attn_output = F.scaled_dot_product_attention(query, lstm_out, lstm_out, dropout_p=0.1)  # query[배치 크기, 타겟 길이, 임베딩 차원], key[배치 크기, 문장 길이, 임베딩 차원], value[배치 크기, 문장 길이, 임베딩 차원]
        attn_output = attn_output.mean(dim=1)  # 문장 기준 평균 내서 h_n과 차원 맞추기

        combined_output = h_n + attn_output  # 합치기(여러 방식 있지만 지금은 단순 더하기)

        logits = self.linear(combined_output)  # linear 통과 시켜서 라벨 개수(8)로 바꾸기
            
        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()  # 손실함수 바꾸는 부분
            print("logits shape:" , logits.shape)
            print(logits)
            print("label shape:" , labels.shape)
            print(labels)
            loss = loss_fct(logits, labels.float())  # 라벨 실수 형태로 받아왔었기 때문에 float
            return loss, logits
        
        else:  # 평가일 경우
            return logits


class LSTM_multitask(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(LSTM_multitask, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification", 
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)
        self.ner_model = AutoModel.from_pretrained(model_path)
                                                                                    
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # AutoModel의 차원 768
        self.linear = nn.Linear(256, num_labels)
        self.linear_ner = nn.Linear(256, 2)  # NER은 0 또는 1
        self.num_labels = num_labels


    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
        ner_outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = ner_outputs.last_hidden_state

        _, (h_n, _) = self.bi_lstm(outputs)
        ner_lstm_out, (_, _) = self.bi_lstm(last_hidden_state)

        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        logits = self.linear(h_n)  # linear 통과 시켜서 라벨 개수(8)로 바꾸기
        ner_logits = self.linear_ner(ner_lstm_out)  # linear 통과 시켜서 라벨 개수(2)로 바꾸기

           
        if labels is not None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()  # EC 손실 함수
            ner_loss_fct = nn.CrossEntropyLoss()  # NER 손실 함수
            loss = loss_fct(logits, labels.float())  # EC 손실
            ner_loss = ner_loss_fct(ner_logits.view(-1, 2), target_positions.view(-1))  # NER 손실
            total_loss = loss + ner_loss * 0.1 # 손실 가중치(현재 alpha=0.1) 설정해야 함(우리 태스크는 EC니까 NER은 EC보다 가중치 적어야)
            return total_loss, logits
        
        else:  # 평가일 경우
            return logits
        

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=1, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class loss_function_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs['logits']
        loss_fct = AsymmetricLoss()
        #loss_fct = nn.BCEWithLogitsLoss() 
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


'''
parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, default="/home/nlpgpu7/ellt/eojin/EA/", help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="beomi/KcELECTRA-base-v2022", help="model file path")
g.add_argument("--tokenizer", type=str, default="beomi/KcELECTRA-base-v2022", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=218, help="max sequence length")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=30, help="the numnber of training epochs")
g.add_argument("--learning-rate", type=float, default=4e-5, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--model-choice", type=str, default="AutoModelForSequenceClassification", help="or LSTM_attention or LSTM_multitask or loss_function")
'''

# Parameters
output_dir = "models/꼬렛_v2"
model_path = "beomi/KcELECTRA-base-v2022"
tokenizer = "beomi/KcELECTRA-base-v2022"
max_seq_len = 218
batch_size = 32
valid_batch_size = 64
accumulate_grad_batches = 8
epochs = 30
learning_rate = 4e-5
weight_decay = 0.013
seed = 10
model_choice = "AutoModelForSequenceClassification"  # "or LSTM_attention or LSTM_multitask or loss_function"


def main(output_dir, model_path, tokenizer, max_seq_len, batch_size, valid_batch_size, accumulate_grad_batches, epochs, learning_rate, weight_decay, seed, model_choice):
    # device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("train")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{output_dir}"')

    logger.info(" ====== Arguements ======")
    #for k, v in vars(args).items():
    #    logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {seed}")
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    logger.info(f'[+] Load Dataset')

    # 중복제거 
    train_ds = Dataset.from_json("data/train_dataset_for_꼬렛.jsonl")
    valid_ds = Dataset.from_json("data/valid_dataset_for_꼬렛.jsonl")
    test_ds = Dataset.from_json("data/test_dataset_for_꼬렛.jsonl")

    labels = list(train_ds["output"][0].keys())
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    def preprocess_data(examples):
        # take a batch of texts
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        target_begin = examples["input"]["target"].get("begin")
        target_end = examples["input"]["target"].get("end")

        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_len)
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

    logger.info(f'[+] Load Model from "{model_path}"')


    # config = AutoConfig.from_pretrained(args.model_path)
    # config.output_hidden_states = True
    # config.problem_type = "multi_label_classification"
    # config.num_labels = len(labels)
    # config.id2label = id2label
    # config.label2id = label2id
   
        
    model_choices = {
                    "LSTM_attention": LSTM_attention,
                    "LSTM_multitask": LSTM_multitask
                    #"loss_function": loss_function
                    }

    common_params = {
        'model_path': model_path,
        'problem_type': "multi_label_classification",
        'num_labels': len(labels),
        'id2label': id2label,
        'label2id': label2id
    }

    ModelClass = model_choices.get(model_choice)


    if model_choice in model_choices:
        common_params['output_hidden_states'] = True
        model = ModelClass(**common_params)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
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
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=valid_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
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
            if state.epoch != 30:
                return
            logger.info("Epoch ended. Running inference on test set...")
            test_dataset = encoded_test_ds
        
            trainer = loss_function_Trainer(
                model,
                targs,
                compute_metrics=compute_metrics
            )
        
            predictions, label_ids, _ = trainer.predict(test_dataset)
            sigmoid = torch.nn.Sigmoid()
            threshold_values = sigmoid(torch.Tensor(predictions))
            outputs = (threshold_values >= 0.5).tolist()
        
            j_list = jsonlload("data/test_dataset.jsonl")
            
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
        
            jsonldump(j_list, "outputs/꼬렛_v2.jsonl")

    

    trainer = loss_function_Trainer(  # or loss_function_Trainer
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
    main(output_dir, model_path, tokenizer, max_seq_len, batch_size, valid_batch_size, accumulate_grad_batches, epochs, learning_rate, weight_decay, seed, model_choice)
