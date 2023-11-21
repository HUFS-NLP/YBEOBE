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
# from run.spanEmo import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, Trainer
import logging
import sys


class BertEncoder(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(BertEncoder, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification",
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)
        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None ):
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
        return last_hidden_state





class SpanEmo(nn.Module):
    def __init__(self,model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(SpanEmo, self).__init__()
        self.bert = BertEncoder(model_path, output_hidden_states, problem_type, num_labels, id2label, label2id)
        self.joint_loss = 'joint'   #이 부분 이후에 생략가능 - LCA / BCE 같이만드는 것도 사용 가능
        self.alpha = 0.2

        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=0.5),  # dropout 값 파라미터로 조정 가능 - 논문에서 제시한 최적의 dropout 값으로 설정
            nn.Linear(self.bert.feature_size, 1)
        )


    def forward(self, input_ids, attention_mask, label_idxs, token_type_ids=None, labels=None):


        #Bert encoder
        last_hidden_state = self.bert(input_ids, attention_mask, token_type_ids=None, labels=None)


        #FFN
        batch_size, num_indices = label_idxs.size()  #batch_size=64, num_indixes=8
        logits_list = []                             #where logits will be saved

        for i in range(batch_size):
            logits_i = self.ffn(last_hidden_state[i]).squeeze(-1).index_select(dim=0, index=label_idxs[i])
            logits_list.append(logits_i)

        logits = torch.stack(logits_list, dim=0)

        #FFN old
        #label_idxs = torch.tensor([1, 2, 4, 5, 6, 7, 8, 10], device=input_ids.device)
        #logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)

        #LCAloss 계산
        if labels != None: #train
            cel = F.binary_cross_entropy_with_logits(logits,labels).cuda()
            cl = self.corr_loss(logits, labels)
            loss = ((1 - self.alpha) * cel) + (self.alpha * cl)
            return loss, logits

        else: #test
            return logits

    @staticmethod
    def corr_loss(y_hat, y_true, reduction = 'mean'):
        loss = torch.zeros(y_true.size(0)).cuda()
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() != 0:
                output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                loss[idx] = output.div(num_comparisons)

        return loss.mean() if reduction == 'mean' else loss.sum()


parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

'''
g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, default="/home/nlpgpu7/ellt/eojin/EA/", help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="Twitter/twhin-bert-base", help="model file path")
g.add_argument("--tokenizer", type=str, default="Twitter/twhin-bert-base", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=218, help="max sequence length")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=8, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=30, help="the numnber of training epochs")
g.add_argument("--learning-rate", type=float, default=4e-5, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--model-choice", type=str, default="AutoModelForSequenceClassification", help="or LSTM_attention or LSTM_multitask or loss_function")
'''


# Parameters
output_dir = "models/test_EMO"
model_path = "Twitter/twhin-bert-base"
tokenizer = "Twitter/twhin-bert-base"
max_seq_len = 218
batch_size = 32
valid_batch_size = 64
accumulate_grad_batches = 8
epochs = 40
learning_rate = 1e-5
weight_decay = 0.01
seed = 10
model_choice = "spanEmo"  # "or LSTM_attention or LSTM_multitask or loss_function"


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
    ############## ??? ############## 
    #for k, v in vars(args).items():
    #   logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {seed}")
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    logger.info(f'[+] Load Dataset')
    train_ds = Dataset.from_json("data/train_dataset_for_testEmo.jsonl")
    valid_ds = Dataset.from_json("data/valid_dataset_for_testEmo.jsonl")
    test_ds = Dataset.from_json("data/test_dataset.jsonl")

    #labels = list(train_ds["output"][0].keys())
    labels = ['▁행복한', '▁기대', '▁신뢰', '▁놀라운', '▁싫어', '겁', '▁화', '▁눈물']
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    def preprocess_data(examples):
        # take a batch of texts
        text1 = '행복한 기대하는 신뢰하는 놀라운 싫어하는 겁나는 화나는 눈물나는 감정이 든다'
        text2 = examples["input"]["form"]

        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_len)

        key_mapping = {
            "joy": "▁행복한",
            "anticipation": "▁기대",
            "trust": "▁신뢰",
            "surprise": "▁놀라운",
            "disgust": "▁싫어",
            "fear": "겁",
            "anger": "▁화",
            "sadness": "▁눈물"
        }

        # add labels
        if examples["output"] != "":
            encoding["labels"] = [0.0] * len(labels)
            for key in key_mapping:
                if examples["output"][key] == 'True':
                    encoding["labels"][label2id[key_mapping[key]]] = 1.0

        # 감정레이블의 인덱스 구하기
        input_ids = encoding['input_ids']
        encoding['label_idxs'] = [tokenizer.convert_ids_to_tokens(input_ids).index(labels[idx])
                                  for idx, _ in enumerate(labels)]

        return encoding


    encoded_tds = train_ds.map(preprocess_data, remove_columns=train_ds.column_names)
    encoded_vds = valid_ds.map(preprocess_data, remove_columns=valid_ds.column_names)
    encoded_test_ds = test_ds.map(preprocess_data, remove_columns=train_ds.column_names)

    logger.info(f'[+] Load Model from "{model_path}"')


    model_choices = {
                    "spanEmo": SpanEmo,
                    "AutoModelForSequenceClassification": AutoModelForSequenceClassification
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
        y_true = labels[1]
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
        ### args 삭제, **kwargs 삭제, model=None 삭제
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if state.epoch != 35:
                return
            

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
                5: 0.5, #fear
                6: 0.5, #anger
                7: 0.5 #sadness
            }
            # 예측값에 시그모이드 함수 적용
            predictions, label_ids, _ = trainer.predict(test_dataset)
            Sigmoid = torch.nn.Sigmoid()
            threshold_values = Sigmoid(torch.Tensor(predictions))

            # 각 클래스별로 임계값 적용하여 outputs 생성
            outputs = []
            for thresholded in threshold_values:
                output = []
                for jdx, value in enumerate(thresholded):
                    output.append(float(value) >= threshold_dict.get(jdx, 0.5))
                outputs.append(output)

            # outputs는 이제 각 예측에 대해 8개의 클래스별 boolean 값을 포함한 리스트가 됩니다.

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
                    j_list[idx]["output"][id2label[jdx]] = ["True" if v else "False", threshold_values[idx][jdx].item()]

            jsonldump(j_list, f"outputs/test_EMO.jsonl")
            


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


#if __name__ == "__main__":
#    exit(main(parser.parse_args()))

main(output_dir, model_path, tokenizer, max_seq_len, batch_size, valid_batch_size, accumulate_grad_batches, epochs, learning_rate, weight_decay, seed, model_choice)
