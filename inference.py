import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, Trainer, BertConfig
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, average_precision_score
from run.LSTM_attention import *

# config 불러옴
with open('inference_settings.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

target_models = []

def general_inference(model_name, postprocess=True):
    print(f"Start inference for {model_name}...")
    if model_name == "이상해풀_v7":
        model_name = "이상해풀풀_v7"
    inference_config = config[model_name]
    model_path = inference_config["model_ckpt_path"]
    output_path = inference_config["output_path"]
    batch_size = inference_config["batch_size"]
    max_seq_length = inference_config["max_seq_length"]
    threshold = inference_config["threshold"]
    num_beams = inference_config["num_beams"]

    if model_name == "이상해풀_v7":
        output_path = "outputs/이상해풀_v7.jsonl"

    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer 불러옴
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # test dataset 불러옴
    test_ds_path = "data/test_dataset_sample.jsonl"
    test_ds = Dataset.from_json(test_ds_path)

    # id & label 매칭
    with open("data/label2id.json") as f:
        label2id = json.load(f)
    labels = list(label2id.keys())
    id2label = {}
    for k, v in label2id.items():
        id2label[v] = k

    def preprocess_data(examples):
        # texts를 batch로 받아옴
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        target_begin = examples["input"]["target"].get("begin")
        target_end = examples["input"]["target"].get("end")

        # 인코딩
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_length)

        # target position을 0으로 초기화
        encoding["target_positions"] = [0] * len(encoding['input_ids'])

        # target position을 1로 표시
        if text2 != None:
            encoded_target = tokenizer(text2, add_special_tokens=False)["input_ids"]
            encoded_text = tokenizer(text1, add_special_tokens=False)["input_ids"]

            # target이 시작하는 위치를 찾음
            for i in range(len(encoded_text) - len(encoded_target) + 1):
                if encoded_text[i:i+len(encoded_target)] == encoded_target:
                    target_begin = i + 1 # [CLS] 떄문에 + 1
                    target_end = i + len(encoded_target) + 1 # 나중에 리스트 슬라이싱 때문에 + 1
                    break

            for i in range(target_begin, target_end):
                encoding["target_positions"][i] = 1 # 타겟이면 1, 타겟이 아니면 0

        return encoding

    # test dataset 전처리
    encoded_tds = test_ds.map(preprocess_data, remove_columns=test_ds.column_names).with_format("torch")

    # data loader 설정
    data_loader = DataLoader(encoded_tds, batch_size=batch_size)

    # model 불러옴
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        output_hidden_states=True,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True
    )
    model.to(device)

    # inference
    model.eval()
    torch.set_grad_enabled(False)

    sigmoid = torch.nn.Sigmoid()
    outputs = []

    if postprocess:
        for batch in tqdm(data_loader):
            if model_name in target_models:
                oup = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    target_positions=batch["target_positions"].to(device)
                )

            else:
                oup = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )

            probs = sigmoid(oup.logits).cpu().detach().numpy()
            y_pred = np.zeros(probs.shape)
            has_threshold_exceed = (probs >= threshold).any(axis=1)
        
            for i, exceed in enumerate(has_threshold_exceed):
                if not exceed:
                    max_prob_index = np.argmax(probs[i])
                    y_pred[i, max_prob_index] = 1
                else:
                    y_pred[i, np.where(probs[i] >= threshold)] = 1

            outputs.extend(y_pred.tolist())

    else:
        for batch in tqdm(data_loader):
            if model_name in target_models:
                oup = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    target_positions=batch["target_positions"].to(device)
                )

            else:
                oup = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )

            oup.logits

            probs = sigmoid(oup.logits).cpu().detach().numpy()
            y_pred = np.zeros(probs.shape)
            y_pred[probs >= threshold] = 1

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

    j_list = jsonlload(test_ds_path)
    for idx, oup in enumerate(outputs):
        j_list[idx]["output"] = {}
        for jdx, v in enumerate(oup):
            if v:
                j_list[idx]["output"][id2label[jdx]] = "True"
            else:
                j_list[idx]["output"][id2label[jdx]] = "False"

    jsonldump(j_list, output_path)

def pth_inference(model_name):
    print(f"Start inference for {model_name}...")
    inference_config = config[model_name]
    model_path = inference_config["model_ckpt_path"]
    output_path = inference_config["output_path"]
    batch_size = inference_config["batch_size"]
    max_seq_len = inference_config["max_seq_length"]
    threshold = inference_config["threshold"]
    num_beams = inference_config["num_beams"]

    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer 불러옴
    twhin_list = ["가라도스_v1", "갸라도스_v2", "수댕이_v1", "꼬링크_v1", "꼬링크_v2", "럭시오_v1", "test2"]
    kc_list = ["찌르버드_v2", "찌르호크_v1", "조로아크_v5", "윈디_v1", "샤비_v1"]

    if model_name in twhin_list:
        tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-large")
    elif model_name in kc_list:
        tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    # test dataset 불러옴
    test_ds_path = "data/test_dataset_sample.jsonl"
    test_ds = Dataset.from_json(test_ds_path)

    if model_name != "찌르호크_v1":
        def preprocess_data(examples):
            text1 = examples["input"]["form"]
            text2 = examples["input"]["target"]["form"]
            target_begin = examples["input"]["target"].get("begin")
            target_end = examples["input"]["target"].get("end")

            # encode them
            encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_len)
            # add labels
            if examples["output"] != "":
                encoding["labels"] = [0.0] * 8
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
    else:
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
                encoding["labels"] = [0.0] * 8
                for key, idx in label2id.items():
                    if examples["output"][key] == 'True':
                        encoding["labels"][idx] = 1.0

            if text2 != None:
                encoded_target = tokenizer(text2, add_special_tokens=False)["input_ids"]
                encoded_text = tokenizer(text1, add_special_tokens=False)["input_ids"]
                
                for i in range(len(encoded_text) - len(encoded_target) + 1):
                    if encoded_text[i:i+len(encoded_target)] == encoded_target:
                        target_begin = i + 1
                        target_end = i + len(encoded_target) + 1
                        break
            else:
                target_begin, target_end = -1, -1

            encoding["target_positions"] = target_begin, target_end
            
            return encoding


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

        tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        youden_j = sensitivity + specificity - 1

        precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')

        recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')


        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy,
                   'youden_j': youden_j,
                   'precision': precision,
                   'recall': recall
                   }
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return multi_label_metrics(predictions=preds, labels=p.label_ids)

    def jsonlload(fname):
        with open(fname, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            j_list = [json.loads(line) for line in lines]
        return j_list

    def jsonldump(j_list, fname):
        with open(fname, "w", encoding='utf-8') as f:
            for json_data in j_list:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

    with open("data/label2id.json") as f:
        label2id = json.load(f)
    id2label = {idx: label for label, idx in label2id.items()}

    # test dataset 전처리
    encoded_test_ds = test_ds.map(preprocess_data, remove_columns=test_ds.column_names)

    # lstm_attention2 models
    lstm_attention2_list = ["윈디_v1", "샤비_v1", "가라도스_v1", "갸라도스_v2", "수댕이_v1", "꼬링크_v1", "꼬링크_v2", "럭시오_v1", "test2"]
    lstm_attention_plus_list = ["찌르버드_v2"]
    lstm_attention_tp_list = ["찌르호크_v1"]
    lstm_attention1_list = ["조로아크_v5"]

    common_params = {
        "model_path": model_path,
        "problem_type": "multi_label_classification",
        "num_labels": 8,
        "id2label": id2label,
        "label2id": label2id,
        "output_hidden_states": True,
    }

    if model_name in twhin_list:
        common_params["model_path"] = "Twitter/twhin-bert-large"
        common_params["isKc"] = False
    elif model_name in kc_list:
        common_params["model_path"] = "beomi/KcELECTRA-base-v2022"
        common_params["isKc"] = True

    if model_name == "윈디_v1":
        common_params["isWindy"] = True
    else:
        common_params["isWindy"] = False

    if model_name in lstm_attention2_list:
        model = LSTM_attention(**common_params)
    elif model_name == "찌르버드_v2":
        model = bird_LSTM_attention(**common_params)
    elif model_name == "찌르호크_v1":
        model = hawk_LSTM_attention(**common_params)
    elif model_name == "조로아크_v5":
        model = ark_LSTM_attention(**common_params)

    # saved_state_dict = torch.load(model_path)
    # model_state_dict = model.state_dict()

    # with open("saved_state_dict.txt", "w") as f:
    #     for key, value in saved_state_dict.items():
    #         f.write(f"{key}: {value.shape}\n")

    # with open("model_state_dict.txt", "w") as f:
    #     for key, value in model_state_dict.items():
    #         f.write(f"{key}: {value.shape}\n")

    # import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()

    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to(device)

    trainer = Trainer(
        model,
        compute_metrics=compute_metrics,
    )

    threshold_dict = {
        0: 0.5,
        1: 0.5,
        2: 0.5,
        3: 0.5,
        4: 0.5,
        5: 0.5,
        6: 0.5,
        7: 0.5
    }

    predictions, label_ids, _ = trainer.predict(encoded_test_ds)
    sigmoid = torch.nn.Sigmoid()
    threshold_values = sigmoid(torch.Tensor(predictions))

    outputs = []
    for thresholded in threshold_values:
        output = []
        for jdx, value in enumerate(thresholded):
            output.append(float(value) >= threshold_dict.get(jdx, 0.5))
        outputs.append(output)


        j_list = jsonlload(test_ds_path)

        for idx, oup in enumerate(outputs):
            j_list[idx]["output"] = {}

            # oup에서 True 또는 1인 값의 개수를 확인
            true_count = sum(oup)

            if not any(oup):
                max_index = threshold_values[idx].argmax().item()
                oup[max_index] = True

            for jdx, v in enumerate(oup):
                j_list[idx]["output"][id2label[jdx]] = ["True" if v else "False", threshold_values[idx][jdx].item()]

    jsonldump(j_list, f"outputs/temp_{model_name}.jsonl")

    with open(f"outputs/temp_{model_name}.jsonl", "r", encoding="utf-8") as reader:
        with open(f"outputs/{model_name}.jsonl", "w", encoding="utf-8") as writer:
            for line in reader:
                entry = json.loads(line)
                outputs = {}
                outputs["joy"] = entry["output"]["joy"][0]
                outputs["anticipation"] = entry["output"]["anticipation"][0]
                outputs["trust"] = entry["output"]["trust"][0]
                outputs["surprise"] = entry["output"]["surprise"][0]
                outputs["disgust"] = entry["output"]["disgust"][0]
                outputs["fear"] = entry["output"]["fear"][0]
                outputs["anger"] = entry["output"]["anger"][0]
                outputs["sadness"] = entry["output"]["sadness"][0]

                writer.write(json.dumps({
                    "id": entry["id"],
                    "input": entry["input"],
                    "output": outputs
                }, ensure_ascii=False) + "\n")

    os.remove(f"outputs/temp_{model_name}.jsonl")

def palkia_inference(model_name="펄기아_v4"):
    print(f"Start inference for {model_name}...")
    inference_config = config[model_name]
    model_path = inference_config["model_ckpt_path"]
    output_path = inference_config["output_path"]
    batch_size = inference_config["batch_size"]
    max_seq_length = inference_config["max_seq_length"]
    threshold = inference_config["threshold"]
    num_beams = inference_config["num_beams"]

    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer 불러옴
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # test dataset 불러옴
    test_ds_path = "data/test_dataset_sample.jsonl"
    test_ds = Dataset.from_json(test_ds_path)

    # id & label 매칭
    with open("data/label2id.json") as f:
        label2id = json.load(f)
    labels = list(label2id.keys())
    id2label = {}
    for k, v in label2id.items():
        id2label[v] = k

    def preprocess_data(examples):
        # texts를 batch로 받아옴
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        target_begin = examples["input"]["target"].get("begin")
        target_end = examples["input"]["target"].get("end")

        # 인코딩
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_length)

        # target position을 0으로 초기화
        encoding["target_positions"] = [0] * len(encoding['input_ids'])

        # target position을 1로 표시
        if text2 != None:
            encoded_target = tokenizer(text2, add_special_tokens=False)["input_ids"]
            encoded_text = tokenizer(text1, add_special_tokens=False)["input_ids"]

            # target이 시작하는 위치를 찾음
            for i in range(len(encoded_text) - len(encoded_target) + 1):
                if encoded_text[i:i+len(encoded_target)] == encoded_target:
                    target_begin = i + 1 # [CLS] 떄문에 + 1
                    target_end = i + len(encoded_target) + 1 # 나중에 리스트 슬라이싱 때문에 + 1
                    break

            for i in range(target_begin, target_end):
                encoding["target_positions"][i] = 1 # 타겟이면 1, 타겟이 아니면 0

        return encoding

    # test dataset 전처리
    encoded_tds = test_ds.map(preprocess_data, remove_columns=test_ds.column_names).with_format("torch")

    # data loader 설정
    data_loader = DataLoader(encoded_tds, batch_size=batch_size)

    # model 불러옴
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        output_hidden_states=True,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True
    )
    model.to(device)

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

    model.eval()

    trainer = Trainer(
        model,
        compute_metrics=compute_metrics,
    )

    threshold_dict = {
        0: 0.24,
        1: 0.77,
        2: 0.12,
        3: 0.729,
        4: 0.99,
        5: 0.78,
        6: 0.09,
        7: 0.95
    }

    predictions, label_ids, _ = trainer.predict(encoded_tds)
    sigmoid = torch.nn.Sigmoid()

    predictions = predictions[0] if isinstance(predictions, tuple) else predictions

    threshold_values = sigmoid(torch.Tensor(predictions))
    # 각 클래스별로 임계값 적용하여 outputs 생성
    outputs = []
    for thresholded in threshold_values:
        output = []
        for jdx, value in enumerate(thresholded):
            output.append(float(value) >= threshold_dict.get(jdx, 0.5))
        outputs.append(output)

    # outputs는 이제 각 예측에 대해 8개의 클래스별 boolean 값을 포함한 리스트가 됩니다.

    j_list = jsonlload(test_ds_path)
    
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

    jsonldump(j_list, output_path)

# general_inference("test_T5")
pth_inference("test2")
# pth_inference("가라도스_v1")
# pth_inference("갸라도스_v2")
# pth_inference("꼬링크_v1")
# pth_inference("꼬링크_v2")
# general_inference("꼬마돌_v1")
# general_inference("나옹_v1")
# pth_inference("럭시오_v1")
# general_inference("리자드_v1")
# general_inference("리자몽_v2")
# general_inference("리자몽_v3")
# pth_inference("샤비_v1")
# pth_inference("수댕이_v1")
# general_inference("식스테일_v5")
# pth_inference("윈디_v1")
# general_inference("이브이_v1")
# general_inference("이상해씨_v2")
# general_inference("이상해풀_v6")
# general_inference("이상해풀_v7")
# general_inference("이상해풀풀_v7")
# pth_inference("조로아크_v5")
# general_inference("주리비얀_v2")
# pth_inference("찌르버드_v2")
# pth_inference("찌르호크_v1")
# general_inference("파이리_v2")
# general_inference("파이리_v4")
# palkia_inference("펄기아_v3")
# palkia_inference("펄기아_v4")

"""
size mismatch for model.bert.pooler.dense.bias: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([768]).
size mismatch for model.classifier.weight: copying a param with shape torch.Size([8, 1024]) from checkpoint, the shape in current model is torch.Size([8, 768]).
"""
