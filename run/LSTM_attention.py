import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoModel, Trainer
import os
import json


# 윈디: bi_lstm param 초기화 x, 샤비, 꼬링크, test2, 수댕이, 갸라도스 bi_lstm param 초기화 o (주석된 부분 참고)
# 윈디, 샤비: ELECTRA base의 차원 768, 꼬링크, test2, 수댕이, 갸라도스: bert large의 차원 1024
class LSTM_attention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id, isKc, isWindy):
        super(LSTM_attention, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification",
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)
        # if isKc:
        #     self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # ELECTRA base의 차원 768
        # else:
        #     self.bi_lstm = nn.LSTM(1024, 128, bidirectional=True, batch_first=True)  # bert large의 차원 1024
        self.bi_lstm = nn.LSTM(self.model.config.hidden_size, 128, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.gate_linear = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(256, num_labels)
        self.num_labels = num_labels
        self.isKc = isKc

        if not isWindy:
            for name, param in self.bi_lstm.named_parameters():
                if 'weight_ih' in name:
                    init.orthogonal_(param.data)  # Orthogonal 초기화
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)  # Orthogonal 초기화
                elif 'bias' in name:
                    param.data.fill_(0)  # Bias는 0으로 초기화

            # Linear 계층 초기화
            init.xavier_normal_(self.gate_linear.weight)
            init.xavier_normal_(self.linear.weight)

    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
        if self.isKc:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             labels=labels).hidden_states[-1]
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 labels=labels).hidden_states[-1]
        lstm_out, (h_n, _) = self.bi_lstm(outputs)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        h_n = self.dropout(h_n)

        max_len = 1  # 최대 타겟 길이 초기화

        query_list = []
        for idx, target_positions in enumerate(target_positions):
            non_zero_indices = (target_positions == 1).nonzero(as_tuple=True)
            if len(non_zero_indices) > 1:
                target_indices = non_zero_indices[1]
            else:
                target_indices = None  # Find indices where token_type_ids is 1

            if target_indices is not None and len(target_indices) > 0:  # If target exists
                s, e = target_indices[0], target_indices[-1]  # Start and end positions
                query = lstm_out[idx, s:e + 1, :]  # Extract the query
                max_len = max(max_len, e - s + 1)  # Update max_len
                query_list.append(query)
            else:
                query_list.append(None)  # Use None for later handling

        # Pad each query to max_len
        padded_query_list = []
        for query in query_list:
            if query is not None:
                pad_len = max_len - query.size(0)
                if pad_len > 0:
                    query = F.pad(query, (0, 0, 0, pad_len))  # Padding
                padded_query_list.append(query)
            else:
                padded_query_list.append(torch.zeros((max_len, lstm_out.size(2)), device=lstm_out.device))

        query = torch.stack(padded_query_list, dim=0)
        key = lstm_out
        value = lstm_out

        attn_mask = attention_mask.unsqueeze(1).repeat(1, max_len, 1).float()  # Create attention_mask for scaled_dot_product_attention

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.1)
        attn_output = attn_output.mean(dim=1)

        gate = torch.sigmoid(self.gate_linear(torch.cat((h_n, attn_output), dim=1)))
        gate = self.dropout(gate)
        combined_output = gate * h_n + (1 - gate) * attn_output

        combined_output = h_n + attn_output

        logits = self.linear(combined_output)

        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())  # 라벨 실수 형태로 받아왔었기 때문에 float
            return loss, logits

        else:  # 평가일 경우
            return logits

# 조로아크
class ark_LSTM_attention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id, isKc, isWindy):
        super(ark_LSTM_attention, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification", 
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)        
                                                               
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # AutoModelForSequenceClassification의 차원 768
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(256, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
   
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
       
        lstm_out, (h_n, _) = self.bi_lstm(outputs)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # [num_directions, batch_size, hidden_size]

        target_indices = (target_positions == 1).nonzero(as_tuple=True)[1]  # target_positions == 1인 것이 타겟. [0]은 배치 기준, [1]은 문장 기준 # 잘못된 것 같은데?
        query = lstm_out[:, target_indices, :]  # [배치 크기, 타겟 길이, 임베딩 차원]

        attn_output = F.scaled_dot_product_attention(query, lstm_out, lstm_out, dropout_p=0.1)  # query[배치 크기, 타겟 길이, 임베딩 차원], key[배치 크기, 문장 길이, 임베딩 차원], value[배치 크기, 문장 길이, 임베딩 차원]
        attn_output = attn_output.mean(dim=1)  # 문장 기준 평균 내서 h_n과 차원 맞추기
        combined_output = h_n + attn_output		# 단순 더하기

        logits = self.linear(combined_output)  # linear 통과 시켜서 라벨 개수(8)로 바꾸기
            
        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())  # 라벨 실수 형태로 받아왔었기 때문에 float
            return loss, logits
        
        else:  # 평가일 경우
            return logits


# 찌르버드: token_type_ids로 타겟 받아왔었음. 그리고 max seq가 128
class bird_LSTM_attention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id, isKc, isWindy):
        super(bird_LSTM_attention, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification", 
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)        
                                                               
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # AutoModelForSequenceClassification의 차원 768
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(256, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
   
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
       
        lstm_out, (h_n, _) = self.bi_lstm(outputs)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # [num_directions, batch_size, hidden_size]

        target_indices = (token_type_ids == 1).nonzero(as_tuple=True)[1]  # target_positions == 1인 것이 타겟. [0]은 배치 기준, [1]은 문장 기준 # 잘못된 것 같은데?
        query = lstm_out[:, target_indices, :]  # [배치 크기, 타겟 길이, 임베딩 차원]

        attn_output = F.scaled_dot_product_attention(query, lstm_out, lstm_out)  # query[배치 크기, 타겟 길이, 임베딩 차원], key[배치 크기, 문장 길이, 임베딩 차원], value[배치 크기, 문장 길이, 임베딩 차원]
        attn_output = attn_output.mean(dim=1)  # 문장 기준 평균 내서 h_n과 차원 맞추기
        combined_output = h_n + attn_output		# 단순 더하기

        logits = self.linear(combined_output)  # linear 통과 시켜서 라벨 개수(8)로 바꾸기
            
        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())  # 라벨 실수 형태로 받아왔었기 때문에 float
            return loss, logits
        
        else:  # 평가일 경우
            return logits


# 찌르호크
# 데이터 전처리 단계도 달라짐        
class hawk_LSTM_attention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id, isKc, isWindy):
        super(hawk_LSTM_attention, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                              output_hidden_states=True,
                                                                              problem_type="multi_label_classification", 
                                                                              num_labels=num_labels,
                                                                              id2label=id2label,
                                                                              label2id=label2id)  
                                                                                    
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(256, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
        lstm_out, (h_n, c_n) = self.bi_lstm(outputs)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        attn_outputs = torch.zeros(lstm_out.size(0), lstm_out.size(2)).to(lstm_out.device)
        
        for i in range(lstm_out.size(0)):
            if np.array_equal(target_positions[i], np.array([-1, -1])):
                query = lstm_out[i, target_positions[i], :]
            else:
                query = lstm_out
            
            # query = query.unsqueeze(0)
            
            attn_output = F.scaled_dot_product_attention(query=query, key=lstm_out[i].unsqueeze(0), value=lstm_out[i].unsqueeze(0))
            attn_output = attn_output.mean(dim=1).squeeze(0)
            attn_output = attn_output.mean(dim=0)
            attn_outputs[i] = attn_output

        combined_output = h_n + attn_outputs

        logits = self.linear(combined_output)
            
        if labels != None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, logits
        
        else:
            return logits
