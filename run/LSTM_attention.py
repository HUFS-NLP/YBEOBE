import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel


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
        


class loss_function(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(loss_function, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        output_hidden_states=True,
                                                                        problem_type="multi_label_classification", 
                                                                        num_labels=num_labels,
                                                                        id2label=id2label,
                                                                        label2id=label2id)                                                                          
        self.linear = nn.Linear(768, num_labels)  # # AutoModelForSequenceClassification의 차원 768
        self.num_labels = num_labels


    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
        logits = self.linear(outputs)
            
        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()  # 손실함수 바꾸는 부분
            loss = loss_fct(logits, labels.float())
            return loss, logits
        
        else:  # 평가일 경우
            return logits



"""현재 LSTM_multitask는 모델 2개 받아오기 때문에 시간 2배 걸림. 따라서 AutoModel로만 학습해도 되는지 확인해보고 되면 아래 거"""


# class LSTM_multitask(nn.Module):
#     def __init__(self, model_path, num_labels):
#         super(LSTM_multitask, self).__init__()
#         self.base_model = AutoModel.from_pretrained(model_path)
#         self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
#         self.linear = nn.Linear(256, num_labels)
#         self.linear_ner = nn.Linear(256, 2)  

#     def forward(self, input_ids, attention_mask, target_positions, labels, token_type_ids=None):
#         outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         last_hidden_state = outputs.last_hidden_state
        
#         lstm_out, (h_n, c_n) = self.bi_lstm(last_hidden_state)
#         h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
#         logits = self.linear(h_n)
#         ner_logits = self.linear_ner(lstm_out)

#         if labels is not None:
#             loss_fct = nn.BCEWithLogitsLoss()
#             ner_loss_fct = nn.CrossEntropyLoss()
            
#             loss = loss_fct(logits, labels.float())  # EC loss
#             ner_loss = ner_loss_fct(ner_logits.view(-1, 2), target_positions.view(-1))
            
#             total_loss = loss + ner_loss * 0.1  # 손실 가중치(현재 alpha=0.5) 설정해야 함(우리 태스크는 EC니까 NER은 EC보다 가중치 적어야)
            
#             return total_loss, logits
        
#         else:
#             return logits
