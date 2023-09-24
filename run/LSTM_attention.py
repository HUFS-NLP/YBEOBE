import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class LSTM_attention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(LSTM_attention, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                              output_hidden_states=True,
                                                                              problem_type="multi_label_classification", 
                                                                              num_labels=num_labels,
                                                                              id2label=id2label,
                                                                              label2id=label2id)  
        
        # LSTM 추가                                                                            
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(256, num_labels)  # biLSTM이라 128*2
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]  # 마지막 은닉층
        lstm_out, (h_n, c_n) = self.bi_lstm(outputs)  # lstm_out: 개별 정보, h_n: 문장 전체 정보
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # biLSTM이라 두 층 받아오고, 문장길이 기준으로 함치기

        target_indices = (token_type_ids == 1).nonzero(as_tuple=True)[1]  # token_type_ids == 1인 것이 타겟. [0]은 배치 기준, [1]은 문장 기준
        query = lstm_out[:, target_indices, :]  # [배치 크기, 타겟 길이, 임베딩 차원]

        attn_output = F.scaled_dot_product_attention(query, lstm_out, lstm_out)  # query[배치 크기, 타겟 길이, 임베딩 차원], key[배치 크기, 문장 길이, 임베딩 차원], value[배치 크기, 문장 길이, 임베딩 차원]
        attn_output = attn_output.mean(dim=1)  # 문장 기준 평균 내서 h_n과 차원 맞추기

        combined_output = h_n + attn_output  # 합치기(여러 방식 있지만 지금은 단순 더하기)

        logits = self.linear(combined_output)  # linear 통과 시켜서 라벨 개수(8)로 바꾸기
            
        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()  # 손실함수 바꾸는 부분
            loss = loss_fct(logits, labels.float())  # 라벨 실수 형태로 받아왔었기 때문에 float
            return loss, logits
        
        else:  # 평가일 경우
            return logits
