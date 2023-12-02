import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoModelForSequenceClassification, AutoConfig
import numpy as np
import random


class LSTMAttention(nn.Module):
    def __init__(self, model_path, output_hidden_states, problem_type, num_labels, id2label, label2id):
        super(LSTMAttention, self).__init__()
        config = AutoConfig.from_pretrained(model_path)
        config.output_hidden_states = True
        config.problem_type = "multi_label_classification"
        config.num_labels = num_labels
        config.id2label = id2label
        config.label2id = label2id

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        self.bi_lstm = nn.LSTM(self.model.config.hidden_size, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(256, 8)

        for name, param in self.bi_lstm.named_parameters():
            if 'weight_ih' in name:
                init.orthogonal_(param.data)  # Orthogonal 초기화
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)  # Orthogonal 초기화
            elif 'bias' in name:
                param.data.fill_(0)  # Bias는 0으로 초기화

        # Linear 계층 초기화
        init.xavier_normal_(self.linear.weight)


    def forward(self, input_ids, attention_mask, target_positions, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
        lstm_out, (h_n, _) = self.bi_lstm(outputs)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        h_n = self.dropout(h_n)

        max_len = 1  # 최대 타겟 길이 초기화

        query_list = []
        for idx, target_position in enumerate(target_positions):
            non_zero_indices = (target_position == 1).nonzero(as_tuple=True)  # 타겟 찾기
            if non_zero_indices[0].numel() > 0:  # 첫 번째 요소에 있는 non-zero 인덱스의 수 확인
                target_indices = non_zero_indices[0]
            else:
                target_indices = None

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

        combined_output = h_n + attn_output

        logits = self.linear(combined_output)

        if labels != None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, logits

        else:
            return logits
