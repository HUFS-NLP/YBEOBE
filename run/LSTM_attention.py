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

        self.bi_lstm = nn.LSTM(1024, 128, bidirectional=True, batch_first=True)  # twhin bert의 차원 1024
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(256, num_labels)
        self.num_labels = num_labels

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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
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
                query_list.append(None)

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

        attn_mask = attention_mask.unsqueeze(1).repeat(1, max_len, 1).float()

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.1)
        attn_output = attn_output.mean(dim=1)

        combined_output = h_n + attn_output

        logits = self.linear(combined_output)

        if labels != None:  # 학습일 경우
            loss_fct = nn.BCEWithLogitsLoss()
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
        logits = outputs.get("logits")
        loss_fct = AsymmetricLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss



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
