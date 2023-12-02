import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, Trainer


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
            nn.Dropout(p=0.1),  # dropout 값 파라미터로 조정 가능 - 논문에서 제시한 최적의 dropout 값으로 설정
            nn.Linear(self.bert.feature_size, 1)
        )


    def forward(self, input_ids, attention_mask, label_idxs, token_type_ids=None, labels=None):


        #Bert encoder
        last_hidden_state = self.bert(input_ids, attention_mask, token_type_ids=None, labels=None)


        #FFN
        batch_size, num_indices = label_idxs.size()
        logits_list = []                             #where logits will be saved

        for i in range(batch_size):
            logits_i = self.ffn(last_hidden_state[i]).squeeze(-1).index_select(dim=0, index=label_idxs[i])
            logits_list.append(logits_i)

        logits = torch.stack(logits_list, dim=0)


        #LCAloss 계산
        if labels != None: #train

            cel = F.binary_cross_entropy_with_logits(logits,labels).cuda()
            cl = self.corr_loss(logits, labels)
            loss = ((1 - self.alpha) * cel) + (self.alpha * cl)
            #logger.info(" ====== Good ======")
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
