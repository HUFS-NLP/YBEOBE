import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    )


# baseline 코드에 넣어 사용 or baseline에서 from run.LSTM_attention import LSTM_attention와 같이 불러와서 사용

class LSTM_attention(nn.Module):
        def __init__(self):
            super(LSTM_attention, self).__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(args.model_path,
                                                                              output_hidden_states=True,
                                                                              problem_type="multi_label_classification", 
                                                                              num_labels=len(labels),
                                                                              id2label=id2label,
                                                                              label2id=label2id)  
                                                                                    
            self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # make sure to match the dimensions with the chosen model version
            self.linear = nn.Linear(256, len(labels))
            self.num_labels = len(labels)

        def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels).hidden_states[-1]
            lstm_out, (h_n, c_n) = self.bi_lstm(outputs)
            h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

            target_indices = (token_type_ids == 1).nonzero(as_tuple=True)[1]
            query = lstm_out[:, target_indices, :]
            
            attn_output = F.scaled_dot_product_attention(query, lstm_out, lstm_out)
            attn_output = attn_output.mean(dim=1)

            combined_output = h_n + attn_output

            logits = self.linear(combined_output)
            

            if labels != None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
                return loss, logits
        
            else:
                return logits

model = LSTM_attention()
