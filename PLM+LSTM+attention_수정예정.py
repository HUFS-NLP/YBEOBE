from sklearn.metrics import f1_score
import numpy as np
from transformers import ElectraTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
import json
from transformers import ElectraForSequenceClassification, ElectraModel
from torch.optim import Adam
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from tqdm import tqdm


# Initialize KO-ELECTRA tokenizer
# tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
# Hyperparameters
vocab_size = tokenizer.vocab_size
# embedding_dim = 768  # This should match the dimensionality of your word vectors
max_seq_length = 218
batch_size = 64
epochs = 30
learning_rate = 4e-5

multi_binarizer = MultiLabelBinarizer()

def map_labels_to_integers(ner_labels_list):
    label_to_int = {'O': 0, 'T': 1}
    return [[label_to_int[label] for label in labels] for labels in ner_labels_list]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# data_path = "/home/nlpgpu9/ellt/eojin/EA/attention/output_with_senti.jsonl"  # Replace with the actual path

# texts = []
# senti = []

# target_positions = []

# with open(data_path, "r", encoding="utf-8") as file:
#     for line in file:
#         entry = json.loads(line)
#         texts.append(entry["input"]["form"])
#         senti.append(entry["senti"])

    
#         # Process target positions
#         target_form = entry["input"].get("target", {}).get("form")
#         target_start = entry["input"].get("target", {}).get("begin")
#         target_end = entry["input"].get("target", {}).get("end")
        
#         # If target_form exists, find its position in tokenized input_ids
#         if target_form:
#             encoded_target = tokenizer(target_form, add_special_tokens=False)["input_ids"]
#             encoded_text = tokenizer(entry["input"]["form"], add_special_tokens=False)["input_ids"]
            
#             for i in range(len(encoded_text) - len(encoded_target) + 1):
#                 if encoded_text[i:i+len(encoded_target)] == encoded_target:
#                     target_start = i
#                     target_end = i + len(encoded_target) - 1
#                     break
#         else:
#             target_start, target_end = -1, -1
        
#         target_positions.append((target_start, target_end))

# target_positions_tensor = torch.tensor(target_positions, dtype=torch.int)
# print(target_positions[:5])

def process_data(data, tokenizer, max_seq_length=max_seq_length):
    input_ids_list = []
    attention_masks_list = []
    ner_labels_list = []
    senti_labels_list = []
    target_positions = []


    for item in data:
        text = item["input"]["form"]
        target_form = item["input"]["target"].get("form")
        target_begin = item["input"]["target"].get("begin")
        target_end = item["input"]["target"].get("end")
        senti = item["senti"]

        encoded = tokenizer(text, truncation=True, padding='max_length', max_length=max_seq_length)
        input_ids = encoded['input_ids']
        attention_masks = encoded['attention_mask']


        # if target_form:
        #     encoded_target = tokenizer(target_form, add_special_tokens=False)["input_ids"]
        #     encoded_text = tokenizer(entry["input"]["form"], add_special_tokens=False)["input_ids"]
            
        #     for i in range(len(encoded_text) - len(encoded_target) + 1):
        #         if encoded_text[i:i+len(encoded_target)] == encoded_target:
        #             target_start = i
        #             target_end = i + len(encoded_target) - 1
        #             break
        # else:
        #     target_start, target_end = None, None

        # target_positions.append((target_start, target_end))

        # Create NER labels
        ner_labels = ['O'] * max_seq_length  # Initialize with 'O'

        # If target_begin or target_end != None, set the labels accordingly
        if target_begin != None and target_end != None:
            for i in range(target_begin, target_end + 1):
                ner_labels[i] = 'T'
            target_positions.append((target_begin, target_end))

        else:
            target_positions.append((-1, -1))

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_masks)
        ner_labels_list.append(ner_labels)
        senti_labels_list.append(senti)

    # Convert lists to tensors
    input_ids_tensor = torch.tensor(input_ids_list)
    attention_masks_tensor = torch.tensor(attention_masks_list)
    ner_labels_list = map_labels_to_integers(ner_labels_list)
    ner_labels_tensor = torch.tensor(ner_labels_list)
    target_positions_tensor = torch.tensor(target_positions, dtype=torch.int)

    # Multi-label binarization for sentiment labels
    senti_labels_binarized = multi_binarizer.fit_transform(senti_labels_list)
    senti_labels_tensor = torch.tensor(senti_labels_binarized, dtype=torch.float)

    return input_ids_tensor, attention_masks_tensor, ner_labels_tensor, senti_labels_tensor, target_positions_tensor


# Read JSONL file
file_path = '/home/nlpgpu9/ellt/eojin/EA/attention/output_with_senti.jsonl'  # Replace with your JSONL file path
sample_data = read_jsonl(file_path)

dev_file_path = '/home/nlpgpu9/ellt/eojin/EA/baseline/dev_output.jsonl'
dev_sample_data = read_jsonl(dev_file_path)

# Process the data
input_ids_tensor, attention_masks_tensor, ner_labels_tensor, senti_labels_tensor, target_positions_tensor = process_data(sample_data, tokenizer)
dev_input_ids_tensor, dev_attention_masks_tensor, dev_ner_labels_tensor, dev_senti_labels_tensor, dev_target_positions_tensor = process_data(dev_sample_data, tokenizer)

class MyDataset(Dataset):
    def __init__(self, input_ids, attention_masks, ner_labels, senti_labels, target_positions_tensor):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.senti_labels = senti_labels
        self.ner_labels = ner_labels
        self.target_positions = target_positions_tensor

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.senti_labels[idx], self.ner_labels[idx], self.target_positions[idx]

# Create a TensorDataset from your tensors
dataset = MyDataset(input_ids_tensor, attention_masks_tensor, senti_labels_tensor, ner_labels_tensor, target_positions_tensor)
dev_dataset = MyDataset(dev_input_ids_tensor, dev_attention_masks_tensor, dev_senti_labels_tensor, dev_ner_labels_tensor, dev_target_positions_tensor)

# Create a DataLoader
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)


class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.roberta = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")  # you can choose other versions like "roberta-large"
        self.embedding = nn.Embedding(len(tokenizer), 768)
        self.bi_lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)  # make sure to match the dimensions with the chosen Roberta version
        self.linear_senti = nn.Linear(256, 8)  # 8 classes for sentiment analysis
        self.linear_ner = nn.Linear(256, 2)  # 2 classes for NER
        self.linear_senti_combined = nn.Linear(512, 8)

    def forward(self, input_ids, attention_masks, target_start, target_end):
        outputs = self.roberta(input_ids, attention_mask=attention_masks)
        last_hidden_state = outputs.last_hidden_state
        output, (h_n, c_n) = self.bi_lstm(last_hidden_state)
        
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.bi_lstm(embedded)

        max_len = 1
        for start, end in zip(target_start, target_end):
            if start != -1 and end != -1:
                max_len = max(max_len, end - start)
                # print(max_len)
        # batch_size, sequence_length, hidden_size
        query_list = []
        for idx, (start, end) in enumerate(zip(target_start, target_end)):
            if start != -1 and end != -1:  # Check if start and end are not None
                query = lstm_output[idx, start:end, :]
                pad_len = max_len - (end - start)
                # print(start, end)
                if pad_len > 0:
                    query = F.pad(query, (0, 0, 0, pad_len))  # Padding to max_len
                query_list.append(query)
            else:
        # Handle the case when start or end is None
        # (e.g., create a zero tensor of the shape you need)
                query_list.append(torch.zeros((max_len, lstm_output.size(2)), device=lstm_output.device))
        # print([q.shape for q in query_list])
        # print("Pad length:", pad_len)
        # print("Start:", start)
        # print("End:", end)
        # print("Max length:", max_len)
        query = torch.stack(query_list, dim=0)
        key = lstm_output
        value = lstm_output

        # print("Query size: ", query.size())
        # print("Key size: ", key.size())
        # print("Value size: ", value.size())


        target_lengths = [end - start if end != -1 and start != -1 else 0 for start, end in zip(target_start, target_end)]
        
        if attention_masks != None:
            attn_mask = attention_masks  # shape: (batch_size, seq_length)
            max_target_length = max(target_lengths)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, max_target_length, 1)  # shape: (batch_size, target_length, seq_length)


        # attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)  # shape: (batch_size, target_length, hidden_size)
        attn_output = F.scaled_dot_product_attention(query, key, value)
        attn_output = attn_output.mean(dim=1)
        
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        combined_output = torch.cat((h_n, attn_output), dim=1)

        # concat_output = torch.cat((attn_output, h_n), dim=1)

        # combined = torch.cat((h_n, attn_output), dim=1)
        senti_out = self.linear_senti_combined(combined_output)
        ner_out = self.linear_ner(output)

        return senti_out, ner_out


# Initialize the model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = EmotionModel()
model.to(device)

# Optimizer and Loss Functions
optimizer = Adam(model.parameters(), lr=learning_rate)
senti_loss_fn = BCEWithLogitsLoss().to(device)
ner_loss_fn = nn.CrossEntropyLoss().to(device)
dev_senti_loss_fn = BCEWithLogitsLoss().to(device)
dev_ner_loss_fn = nn.CrossEntropyLoss().to(device)

save_path = '/home/nlpgpu9/ellt/eojin/EA/attention/PLM_LSTM_attention_v1/'  # 모델을 저장할 디렉터리
os.makedirs(save_path, exist_ok=True)  # 디렉터리가 없으면 생성

# Training Loop
for epoch in range(epochs):
    model.train()
    total_senti_loss = 0
    total_ner_loss = 0
    all_senti_labels = []
    all_senti_preds = []

    train_loop = tqdm(train_dataloader, leave=True, position=0)  # tqdm 적용
    for i, batch in enumerate(train_loop):
        train_loop.set_description(f"Epoch {epoch+1}")
        input_ids, attention_masks, ner_labels, senti_labels, target_positions = batch

        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        ner_labels = ner_labels.to(device)
        senti_labels = senti_labels.to(device)

        # batch_input_ids, batch_attention_masks, batch_target_positions = batch
        batch_target_start = [pos[0] for pos in target_positions]
        batch_target_end = [pos[1] for pos in target_positions]

        optimizer.zero_grad()
        senti_out, ner_out = model(input_ids, attention_masks, batch_target_start, batch_target_end)

        senti_loss = senti_loss_fn(senti_out, senti_labels)
        ner_out = ner_out.view(-1, ner_out.shape[-1])

        ner_labels = ner_labels.view(-1)
        ner_loss = ner_loss_fn(ner_out, ner_labels)

        total_loss = senti_loss + ner_loss
        total_senti_loss += senti_loss.item()
        total_ner_loss += ner_loss.item()

        total_loss.backward()
        optimizer.step()

        senti_preds = torch.sigmoid(senti_out)
        senti_preds = (senti_preds > 0.5).cpu().numpy().astype(int)
        senti_labels = senti_labels.cpu().numpy().astype(int)

        all_senti_labels.extend(senti_labels.tolist())
        all_senti_preds.extend(senti_preds.tolist())

    senti_f1_score = f1_score(all_senti_labels, all_senti_preds, average='micro')
    print(f'Epoch {epoch+1}, Senti Loss: {total_senti_loss/len(train_dataloader)}, NER Loss: {total_ner_loss/len(train_dataloader)}, Senti F1 Score: {senti_f1_score}')
    train_loop.set_postfix({"Senti Loss": total_senti_loss / (i + 1), "NER Loss": total_ner_loss / (i + 1), "Senti F1 Score": senti_f1_score})

    torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))

    model.eval()
    total_dev_senti_loss = 0
    total_dev_ner_loss = 0
    all_dev_senti_labels = []
    all_dev_senti_preds = []

    dev_loop = tqdm(dev_dataloader, leave=True, position=0)
    for i, batch in enumerate(dev_loop):
        dev_loop.set_description(f"Dev Epoch {epoch+1}")
        dev_input_ids, dev_attention_masks, dev_ner_labels, dev_senti_labels, dev_target_positions = batch

        dev_input_ids = dev_input_ids.to(device)
        dev_attention_masks = dev_attention_masks.to(device)
        dev_ner_labels = dev_ner_labels.to(device)
        dev_senti_labels = dev_senti_labels.to(device)

        dev_batch_target_start = [pos[0] for pos in dev_target_positions]
        dev_batch_target_end = [pos[1] for pos in dev_target_positions]

        with torch.no_grad():
            dev_senti_out, dev_ner_out = model(dev_input_ids, dev_attention_masks, dev_batch_target_start, dev_batch_target_end)

        dev_senti_loss = dev_senti_loss_fn(dev_senti_out, dev_senti_labels)
        dev_ner_out = dev_ner_out.view(-1, dev_ner_out.shape[-1])

        dev_ner_labels = dev_ner_labels.view(-1)
        dev_ner_loss = dev_ner_loss_fn(dev_ner_out, dev_ner_labels)

        total_dev_senti_loss += dev_senti_loss.item()
        total_dev_ner_loss += dev_ner_loss.item()

        dev_senti_preds = torch.sigmoid(dev_senti_out)
        dev_senti_preds = (dev_senti_preds > 0.5).cpu().numpy().astype(int)
        dev_senti_labels = dev_senti_labels.cpu().numpy().astype(int)

        all_dev_senti_labels.extend(dev_senti_labels.tolist())
        all_dev_senti_preds.extend(dev_senti_preds.tolist())

    dev_senti_f1_score = f1_score(all_dev_senti_labels, all_dev_senti_preds, average='micro')
    print(f'Dev Epoch {epoch+1}, Dev Senti Loss: {total_dev_senti_loss/len(dev_dataloader)}, Dev NER Loss: {total_dev_ner_loss/len(dev_dataloader)}, Dev Senti F1 Score: {dev_senti_f1_score}')
    dev_loop.set_postfix({"Dev Senti Loss": total_dev_senti_loss / (i + 1), "Dev NER Loss": total_dev_ner_loss / (i + 1), "Dev Senti F1 Score": dev_senti_f1_score})
      # Reset the model back to training model

# def test_process_data(data, tokenizer, max_seq_length=max_seq_length):
#     input_ids_list = []
#     attention_masks_list = []
#     ner_labels_list = []
#     # senti_labels_list = []
#     # multi_binarizer = MultiLabelBinarizer()

#     for item in data:
#         text = item["input"]["form"]
#         target_begin = item["input"]["target"].get("begin")
#         target_end = item["input"]["target"].get("end")
#         # senti = item["senti"]

#         encoded = tokenizer(text, truncation=True, padding='max_length', max_length=max_seq_length)
#         input_ids = encoded['input_ids']
#         attention_masks = encoded['attention_mask']

#         # Create NER labels
#         ner_labels = ['O'] * max_seq_length  # Initialize with 'O'

#         # If target_begin or target_end != None, set the labels accordingly
#         if target_begin != None and target_end != None:
#             for i in range(target_begin, target_end + 1):
#                 ner_labels[i] = 'T'

#         input_ids_list.append(input_ids)
#         attention_masks_list.append(attention_masks)
#         ner_labels_list.append(ner_labels)
#         # senti_labels_list.append(senti)

#     # Convert lists to tensors
#     input_ids_tensor = torch.tensor(input_ids_list)
#     attention_masks_tensor = torch.tensor(attention_masks_list)
#     ner_labels_list = map_labels_to_integers(ner_labels_list)
#     ner_labels_tensor = torch.tensor(ner_labels_list)

#     # # Multi-label binarization for sentiment labels
#     # senti_labels_binarized = multi_binarizer.fit_transform(senti_labels_list)
#     # senti_labels_tensor = torch.tensor(senti_labels_binarized, dtype=torch.float)

#     return input_ids_tensor, attention_masks_tensor, ner_labels_tensor # senti_labels_tensor

# def evaluate_model(test_data, model, tokenizer, multi_binarizer, device):
#     model.eval()  # Set the model to evaluation mode
#     output_jsonl = []

#     # Process the test data
#     test_input_ids_tensor, test_attention_masks_tensor, _ = test_process_data(test_data, tokenizer)

#     # Create a DataLoader for the test set
#     test_dataset = TensorDataset(test_input_ids_tensor, test_attention_masks_tensor)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     with torch.no_grad():
#         for i, (input_ids, attention_masks) in enumerate(test_dataloader):
#             input_ids = input_ids.to(device)
#             attention_masks = attention_masks.to(device)

#             # Forward pass
#             senti_out, _ = model(input_ids, attention_masks)

#             # Get sentiment predictions
#             senti_preds = torch.sigmoid(senti_out)
#             senti_preds = (senti_preds > 0.5).cpu().numpy().astype(int)  # Convert to boolean labels
#             senti_preds_labels = multi_binarizer.inverse_transform(senti_preds)
#             for j, preds in enumerate(senti_preds_labels):
#                 output = {}
#                 label_order = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]

#                 for label in label_order:
#                     output[label] = str(label in [l.lower() for l in preds])  # Convert the label to lowercase and check if it exists in the prediction

#                 # Create the output json object
#                 json_output = {
#                     "id": test_data[i*batch_size + j]["id"],
#                     "input": test_data[i*batch_size + j]["input"],
#                     "output": output
#                 }
#                 output_jsonl.append(json_output)

#     return output_jsonl

# # You should train your model first before evaluating.
# # Assuming `model` is your trained model:

# # Read your test data
# test_file_path = 'new_test.jsonl'  # Replace with your test JSONL file path
# test_data = read_jsonl(test_file_path)

# # Evaluate the model
# output_jsonl = evaluate_model(test_data, model, tokenizer, multi_binarizer, device)

# # Save the output to a JSONL file
# with open('Purin.jsonl', 'w', encoding='utf-8') as f:
#     for item in output_jsonl:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")