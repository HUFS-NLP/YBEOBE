from data_handler.py import JsonHandler

import numpy as np
from transformers import ElectraTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_masks):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        lstm_output = lstm_output[:, -1, :]  # Get the last output
        logits = self.linear(lstm_output)
        return logits


class BaseLine:
    def __init__(self):
        self.dh = JsonHandler()
        self.dh.do_handle()

        self.mlb = MultiLabelBinarizer()
        self.sub_labels = self.mlb.fit_transform(self.dh.train_senti_list)

        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

        self.train_input_ids, self.train_attention_masks = self.encode(self.dh.train_text_list)

        self.dev_input_ids, self.dev_attention_masks = self.encode(self.dh.dev_text_list)

        mlb = MultiLabelBinarizer()
        self.train_sub_labels = mlb.fit_transform(self.dh.train_senti_list)
        self.dev_sub_labels = mlb.fit_transform(self.dh.dev_senti_list)

        # Define model parameters
        self.input_size = len(self.tokenizer) # Vocabulary size
        self.hidden_size = 128
        self.num_labels = self.sub_labels.shape[1] # Number of sub-category labels

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create LSTM model
        self.lstm = LSTM(self.input_size, self.hidden_size, self.num_labels).to(self.device)

        # Choose model to train
        self.model = self.lstm

        # Move data to GPU
        self.train_input_ids = self.train_input_ids.to(self.device)
        self.train_attention_masks = self.train_attention_masks.to(self.device)
        self.train_sub_labels = torch.tensor(self.train_sub_labels, dtype=torch.float32).to(self.device)

        self.dev_input_ids = self.dev_input_ids.to(self.device)
        self.dev_attention_masks = self.dev_attention_masks.to(self.device)
        self.dev_sub_labels = torch.tensor(self.dev_sub_labels, dtype=torch.float32).to(self.device)  # Assuming dev_sub_labels is defined

        # Training loop
        self.epochs = 30
        self.batch_size = 32
        self.learning_rate = 0.004

        # Loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

    def encode(self, text_list):
        input_ids = []
        attention_masks = []

        for text in text_list:
            encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def calculate_accuracy(self, predictions, targets):
        # Convert logits to predicted labels (0 or 1)
        predicted_labels = torch.round(torch.sigmoid(predictions))

        # compare predicted labels with target to check for correct predictions
        correct_predictions = torch.all(predicted_labels == targets, dim=1)

        # Calculate accuracy
        accuracy = torch.sum(correct_predictions).item() / len(targets)

        return accuracy

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def train_model(self):
        self.reset_parameters()
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_accuracy = 0.0

            for i in range(0, len(self.train_input_ids), self.batch_size):
                batch_input_ids = self.train_input_ids[i:i + self.batch_size]
                batch_attention_masks = self.train_attention_masks[i:i + self.batch_size]
                batch_sub_labels = self.train_sub_labels[i:i + self.batch_size]

                self.optimizer.zero_grad()
                logits = self.model(batch_input_ids, batch_attention_masks)
                loss = self.criterion(logits, batch_sub_labels)
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy for this batch
                batch_accuracy = self.calculate_accuracy(logits, batch_sub_labels)
                total_accuracy += batch_accuracy
                total_loss += loss.item()

            avg_loss = total_loss / (len(self.train_input_ids) // self.batch_size)
            avg_accuracy = total_accuracy / (len(self.train_input_ids) // self.batch_size)

            print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        print("Training complete!")

    def evaluate_model(self):
        self.model.eval()

        total_dev_loss = 0.0
        total_dev_accuracy = 0.0

        with torch.no_grad():
            for i in range(0, len(self.dev_input_ids), self.batch_size):
                batch_dev_input_ids = self.dev_input_ids[i:i + self.batch_size]
                batch_dev_attention_masks = self.dev_attention_masks[i:i + self.batch_size]
                batch_dev_sub_labels = self.dev_sub_labels[i:i + self.batch_size]

                dev_logits = self.model(batch_dev_input_ids, batch_dev_attention_masks)
                dev_loss = self.criterion(dev_logits, batch_dev_sub_labels)

                # Calculate accuracy for this batch
                dev_batch_accuracy = self.calculate_accuracy(dev_logits, dev_batch_sub_labels)
                total_dev_accuracy += dev_batch_accuracy
                total_dev_loss += dev_loss.item()

            avg_dev_loss = total_dev_loss / (len(self.dev_input_ids) // self.batch_size)
            avg_dev_accuracy = total_dev_accuracy / (len(self.dev_input_ids) // self.batch_size)

        print(f"Validation Loss: {avg_dev_loss:.4f}, Validation Accuracy: {avg_dev_accuracy:.4f}")
