import json
import os

import torch
import torch.nn as nn
from tqdm import trange
from transformers import XLMRobertaModel, AutoTokenizer
from transformers import ElectraModel, ElectraTokenizer, pipeline
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from datasets import load_metric
from sklearn.metrics import f1_score
import pandas as pd
import copy


class BaseInfo:
    """A configuration class containing basic information (constants) 
    related to Aspect-Based Sentiment Analysis (ABSA) task using KoElectra models"""
    def __init__(self):
        self.PADDING_TOKEN = 1
        self.S_OPEN_TOKEN = 0
        self.S_CLOSE_TOKEN = 2

        self.do_eval = True

        self.category_extraction_model_path = '/home/nlpgpu7/ellt/eojin/ABSA/Electra/category_electra/'
        self.polarity_classification_model_path = '/home/nlpgpu7/ellt/eojin/ABSA/Electra/polarity_electra/'

        self.test_category_extraction_model_path = '/home/nlpgpu7/ellt/eojin/ABSA/Electra/category_electra/saved_model_epoch_20.pt'
        self.test_polarity_classification_model_path = '/home/nlpgpu7/ellt/eojin/ABSA/Electra/polarity_electra/saved_model_epoch_20.pt'

        self.train_data_path = '/home/nlpgpu7/ellt/eojin/ABSA/data/nikluge-ea-2022-train.jsonl'
        self.dev_data_path = '/home/nlpgpu7/ellt/eojin/ABSA/data/nikluge-ea-2022-dev.jsonl'
        self.test_data_path = '/home/nlpgpu7/ellt/eojin/ABSA/datanikluge-ea-2022-test.jsonl'

        self.max_len = 256
        self.batch_size = 8
        self.base_model = 'xml-roberta-base'  # 'BERT' or 'KoELECTRA'

        self.koelectra_model = 'monologg/koelectra-base-v3-discriminator'
        self.koelectra_tokenizer = 'monologg/koelectra-base-v3-discriminator'

        self.learning_rate = 3e-6
        self.eps = 1e-8
        self.num_train_epochs = 20
        self.classifier_hidden_size = 768
        self.classifier_dropout_prob = 0.1
        self.max_grad_norm = 1.0

        self.entity_property_pair = [
            '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
            '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성',
            '패키지/구성품#일반', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
            '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
        ]

        self.tf_id_to_name = ['True', 'False']
        self.tf_name_to_id = {self.tf_id_to_name[i]: i for i in range(len(self.tf_id_to_name))}

        self.polarity_id_to_name = ['positive', 'negative', 'neutral']
        self.polarity_name_to_id = {self.polarity_id_to_name[i]: i for i in range(len(self.polarity_id_to_name))}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.special_tokens_dict = {
            'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&',
                                          '&bank-account&', '&num&', '&online-account&']
        }


class DataHandler:
    """A utility class for handling JSON and JSONL files"""
    def __init__(self):
        """Initializes the DataHandler object with default values"""
        self.j = None
        self.j_load = None
        self.json_list = None

    def jsonload(self, fname, encoding="utf-8"):
        """Loads a JSON file and stores its content in the j_load attribute"""
        with open(fname, 'r', encoding=encoding) as f:
            self.j_load = json.load(f)

        return self.j_load

    def jsondump(self, j, fname):
        """Dumps a JSON object to a file"""
        with open(fname, "w", encoding="UTF8") as f:
            json.dump(j, f, ensure_ascii=False)

    def jsonlload(self, fname, encoding="utf-8"):
        """Loads a JSONL file and stores its content in the json_list attribute"""
        self.json_list = []
        with open(fname, encoding=encoding) as f:
            for line in f.readlines():
                self.json_list.append(json.loads(line))
        return self.json_list


class SimpleClassifier(nn.Module):
    """Feedforward neural network classifier built using PyTorch's nn.Module.

    The classifier is designed with a dense layer followed by 
    - dropout layer, 
    - activation function, 
    - output layer. 
    
    The configuration for the classifier, such as the hidden size and dropout probability, is fetched from the `BaseInfo` class"""
    def __init__(self, num_label):
        """Initializes SimpleClassifier with layers defined based on configurations from BaseInfo class"""
        super().__init__()
        bi = BaseInfo()
        self.dense = nn.Linear(bi.classifier_hidden_size, bi.classifier_hidden_size)
        self.dropout = nn.Dropout(bi.classifier_dropout_prob)
        self.output = nn.Linear(bi.classifier_hidden_size, num_label)

    def forward(self, features):
        """Forward pass of the classifier"""
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class RoBertaBaseClassifier(nn.Module):
    """We will not use this class in the final model. RoBERTa is provided in the original baseline code, but we will use KoElectra instead."""
    def __init__(self, num_label, len_tokenizer):
        super(RoBertaBaseClassifier, self).__init__()

        self.num_label = num_label
        bi = BaseInfo()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(bi.base_model)
        self.xlm_roberta.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(self.num_label)

        self.logits = None
        self.loss = None

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
        )

        sequence_output = outputs[0]
        self.logits = self.labels_classifier(sequence_output)

        self.loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            self.loss = loss_fct(self.logits.view(-1, self.num_label),
                                 labels.view(-1))

        return self.loss, self.logits


class KoElectraBaseClassifier(nn.Module):
    """A classifier built on top of the KoELECTRA model using PyTorch's nn.Module

    The classifier utilizes a base ElectraModel followed by a simple feed-forward neural network defined in the `SimpleClassifier` class for classification tasks"""
    def __init__(self, num_label, len_tokenizer):
        """Initializes the `KoElectraBaseClassifier` with the ElectraModel and additional layers"""
        super(KoElectraBaseClassifier, self).__init__()

        self.num_label = num_label
        bi = BaseInfo()

        self.electra = ElectraModel.from_pretrained(bi.koelectra_model)
        self.electra.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(self.num_label)

        self.logits = None
        self.loss = None

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the classifier"""
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        self.logits = self.labels_classifier(sequence_output)

        self.loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            self.loss = loss_fct(
                self.logits.view(-1, self.num_label),
                labels.view(-1)
            )

        return self.loss, self.logits


class DataParser:
    """Data parser to tokenize and align labels for ABSA tasks.

    This class processes raw input data, tokenizes sentences, and aligns labels to tokens.

    It provides functionalities to convert raw text data and its annotations into formats suitable for model training"""
    def __init__(self):
        """Initializes the DataParser with attributes from BaseInfo and sets up empty dictionaries"""
        bi = BaseInfo()
        self.entity_property_pair = bi.entity_property_pair
        self.entity_property_data_dict = None
        self.polarity_data_dict = None

    def tokenize_and_align_lables(self, tokenizer, form, annotations, max_len):
        """Tokenizes the given sentences and aligns the labels based on the entity-property pair and annotations"""
        bi = BaseInfo()

        self.entity_property_data_dict = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        self.polarity_data_dict = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for pair in self.entity_property_pair:
            isPairInOpinion = False
            if pd.isna(form):
                break
            tokenized_data = tokenizer(form, pair, padding='max_length', max_length=max_len, truncation=True)

            for annotation in annotations:
                entity_property = annotation[0]
                polarity = annotation[2]

                if polarity == '------------':
                    continue

                if entity_property == pair:
                    self.entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                    self.entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                    self.entity_property_data_dict['labels'].append(bi.tf_name_to_id['True'])

                    self.polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                    self.polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                    self.polarity_data_dict['labels'].append(bi.polarity_name_to_id[polarity])
                    
                    isPairInOpinion = True
                    break

            if not isPairInOpinion:
                self.entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                self.entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                self.entity_property_data_dict['labels'].append(bi.tf_name_to_id['False'])

        return self.entity_property_data_dict, self.polarity_data_dict

    def get_dataset(self, raw_data, tokenizer, max_len):
        """Processes the raw data and returns it in the form of PyTorch's TensorDataset"""
        input_ids_list = []
        attention_mask_list = []
        token_labels_list = []

        polarity_input_ids_list = []
        polarity_attention_mask_list = []
        polarity_token_labels_list = []

        for utterance in raw_data:
            entity_property_data_dict, polarity_data_dict = self.tokenize_and_align_lables(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)

            input_ids_list.extend(entity_property_data_dict['input_ids'])
            attention_mask_list.extend(entity_property_data_dict['attention_mask'])
            token_labels_list.extend(entity_property_data_dict['labels'])

            polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
            polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
            polarity_token_labels_list.extend(polarity_data_dict['labels'])

        self.entity_dataset = TensorDataset(
            torch.tensor(input_ids_list), torch.tensor(attention_mask_list), torch.tensor(token_labels_list)
        )

        self.polarity_dataset = TensorDataset(
            torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
            torch.tensor(polarity_token_labels_list)
        )

        return self.entity_dataset, self.polarity_dataset


class Trainer:
    """A class to handle training and evaluation of sentiment analysis models"""
    def __init__(self):
        """Initializes the Trainer with default values for count_list, hit_list, and acc_list."""
        self.count_list = None
        self.hit_list = None
        self.acc_list = None

    def evaluation(self, y_true, y_pred, label_len):
        """Evaluates and prints the performance of the model based on true and predicted labels."""
        self.count_list = [0 for _ in range(label_len)]
        self.hit_list = [0 for _ in range(label_len)]

        for i in range(len(y_true)):
            self.count_list[y_true[i]] += 1
            if y_true[i] == y_pred[i]:
                self.hit_list[y_true[i]] += 1

        self.acc_list = [self.hit_list[i] / self.count_list[i] for i in range(label_len)]

        print('count_list: ', self.count_list)
        print('hit_list: ', self.hit_list)
        print('acc_list: ', self.acc_list)
        print('accuracy: ', sum(self.hit_list) / sum(self.count_list))
        print('macro_accuracy: ', sum(self.acc_list) / label_len)
        # print('y_true: ', y_true)

        y_true = list(map(int, y_true))
        y_pred = list(map(int, y_pred))

        print('f1_score: ', f1_score(y_true, y_pred, average=None))
        print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
        print('f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))

    def train_sentiment_analysis(self):
        """Trains sentiment analysis models, prints training progress, and saves the trained models.

        The method includes procedures for:
        - Data loading and tokenization.
        - Model initialization.
        - Optimizer and scheduler setup.
        - Training loop for both category extraction and polarity classification.
        - Evaluation during training if `bi.do_eval` is set to True.

        Uses configurations and paths specified in the BaseInfo class and relies on external classes/methods like 
        - DataHandler, 
        - DataParser, 
        - KoElectraBaseClassifier, 
        - AdamW, 
        - get_linear_schedule_with_warmup."""
        bi = BaseInfo()
        dh = DataHandler()
        dp = DataParser()

        print('train_sentiment_analysis')
        print('category_extraction model would be saved at ', bi.category_extraction_model_path)
        print('polarity model would be saved at ', bi.polarity_classification_model_path)

        print('loading train data')
        train_data = dh.jsonlload(bi.train_data_path)
        dev_data = dh.jsonlload(bi.dev_data_path)

        print('tokenizing train data')
        tokenizer = AutoTokenizer.from_pretrained(bi.koelectra_tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(bi.special_tokens_dict)
        print('We have added', num_added_tokens, 'tokens')

        entity_property_train_data, polarity_train_data = dp.get_dataset(train_data, tokenizer, bi.max_len)
        entity_property_dev_data, polarity_dev_data = dp.get_dataset(dev_data, tokenizer, bi.max_len)

        entity_property_train_dataloader = DataLoader(
            entity_property_train_data, shuffle=True, batch_size=bi.batch_size)
        entity_property_dev_dataloader = DataLoader(
            entity_property_dev_data, shuffle=True, batch_size=bi.batch_size)

        polarity_train_dataloader = DataLoader(
            polarity_train_data, shuffle=True, batch_size=bi.batch_size)
        polarity_dev_dataloader = DataLoader(
            polarity_dev_data, shuffle=True, batch_size=bi.batch_size)

        print('loading model')
        entity_property_model = KoElectraBaseClassifier(len(bi.tf_name_to_id), len(tokenizer))
        entity_property_model.to(bi.device)

        polarity_model = KoElectraBaseClassifier(len(bi.polarity_name_to_id), len(tokenizer))
        polarity_model.to(bi.device)

        print('end loading model')

        # entity_property_model_optimizer_setting
        FULL_ENDING = True
        if FULL_ENDING:
            entityOproperty_param_optimiizer = list(entity_property_model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            entity_property_optimizer_grouped_parameters = [
                {'params': [p for n, p in entityOproperty_param_optimiizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in entityOproperty_param_optimiizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            entity_property_param_optimiizer = list(entity_property_model.classifier.named_parameters())
            entity_property_optimizer_grouped_parameters = [
                {'params': [p for n, p in entity_property_param_optimiizer]}]

        entity_property_optimizer = AdamW(
            entity_property_optimizer_grouped_parameters, lr=bi.learning_rate, eps=bi.eps)

        epochs = bi.num_train_epochs
        max_grad_norm = bi.max_grad_norm
        total_steps = epochs * len(entity_property_train_dataloader)

        entity_property_scheduler = get_linear_schedule_with_warmup(
            entity_property_optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # polarity_model_optimizer_setting
        if FULL_ENDING:
            polarity_param_optimiizer = list(polarity_model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']

            polarity_optimizer_grouped_parameters = [
                {'params': [p for n, p in polarity_param_optimiizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in polarity_param_optimiizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            polarity_param_optimiizer = list(polarity_model.classifier.named_parameters())
            polarity_optimizer_grouped_parameters = [{'params': [p for n, p in polarity_param_optimiizer]}]

        polarity_optimizer = AdamW(
            polarity_optimizer_grouped_parameters,
            lr=bi.learning_rate,
            eps=bi.eps
        )

        epochs = bi.num_train_epochs
        max_grad_norm = bi.max_grad_norm
        total_steps = epochs * len(polarity_train_dataloader)

        polarity_scheduler = get_linear_schedule_with_warmup(
            polarity_optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        epoch_step = 0

        for _ in trange(epochs, desc='Epoch'):
            entity_property_model.train()
            epoch_step += 1

            entity_property_total_loss = 0

            for step, batch in enumerate(entity_property_train_dataloader):
                batch = tuple(t.to(bi.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                entity_property_model.zero_grad()

                loss, _ = entity_property_model(b_input_ids, b_input_mask, b_labels)

                loss.backward()

                entity_property_total_loss += loss.item()
                # print('batch_loss: ', loss.item())

                torch.nn.utils.clip_grad_norm_(entity_property_model.parameters(), max_grad_norm)
                entity_property_optimizer.step()
                entity_property_scheduler.step()

            avg_train_loss = entity_property_total_loss / len(entity_property_train_dataloader)

            print("Entity_Property_Epoch: ", epoch_step)
            print("Average train loss: {}".format(avg_train_loss))

            model_saved_path = bi.category_extraction_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'

            torch.save(entity_property_model.state_dict(), model_saved_path)

            if bi.do_eval:
                entity_property_model.eval()

                pred_list = []
                label_list = []

                for batch in entity_property_dev_dataloader:
                    batch = tuple(t.to(bi.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    with torch.no_grad():
                        loss, logits = entity_property_model(b_input_ids, b_input_mask, b_labels)

                    predictions = torch.argmax(logits, dim=-1)
                    pred_list.extend(predictions)
                    label_list.extend(b_labels)

                self.evaluation(label_list, pred_list, len(bi.tf_id_to_name))

            # polarity train
            polarity_total_loss = 0
            polarity_model.train()

            for step, batch in enumerate(polarity_train_dataloader):
                batch = tuple(t.to(bi.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                polarity_model.zero_grad()

                loss, _ = polarity_model(b_input_ids, b_input_mask, b_labels)

                loss.backward()

                polarity_total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(polarity_model.parameters(), max_grad_norm)
                polarity_optimizer.step()
                polarity_scheduler.step()

            avg_train_loss = polarity_total_loss / len(polarity_train_dataloader)
            print("Entity_Property_Epoch: ", epoch_step)
            print("Average train loss: {}".format(avg_train_loss))

            model_saved_path = bi.polarity_classification_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
            torch.save(polarity_model.state_dict(), model_saved_path)

            if bi.do_eval:
                polarity_model.eval()

                pred_list = []
                label_list = []

                for batch in polarity_dev_dataloader:
                    batch = tuple(t.to(bi.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    with torch.no_grad():
                        loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)

                    predictions = torch.argmax(logits, dim=-1)
                    pred_list.extend(predictions)
                    label_list.extend(b_labels)

                self.evaluation(label_list, pred_list, len(bi.polarity_id_to_name))

        print("Training complete!")


class EvaluateModel:
    """Evaluating models trained for ABSA tasks"""
    def __init__(self):
        """Initializes the EvaluateModel class"""
        pass

    def predict_from_korean_form(self, tokenizer, ce_model, pc_model, data):
        """Generate predictions for given sentences using the provided models"""
        bi = BaseInfo()

        ce_model.to(bi.device)
        ce_model.eval()
        for sentence in data:
            form = sentence['sentence_form']
            sentence['annotation'] = []
            if type(form) != str:
                print("form type is arong: ", form)
                continue

            for pair in bi.entity_property_pair:
                tokenized_data = tokenizer(form, pair, padding='max_length', max_length=256, truncation=True)

                input_ids = torch.tensor([tokenized_data['input_ids']]).to(bi.device)
                attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(bi.device)
                with torch.no_grad():
                    _, ce_logits = ce_model(input_ids, attention_mask)

                ce_predictions = torch.argmax(ce_logits, dim=-1)

                ce_result = bi.tf_id_to_name[ce_predictions[0]]

                if ce_result == 'True':
                    with torch.no_grad():
                        _, pc_logits = pc_model(input_ids, attention_mask)

                    pc_predictions = torch.argmax(pc_logits, dim=-1)
                    pc_result = bi.polarity_id_to_name[pc_predictions[0]]

                    sentence['annotation'].append([pair, pc_result])

        return data

    def evaluation_f1(self, true_data, pred_data):
        """Evaluate the precision, recall, and F1 score for the predictions."""
        true_data_list = true_data
        pred_data_list = pred_data

        ce_eval = {
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'TN': 0
        }

        pipeline_eval = {
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'TN': 0
        }

        for i, true_data in enumerate(true_data_list):
            # TP, FN checking
            is_ce_found = False
            is_pipeline_found = False
            for y_ano in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                for p_ano in pred_data_list[i]['annotation']:
                    p_category = p_ano[0]
                    p_polarity = p_ano[1]

                    if y_category == p_category:
                        is_ce_found = True
                        if y_polarity == p_polarity:
                            is_pipeline_found = True

                        break

                if is_ce_found is True:
                    ce_eval['TP'] += 1
                else:
                    ce_eval['FN'] += 1

                if is_pipeline_found is True:
                    pipeline_eval['TP'] += 1
                else:
                    pipeline_eval['FN'] += 1

                is_ce_found = False
                is_pipeline_found = False

            # FP checking
            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                for y_ano in true_data_list[i]['annotation']:
                    y_category = y_ano[0]
                    y_polarity = y_ano[2]

                    if y_category == p_category:
                        is_ce_found = True
                        if y_polarity == p_polarity:
                            is_pipeline_found = True

                        break

                if is_ce_found is False:
                    ce_eval['FP'] += 1

                if is_pipeline_found is False:
                    pipeline_eval['FP'] += 1
                is_ce_found = False
                is_pipeline_found = False

        ce_precision = ce_eval['TP'] / (ce_eval['TP'] + ce_eval['FP'])
        ce_recall = ce_eval['TP'] / (ce_eval['TP'] + ce_eval['FN'])

        ce_result = {
            'Precision': ce_precision,
            'Recall': ce_recall,
            'F1': 2 * ce_recall * ce_precision / (ce_recall + ce_precision)
        }

        pipeline_precision = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FP'])
        pipeline_recall = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FN'])

        pipeline_result = {
            'Precision': pipeline_precision,
            'Recall': pipeline_recall,
            'F1': 2 * pipeline_recall * pipeline_precision / (pipeline_recall + pipeline_precision)
        }

        return {
            'category extraction result': ce_result,
            'entire pipeline result': pipeline_result
        }

    def test_sentiment_analysis(self):
        """Perform sentiment analysis on a test dataset using pretrained models, and then evaluate the results
        using F1 score, precision, and recall."""
        print('Test start')
        bi = BaseInfo()
        dh = DataHandler()
        dp = DataParser()

        tokenizer = AutoTokenizer.from_pretrained(bi.koelectra_tokenizer)
        num_added_toks = tokenizer.add_special_tokens(bi.special_tokens_dict)
        test_data = dh.jsonlload(bi.dev_data_path)

        entity_property_test_data, polarity_test_data = dp.get_dataset(test_data, tokenizer, bi.max_len)

        entity_property_test_dataloader = DataLoader(entity_property_test_data, shuffle=True,
                                                     batch_size=bi.batch_size)

        polarity_test_dataloader = DataLoader(polarity_test_data, shuffle=True,
                                              batch_size=bi.batch_size)

        model = KoElectraBaseClassifier(len(bi.tf_id_to_name), len(tokenizer))
        model.load_state_dict(torch.load(bi.test_category_extraction_model_path, map_location=bi.device), strict=False)
        model.to(bi.device)
        model.eval()

        polarity_model = KoElectraBaseClassifier(len(bi.polarity_id_to_name), len(tokenizer))
        polarity_model.load_state_dict(
            torch.load(bi.test_polarity_classification_model_path, map_location=bi.device), strict=False)
        polarity_model.to(bi.device)
        polarity_model.eval()

        pred_data = self.predict_from_korean_form(tokenizer, model, polarity_model, copy.deepcopy(test_data))

        # jsondump(pred_data, './pred_data.json')
        # pred_data = jsonload('./pred_data.json')

        print('F1 result: ', self.evaluation_f1(test_data, pred_data))


# Train = Trainer()
# Train.train_sentiment_analysis()

# Test = EvaluateModel()
# Test.test_sentiment_analysis()
