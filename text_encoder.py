import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
#from transformers.models import bert
from data import task_a, task_b, task_c, all_tasks, read_test_file, read_test_file_all, all_task_hijack, read_hijack_file, read_test_file_hijack 
from config import HASH_PATH, TRAIN_PATH, TEST_PATH
from cli import get_args
from utils import load
from datasets import HijackDataset, HuggingfaceDataset, HuggingfaceMTDataset, ImbalancedDatasetSampler
from models.bert import BERT, RoBERTa
from models.gated import GatedModel
from models.mtl import MTL_Transformer_LSTM
from transformers import BertTokenizer, BertModel
from trainer import Trainer
import datetime
import warnings

def text_encode_train():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    # read file
    data = pd.read_csv(TRAIN_PATH,encoding='utf-8').iloc[:,0:5]
    # content & hashtag
    se1_encoded = []
    se2_encoded = []

    for _, row in data.iterrows():
        s1 = row['s1']
        s2 = row['s2']
        
        #
        s1_input_text = '[CLS] ' + s1 + ' [SEP]'
        s2_input_text = '[CLS] ' + s2 + ' [SEP]'
        
        # input
        s1_inputs = tokenizer.encode_plus(
            s1_input_text,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        s2_inputs = tokenizer.encode_plus(
            s2_input_text,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # encode
        with torch.no_grad():
            s1_outputs = model(**s1_inputs)
            s1_embeddings = s1_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            s2_outputs = model(**s2_inputs)
            s2_embeddings = s2_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # 
        se1_encoded.append(s1_embeddings)
        se2_encoded.append(s2_embeddings)
        encoded_data = pd.DataFrame({'se1': se1_encoded, 'se2': se2_encoded, 'la':data['subtask_a'], 'lb':data['subtask_b'], 'lc':data['subtask_c']})
        # data merge
        #encoded_data = pd.concat([data, encoded_data], axis=1)

        # save
        encoded_data.to_csv('encoded_hashtag_train.csv', index=False)


def text_encode_test():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    # read file
    data = pd.read_csv(TEST_PATH,encoding='utf-8').iloc[:,0:5]
    # content & hashtag
    se1_encoded = []
    se2_encoded = []

    for _, row in data.iterrows():
        s1 = row['s1']
        s2 = row['s2']
        
        #
        s1_input_text = '[CLS] ' + s1 + ' [SEP]'
        s2_input_text = '[CLS] ' + s2 + ' [SEP]'
        
        # input
        s1_inputs = tokenizer.encode_plus(
            s1_input_text,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        s2_inputs = tokenizer.encode_plus(
            s2_input_text,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # encode
        with torch.no_grad():
            s1_outputs = model(**s1_inputs)
            s1_embeddings = s1_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            s2_outputs = model(**s2_inputs)
            s2_embeddings = s2_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # 
        se1_encoded.append(s1_embeddings)
        se2_encoded.append(s2_embeddings)
        encoded_data = pd.DataFrame({'se1': se1_encoded, 'se2': se2_encoded, 'la':data['subtask_a'], 'lb':data['subtask_b'], 'lc':data['subtask_c']})
        # data merge
        #encoded_data = pd.concat([data, encoded_data], axis=1)

        # save
        encoded_data.to_csv('encoded_hashtag_test.csv', index=False)

