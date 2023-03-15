import csv
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from typing import *
from pathlib import Path as path

from config import get_default_config

# warnings.filterwarnings("ignore")


train_data_excel = r'./data/randomdata_1000 20230213_training_dataset.xlsx'
test_data_excel = r'./data/randomdata10k_test_dataset.xlsx'


def read_txt(file_path):
    with open(file_path, 'r')as f:
        content = f.readlines()
    return content

def save_txt(content: list, file_path):
    with open(file_path, 'w')as f:
        for line in content:
            f.write(str(line)+'\n')

def read_excel(file_path):
    content = pd.read_excel(file_path, sheet_name=0)
    content = np.array(content)
    return content

def save_excel(lines: List[list], heads: List[str], excel_file, sheet_name, start_column=0, mode='w'):
    with pd.ExcelWriter(excel_file, mode=mode) as writer:
        cur_line = pd.DataFrame(list(zip(*lines)), columns=heads)
        # print(cur_line)
        cur_line.to_excel(writer, sheet_name=sheet_name, startcol=start_column,
                          index=False, header=True)
    # print(f'{excel_file} {sheet_name} is saved')


def deal_train_data():
    train_content = read_excel(train_data_excel)
    # print(train_content)
    # print(train_content.shape)
    # print(train_content[0])
    # tar = train_content[0]
    # for p in range(0, len(tar), 5):
    #     print(tar[p:p+5])
    '''
    shape: 1000 * 20
    meaning: 
        sn scode Coname Coname_Scode Qsubj 
        Reply Qcount Acount Qpuretext Apuretext 
        Qpurecount Apurecount Qtm Atm timeliness_mins 
        timeliness_hours timeliness_days - drop non_answer
    Qsubj: 4
    Reply: 5
    non_answer: 19
    '''
    return train_content[:, (4, 5, 19)]
    
def deal_test_data():
    test_content = read_excel(test_data_excel)
    # print(test_content)
    # print(test_content.shape)
    # print(test_content[0])
    # tar = test_content[0]
    # for p in range(0, len(tar), 5):
    #     print(tar[p:p+5])
    '''
    shape: 1000 * 17
    meaning: 
        sn scode Coname Coname_Scode Qsubj 
        Reply non_answer - - -
        - - - - -
        - drop 
    Qsubj: 4
    Reply: 5
    '''
    return test_content[:, (4, 5)]


class CustomDataset(Dataset):
    def __init__(self, data, config, phase='train') -> None:
        super().__init__()
        self.data = data
        self.config = config
        self.phase = phase
    
    def __len__(self):
        return len(self.data)
    
    def deal_sentence(self, sentence:str):
        return str(sentence)
        sentence = sentence.strip().split()
        ans_sentence = []
        for word in sentence:
            if word[0] != '@':
                ans_sentence.append(word)      
        return ' '.join(ans_sentence)
    
    def __getitem__(self, index):
        if self.phase == 'train':
            sentence1, sentence2, label = self.data[index]
            if self.config.base:
                return (self.deal_sentence(sentence1), self.deal_sentence(sentence2)), label
            elif self.config.clip:
                return (self.deal_sentence(sentence1), self.deal_sentence(sentence2)), label
            else:
                raise 'wrong config' 
        else:
            sentence1, sentence2 = self.data[index]
            return self.deal_sentence(sentence1), self.deal_sentence(sentence2)
        

if __name__ == '__main__':

    # print(deal_train_data())
    # print(deal_test_data())
    sample_config = get_default_config()
    sample_train_data = deal_train_data()
    sample_train_data = CustomDataset(sample_train_data, sample_config)
    sample_train_data = DataLoader(sample_train_data, batch_size=3, shuffle=False)
    for sample_input in sample_train_data:
        print(sample_input)
        print()
        break
    # exit()
    
    from model.xlm_roberta import BertModel
    sample_model = BertModel(sample_config)
    sample_criterion = nn.CrossEntropyLoss(reduction='sum')
    for sample_x, sample_y in sample_train_data:
        sample_output = sample_model(sample_x)
        print(sample_output)
        loss = sample_criterion(sample_output, sample_y.to(torch.long))
        print(loss)
        loss.backward()
        break
    
    pass
    