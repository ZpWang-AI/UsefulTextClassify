import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from config import CustomConfig
from corpus import preprocess_test_data, CustomDataset, test_data_file_list
from model.bertModel import BertModel


def inference_main(config: CustomConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    save_res_path = f'./data/result_{config.version}.xlsx'
    
    model = BertModel(config)
    model.load_state_dict(torch.load(config.test_model_path))
    model.to(config.device)
    model.eval()
    
    test_data_file = config.test_data_file
    test_data = preprocess_test_data(test_data_file)
    # print(test_data)
    
    test_dataset = CustomDataset(test_data, config, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    preds = []
    for input_x in tqdm(test_dataloader):
        with torch.no_grad():
            output = model.predict(input_x)
        preds.append(output)
    preds = torch.concat(preds)
    preds = preds.cpu().numpy()
    # print(preds)
    
    if test_data_file == test_data_file_list[0]:
        test_data_content = pd.read_excel(test_data_file, sheet_name=0)
        test_data_content['non_answer'] = preds
        writer = pd.ExcelWriter(save_res_path)
        test_data_content.to_excel(writer)
        writer.save()
    elif test_data_file == test_data_file_list[1]:
        test_data_content = pd.read_excel(test_data_file, sheet_name=1)
        test_data_content['non_answer (not marked)'] = preds
        writer = pd.ExcelWriter(save_res_path)
        test_data_content.to_excel(writer)
        writer.save()
    else:
        raise 'Wrong test file in inference.py'
    
    
if __name__ == '__main__':
    custom_config = CustomConfig()
    custom_config.batch_size = 16
    custom_config.version = 'train1test1_test2'
    custom_config.device = 'cuda'
    custom_config.cuda_id = '9'
    custom_config.test_data_file = test_data_file_list[1]
    custom_config.test_model_path = r'./saved_res/2023-03-20_13:33:05_train 1 test 2/saved_model/2023-03-20_13-33-05_epoch10_830.pth'
    inference_main(custom_config)
    