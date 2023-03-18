import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_default_config
from corpus import preprocess_test_data, CustomDataset, test_data_file_list
from model.xlm_roberta import BertModel


def inference_main():
    config = get_default_config()
    config.batch_size = 64
    config.device = 'cuda'
    config.cuda_id = '9'
    test_data_file = test_data_file_list[1]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    model_params_path = './saved_model/2023_03_18-11_17_46_base_910_epoch10.pth'
    save_res_path = './data/result.xlsx'
    
    model = BertModel(config)
    model.load_state_dict(torch.load(model_params_path))
    model.to(config.device)
    model.eval()
    
    test_data = preprocess_test_data(test_data_file)
    print(test_data)
    
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
    inference_main()
    