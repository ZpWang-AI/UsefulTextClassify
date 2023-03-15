import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_default_config
from corpus import read_excel, save_excel, deal_test_data, CustomDataset, test_data_excel
from model.xlm_roberta import BertModel


def inference_main():
    config = get_default_config()
    config.batch_size = 64
    config.device = 'cuda'
    config.cuda_id = '9'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    model_params_path = './saved_model/2023_03_14-10_13_01_base_915_epoch10.pth'
    save_res_path = './data/result.xlsx'
    
    model = BertModel(config)
    model.load_state_dict(torch.load(model_params_path))
    model.to(config.device)
    model.eval()
    
    test_data = deal_test_data()
    
    test_dataset = CustomDataset(test_data, config, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    preds = []
    for input_x in tqdm(test_dataloader):
        with torch.no_grad():
            output = model.predict(input_x)
        preds.append(output)
    preds = torch.concat(preds)
    preds = preds.cpu().numpy()
    print(preds)
    
    test_data_content = pd.read_excel(test_data_excel, sheet_name=0)
    test_data_content['non_answer'] = preds
    writer = pd.ExcelWriter(save_res_path)
    test_data_content.to_excel(writer)
    writer.save()
    
    
if __name__ == '__main__':
    inference_main()
    