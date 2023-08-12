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
    print(config.as_dict())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
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
    pb = tqdm(total=len(test_dataloader))
    for p, input_x in enumerate(test_dataloader):
        with torch.no_grad():
            output = model.predict(input_x)
        preds.append(output)
        p += 1
        if p % config.pb_frequency == 0:
            pb.update(config.pb_frequency)
        elif p == len(test_dataloader):
            pb.update(pb.total-pb.n)
    pb.close()
    preds = torch.concat(preds)
    preds = preds.cpu().numpy()
    # print(preds)
    
    save_res_path = f'./data/result_{config.version}.xlsx'
    if test_data_file == test_data_file_list[0]:
        test_data_content = pd.read_excel(test_data_file, sheet_name=0)
        test_data_content['non_answer'] = preds
        writer = pd.ExcelWriter(save_res_path)
        test_data_content.to_excel(writer)
        writer.close()
    elif test_data_file == test_data_file_list[1]:
        test_data_content = pd.read_excel(test_data_file, sheet_name=1)
        test_data_content['non_answer (not marked)'] = preds
        writer = pd.ExcelWriter(save_res_path)
        test_data_content.to_excel(writer)
        writer.close()
    elif test_data_file == test_data_file_list[2]:
        save_res_path = f'./data/result_{config.version}.csv'
        with open(save_res_path, 'w')as f:
            for p, d in enumerate(list(preds)):
                f.write(f'{p},{d}\n')
    elif test_data_file == test_data_file_list[4]:
        test_data_content = pd.read_excel(test_data_file, sheet_name=0)
        test_data_content['real_questions'] = preds
        writer = pd.ExcelWriter(save_res_path)
        test_data_content.to_excel(writer)
        writer.close()
    else:
        raise BaseException( 'Wrong test file in inference.py')
    
    
if __name__ == '__main__':
    def get_config_infer1():
        custom_config = CustomConfig()
        custom_config.batch_size = 32
        custom_config.device = 'cuda'
        custom_config.cuda_id = '3'

        # custom_config.test_data_file = test_data_file_list[1]
        # custom_config.version = 'train1test2_test2'
        # custom_config.test_model_path = r'./saved_res/2023-03-20_13:33:05_train 1 test 2/saved_model/2023-03-20_13-33-05_epoch10_830.pth'
        
        # custom_config.test_data_file = test_data_file_list[1]
        # custom_config.version = 'train1test1_test2'
        # custom_config.test_model_path = r'./saved_res/2023-03-20_13:26:12_train 1 test 1/saved_model/2023-03-20_13-26-12_epoch10_857.pth'
        
        custom_config.test_data_file = test_data_file_list[2]
        custom_config.version = 'train1test2_txt1'
        custom_config.test_model_path = r'./saved_res/2023-03-20_13_33_05_train_1_test_2/saved_model/2023-03-20_13-33-05_epoch10_830.pth'
        custom_config.pb_frequency = 100
        return custom_config

    def get_config_infer_question_base():
        custom_config = CustomConfig()
        custom_config.input_feature = 'qsubj only'
        custom_config.device = 'cuda'
        custom_config.cuda_id = '4'
        custom_config.batch_size = 32

        custom_config.test_data_file = test_data_file_list[4]
        custom_config.version = 'classify_real_questions_base'
        custom_config.test_model_path = r'./saved_res/2023-08-12_13_23_24_question_base/saved_model/2023-08-12_13_23_24_epoch5_971.pth'
        custom_config.pb_frequency = 100
        return custom_config
    
    infer_config = get_config_infer_question_base()
    inference_main(infer_config)