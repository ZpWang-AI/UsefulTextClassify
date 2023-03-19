import os
import time
import datetime

from pathlib import Path as path


class CustomConfig:
    version = 'base'
    
    # model_name = 'xlm-roberta-base'
    model_name = 'hfl/chinese-roberta-wwm-ext'
    device = 'cpu'
    cuda_id = '0'
    
    train_data_file = './data/randomdata_1000 20230213_training_dataset.xlsx'
    pretrained_model_fold = './saved_model'
    save_res_file = './saved_res.txt'
    
    input_feature = 'reply only'  # reply only, qsubj+reply, reply+qsubj
    
    base = True
    clip = False
    just_test = False
    
    epochs = 10
    batch_size = 8
    save_model_epoch = 5
