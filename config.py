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
    dev_data_file = ''
    pretrained_model_fold = './saved_model'
    save_res_fold = './saved_res'
    # save_res_file = './saved_res.txt'
    
    input_feature = 'reply only'  # reply only, qsubj+reply, reply+qsubj
    
    base = True
    clip = False
    just_test = False
    
    epochs = 10
    batch_size = 8
    save_model_epoch = 5
    pb_frequency = 10
    train_ratio = 0.8
    lr = 5e-5
    
    def as_list(self):
        return [[attr, getattr(self, attr)] for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("__")]
    
    def as_dict(self):
        return dict(self.as_list())
   

if __name__ == '__main__':
    sample_config = CustomConfig()
    print(sample_config.as_list())
    print(sample_config.as_dict())
    pass