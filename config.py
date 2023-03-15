import os
import time
import datetime

from pathlib import Path as path


class Config:
    version = 'base'
    
    # model_name = 'xlm-roberta-base'
    model_name = 'hfl/chinese-roberta-wwm-ext'
    device = 'cpu'
    cuda_id = '0'
    
    save_model_fold = './saved_model'
    save_res_file = './saved_res.txt'
    
    base = True
    clip = False
    just_test = False
    
    epochs = 10
    batch_size = 8
    save_model_epoch = 5
    
    def __init__(self) -> None:
        cur_time = datetime.datetime.now().strftime('%Y_%m_%d-%H:%M:%S')
        self.info = f'{cur_time}_{self.version}'
        if not path(self.save_model_fold).exists():
            os.mkdir(self.save_model_fold)
        if not path(self.save_res_file).exists():
            with open(self.save_res_file, 'w')as f:
                f.write('')
        pass
    
    
def get_default_config():
    return Config()


def get_cuda_config():
    config = get_default_config()
    config.device = 'cuda'
    return config


def get_clip_config():
    config = Config()
    config.version = 'clip'
    config.base = False
    config.clip = True
    return config