import os
import torch
import torch.nn as nn
import numpy as np
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torcheval.metrics.functional import (
    binary_f1_score,
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_confusion_matrix,
    # multiclass_f1_score, 
    # multiclass_accuracy,
    # multiclass_precision,
    # multiclass_recall,
    # multiclass_confusion_matrix,
)

from utils import *
from config import *
from corpus import (train_data_file_list,
                    test_data_file_list,
                    preprocess_train_data, 
                    preprocess_test_data, 
                    CustomDataset)
from model.bertModel import BertModel

logging.getLogger('transformers').setLevel(logging.ERROR)


@clock
def eval_main(model, eval_dataloader, logger):
    model.eval()
    pred, groundtruth = [], []
    with torch.no_grad():
        for x, y in eval_dataloader:
            output = model.predict(x)
            pred.append(output)
            groundtruth.append(y)
    pred = torch.cat(pred).cpu()
    groundtruth = torch.cat(groundtruth)
    # print(pred)
    # print()
    # print(groundtruth)
    
    eval_res = [
        ['f1       ', binary_f1_score(pred, groundtruth)],
        ['accuracy ', binary_accuracy(pred, groundtruth)],
        ['precision', binary_precision(pred, groundtruth)],
        ['recall   ', binary_recall(pred, groundtruth)],
    ]
    eval_res = list(map(lambda x: (x[0], float(x[1])), eval_res))
    confusion_matrix = binary_confusion_matrix(pred, groundtruth)
    confusion_matrix = np.array(confusion_matrix)
    # print(eval_res)
    # print(confusion_matrix)
    
    def show_res():
        for name, num in eval_res:
            logger.info(f'{name}: {num*100:.2f}%')
        logger.info(
            f'\n\t pred_0\t pred_1',
            f'gt_0 \t {confusion_matrix[0][0]} \t {confusion_matrix[0][1]}',
            f'gt_1 \t {confusion_matrix[1][0]} \t {confusion_matrix[1][1]}',
            sep='\n'
        )
        pass
    
    show_res()
    return {k.strip(): v for k, v in eval_res}
    

@clock(sym='-----')
def train_main(config: CustomConfig):    
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    start_time = get_cur_time()
    saved_res_fold = path(config.save_res_fold) / path(f'{start_time}_{config.version}')
    saved_model_fold = saved_res_fold / path('saved_model')
    saved_model_fold.mkdir(parents=True, exist_ok=True)
    
    logger = MyLogger(
        fold=saved_res_fold, file=f'{start_time}_{config.version}.out',
        just_print=config.just_test, 
        # log_with_time=(not config.just_test),
        log_with_time=False,
    )
    
    logger.info(config.as_list())
    logger.info(get_cur_time())
    
    train_data = preprocess_train_data(config.train_data_file)
    if not config.dev_data_file:
        train_data, dev_data = train_test_split(train_data, train_size=config.train_ratio, shuffle=True)
    else:
        dev_data = preprocess_train_data(config.dev_data_file)
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
        
    model = BertModel(config)
    model.get_pretrained_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # scheduler = StepLR(optimizer, step_size=10)
    
    logger.info('=== start training ===')
    for epoch in range(1, config.epochs+1):
        model.train()
        tot_loss = AverageMeter()
        pb = tqdm(total=len(train_data), desc=f'epoch{epoch}')
        for p, (x, y) in enumerate(train_data):
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            tot_loss.add(loss)
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            if config.just_test:
                break
            p += 1
            if p % config.pb_frequency == 0 or p == len(train_data):
                pb.update(min(config.pb_frequency, pb.total-pb.n))
                # logger.info(f'loss: {tot_loss.average:.6f}, lr: {scheduler.get_lr()[0]}')
                logger.info(f'batch[{p}/{len(train_data)}], loss: {tot_loss.average:.6f}')
        pb.close()
        logger.info(f'epoch{epoch} ends')
        eval_res = eval_main(model, dev_data, logger)
        
        if config.just_test:
            break   
        if epoch % config.save_model_epoch == 0:
            saved_model_file = (
                f'{start_time.replace(":", "-")}_'
                f'epoch{epoch}_'
                f'{int(eval_res["f1"]*1000)}'
                '.pth'
            )
            torch.save(
                model.state_dict(), 
                saved_model_fold / saved_model_file
            )
    logger.info('=== finish training ===')
    logger.info(get_cur_time())
    logger.close()
    del logger


if __name__ == '__main__':
    def get_config_base_test():
        custom_config = CustomConfig()
        custom_config.device = 'cuda'
        custom_config.cuda_id = '8'
        
        # custom_config.just_test = True
        custom_config.save_model_epoch = 1
        custom_config.pb_frequency = 20
        custom_config.batch_size = 8
        custom_config.lr = 5e-5
        
        custom_config.train_data_file = train_data_file_list[2]
        custom_config.dev_data_file = ''
        return custom_config
    
    def get_config_11():
        config = get_config_base_test()
        config.version = 'train 1 test 1'
        config.train_data_file = train_data_file_list[1]
        config.dev_data_file = ''
        return config
        
    def get_config_22():
        config = get_config_base_test()
        config.version = 'train 2 test 2'
        config.train_data_file = train_data_file_list[2]
        config.dev_data_file = ''
        return config
    
    def get_config_12():
        config = get_config_base_test()
        config.version = 'train 1 test 2'
        config.train_data_file = train_data_file_list[1]
        config.dev_data_file = train_data_file_list[2]
        return config
        
    def get_config_21():
        config = get_config_base_test()
        config.version = 'train 2 test 1'
        config.train_data_file = train_data_file_list[2]
        config.dev_data_file = train_data_file_list[1]
        return config
    
    def get_config_mix_12():
        config = get_config_base_test()
        config.version = 'train mix12 test mix12'
        config.train_data_file = train_data_file_list[1]+'@'+train_data_file_list[2]
        config.dev_data_file = ''
        return config
    
    # train_main(get_config_base_test())
    # train_main(get_config_base_test())
    # exit()
    train_main(get_config_11())
    train_main(get_config_22())
    train_main(get_config_12())
    train_main(get_config_21())
    train_main(get_config_mix_12())
    
    
