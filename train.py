import os
import torch
import torch.nn as nn
import numpy as np
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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

from utils import clock
from config import *
from corpus import (train_data_file_list,
                    test_data_file_list,
                    preprocess_train_data, 
                    preprocess_test_data, 
                    CustomDataset)
from model.xlm_roberta import BertModel

logging.getLogger('transformers').setLevel(logging.ERROR)


@clock
def eval_main(model, eval_dataloader):
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
            print(f'{name}: {num*100:.2f}%')
        print(
            f'\t pred_0\t pred_1',
            f'gt_0 \t {confusion_matrix[0][0]} \t {confusion_matrix[0][1]}',
            f'gt_1 \t {confusion_matrix[1][0]} \t {confusion_matrix[1][1]}',
            sep='\n'
        )
        pass
    
    show_res()
    return dict(eval_res)
    

@clock(sym='-----')
def train_main():
    config = get_default_config()
    # config = get_cuda_config()
    config.device = 'cuda'
    config.cuda_id = '9'
    # config.just_test = True
    train_data_file = train_data_file_list[1]
    config.save_model_epoch = 1
    config.batch_size = 8
    
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    train_data = preprocess_train_data(train_data_file)
    train_data, dev_data = train_test_split(train_data, train_size=0.8, shuffle=True)
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
        
    model = BertModel(config)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    print('=== start training ===')
    for epoch in range(1, config.epochs+1):
        model.train()
        tot_loss = 0
        for x, y in tqdm(train_data, desc=f'epoch{epoch}'):
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            tot_loss += loss
            optimizer.step()
            optimizer.zero_grad()
            if config.just_test:
                break
        print('loss:', tot_loss/len(train_data))
        eval_res = eval_main(model, dev_data)
        
        if config.just_test:
            break   
        if epoch % config.save_model_epoch == 0:
            torch.save(
                model.state_dict(), 
                f"{config.save_model_fold}/{config.info.replace(':', '_')}_epoch{epoch}_{int(eval_res['accuracy ']*1000)}.pth"
            )


if __name__ == '__main__':
    train_main()
