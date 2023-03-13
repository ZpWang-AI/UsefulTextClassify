import os
import torch
import torch.nn as nn
import numpy as np
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    multiclass_f1_score, 
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_confusion_matrix,
)

from utils import clock
from config import *
from corpus import deal_train_data, deal_test_data, CustomDataset
from model.xlm_roberta import BertModel

logging.getLogger('transformers').setLevel(logging.ERROR)


@clock
def eval_main(model, eval_dataloader):
    model.eval()
    pred, groundtruth = [], []
    with torch.no_grad():
        for x, y in eval_dataloader:
            output = model(x)
            pred.append(output)
            groundtruth.append(y)
    pred = torch.cat(pred)
    groundtruth = torch.cat(groundtruth)
    # print(pred)
    # print(groundtruth)
    
    eval_res = [
        ['macro_f1 ', multiclass_f1_score(pred, groundtruth, num_classes=2, average='macro')],
        ['micro_f1 ', multiclass_f1_score(pred, groundtruth, num_classes=2, average='micro')],
        ['accuracy ', multiclass_accuracy(pred, groundtruth, num_classes=2)],
        ['precision', multiclass_precision(pred, groundtruth, num_classes=2)],
        ['recall   ', multiclass_recall(pred, groundtruth, num_classes=2)],
    ]
    eval_res = list(map(lambda x: (x[0], float(x[1])), eval_res))
    confusion_matrix = multiclass_confusion_matrix(pred, groundtruth, num_classes=2)
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
    device = config.device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    
    train_data = deal_train_data()
    train_data, dev_data = train_test_split(train_data, train_size=0.8, shuffle=True)
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
        
    model = BertModel(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    
    print('=== start training ===')
    for epoch in range(1, config.epochs+1):
        model.train()
        for x, y in tqdm(train_data, desc=f'epoch{epoch}'):
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break
        eval_res = eval_main(model, dev_data)
        
        torch.save(
            model.state_dict(), 
            f'{config.save_model_fold}/{config.version}_{int(eval_res["micro_f1 "]*1000)}_epoch{epoch}.pth'
        )
        break   


if __name__ == '__main__':
    train_main()
