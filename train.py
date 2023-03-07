import os
import torch
import torch.nn as nn

from config import Config, get_default_config
from corpus import deal_train_data, deal_test_data, CustomDataset
from model.xlm_roberta import BertModel

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def train_main():
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    config = get_default_config()
    train_data = deal_train_data()
    train_data, dev_data = train_test_split(train_data, train_size=0.8, shuffle=True)
    train_data = CustomDataset(train_data, config)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_data = CustomDataset(dev_data, config)
    dev_data = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
    
    model = BertModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    
    print(
        torch.cuda.memory_allocated(), 
        torch.cuda.memory_reserved(), 
        # torch.cuda.memory_summary(), 
        # torch.cuda.memory_usage(),
        torch.cuda.max_memory_allocated(),
    )
    return 
    for epoch in range(1, config.epochs+1):
        for x, y in train_data:
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break
        break   


if __name__ == '__main__':
    train_main()