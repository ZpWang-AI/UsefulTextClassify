import csv
import os
import warnings

from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
# warnings.filterwarnings('')


def get_data1():
    data_path1 = r'D:\NewDesktop\本科生学务\2023-大四下\毕设\毕设data\hate-speech-and-offensive-language-master\labeled_data.csv'

    with open(data_path1, 'r')as f:
        reader = csv.reader(f)
        content = [p[-2:] for p in reader][1:]
        content = [[p[1], int(p[0])]for p in content]

        # for i in content[:5]:
        #     print(i)
    return content


def get_data2():
    data_path2 = r''


class CustomDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def deal_sentence(self, sentence:str):
        return sentence
        sentence = sentence.strip().split()
        ans_sentence = []
        for word in sentence:
            if word[0] != '@':
                ans_sentence.append(word)      
        return ' '.join(ans_sentence)
    
    def __getitem__(self, index):
        sentence, label = self.data[index]
        return self.deal_sentence(sentence), label
    

if __name__ == '__main__':
    sample_data = get_data1()
    sample_dataset = CustomDataset(sample_data)
    sample_dataloader = DataLoader(sample_dataset, batch_size=3)
    for a, b in sample_dataloader:
        print(a, b)
        break
    
    from models.bert import MyBert
    sample_model = MyBert()
    for a, b in sample_dataloader:
        output = sample_model(a)
        print(output)
        break