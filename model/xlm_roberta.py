# import os
import torch
import torch.nn as nn

from torch.nn import functional as F
from transformers import (XLMRobertaModel, 
                          XLMRobertaTokenizer, 
                          XLMRobertaForSequenceClassification, 
                          XLMRobertaConfig)

# os.environ['CUDA_VISIBLE_DEVICES']='5'
# os.environ['TOKENIZERS_PARALLELISM']='true'

model_name = "xlm-roberta-base"
# tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
# model = XLMRobertaModel.from_pretrained(model_name)


class BertModel(nn.Module):
    def __init__(self, bert_name='xlm-roberta-base') -> None:
        super().__init__()
        self.config = XLMRobertaConfig.from_pretrained(bert_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(bert_name, config=self.config)
        self.model = XLMRobertaModel.from_pretrained(bert_name, config=self.config)
        self.classify_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 2),
            nn.Softmax(dim=1),
        )
        
    def forward(self, sentences):
        encoded_sentences = self.tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt')
        output = self.model(**encoded_sentences)
        # print(output)
        # print(output.last_hidden_state)
        # print(output.last_hidden_state.shape)
        cls = output.last_hidden_state[:,0,:]
        # print(cls.shape)
        pred = self.classify_head(cls)
        # predicted_labels = torch.argmax(output.logits, dim=1)
        # print(predicted_labels)
        return pred


if __name__ == '__main__':
    pass
    
