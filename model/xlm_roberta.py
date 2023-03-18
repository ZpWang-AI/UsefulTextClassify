# import os
import torch
import torch.nn as nn

from torch.nn import functional as F
from transformers import (XLMRobertaModel, 
                          XLMRobertaTokenizer, 
                          XLMRobertaForSequenceClassification, 
                          XLMRobertaConfig,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          AutoTokenizer
                          )

# os.environ['CUDA_VISIBLE_DEVICES']='5'
# os.environ['TOKENIZERS_PARALLELISM']='true'

model_name = "xlm-roberta-base"
# tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
# model = XLMRobertaModel.from_pretrained(model_name)


class BertModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        
    def forward(self, sentences):
        # sentences = [sa+self.tokenizer.sep_token+sb for sa, sb in zip(*sentences)]
        sentences = [sb+self.tokenizer.sep_token+sa for sa, sb in zip(*sentences)]
        # sentences = [sb for sa, sb in zip(*sentences)]
        
        encoded_sentences = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_sentences = encoded_sentences.to(self.config.device)
        output = self.model(**encoded_sentences)
        return output.logits
    
    def predict(self, sentences):
        logits = self(sentences)
        return torch.argmax(logits, dim=1)


if __name__ == '__main__':
    pass
    
