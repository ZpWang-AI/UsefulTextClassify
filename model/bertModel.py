# import os
import torch
import torch.nn as nn

from pathlib import Path as path
from torch.nn import functional as F
from transformers import (
    XLMRobertaConfig,
    XLMRobertaModel, 
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification, 
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class BertModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=self.config.pretrained_model_fold)
        self.model_config = AutoConfig.from_pretrained(config.model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_config(self.model_config)
    
    def get_pretrained_model(self):
        path(self.config.pretrained_model_fold).mkdir(parents=True, exist_ok=True)
        self.model = self.model.from_pretrained(self.config.model_name, cache_dir=self.config.pretrained_model_fold)
        self.to(self.config.device)
        
    def forward(self, sentences):
        if self.config.input_feature == 'reply only':
            sentences = [sb for sa, sb in zip(*sentences)]
        elif self.config.input_feature == 'qsubj+reply':
            sentences = [sa+self.tokenizer.sep_token+sb for sa, sb in zip(*sentences)]    
        elif self.config.input_feature == 'reply+qsubj':
            sentences = [sb+self.tokenizer.sep_token+sa for sa, sb in zip(*sentences)]
        else:
            raise 'Wrong Config.input_feature'
        
        encoded_sentences = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_sentences = encoded_sentences.to(self.config.device)
        output = self.model(**encoded_sentences)
        return output.logits
    
    def predict(self, sentences):
        logits = self(sentences)
        return torch.argmax(logits, dim=1)


if __name__ == '__main__':
    class SampleConfig:
        model_name = 'hfl/chinese-roberta-wwm-ext'
        device = 'cuda'
        cuda_id = '9'
        
        pretrained_model_fold = './saved_model'
        input_feature = 'reply only'  # reply only, qsubj+reply, reply+qsubj
        
    sample_sentence_pairs = [
        ['你好', '我好'],
        ['他好', '谢谢'*100000],
    ]
    sample_model = BertModel(SampleConfig())
    sample_model.get_pretrained_model()
    sample_model.to(SampleConfig.device)
    sample_model.eval()
    sample_output = sample_model(sample_sentence_pairs)
    print(sample_output)
    print(sample_output.shape)
    sample_preds = sample_model.predict(sample_sentence_pairs)
    print(sample_preds)
    print(sample_preds.shape)
    pass
    