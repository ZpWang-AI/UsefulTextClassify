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

from config import CustomConfig


class BertModel(nn.Module):
    def __init__(self, config: CustomConfig) -> None:
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config.model_name, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_config(self.model_config)
    
    def get_pretrained_model(self, cache_dir='./saved_model'):
        path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.model = self.model.from_pretrained(self.config.model_name, cache_dir=cache_dir)
        self.to(self.config.device)
        
    def forward(self, sentences):
        if self.config.input_feature == 'reply':
            sentences = [sb for sa, sb in zip(*sentences)]
        # sentences = [sa+self.tokenizer.sep_token+sb for sa, sb in zip(*sentences)]
        # sentences = [sb+self.tokenizer.sep_token+sa for sa, sb in zip(*sentences)]
        
        encoded_sentences = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_sentences = encoded_sentences.to(self.config.device)
        output = self.model(**encoded_sentences)
        return output.logits
    
    def predict(self, sentences):
        logits = self(sentences)
        return torch.argmax(logits, dim=1)


if __name__ == '__main__':
    class SampleConfig:
        model_name = 'xlm-roberta-base'
        device = 'cuda'
        
    sample_sentences = ['a sample sentence', 
                        'two sample sentences',
                        'three sample sentences',
                        'four sample sentences '*1000,
                        ]
    sample_model = BertModel(SampleConfig())
    sample_model.get_pretrained_encoder()
    sample_model.freeze_encoder()
    sample_model.to(SampleConfig.device)
    sample_model.eval()
    sample_output = sample_model(sample_sentences)
    print(sample_output)
    print(sample_output.shape)
    sample_preds = sample_model.predict(sample_sentences)
    print(sample_preds)
    print(sample_preds.shape)
    pass
    