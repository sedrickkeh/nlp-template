import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, out_dim):
        super(BertClassifier, self).__init__()
        self.out_dim = out_dim
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.out_dim)

        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        input_ids, attention_mask, segment_ids = batch
        batch_size = input_ids.size(0)
        outputs = self.model(input_ids, segment_ids, attention_mask, labels=None)
        logits = outputs[0]
        logits = logits.view(batch_size, -1, self.out_dim)
        out = logits[:, -1]
        out = self.softmax(out)
        return out

    def get_bert_features(self, batch):
        input_ids, attention_mask, segment_ids = batch
        
        features = self.model.bert(input_ids, segment_ids, attention_mask)

        return features