import torch
import torch.nn as nn
import numpy as np

from transformers import BertModel
from model.layers import LSTMEncoder
from model.crf import CRF
from sklearn.metrics import f1_score

class NER(nn.Module):
    def __init__(self, bert_type, d_bert, d_share, d_hidden, num_layers, num_tags, drop_out):
        '''
        bert_type: bert_type or bert path
        d_bert: dimension of bert output
        d_share: dimension of the shared_inform
        d_hidden: dimension of the hidden state
        num_layers: layers of the lstm encoder
        num_tags: numbers of tags
        '''
        super(NER, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(bert_type)
        self.lstm_encoder = LSTMEncoder(d_bert + d_share, d_hidden, num_layers, drop_out)
        self.classifier = nn.Linear(d_hidden, num_tags) 
        self.crf = CRF(num_tags, use_cuda = True if torch.cuda.is_available() else False)

    def forward(self, x, shared_inform):
        '''
        x: sentence to be tagged
        shared_inform: the shared information between source and target domain, retrive from doamin_classifier
        '''

        # get x_encoded from bert layer
        x_encoded = self.bert_encoder(input_ids=x)[0]
        
        # combine shared_inform and x_encodded
        encoding_with_external_inform = torch.cat([x_encoded, shared_inform], dim=-1)
        x_len = torch.sum(x != 0, dim=-1)
        encoding_with_external_inform = self.lstm_encoder(encoding_with_external_inform, x_len)

        # get logits
        out = self.classifier(encoding_with_external_inform)
        
        return out

    def loss_fn(self, logits, mask, y):
        loss = self.crf.negative_log_loss(logits, mask, y)
        return loss
    
    def step(self, optimizer, scheduler=None):
        optimizer.step()
        try:
            scheduler.step()
        except:
            pass

        optimizer.zero_grad()

    def predict_with_pad(self, logits, mask):
        predicts = self.crf.get_batch_best_path(logits, mask)
        return predicts

    def predict_without_pad(self, logits, mask):
        predicts = self.crf.get_batch_best_path(logits, mask)
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts
    
    def evaluate(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy().reshape(-1)
        y_true = y_true.cpu().numpy().reshape(-1)
        f1 = f1_score(y_true, y_pred, average="weighted")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return acc, f1
