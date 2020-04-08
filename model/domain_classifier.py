import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import BertEncoder

class DomainClassifier(nn.Module):
    def __init__(self, bert_type, bert_dim, encode_dim, encode_num_layers, seq_len, drop_out, num_domains):
        super(DomainClassifier, self).__init__()
        
        # encoding layer
        self.encoder = BertEncoder(bert_type, bert_dim, encode_dim, encode_num_layers, drop_out)

        # polling layer
        self.pooling = nn.MaxPool1d(kernel_size=seq_len, stride=1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(encode_dim, encode_dim // 2),
            nn.ReLU(),
            nn.Linear(encode_dim // 2, num_domains)
        )

    def forward(self, x):
        # encoding
        x_encoded = self.encoder(input_ids=x)

        # max_pool
        x_pool = self.pooling(x_encoded.permute(0, 2, 1)).squeeze()

        # classify
        out = self.classifier(x_pool)
        out = F.log_softmax(out, dim=-1)

        return out, x_encoded

    def loss_fn(self, logits, target):
        loss = F.nll_loss(logits, target)
    
    def step(self, adversial_optimizer, classifier_optimizer, scheduler=None):
        # step optimizer
        adversial_optimizer.step()
        classifier_optimizer.step()

        # scheduler if exsits
        try:
            scheduler.step()
        except:
            continue

        # zeros grad
        adversial_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
