import torch
import torch.nn as nn

from model import NER
from mdoel import DomainClassifier

class OverallModel(nn.Module):
    def __init__(self, bert_type, bert_dim, domain_clf_encode_dim, domain_clf_num_layer, \
        seq_len, domain_clf_dropout, num_domains, ner_d_hidden, ner_num_layers, ner_num_tags, ner_dropout):
        super(OverallModel).__init__()
        self.domain_classifier = DomainClassifier(bert_type, bert_dim, domain_clf_encoe_dim, \
            domain_clf_num_layer, seq_len, domain_clf_dropout, num_domains)
        self.source_ner_model = NER(bert_type, bert_dim, domain_clf_encode_dim, ner_d_hidden, \
            ner_num_layers, ner_num_tags, ner_dropout)

    def forward(self, x):
        # domain classifier
        task_pred, shared_inform = self.domain_classifier(x)

        # source model
        encoding = self.source_ner_model(x, shared_inform)

        return (task_pred, shared_inform, encoding)

    def loss_fn(self, task_pred, encoding, task_truth, ner_truth, out_ner_mask):
        domain_loss = self.domain_classifier.loss_fn(task_pred, task_truth)
        source_ner_loss = self.source_ner_model.loss_fn(encoding, out_ner_mask, ner_truth)
        combined_loss = domain_loss + source_ner_loss

        return domain_loss, source_ner_loss, combined_loss

    def step(self, loss, domain_clf_optimizer, domain_adversial_optimizer, source_ner_optimizer, \
        domain_clf_scheduler, domain_adversial_scheduler=None, source_scheduler=None):
        loss.backward()
        self.domain_classifier.step(domain_adversial_optimizer, domain_clf_optimizer, \
            domain_adversial_scheduler, domain_clf_scheduler)
        self.source_ner_model.step(loss, source_ner_optimizer, source_scheduler)
    
    def evaluate(self, task_pred, encoding, out_ner_mask, task_truth, ner_truth):
        # domain classifier
        domain_pred = torch.argmax(task_pred, dim=-1)
        domain_acc = (domain_pred == task_truth) / task_truth.shape[0]

        # ner model
        ner_acc, ner_f1 = 0, 0

        return (domain_acc, ner_acc, ner_f1)
