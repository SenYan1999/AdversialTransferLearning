import torch
import torch.nn as nn

from model import NER
from model import DomainClassifier

class OverallModel(nn.Module):
    def __init__(self, bert_type, bert_dim, domain_clf_encode_dim, domain_clf_num_layer, \
        seq_len, domain_clf_dropout, num_domains, ner_d_hidden, ner_num_layers, ner_num_tags, ner_dropout):
        super(OverallModel, self).__init__()
        self.domain_classifier = DomainClassifier(bert_type, bert_dim, domain_clf_encode_dim, \
                domain_clf_num_layer, seq_len, domain_clf_dropout, num_domains)
        self.source_ner_model = NER(bert_type, bert_dim, domain_clf_encode_dim, ner_d_hidden, \
                ner_num_layers, ner_num_tags, ner_dropout)

    def forward(self, x, mode):
        if mode == 'source':
            # domain classifier
            task_pred, shared_inform = self.domain_classifier(x)

            # source model
            encoding = self.source_ner_model(x, shared_inform)

            return (task_pred, shared_inform, encoding)
        elif mode == 'target':
            # domain classifier
            task_pred, shared_inform = self.domain_classifier(x)

            return task_pred, shared_inform, None
        else:
            raise('Error Mode')

    def loss_fn(self, task_pred, encoding, task_truth, ner_truth, out_ner_mask):
        domain_loss = self.domain_classifier.loss_fn(task_pred, task_truth)
        source_ner_loss = self.source_ner_model.loss_fn(encoding, out_ner_mask, ner_truth)
        combined_loss = domain_loss + source_ner_loss

        return domain_loss, source_ner_loss, combined_loss

    def step(self, loss, domain_clf_optimizer, domain_adversial_optimizer, source_ner_optimizer, \
        domain_clf_scheduler=None, domain_adversial_scheduler=None, source_scheduler=None):
        loss.backward()
        self.domain_classifier.step(domain_adversial_optimizer, domain_clf_optimizer, \
            domain_adversial_scheduler, domain_clf_scheduler)
        self.source_ner_model.step(source_ner_optimizer, source_scheduler)
    
    def evaluate(self, task_pred, encoding, out_ner_mask, task_truth, ner_truth):
        # domain classifier
        domain_pred = torch.argmax(task_pred, dim=-1)
        domain_acc = torch.sum(domain_pred == task_truth) / task_truth.shape[0]

        # ner model
        ner_pred = self.source_ner_model.predict_with_pad(encoding, out_ner_mask)
        ner_acc, ner_f1 = self.source_ner_model.evaluate(ner_pred, ner_truth)

        return (domain_acc.item(), ner_acc, ner_f1)
