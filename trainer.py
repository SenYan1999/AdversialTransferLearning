import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from utils import logger

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, overall_model, source_ner_optimizer, \
        domain_clf_adversial_optimizer, domain_clf_optimizer):
        '''
        train_dataloader: contains both source and target domain corpus to train
        dev_dataloader: contains both source and target domain corpus to evaludate
        overall_model: the overall model
        source_ner_optimizer: optimizer of source domain ner task
        domain_clf_adversial_optimizer: optimizer of domain classifier(to optim layers before classifier with reversed grad)
        domain_clf_optimizer: optimizer of domain classifier(to optim classifier layer)
        '''

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.source_ner_optimizer = source_ner_optimizer
        self.domain_clf_adversial_optimizer = domain_clf_adversial_optimizer
        self.domain_clf_optimizer = domain_clf_optimizer
        self.overall_model = overall_model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.overall_model.to(self.device)

    def train_epoch(self, epoch):
        # step 1: define training setting
        logger.info('Epoch: %2d: Training Model...' % epoch)
        logger.info('')
        self.overall_model.train()
        pbar_source = tqdm(total = len(self.train_dataloader))

        # step 2: training
        logger.info('Epoch: %2d: Training source domain...' % epoch)
        domain_loss_list, domain_acc_list, ner_loss_list, ner_acc_list, ner_f1_list = [], [], [], [], []
        for batch in self.train_dataloader:
            for mode in ['source', 'target']:
                guid, x, x_mask, x_segment, y, out_ner_mask, label = map(lambda i: i.to(self.device), batch)

                # apply model
                task_logits, shared_inform, encoding = self.overall_model(x, mode)

                if mode == 'source':
                    idx = torch.where(label == 0)[0]
                    # compute loss
                    domain_loss, ner_loss, combine_loss = self.overall_model.loss_fn(task_logits[idx], encoding[idx], label[idx], y[idx], out_ner_mask[idx])

                    # backward and step
                    self.overall_model.step(combine_loss, self.domain_clf_optimizer, self.domain_clf_adversial_optimizer, self.source_ner_optimizer)

                    # evaluate
                    domain_acc, ner_acc, ner_f1 = self.overall_model.evaluate(task_logits[idx], encoding[idx], out_ner_mask[idx], label[idx], y[idx])

                    # append metrics to list
                    domain_loss_list.append(domain_loss.item())
                    domain_acc_list.append(domain_acc)
                    ner_loss_list.append(ner_loss.item())
                    ner_acc_list.append(ner_acc)
                    ner_f1_list.append(ner_f1)
                
                else:
                    idx = torch.where(label == 1)[0]
                    if idx.sum() == 0:
                        continue
                    domain_loss = self.overall_model.domain_classifier.loss_fn(task_logits[idx], label[idx])
                    domain_loss.backward()
                    self.domain_clf_adversial_optimizer.step()
                    self.domain_clf_optimizer.step()
                    self.domain_clf_adversial_optimizer.zero_grad()
                    self.domain_clf_optimizer.zero_grad()

            pbar_source.set_description('Epoch: %2d | Adversial Loss: %2.3f | Source Ner Loss: %1.3f | Domain ACC: %1.3f | NER ACC: %1.3f  ' % (epoch, domain_loss.item(), \
                ner_loss.item(), domain_acc, ner_acc))
            pbar_source.update(1)
        pbar_source.close()
        logger.info('Source Domain:')
        logger.info('Epoch: %2d | Adversial Loss: %2.3f | Source Ner Loss: %1.3f | Domain Acc: %1.3f | NER ACC: %1.3f | NER F1: %1.3f' %
                    (epoch, np.mean(domain_loss_list), np.mean(ner_loss_list), np.mean(domain_acc_list), np.mean(ner_acc_list), np.mean(ner_f1_list)))
        
    def train(self, num_epoch, save_path):
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
