import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from utils import logger

class Trainer:
    def __init__(self, train_source_dataloader, train_target_dataloader, dev_source_dataloader, \
        dev_target_dataloader, overall_model, source_ner_optimizer, domain_clf_adversial_optimizer, \
            domain_clf_optimizer):
        '''
        train_source_dataloader: contains source domain corpus to train
        train_target_dataloader: contains target domain corpus to train
        dev_source_dataloader: contains source domain corpus to evaludate
        overall_model: the overall model
        source_ner_optimizer: optimizer of source domain ner task
        domain_clf_adversial_optimizer: optimizer of domain classifier(to optim layers before classifier with reversed grad)
        domain_clf_optimizer: optimizer of domain classifier(to optim classifier layer)
        '''

        self.train_source_dataloader = training_target_dataloader
        self.train_target_dataloader = training_target_dataloader
        self.dev_source_dataloader = dev_source_dataloader
        self.dev_target_dataloader = dev_target_dataloader
        self.source_ner_optimizer = source_ner_optimizer
        self.domain_clf_adversial_optimizer = domain_clf_adversial_optimizer
        self.domain_clf_optimzir = domain_clf_optimzier
        self.overall_model = overall_model

    def train_epoch(self, epoch):
        # step 1: define training setting
        logger.info('Epoch: %2d: Training Model...' % epoch)
        logger.info('')
        self.overall_model.train()
        pbar_source = tqdm(total = len(self.train_source_dataloader))
        pbar_target = tqdm(total = len(self.train_target_dataloader))

        # step 2: training source domain
        logger.info('Epoch: %2d: Training source domain...' % epoch)
        domain_loss_list, domain_acc_list, ner_loss_list, ner_acc_list, ner_f1_list = [], [], [], [], []
        for batch in self.train_source_dataloader:
            x, out_ner_mask, y, label = map(lambda i: i.to(self.device), batch)

            # apply model
            task_logits, shared_inform, encoding = self.overall_model(x, 'source')

            # compute loss
            domain_loss, ner_loss, combine_loss = self.overall_model.loss_fn(task_logits, encoding, label, y, out_ner_mask)

            # backward and step
            self.overall_model.step(combine_loss, self.domain_clf_optimzer, self.domain_clf_adversial_optimizer, self.source_ner_optimizer)

            # evaluate
            domain_acc, ner_acc, ner_f1 = self.overall_model.evaluate(task_logits, encoding, out_ner_mask, label, y)

            # append metrics to list
            domain_loss_list.append(domain_loss.item())
            domain_acc_list.append(domain_acc)
            ner_loss_list.append(ner_loss.item())
            ner_acc_list.append(ner_acc)
            ner_f1_list.append(ner_f1)

            pbar.set_description('Epoch: %2d | Adversial Loss: %2.3f | Source Ner Loss: %1.3f | Domain ACC: %1.3f | NER ACC: %1.3f  ' % (epoch, domain_loss.item(), \
                ner_loss.item(), domain_acc, ner_acc))
            pbar.update(1)
        pbar.close()
        logger.info('Source Domain:')
        logger.info('Epoch: %2d | Adversial Loss: %2.3f | Source Ner Loss: %1.3f | Domain Acc: %1.3f | NER ACC: %1.3f | NER F1: %1.3f' %
                    (epoch, np.mean(domain_loss_list), np.mean(ner_loss_list), np.mean(domain_acc_list), np.mean(ner_acc_list), np.mean(ner_f1_list)))
        
        # step 2: training target domain
        logger.info('Epoch: %2d: Training target domain...' % epoch)
        domain_loss_list, domain_acc_list = [], [], [], [], []
        for batch in self.train_source_dataloader:
            x, out_ner_mask, y, label = map(lambda i: i.to(self.device), batch)

            # apply model
            task_logits, shared_inform, encoding = self.overall_model(x, 'target')

            # compute loss
            domain_loss, ner_loss, combine_loss = self.overall_model.loss_fn(task_logits, encoding, label, y, out_ner_mask)

            # backward and step
            self.overall_model.step(combine_loss, self.domain_clf_optimzir, self.domain_clf_adversial_optimizer, self.source_ner_optimizer)

            # evaluate
            domain_acc, ner_acc, ner_f1 = self.overall_model.evaluate(task_logits, encoding, out_ner_mask, label, y)

            # append metrics to list
            domain_loss_list.append(domain_loss.item())
            domain_acc_list.append(domain_acc)

            pbar.set_description('Epoch: %2d | Adversial Loss: %2.3f Domain Acc:' % (epoch, domain_loss.item(), domain_acc))
            pbar.update(1)
        pbar.close()
        logger.info('Target Domain:')
        logger.info('Epoch: %2d | Adversial Loss: %2.3f | Domain Acc: %1.3f' % (epoch, np.mean(domain_loss_list), np.mean(domain_acc_list)))


    def train(self, num_epoch, save_path):
        for epoch in range(num_epoch):
            self.train_epoch(epoch)
