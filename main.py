import torch
import os

from torch.utils.data import DataLoader
from torch.optim import Adam
from model import OverallModel, AdversialAdam
from trainer import Trainer
from utils import InputDataset
from args import args

def prepare_data():
    for mode in ['train', 'dev', 'test']:
        data = InputDataset(mode)
        torch.save(data, os.path.join(args.data_dir, mode+'.pt'))

def train():
    # prepare dataset
    train_data = torch.load(os.path.join(args.data_dir, 'train.pt'))
    dev_data = torch.load(os.path.join(args.data_dir, 'dev.pt'))

    # prepare dataloader
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)

    # prepare model
    model = OverallModel(args.bert_type, args.d_bert, args.domain_clf_encoding_dim, args.domain_clf_encoding_num_layers, \
        args.max_len, args.domain_clf_dropout, args.domain_clf_num_domains, args.ner_d_hidden, args.ner_num_layers, \
            args.ner_num_tags, args.ner_dropout)
    
    # prepare optimizer
    domain_encoder_params = filter(lambda x:id(x) not in list(map(id,model.domain_classifier.classifier.parameters())), \
        model.domain_classifier.parameters())
    domain_clf_params = model.domain_classifier.classifier.parameters()
    source_ner_params = model.source_ner_model.parameters()

    domain_adversial_optimizer = AdversialAdam(domain_encoder_params, lr=args.learning_rate)
    domain_optimizer = Adam(domain_clf_params, lr=args.learning_rate)
    source_ner_optimizer = Adam(source_ner_params, lr=args.learning_rate)

    # define trainer
    trainer = Trainer(train_dataloader, dev_dataloader, model, source_ner_optimizer, domain_adversial_optimizer, domain_optimizer)

    # begin training process
    trainer.train(args.num_epochs, '')


if __name__ == "__main__":
    if args.prepare_data:
        prepare_data()

    if args.train:
        train()
