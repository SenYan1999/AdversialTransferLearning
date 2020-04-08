import argparse

# define args parser
parser = argparse.ArgumentParser()

# data prepropare
parser.add_argument('--max_len', default=150, type=int)

# model
parser.add_argument('--bert_type', default='pretrained_model/', type=str)
parser.add_argument('--bert_tokenizer_type', default='pretrained_model/vocab.txt', type=str)
parser.add_argument('--d_bert', default=768, type=int)
# domain classifier
parser.add_argument('--domain_clf_encoding_dim', default=512, type=int)
parser.add_argument('--domain_clf_encoding_num_layers', default=3, type=int)
parser.add_argument('--domain_clf_dropout', default=0.2, type=float)
parser.add_argument('--domain_clf_num_domains', default=2, type=int)
# ner model
parser.add_argument('--ner_d_hidden', default=512, type=int)
parser.add_argument('--ner_num_layers', default=3, type=int)
parser.add_argument('--ner_dropout', default=0.2, type=float)
parser.add_argument('--num_tags', default=50, type=int)

# training processing

# logging
parser.add_argument('--log_file', default='log/adversial_transfer.out')
