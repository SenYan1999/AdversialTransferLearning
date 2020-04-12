import argparse

# define args parser
parser = argparse.ArgumentParser()

# program mode
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--train', action='store_true')

# data path
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--train_data', default='')

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
parser.add_argument('--ner_num_tags', default=50, type=int)

# training processing
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)

# logging
parser.add_argument('--log_file', default='log/adversial_transfer.out')

args= parser.parse_args()
