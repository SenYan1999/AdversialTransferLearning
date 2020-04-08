import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel

class LSTMEncoder(nn.Module):
    def __init__(self, in_size, out_size, num_layers, drop_out, bidirectional=True):
        super(LSTMEncoder, self).__init__()

        self.enc = nn.LSTM(input_size=in_size, hidden_size=out_size // 2, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.)
        self.dropout = drop_out

    def forward(self, x, x_len: torch.Tensor):
        origin_len = x.shape[1]
        lengths, sorted_idx = x_len.sort(0, descending=True)
        x = x[sorted_idx]
        inp = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.enc(inp)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=origin_len)
        _, unsorted_idx = sorted_idx.sort(0)
        out = out[unsorted_idx]
        out = F.dropout(out, self.dropout)
        return out

class BertEncoder(nn.Module):
    def __init__(self, bert_type, in_size, out_size, num_layers, drop_out, bidirection=True):
        super(BertEncoder, self).__init__()

        self.encoder = BertModel.from_pretrained(bert_type)
        self.lstm = LSTMEncoder(in_size, out_size, num_layers, drop_out, bidirection)
    
    def forward(self, input_ids):
        x_len = torch.sum(input_ids != 0, dim=-1)
        x_encoded = self.encoder(input_ids = input_ids)[0]
        x_encoded = self.lstm(x_encoded, x_len)

        return x_encoded

class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(in_size, in_size // 2)
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(in_size // 2, in_size // 4)
        self.activation_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_size // 4, out_size)

    def forward(x):
        out = self.layer_1(x)
        out = self.activation_1(out)
        out = self.layer_2(out)
        out = self.activation_2(out)
        out = self.layer_3(out)
        
        return out
