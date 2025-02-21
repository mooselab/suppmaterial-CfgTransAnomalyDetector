import torch
from torch import nn, Tensor
import math
from typing import Callable

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device = torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class TimeIntervalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, device = torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def time_interval_encoding(self, embedded: Tensor, time_intervals: Tensor) -> Tensor:
        # modified div_term, no need to have interval here
        # Experiment done: expand the period to 100000*2pi, no significant difference.
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)).to(self.device)
        interval_embedding = torch.zeros_like(embedded).to(self.device)
        time_intervals = time_intervals.unsqueeze(dim=2).to(self.device)
        interval_embedding[:, :, 0::2] = torch.sin(time_intervals*div_term)
        interval_embedding[:, :, 1::2] = torch.cos(time_intervals*div_term)
        return interval_embedding
    

    def forward(self, x: Tensor, time_intervals: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # print(x.shape)
        # print(time_intervals.shape)
        encoded_time = self.time_interval_encoding(x, time_intervals)
        x = x + encoded_time
        return self.dropout(x)
    
    
class Time2VecEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, act_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sin, device = torch.device('cpu')):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model  # embedding size
        self.device = device
        self.w0 = nn.parameter.Parameter(torch.randn(1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, d_model-1))
        self.b = nn.parameter.Parameter(torch.randn(d_model-1))
        self.f = act_fn

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def time_interval_encoding(self, time_intervals: Tensor) -> Tensor:
        # print("embedded:", embedded.shape)
        # print("time_intervals:", time_intervals.shape)
        # print("self.w:", self.w.shape)
        # print("self.b:", self.b.shape)
        # print("self.w0:", self.w0.shape)
        # print("self.b0:", self.b0.shape)
        v_0 = (time_intervals.to(torch.float32)* self.w0 + self.b0).unsqueeze(-1)
        v_rest = self.f(time_intervals.to(torch.float32).unsqueeze(-1)* self.w + self.b)
        ret = torch.cat([v_0, v_rest], -1)
        return ret

    def forward(self, x: Tensor, time_intervals: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        t2v_encoding = self.time_interval_encoding(time_intervals)
        x = x + t2v_encoding
        # x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def main():
    log_file_path = './dataset/BGL/parsed_result/BGL_train.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    pt_embed_path = './dataset/BGL/model/bert/sentence_embedding_params.pth'
    min_sequence_length = 16
    max_sequence_length = 128  # Define your maximum sequence length
    time_mode = 'first'
    time_unit = 'millisecond'
    from dataset import LogSequenceDataset
    from tokenizer import LogSeqTokenizer
    from torch.utils.data import DataLoader
    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = LogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, min_sequence_length, max_sequence_length, time_mode, time_unit, anomaly_detection = True)

    batch_size = 2
    data_loader = DataLoader(log_dataset, batch_size=batch_size, shuffle=False)

    pt_embedding = torch.load(pt_embed_path)
    vocab_size = pt_embedding['weight'].shape[0]
    embedding_size = pt_embedding['weight'].shape[1]
    max_sequence_length = 128

    for batch in data_loader:
        # print(batch['tid_seq'])
        time_intervals = batch['time_interval']
        embedding_layer = nn.Embedding(vocab_size, embedding_size)
        embedding_layer.load_state_dict(torch.load(pt_embed_path))
        embedded = embedding_layer(batch['tid_seq'])
        
        pe_layer = TimeIntervalEncoding(embedding_size)
        pe = pe_layer(embedded, time_intervals)

        break
if __name__ == "__main__":
    main()