import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import math
from utils import LogSequenceDataset, LogSeqTokenizer, PositionalEncoding, TimeIntervalEncoding, Time2VecEncoding, summary

class LogSeqTransformer(nn.Module):
    def __init__(self, pt_embed_path : str, num_heads: int, ff_hidden_size: int, num_layers: int, se_dropout: float = 0.1, tf_dropout: float = 0.1, max_sequence_length: int = 2048, seq_enc_method: str = 'time2vec'):
        super(LogSeqTransformer, self).__init__()
        pt_embed_param = torch.load(pt_embed_path)
        vocab_size = pt_embed_param['weight'].shape[0]
        self.d_model = pt_embed_param['weight'].shape[1]
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.embedding.load_state_dict(pt_embed_param)
        self.seq_enc_method = seq_enc_method
        if self.seq_enc_method in ['temporal', 'temporal_only']:
            self.sequential_encoding = TimeIntervalEncoding(self.d_model, dropout = se_dropout, device=next(self.parameters()).device)
        elif self.seq_enc_method == 'positional':
            self.sequential_encoding = PositionalEncoding(self.d_model, dropout = se_dropout, max_len = max_sequence_length, device=next(self.parameters()).device)
        elif self.seq_enc_method in ['time2vec', 'time2vec_only']:
            self.sequential_encoding = Time2VecEncoding(self.d_model, dropout = se_dropout, device=next(self.parameters()).device)
        elif self.seq_enc_method == 'None':
            self.sequential_encoding = None
        else:
            raise NotImplementedError("Sequential encoding not implemented.")
        print('time_encoding:', self.seq_enc_method)
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, num_heads, ff_hidden_size, tf_dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.cls = nn.Linear(self.d_model, vocab_size)

    def to(self, device):
        super().to(device)
        if self.sequential_encoding != None:
            self.sequential_encoding.to(device)
        return self

    def forward(self, logSeq, time_intervals = None):
        embedded = self.embedding(logSeq)
        if self.seq_enc_method in ['temporal', 'time2vec']:
            embedded = self.sequential_encoding(embedded, time_intervals)
        elif self.seq_enc_method in ['temporal_only', 'time2vec_only']:
            embedded = self.sequential_encoding(torch.zeros_like(embedded), time_intervals)
        elif self.seq_enc_method == 'positional':
            embedded = self.sequential_encoding(embedded)
        elif self.seq_enc_method == 'None':
            embedded = embedded
        else:
            raise Exception("Illegal encoding method.")
        transformer_encoder = self.transformer_encoder(embedded)
        # print('TF_encoder output:', transformer_encoder.shape)
        logits = self.cls(transformer_encoder)
        return logits, transformer_encoder


def main():
    log_file_path = './dataset/BGL/parsed_result/BGL_train.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    pt_embed_path = './dataset/BGL/model/embedding_params.pth'
    min_sequence_length = 16
    max_sequence_length = 128  # Define your maximum sequence length
    time_mode = 'first'
    time_unit = 'millisecond'

    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = LogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, min_sequence_length, max_sequence_length, time_mode, time_unit)

    num_heads = 8
    # hidden_size = 512
    hidden_size = 2048  # default by transformer
    num_layers = 1
    se_dropout = 0.1
    tf_dropout = 0.1

    # Sample input sequence
    # input_sequence = torch.randint(0, vocab_size, (10, 5))  # (sequence length, batch size)

    # Creating the transformer model
    model = LogSeqTransformer(pt_embed_path, num_heads, hidden_size, num_layers, se_dropout, tf_dropout, max_sequence_length = max_sequence_length)

    summary(model, [(model.max_sequence_length, ), (model.max_sequence_length, )])

    batch_size = 1
    data_loader = DataLoader(log_dataset, batch_size=batch_size, shuffle=True)

    pt_embedding = torch.load(pt_embed_path)
    vocab_size = pt_embedding['weight'].shape[0]
    embedding_size = pt_embedding['weight'].shape[1]

    for batch in data_loader:
        # print(batch['tid_seq'])
        time_intervals = batch['time_interval']
        embedding_layer = nn.Embedding(vocab_size, embedding_size)
        embedded = embedding_layer(batch['tid_seq'])
        pe_layer = TimeIntervalEncoding(embedding_size)

        pe = pe_layer(embedded, time_intervals)
        # print(pe)
        output = model(batch['tid_seq'], time_intervals = time_intervals)
        print(output[0].shape)
        print(output[1].shape)
        print(output[1][:,0,:].shape)
        # print(batch)
        break


if __name__ == "__main__":
    main()