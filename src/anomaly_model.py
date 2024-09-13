import torch
import json
# import re
import math
from torch import nn, Tensor
from model import LogSeqTransformer
from torch.utils.data import DataLoader
from utils import LogSequenceDataset, LogSeqTokenizer, RawLogDataset, PositionalEncoding, TimeIntervalEncoding, Time2VecEncoding, summary
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer


# Define the anomaly detection model based on top of the transformer
class AnomalyModel(LogSeqTransformer):
    def __init__(self, pt_embed_path : str, num_heads: int, ff_hidden_size: int, num_layers: int, se_dropout: float = 0.1, tf_dropout: float = 0.1, max_sequence_length: int = 2048, seq_enc_method: str = 'time2vec', tid_mapping_file = None):
        super(AnomalyModel, self).__init__(pt_embed_path, num_heads, ff_hidden_size, num_layers, se_dropout, tf_dropout, max_sequence_length, seq_enc_method)
        # Add an extra layer on top of the base model's first token '<AGG>' output
        if tid_mapping_file != None:
            with open(tid_mapping_file, 'r') as file:
                self.tid_to_eid = json.load(file)
                self.agg_token_embedding = self.embedding(torch.tensor(self.tid_to_eid['<AGG>']).unsqueeze(0))
                self.eos_token_embedding = self.embedding(torch.tensor(self.tid_to_eid['<EOS>']).unsqueeze(0))
                self.pad_token_embedding = self.embedding(torch.tensor(self.tid_to_eid['<PAD>']).unsqueeze(0))
        self.embed_model = SentenceTransformer('./all-MiniLM-L6-v2')
        self.pad_dummy_time = -1
        self.classifier = nn.Linear(self.d_model, 2)  # Experimental: First Token - Binary output

    def to(self, device):
        super().to(device)
        self.device = device
        self.embed_model.to(device)
        return self
    
    def forward(self, logSeq, time_intervals = None, parsed = True):
        if parsed == False:
            idx = torch.cuda.current_device()
            count = torch.cuda.device_count()
            # fix the splitting problem of DataParallel
            segment_length = math.ceil(len(logSeq)/ count)
            start_idx = idx * segment_length
            end_idx = (idx + 1) * segment_length if idx < count - 1 else len(logSeq)
            logSeq = logSeq[start_idx : end_idx]
            # print(len(logSeq))
            # print(time_intervals.shape)
            # print(start_idx,':',end_idx)
            padded_seq, padded_time = self.to_embedding(logSeq, time_intervals)
            # print('Padded_seq:', padded_seq.shape)
            # print('padded_time:', padded_time.shape)
            if self.seq_enc_method in ['temporal', 'time2vec']:
                padded_seq = self.sequential_encoding(padded_seq, padded_time)
            elif self.seq_enc_method in ['temporal_only', 'time2vec_only']:
                padded_seq = self.sequential_encoding(torch.zeros_like(padded_seq), padded_time)
            elif self.seq_enc_method == 'positional':
                padded_seq = self.sequential_encoding(padded_seq)
            elif self.seq_enc_method == 'None':
                padded_seq = padded_seq
            else:
                raise Exception("Illegal encoding method.")
            transformer_encoder = self.transformer_encoder(padded_seq)
            binary_logit_output = self.classifier(transformer_encoder[:,0,:])    # output: transformer_encoder <AGG>
            # print('transformer_encoder:', transformer_encoder.shape)
        else:
            # Call the forward of the base model
            agg_token_output = super(AnomalyModel, self).forward(logSeq, time_intervals)[1][:,0,:]   # output: transformer_encoder <AGG>
            binary_logit_output = self.classifier(agg_token_output)
        return binary_logit_output

    def padding_time_intervals(self, intervals):
        # print(self.device)
        padding_tensor = torch.full((intervals.shape[0], 1), self.pad_dummy_time).to(intervals.device)
        # Concatenating the padding tensor with the original tensor along the second dimension
        intervals = torch.cat((padding_tensor, intervals), dim=1)  # add a -1 corresponding to <AGG>
        intervals = torch.cat((intervals, padding_tensor), dim=1)  # add a -1 corresponding to <EOS>
        padded_intervals = torch.nn.functional.pad(intervals, (0, self.max_sequence_length - intervals.shape[1]), value=self.pad_dummy_time)
        return padded_intervals

    def pad_embeedings(self, output_emb):
        output_emb = torch.cat((self.agg_token_embedding.to(self.device), output_emb), dim=0)  # add the embedding of <AGG>
        output_emb = torch.cat((output_emb, self.eos_token_embedding.to(self.device)), dim=0)  # add the embedding of <EOS>
        extra_length = self.max_sequence_length - output_emb.shape[0]
        if extra_length >0:
            repeated_tensor = self.pad_token_embedding.repeat(extra_length,1).to(self.device)
            output_emb = torch.cat((output_emb, repeated_tensor), dim=0)  # add the embedding of <PAD>
        return output_emb

    def to_embedding(self, logSeq_batch, time_interval):
        # encode and pad the log/time sequence
        padded_time = self.padding_time_intervals(time_interval)
        output_list = []
        for logSeq in logSeq_batch:
            output_emb = torch.tensor(self.embed_model.encode(logSeq)).to(time_interval.device)
            output_emb = self.pad_embeedings(output_emb).unsqueeze_(0)
            output_list.append(output_emb)
        padded_logseq = torch.cat(output_list, dim=0)
        return padded_logseq, padded_time


def main():
    log_file_path = './dataset/BGL/parsed_result/BGL_train.csv'
    raw_log_file_path = './dataset/BGL/BGL_test.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    # pt_embed_path = './dataset/BGL/model/embedding_params.pth'
    # pt_embed_path = './dataset/BGL/model/768_embedding_params.pth'
    pt_embed_path = './dataset/BGL/model/sentence_embedding_params.pth'
    min_sequence_length = 16
    max_sequence_length = 128  # Define your maximum sequence length
    time_mode = 'first'
    time_unit = 'millisecond'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = LogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, min_sequence_length, max_sequence_length, time_mode, time_unit)
    raw_log_dataset = RawLogDataset(raw_log_file_path, max_sequence_length = 128)

    num_heads = 8
    hidden_size = 512
    num_layers = 1
    se_dropout = 0.1
    tf_dropout = 0.1

    # Sample input sequence
    # input_sequence = torch.randint(0, vocab_size, (10, 5))  # (sequence length, batch size)

    # Creating the transformer model
    model = AnomalyModel(pt_embed_path, num_heads, hidden_size, num_layers, se_dropout, tf_dropout, max_sequence_length = max_sequence_length, tid_mapping_file = tid_mapping_path)
    # model.to(device)
    # summary(model, [(model.max_sequence_length, ), (model.max_sequence_length, )])

    batch_size = 128
    # data_loader = DataLoader(log_dataset, batch_size=batch_size, shuffle=True)

    data_loader = DataLoader(raw_log_dataset, batch_size=batch_size, shuffle=False, collate_fn=raw_log_dataset.collate_fn)

    # pt_embedding = torch.load(pt_embed_path)
    # vocab_size = pt_embedding['weight'].shape[0]
    # embedding_size = pt_embedding['weight'].shape[1]


    for index, batch in enumerate(data_loader):
    #     # print(batch['tid_seq'])
    #     # print(len(batch))
    #     # print(type(batch))
        time_intervals = batch['time_interval']
    #     # embedding_layer = nn.Embedding(vocab_size, embedding_size)
    #     # embedded = embedding_layer(batch['tid_seq'])
    #     # pe_layer = TimeIntervalEncoding(embedding_size)
    #     # pe = pe_layer(embedded, time_intervals)
    #     # print(pe)

        output = model(batch['log_seq'], time_intervals = time_intervals, parsed = False)



        print(output)
    #     # print(batch)
        # break


if __name__ == "__main__":
    main()