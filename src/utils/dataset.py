import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
import os
import re
from datetime import datetime
import random
from transformers import BertTokenizer, BertModel
import string

class LogSequenceDataset(Dataset):
    def __init__(self, csv_file, tid_mapping_file, tokenizer, min_sequence_length = 16, max_sequence_length = 2048, time_mode = 'first', time_unit = 'millisecond', anomaly_detection = False, shuffle_log_items = False):
        self.data = pd.read_csv(csv_file)  # Load data from CSV into a DataFrame
        with open(tid_mapping_file, 'r') as file:
            self.tid_to_eid = json.load(file)
            self.agg_token_index = self.tid_to_eid['<AGG>']
            self.pad_token_index = self.tid_to_eid['<PAD>']
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.time_unit = time_unit
        self.tokenizer = tokenizer
        self.time_mode = time_mode
        self.seq_interval = max_sequence_length//2  # TODO: Experimental 
        self.anomaly_detection = anomaly_detection    # TODO: Experimental
        self.fix_seq_len = max_sequence_length  # (lines) TODO: Experimental
        self.shuffle_log_items = shuffle_log_items
        
    def __len__(self):
        if self.anomaly_detection:
            # fix length TODO: experimental
            return math.ceil(len(self.data)/(self.fix_seq_len-2))
        else:
            return (len(self.data)-self.min_sequence_length)//self.seq_interval
    
    def __getitem__(self, idx):
        if self.anomaly_detection:
            start_idx = idx * (self.fix_seq_len-2)
            end_idx = (idx+1) * (self.fix_seq_len-2) # reserve spots for agg and eos token
            sequence = self.data['EventId'].iloc[start_idx:end_idx]  # Extract a slice of data as a sequence
            labels = self.data['Label'].iloc[start_idx:end_idx]
            # times = self.data['Time'].iloc[start_idx:end_idx]
            timestamps = self.data['Timestamp'].iloc[start_idx:end_idx]
            intervals = self.timestamp_to_interval(timestamps, time_mode = self.time_mode)
            # intervals = self.time_to_interval(times, time_mode = self.time_mode, unit = self.time_unit)
            merged_lb = 1 if sum([self.label_to_binary(label) for label in labels])>=1 else 0

            if self.shuffle_log_items:
                sequence = sequence.sample(frac=1).reset_index(drop=True)

            sequence = self.tokenizer.tokenize(sequence, pad = True)
            intervals = self.tokenizer.padding_time_intervals(intervals)
            return {'tid_seq': sequence, 'time_interval':intervals, 'label':merged_lb}
        else:
            # Extract a sequence of varying length starting from the current index
            sequence, intervals, label = self.extract_sequence(idx*self.seq_interval)  # decrease the overlaps
            token_seq = self.tokenizer.tokenize(sequence, pad = True)
            time = self.tokenizer.padding_time_intervals(intervals)
            # input_tensor = self.transform_to_input(sequence)
            # return input_tensor
            return {'tid_seq': token_seq, 'time_interval':time, 'label':label}
    
    def label_to_binary(self, label):
        if label == '-':
            return 0
        else:
            return 1
        
    def timestamp_to_interval(self, timestamps, time_mode = 'first'):
        # time_mode: 'first' : Calculate the intervals between each timestamp and the first timestamp
        # 'gap': between consecutive timestamps
        if time_mode=='first':
            first_timestamp = timestamps.iloc[0]  # Get the first timestamp
            # intervals = (timestamps - first_timestamp).dt.total_seconds().fillna(0)
            intervals = (timestamps - first_timestamp)
        elif time_mode=='gap':
            intervals = timestamps.diff().dt.total_seconds().fillna(0)
        else:
            raise Exception("Unknown time mode.")
        intervals = intervals.astype(int)
        return intervals
    

    def time_to_interval(self, times, time_mode = 'first', unit='millisecond'):
        # time_mode: 'first' : Calculate the intervals between each timestamp and the first timestamp
        # 'gap': between consecutive timestamps

        times = pd.to_datetime(times, format='%Y-%m-%d-%H.%M.%S.%f')
        units_conversion = {'second': 1, 'millisecond': 1000, 'microsecond': 1_000_000, 'nanosecond': 1_000_000_000}
        if time_mode=='first':
            first_time = times.iloc[0]  # Get the first timestamp
            intervals = (times - first_time).dt.total_seconds().fillna(0) * units_conversion.get(unit, 1)
        elif time_mode=='gap':
            intervals = times.diff().dt.total_seconds().fillna(0) * units_conversion.get(unit, 1)
        else:
            raise Exception("Unknown time mode.")
        intervals = intervals.astype(int)
        return intervals

    def extract_sequence(self, idx):
        # Extract a sequence of varying length starting from the current index
        start_idx = idx
        end_idx = min(idx + torch.randint(self.min_sequence_length , self.max_sequence_length - 1 , (1,)).item(), len(self.data))
        
        sequence = self.data['EventId'].iloc[start_idx:end_idx]  # Extract a slice of data as a sequence
        labels = self.data['Label'].iloc[start_idx:end_idx]
        timestamps = self.data['Timestamp'].iloc[start_idx:end_idx]
        intervals = self.timestamp_to_interval(timestamps, time_mode = self.time_mode)
        # times = self.data['Time'].iloc[start_idx:end_idx]
        # intervals = self.timestamp_to_interval(times, time_mode = self.time_mode, unit = self.time_unit)
        merged_lb = 1 if sum([self.label_to_binary(label) for label in labels])>=1 else 0
        return sequence, intervals, merged_lb

class HDFS_Dataset(Dataset):
    def __init__(self, csv_file, tid_mapping_file, tokenizer, max_sequence_length = 2048, time_mode = 'first', shuffle_log_items = False):
        self.data = pd.read_csv(csv_file)  # Load data from CSV into a DataFrame
        with open(tid_mapping_file, 'r') as file:
            self.tid_to_eid = json.load(file)
            self.agg_token_index = self.tid_to_eid['<AGG>']
            self.pad_token_index = self.tid_to_eid['<PAD>']
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.time_mode = time_mode
        self.fix_seq_len = max_sequence_length  # (lines) TODO: Experimental
        self.shuffle_log_items = shuffle_log_items
        
    def __len__(self):
        return len(self.data)
        pass
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = pd.Series(item['Events'].split(','))
        label = item['Label']
        dates = item['Dates'].split(',')
        times = item['Times'].split(',')
        timestamps = pd.to_datetime([date + time for date, time in zip(dates, times)], format='%y%m%d%H%M%S')
        intervals = self.timestamp_to_interval(timestamps, time_mode = self.time_mode)
        if self.shuffle_log_items:
            sequence = sequence.sample(frac=1).reset_index(drop=True)
        sequence, att_mask = self.tokenizer.tokenize(sequence, pad = True)
        intervals = self.tokenizer.padding_time_intervals(intervals)
        return {'tid_seq': sequence, 'time_interval':intervals, 'att_mask': att_mask.to(torch.bool), 'label':label}

    def timestamp_to_interval(self, timestamps, time_mode = 'gap'):
        # time_mode: 'first' : Calculate the intervals between each timestamp and the first timestamp
        # 'gap': between consecutive timestamps
        if time_mode=='first':
            first_timestamp = timestamps[0] # Get the first timestamp
            intervals = (timestamps - first_timestamp).total_seconds().fillna(0)
        elif time_mode=='gap':
            intervals = timestamps.to_series().diff().dt.total_seconds().fillna(0)
        else:
            raise Exception("Unknown time mode.")
        intervals = intervals.astype(int)
        return intervals

class VariantLogSequenceDataset(Dataset):
    def __init__(self, csv_file, tid_mapping_file, tokenizer, min_sequence_length = 16, max_sequence_length = 1024, step_size = 512, time_mode = 'first', shuffle_log_items = False):
        self.data = pd.read_csv(csv_file)  # Load data from CSV into a DataFrame
        with open(tid_mapping_file, 'r') as file:
            self.tid_to_eid = json.load(file)
            self.agg_token_index = self.tid_to_eid['<AGG>']
            self.pad_token_index = self.tid_to_eid['<PAD>']
        self.min_sequence_length = min_sequence_length - 2
        self.max_sequence_length = max_sequence_length - 2
        self.tokenizer = tokenizer
        self.time_mode = time_mode
        self.step_size = step_size
        self.shuffle_log_items = shuffle_log_items
        

    def __len__(self):
        return math.ceil((len(self.data)-self.max_sequence_length)/self.step_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = min(start_idx + random.randint(self.min_sequence_length, self.max_sequence_length), len(self.data)-1)
        # end_idx = min(start_idx + self.max_sequence_length, len(self.data)-1) # the maximum sequence.
        # print('actual length:', end_idx-start_idx)
        sequence = self.data['EventId'].iloc[start_idx:end_idx]  # Extract a slice of data as a sequence
        labels = self.data['Label'].iloc[start_idx:end_idx]
        timestamps = self.data['Timestamp'].iloc[start_idx:end_idx]
        intervals = self.timestamp_to_interval(timestamps, time_mode = self.time_mode)
        merged_lb = 1 if sum([self.label_to_binary(label) for label in labels])>=1 else 0

        if self.shuffle_log_items:
            sequence = sequence.sample(frac=1).reset_index(drop=True)
            # don't need to shuffle the timestamps, as timestamps here make no sense.
            # shuffled_indices = sequence.index
            # timestamps = timestamps.reindex(shuffled_indices).reset_index(drop=True)

        sequence, att_mask = self.tokenizer.tokenize(sequence, pad = True)
        intervals = self.tokenizer.padding_time_intervals(intervals)
        return {'tid_seq': sequence, 'time_interval':intervals, 'att_mask': att_mask.to(torch.bool), 'label':merged_lb}
    

    def label_to_binary(self, label):
        if label == '-':
            return 0
        else:
            return 1
        
    def timestamp_to_interval(self, timestamps, time_mode = 'first'):
        # time_mode: 'first' : Calculate the intervals between each timestamp and the first timestamp
        # 'gap': between consecutive timestamps
        if time_mode=='first':
            first_timestamp = timestamps.iloc[0]  # Get the first timestamp
            # intervals = (timestamps - first_timestamp).dt.total_seconds().fillna(0)
            intervals = (timestamps - first_timestamp)
        elif time_mode=='gap':
            intervals = timestamps.diff().dt.total_seconds().fillna(0)
        else:
            raise Exception("Unknown time mode.")
        intervals = intervals.astype(int)
        return intervals

class NegativeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.filtered_indices = [i for i, item in enumerate(dataset) if item['label'] <= 0]
        self.tid_to_eid = dataset.tid_to_eid

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        return self.dataset[self.filtered_indices[index]]

class RawLogDataset(Dataset):
    def __init__(self, csv_file, HF_path, seq_len=2048, step_size=64, time_mode='first', time_unit='millisecond', cache_dir='./cache'):
        self.data = pd.read_csv(csv_file)
        self.time_unit = time_unit
        self.time_mode = time_mode
        self.seq_len = seq_len
        self.step_size = step_size
        
        # Setup cache
        os.makedirs(cache_dir, exist_ok=True)
        # Extract dataset name from second-to-last directory in path
        path_parts = os.path.normpath(csv_file).split(os.sep)
        dataset_name = path_parts[-2]  # Gets second-to-last component
        cache_subdir = os.path.join(cache_dir, dataset_name)
        os.makedirs(cache_subdir, exist_ok=True)
        self.cache_file = os.path.join(cache_subdir, f"{os.path.basename(csv_file)}.embeddings.pt")
        
        # Initialize BERT only if needed
        if not os.path.exists(self.cache_file):
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(HF_path,'bert-base-uncased'))
            self.bert_model = BertModel.from_pretrained(os.path.join(HF_path,'bert-base-uncased'))
            self.bert_model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model = self.bert_model.to(self.device)
            
            # Preprocess and cache all embeddings
            self.cached_embeddings = self.preprocess_and_cache_embeddings()
        else:
            print(f"Loading cached embeddings from {self.cache_file}")
            self.cached_embeddings = torch.load(self.cache_file)

    def __len__(self):
        # Calculate number of sequences based on data length, sequence length, and step size
        return (len(self.data) - self.seq_len) // self.step_size + 1

    def preprocess_and_cache_embeddings(self):
        print("Preprocessing and caching embeddings... This may take a while.")
        raw_logs = self.data['Content'].values.tolist()
        processed_logs = self.process_special_tokens(raw_logs)
        
        embeddings = []
        batch_size = 32  # Process in batches to speed up
        
        with torch.no_grad():
            for i in range(0, len(processed_logs), batch_size):
                batch = processed_logs[i:i + batch_size]
                batch_embeddings = self.batch_bert_embedding(batch)
                embeddings.extend(batch_embeddings)
                
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(processed_logs)} logs")
        
        embeddings_tensor = torch.stack(embeddings)
        torch.save(embeddings_tensor, self.cache_file)
        print(f"Cached embeddings saved to {self.cache_file}")
        return embeddings_tensor

    def batch_bert_embedding(self, log_msgs):
        # Tokenize all messages in the batch
        inputs = self.tokenizer(log_msgs, 
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=512)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        outputs = self.bert_model(**inputs)
        
        # Extract [CLS] token embeddings for each sequence
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        return [emb for emb in embeddings]

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.seq_len
        
        # Get pre-computed embeddings
        embedded_logs = self.cached_embeddings[start_idx:end_idx]
        
        labels = self.data['Label'].iloc[start_idx:end_idx]
        timestamps = self.data['Timestamp'].iloc[start_idx:end_idx]
        intervals = self.timestamp_to_interval(timestamps, time_mode=self.time_mode).values
        merged_lb = 1 if sum([self.label_to_binary(label) for label in labels])>=1 else 0
        
        return {'log_seq': embedded_logs, 'time_interval':intervals, 'label':merged_lb}

    def process_special_tokens(self, log_msgs):
        processed_logs = []
        for log in log_msgs:
            if isinstance(log, float) and math.isnan(log):
                processed_logs.append('')
                continue
            # log = log.lower()
            # The following preprocessing code is excerpted from https://github.com/LogIntelligence/NeuralLog/blob/main/neurallog/data_loader.py 
            log = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', log)
            log = " ".join([word.lower() if word.isupper() else word for word in log.strip().split()])
            log = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', log))
            log = " ".join([word for word in log.split() if not bool(re.search(r'\d', word))])
            trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
            content = log.translate(trantab)
            log = " ".join([word.lower().strip() for word in content.strip().split()])
            processed_logs.append(log)
        return processed_logs
    
    def label_to_binary(self, label):
        if label == '-':
            return 0
        else:
            return 1
        
    def timestamp_to_interval(self, timestamps, time_mode = 'first'):
        # time_mode: 'first' : Calculate the intervals between each timestamp and the first timestamp
        # 'gap': between consecutive timestamps

        if time_mode=='first':
            first_timestamp = timestamps.iloc[0]  # Get the first timestamp
            # intervals = (timestamps - first_timestamp).dt.total_seconds().fillna(0)
            intervals = (timestamps - first_timestamp)
        elif time_mode=='gap':
            intervals = timestamps.diff().dt.total_seconds().fillna(0)
        else:
            raise Exception("Unknown time mode.")
        intervals = intervals.astype(int)
        return intervals

    def time_to_interval(self, times, time_mode = 'first', unit='millisecond'):
        # time_mode: 'first' : Calculate the intervals between each timestamp and the first timestamp
        # 'gap': between consecutive timestamps

        times = pd.to_datetime(times, format='%Y-%m-%d-%H.%M.%S.%f')
        units_conversion = {'second': 1, 'millisecond': 1000, 'microsecond': 1_000_000, 'nanosecond': 1_000_000_000}
        if time_mode=='first':
            first_time = times.iloc[0]  # Get the first timestamp
            intervals = (times - first_time).dt.total_seconds().fillna(0) * units_conversion.get(unit, 1)
        elif time_mode=='gap':
            intervals = times.diff().dt.total_seconds().fillna(0) * units_conversion.get(unit, 1)
        else:
            raise Exception("Unknown time mode.")
        intervals = intervals.astype(int)
        return intervals
    

    def collate_fn(self, batch):
        # Use this custom collate function to handle lists of strings, fix the problem with default_collate.
        assert all('log_seq' in x for x in batch)
        
        # Stack the log sequences into a single tensor (batch_size x seq_len x embedding_dim)
        log_seqs = torch.stack([x['log_seq'] for x in batch])
        
        # Use default_collate for the other elements
        _, time_interval, label = torch.utils.data.default_collate(batch).values()
        
        return {
            'log_seq': log_seqs,
            'time_interval': time_interval,
            'label': label
        }
    
class HDFSEmbeddedDataset(Dataset):
    def __init__(self, csv_file, tid_mapping_file, embedding_file, seq_len=2048, shuffle_log_items=False):
        self.data = pd.read_csv(csv_file)
        with open(tid_mapping_file, 'r') as file:
            self.tid_to_eid = json.load(file)
            self.pad_token_index = self.tid_to_eid['<PAD>']
        
        # Load pre-trained embeddings
        self.embeddings = torch.load(embedding_file)['weight']
        self.embedding_dim = self.embeddings.shape[1]
        
        self.seq_len = seq_len
        self.shuffle_log_items = shuffle_log_items
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = pd.Series(item['Events'].split(','))
        label = item['Label']

        if self.shuffle_log_items:
            sequence = sequence.sample(frac=1).reset_index(drop=True)

        # Convert log templates to token IDs
        token_ids = torch.tensor([self.tid_to_eid[log] for log in sequence])
        
        # Pad sequence if necessary
        if len(token_ids) < self.seq_len:
            padding = torch.full((self.seq_len - len(token_ids),), self.pad_token_index)
            token_ids = torch.cat([token_ids, padding])
        else:
            token_ids = token_ids[:self.seq_len]
        
        # Convert token IDs to embeddings
        embedded_sequence = self.embeddings[token_ids]

        return {
            'log_seq': embedded_sequence,
            'label': label
        }


    def collate_fn(self, batch):
        log_seqs = torch.stack([x['log_seq'] for x in batch])
        labels = torch.tensor([x['label'] for x in batch])
        
        return {
            'log_seq': log_seqs,
            'label': labels
        }


def main():
    log_file_path = './dataset/HDFS/test.csv'
    tid_mapping_file = './dataset/HDFS/embeddings/template_id_mapping.json'
    embedding_file = './dataset/HDFS/embeddings/bert_768_embedding_params.pth'
    log_dataset = HDFSEmbeddedDataset(log_file_path, tid_mapping_file, embedding_file, seq_len = 300)

    data_loader = DataLoader(log_dataset, batch_size=2, shuffle=False, collate_fn=log_dataset.collate_fn)
    for index, batch in enumerate(data_loader):
        print(len(batch['log_seq']))
        print(len(batch['log_seq'][0]))
        print(batch['log_seq'][0])



if __name__ == "__main__":
    main()

