import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
import re
from datetime import datetime
import random


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
    

class RawLogDataset(Dataset):
    def __init__(self, csv_file, min_sequence_length = 16, max_sequence_length = 2048, time_mode = 'first', time_unit = 'millisecond'):
        self.data = pd.read_csv(csv_file)  # Load data from CSV into a DataFrame
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.time_unit = time_unit
        self.time_mode = time_mode
        self.fix_seq_len = max_sequence_length  # (lines) TODO: Experimental

    def __len__(self):
        return math.ceil(len(self.data)/(self.fix_seq_len-2))
    
    def __getitem__(self, idx):
        start_idx = idx * (self.fix_seq_len-2)
        end_idx = (idx+1) * (self.fix_seq_len-2) # reserve spots for agg and eos token
        raw_log_sequence = self.data['Log'].iloc[start_idx:end_idx].values.tolist()  # Extract a slice of data as a sequence
        # raw_log_sequence = [log.lower() for log in raw_log_sequence]  # buggy, can not handle 'nan' case. some log messages are missing.
        raw_log_sequence = ['' if isinstance(log, float) and math.isnan(log) else log.lower() for log in raw_log_sequence]
        # raw_log_sequence = [re.sub(r'[^a-zA-Z\s,*]', ' ', log) for log in raw_log_sequence]
        raw_log_sequence = [re.sub(r'[^a-zA-Z\s,]', ' ', log) for log in raw_log_sequence]
        labels = self.data['Label'].iloc[start_idx:end_idx]
        # times = self.data['Time'].iloc[start_idx:end_idx]
        timestamps = self.data['Timestamp'].iloc[start_idx:end_idx]
        # intervals = self.timestamp_to_interval(times, time_mode = self.time_mode, unit = self.time_unit).values
        intervals = self.timestamp_to_interval(timestamps, time_mode = self.time_mode).values
        merged_lb = 1 if sum([self.label_to_binary(label) for label in labels])>=1 else 0
        if len(raw_log_sequence) < self.fix_seq_len-2:
            len_padding = self.fix_seq_len-2-len(raw_log_sequence)
            raw_log_sequence.extend(['']*len_padding)
            appended_interval = np.full(len_padding, -1)
            intervals = np.append(intervals, appended_interval)
        # print(len(raw_log_sequence), intervals.shape, merged_lb)
        # if len(raw_log_sequence) <126:
        #     print(raw_log_sequence, intervals)
        return {'log_seq': raw_log_sequence, 'time_interval':intervals, 'label':merged_lb}

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
        _, time_interval, label = torch.utils.data.default_collate(batch).values()
        return {
            'log_seq': [x['log_seq'] for x in batch],
            'time_interval': time_interval,
            'label': label
        }

def main():
    log_file_path = './dataset/BGL/BGL_test.csv'
    # log_dataset = RawLogDataset(log_file_path, max_sequence_length = 128)
    log_dataset = RawLogDataset(log_file_path, max_sequence_length = 128)

    data_loader = DataLoader(log_dataset, batch_size=1, shuffle=False, collate_fn=log_dataset.collate_fn)
    for index, batch in enumerate(data_loader):
        # print(len(batch['log_seq']))
        # print(len(batch['log_seq'][0]))
        print(batch['time_interval'])
        break
        # print(batch['label'].shape)
        # print(batch['log_seq'])

        # print(batch)
        print(index)
        # break

    # print(log_dataset.__getitem__(0))
    # for batch in data_loader:
    #     log_seq, t_interval ,lb = batch.values()
    #     print(len(log_seq))
    #     break

def main_1():
    from tokenizer import LogSeqTokenizer
    log_file_path = './dataset/BGL/parsed_result/BGL_train.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    max_sequence_length = 128  # Define your maximum sequence length
    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = LogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, max_sequence_length=max_sequence_length, time_mode = 'first', anomaly_detection= True)
    # log_dataset = LogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, max_sequence_length=max_sequence_length, time_mode = 'gap')

    # print(len(log_dataset))

    for sequence_index in range(6556,6558):
        sample_sequence, t_interval ,lb = log_dataset[sequence_index].values()
        print(sample_sequence)
        print(len(sample_sequence))
        print(t_interval)
        print(len(t_interval))
        print(lb)
        # Now `sample_sequence` contains a variable-length sequence from your log data
        # You might need to further preprocess or convert it into tensor format depending on your data
        # print(f"Sequence {sequence_index}:")
        # print(sample_sequence)
        # print("=" * 20)

class HDFS_Dataset(Dataset):
    def __init__(self, csv_file, tid_mapping_file, tokenizer, min_sequence_length = 16, max_sequence_length = 2048, time_mode = 'first', shuffle_log_items = False):
        self.data = pd.read_csv(csv_file)  # Load data from CSV into a DataFrame
        with open(tid_mapping_file, 'r') as file:
            self.tid_to_eid = json.load(file)
            self.agg_token_index = self.tid_to_eid['<AGG>']
            self.pad_token_index = self.tid_to_eid['<PAD>']
        self.min_sequence_length = min_sequence_length
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
        sequence = self.tokenizer.tokenize(sequence, pad = True)
        intervals = self.tokenizer.padding_time_intervals(intervals)
        return {'tid_seq': sequence, 'time_interval':intervals, 'label':label}

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


def main_hdfs():
    from tokenizer import LogSeqTokenizer
    log_file_path = './dataset/HDFS/train.csv'
    tid_mapping_path = './dataset/HDFS/template_id_mapping.json'
    max_sequence_length = 128  # Define your maximum sequence length
    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = HDFS_Dataset(log_file_path, tid_mapping_path, tokenizer, max_sequence_length=max_sequence_length, time_mode = 'gap')

    print(log_dataset[0])


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

        sequence = self.tokenizer.tokenize(sequence, pad = True)
        intervals = self.tokenizer.padding_time_intervals(intervals)
        return {'tid_seq': sequence, 'time_interval':intervals, 'label':merged_lb}

    
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

def main_winfree():
    from tokenizer import LogSeqTokenizer
    log_file_path = './dataset/BGL/parsed_result/BGL_train.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    min_sequence_length = 10
    max_sequence_length = 1024  # Define your maximum sequence length
    step_size = (min_sequence_length + max_sequence_length)//2
    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = VariantLogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, min_sequence_length= min_sequence_length, max_sequence_length=max_sequence_length, step_size = step_size, time_mode = 'first')
    neg_dataset = NegativeDataset(log_dataset)

    print('neg_dataset len:', len(neg_dataset))

    print('dataset len:', len(log_dataset))

    # for i, sample in enumerate(log_dataset):
    #     if i <=10:
    #         # print(len(sample['tid_seq']))
    #         print(sample['tid_seq'])
    #     else:
    #         break


if __name__ == "__main__":
    main_winfree()
