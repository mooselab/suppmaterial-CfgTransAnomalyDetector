import json
import torch
from torch.utils.data import DataLoader

class LogSeqTokenizer:
    def __init__(self, tid_mapping_file, max_sequence_length):
        with open(tid_mapping_file, 'r') as file:
            self.vocab = json.load(file)
            self.agg_token_index = self.vocab['<AGG>']
            self.eos_token_index = self.vocab['<EOS>']
            self.pad_token_index = self.vocab['<PAD>']
        self.max_sequence_length = max_sequence_length
        self.pad_dummy_time = -1

    def tokenize(self, sequence, pad = False):
        # Tokenize or encode the sequence (replace this step with your preprocessing logic)
        tkid_sequence = sequence.map(self.vocab).tolist()
        tkid_sequence = [self.agg_token_index] + tkid_sequence + [self.eos_token_index] # Add the aggregation and eos tokens
        if pad:
            padded_sequence = self.padding(tkid_sequence)
            ret_seq = padded_sequence
        else:
            ret_seq = tkid_sequence
        ret_att_mask = (ret_seq==2)
        return ret_seq, ret_att_mask

    def padding(self, sequence):
        # Ensure uniform sequence length (padding or truncating) and convert numerical sequence to PyTorch tensor
        padded_sequence = torch.nn.functional.pad(torch.tensor(sequence), (0, self.max_sequence_length - len(sequence)), value=self.pad_token_index)
        return padded_sequence
    
    def padding_time_intervals(self, intervals):
        intervals = [self.pad_dummy_time] + intervals.tolist() + [self.pad_dummy_time] # aligning the time serie to the template sequence
        padded_intervals = torch.nn.functional.pad(torch.tensor(intervals), (0, self.max_sequence_length - len(intervals)), value=self.pad_dummy_time)
        return padded_intervals

def main():
    from dataset import LogSequenceDataset
    log_file_path = './dataset/BGL/parsed_result/BGL_train_0.8.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    max_sequence_length = 128  # Define your maximum sequence length
    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    log_dataset = LogSequenceDataset(log_file_path, tid_mapping_path, tokenizer, min_sequence_length=16, max_sequence_length=max_sequence_length)

    # Create a DataLoader
    batch_size = 2
    data_loader = DataLoader(log_dataset, batch_size=batch_size, shuffle=True)

    for batch in data_loader:
        print(batch)
        break

if __name__ == "__main__":
    main()