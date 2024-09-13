import torch
from torch.utils.data import DataLoader
from anomaly_model import AnomalyModel
# from anomaly_model_bert import AnomalyModel
from utils import LogSequenceDataset, VariantLogSequenceDataset, RawLogDataset, HDFS_Dataset, LogSeqTokenizer, EarlyStopping, summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import datetime
import os
import math
import random
import numpy as np


def calculate_metrics(true_labels, predicted_probs, threshold=0.5):
    predicted_labels = (predicted_probs > threshold).float()
    true_labels_np = true_labels.cpu().numpy()
    predicted_labels_np = predicted_labels.cpu().numpy()
    true_labels_flat = true_labels_np.flatten()
    predicted_labels_flat = predicted_labels_np.flatten()
    conf_matrix = confusion_matrix(true_labels_flat, predicted_labels_flat)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall / (precision + recall)
    specificity = tn/(tn+fp)
    return precision, recall, specificity, f1

class BinaryAnomalyEval:
    def __init__(self, model, device, val_loader):
        self.device = device
        self.model = model
        if isinstance(model, torch.nn.DataParallel): 
            if self.model.module.sequential_encoding!=None: self.model.module.sequential_encoding.to(self.device)
            self.move_tensor = False
        else: 
            if self.model.sequential_encoding!=None:
                self.model.sequential_encoding.to(self.device)
            self.move_tensor = True
        self.val_loader = val_loader

    def validate_model(self):
        print('--- Validation ---')
        random.seed(42)
        self.model.eval()
        all_true_labels = []
        all_predicted_probs = []
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                u_seq, u_interval ,u_label = data.values()
                if self.move_tensor:
                    u_seq = u_seq.to(self.device)
                    u_interval = u_interval.to(self.device)

                u_label = u_label.to(self.device)
                output_logits = self.model(u_seq, u_interval)

                predicted_probs = torch.argmax(output_logits, dim=1)
                all_true_labels.append(u_label)
                all_predicted_probs.append(predicted_probs)

        all_true_labels = torch.cat(all_true_labels, dim=0)
        print(sum(all_true_labels),"/",len(all_true_labels))
        all_predicted_probs = torch.cat(all_predicted_probs, dim=0)

        precision, recall, specificity, f1 = calculate_metrics(all_true_labels, all_predicted_probs)
        return precision, recall, specificity, f1

def main():
    cpt_save_path = './checkpoint'

    # Dataset
    dataset = 'HDFS'
    val_log_path = f'./dataset/{dataset}/test.csv'
    tid_mapping_path = f'./dataset/{dataset}/template_id_mapping.json'
    pt_embed_path = f'./dataset/{dataset}/sentence_embedding_params.pth'  # 384d
    print(dataset)
    # min_sequence_length = 64  # not applicable when doing anomaly detection
    min_sequence_length = 128
    max_sequence_length = 512
    step_size = 64
    # 'temporal'  'positional' 'time2vec', 'None','temporal_only', 'time2vec_only'
    seq_enc_method = 'time2vec_only'
    time_mode = 'first'
    num_heads = 8
    ff_hidden_size = 2048  # default
    num_layers = 2
    val_batch_size = 32
    # se_dropout = 0.1
    se_dropout = 0
    tf_dropout = 0.1
    # fixed the seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:" + str(0))


    if dataset == "HDFS":
        max_sequence_length = 300
        tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
        val_dataset = HDFS_Dataset(val_log_path, tid_mapping_path, tokenizer, max_sequence_length = max_sequence_length, time_mode = time_mode)
    else:
        tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
        val_dataset = VariantLogSequenceDataset(val_log_path, tid_mapping_path, tokenizer, min_sequence_length = min_sequence_length, max_sequence_length = max_sequence_length, step_size = step_size, time_mode = time_mode)

    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model = AnomalyModel(pt_embed_path, num_heads, ff_hidden_size, num_layers, se_dropout, tf_dropout, max_sequence_length = max_sequence_length, seq_enc_method = seq_enc_method, tid_mapping_file = tid_mapping_path)
    model.to(device)

    checkpoint_path = cpt_save_path + '/' + 'bin_d'+dataset+'_l2_h8_'+seq_enc_method+'_len128-300_step64_modefirst.cpt'
    # checkpoint_path = cpt_save_path + '/' + 'bin_d'+dataset+'_l2_h8_'+seq_enc_method+'_len128-512_step64_modefirst.cpt'
    # checkpoint_path = cpt_save_path + '/' + 'bin_d'+dataset+'_l2_h8_'+seq_enc_method+'_len128-512_step64_modefirst_sentence_embedding_params.pth.cpt'
    # checkpoint_path = cpt_save_path + '/' + 'bin_d'+dataset+'_l2_h8_'+seq_enc_method+'_len128-512_step64_modefirst_random_384_embedding_params.pth.cpt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    binary_anomaly_train = BinaryAnomalyEval(model, device, val_loader)

    # str_cfg = f'bin_d{dataset}_l{num_layers}_h{num_heads}_{seq_enc_method}_len{min_sequence_length}-{max_sequence_length}_step{step_size}_mode{time_mode}'

    precision, recall, specificity, f1 = binary_anomaly_train.validate_model()

    print(f'Precision:{precision}; Recall:{recall}; Spec:{specificity}; F1:{f1}')


    return

if __name__ == "__main__":
    main()