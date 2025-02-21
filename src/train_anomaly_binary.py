import torch
from torch.utils.data import DataLoader
from anomaly_model import AnomalyModel
from utils import VariantLogSequenceDataset, HDFS_Dataset, LogSeqTokenizer, EarlyStopping
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import confusion_matrix
import datetime
import os
import math
import random
import numpy as np
import argparse
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for anomaly detection training.")
    
    parser.add_argument(
        '--dataset', 
        type=str,
        default='BGL',
        help="Dataset to use for training"
    )
    
    parser.add_argument(
        '--env', 
        type=str,
        default='local',
        choices=['local', 'cluster'],
        help="Specify the environment (local or cluster)"
    )
    
    parser.add_argument(
        '--seq_enc_method',
        type=str,
        default='temporal',
        choices=['temporal', 'positional', 'time2vec', 'None', 'temporal_only', 'time2vec_only'],
        help="Sequence encoding method"
    )
    
    parser.add_argument(
        '--d_model', 
        type=int, 
        default=64, 
        help="Specify the d_model. Default is 64."  # empirically best value for all four datasets
    )
    
    parser.add_argument(
        '--embed_method',
        type=str,
        default='sentence',
        choices=['sentence', 'random', 'onehot'],
        help="Log embedding method"
    )

    return parser.parse_args()


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

class BinaryAnomalyTrain:
    def __init__(self, model, device, train_loader, val_loader, test_loader, criterion, optimizer, scheduler = None, freeze_embedding = True, es_patience = 10, cpt_save_path = './checkpoint'):
        self.device = device
        self.model = model
        if isinstance(model, torch.nn.DataParallel): 
            if self.model.module.sequential_encoding!=None: self.model.module.sequential_encoding.to(self.device)
            self.move_tensor = False
        else: 
            if self.model.sequential_encoding!=None:
                self.model.sequential_encoding.to(self.device)
            self.move_tensor = True
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.mask_token_id = self.train_loader.dataset.tid_to_eid['<MASK>']
        self.eos_token_id = self.train_loader.dataset.tid_to_eid['<EOS>']
        self.earlystopping = EarlyStopping(patience=es_patience, verbose=True, save_path=cpt_save_path)

        if freeze_embedding:
            if isinstance(model, torch.nn.DataParallel):
                underlying_model = model.module
                underlying_model.embedding.weight.requires_grad = False
                for param in underlying_model.embed_model.parameters():
                    param.requires_grad = False
            else:
                model.embedding.weight.requires_grad = False
                for param in model.embed_model.parameters():
                    param.requires_grad = False

    def train_model(self, n_epochs, tb_path = './runs', str_cfg=''):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(os.path.join(tb_path, str_cfg+'_'+timestamp))
        self.earlystopping.filename = str_cfg+'.cpt'
        scheduler_exist = False if self.scheduler==None else True
        scheduler_step_within_batch = isinstance(self.scheduler, OneCycleLR)
        print('Update scheduler in each batch:', scheduler_step_within_batch)

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            print(f'--- Training Epoch: {epoch:3d} ---')
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                u_seq, u_interval, u_att_mask, u_label = data.values()
                if self.move_tensor:
                    u_seq = u_seq.to(self.device)
                    u_interval = u_interval.to(self.device)
                    u_att_mask = u_att_mask.to(self.device)
                u_label = u_label.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(u_seq, u_att_mask, u_interval)
                loss = self.criterion(logits, u_label)
                if math.isnan(loss.item()):
                    print('Loss nan!')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                if scheduler_exist and scheduler_step_within_batch: self.scheduler.step()
                total_loss += loss.item()

            if scheduler_exist and (not scheduler_step_within_batch): self.scheduler.step()
            avg_train_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{n_epochs}] - Avg. Training Loss: {avg_train_loss:.4f}")
            writer.add_scalar('Loss/Training', avg_train_loss, epoch)

            val_loss, precision, recall, specificity, f1 = self.validate_model()

            self.earlystopping(val_loss,self.model)
            # self.earlystopping(-f1,self.model)
            if self.earlystopping.early_stop:
                print("Early stopping")
                break
            if epoch == n_epochs-1:
                self.earlystopping.save_checkpoint(-1, self.model) 
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"F1 score: {f1:.3f}")
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Precision/Validation', precision, epoch)
            writer.add_scalar('Recall/Validation', recall, epoch)
            writer.add_scalar('F1/Validation', f1, epoch)
            writer.add_scalar('Specificity/Validation', specificity, epoch)

            writer.close()


    def validate_model(self):
        print('--- Validation ---')
        self.model.eval()
        total_val_loss = 0.0
        all_true_labels = []
        all_predicted_probs = []
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                u_seq, u_interval, u_att_mask, u_label = data.values()
                if self.move_tensor:
                    u_seq = u_seq.to(self.device)
                    u_interval = u_interval.to(self.device)
                    u_att_mask = u_att_mask.to(self.device)
                u_label = u_label.to(self.device)
                output_logits = self.model(u_seq, att_mask = u_att_mask, time_intervals = u_interval)
                loss = self.criterion(output_logits, u_label)
                if math.isnan(loss.item()):
                    print('Loss nan!')
                total_val_loss += loss.item()
                predicted_probs = torch.argmax(output_logits, dim=1)
                all_true_labels.append(u_label)
                all_predicted_probs.append(predicted_probs)

        all_true_labels = torch.cat(all_true_labels, dim=0)
        all_predicted_probs = torch.cat(all_predicted_probs, dim=0)

        precision, recall, specificity, f1 = calculate_metrics(all_true_labels, all_predicted_probs)
        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss, precision, recall, specificity, f1
    
    def evaluate_model(self):
        print('--- Evaluation ---')
        random.seed(42)
        self.model.eval()
        total_val_loss = 0.0
        all_gt_labels = []
        all_predicted_probs = []
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                log_seq, att_mask,time_intervals ,labels = data.values()
                output_logits = self.model(log_seq, att_mask, time_intervals, parsed = False)
                if self.move_tensor:
                    u_seq = u_seq.to(self.device)
                    u_interval = u_interval.to(self.device)
                log_seq = log_seq.to(self.device)
                time_intervals = time_intervals.to(self.device)
                labels = labels.to(self.device)
                loss = self.criterion(output_logits, labels)
                total_val_loss += loss.item()

                predicted_probs = torch.argmax(output_logits, dim=1)
                all_gt_labels.append(labels)
                all_predicted_probs.append(predicted_probs)

        all_gt_labels = torch.cat(all_gt_labels, dim=0)
        all_predicted_probs = torch.cat(all_predicted_probs, dim=0)

        precision, recall, specificity, f1 = calculate_metrics(all_gt_labels, all_predicted_probs)
        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss, precision, recall, specificity, f1

def main(args):
    # Load paths configuration
    with open('paths.yaml', 'r') as f:
        paths_config = yaml.safe_load(f)
    
    # Get environment-specific paths and common dataset structure
    env_paths = paths_config[args.env]
    dataset_structure = paths_config['dataset']
    dataset = args.dataset
    
    # Set up base paths
    cpt_save_path = env_paths['checkpoint']
    tb_log_path = env_paths['tensorboard']

    # Common path construction
    dataset_prefix = os.path.join(env_paths['dataset_prefix'], dataset)
    train_log_path = os.path.join(dataset_prefix, dataset_structure['train'])
    val_log_path = os.path.join(dataset_prefix, dataset_structure['test'])  # Using test as validation
    tid_mapping_path = os.path.join(dataset_prefix, dataset_structure['embeddings']['template_id_mapping'])
    pt_embed_path = os.path.join(dataset_prefix, dataset_structure['embeddings'][args.embed_method+'_embedding'])
    pt_embed_filename = os.path.splitext(os.path.basename(pt_embed_path))[0]

    min_sequence_length = 128
    max_sequence_length = 512
    step_size = 64
    seq_enc_method = args.seq_enc_method

    time_mode = 'first'
    d_model = args.d_model
    num_heads = 8
    ff_hidden_size = 2048  # default
    num_layers = 2
    lr = 5e-4
    n_epochs = 100

    gpu_cnt = torch.cuda.device_count()
    batch_factor = gpu_cnt if gpu_cnt>1 else 1
    batch_factor = 1
    train_batch_size = 32 * batch_factor
    val_batch_size = 32

    se_dropout = 0
    tf_dropout = 0.1

    freeze_embedding = True
    es_patience = 100

    # fixed the seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + str(0))


    if dataset == "HDFS":
        max_sequence_length = 300
        tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
        train_dataset = HDFS_Dataset(train_log_path, tid_mapping_path, tokenizer, max_sequence_length = max_sequence_length, time_mode = time_mode)
        val_dataset = HDFS_Dataset(val_log_path, tid_mapping_path, tokenizer, max_sequence_length = max_sequence_length, time_mode = time_mode)
    else:
        tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
        train_dataset = VariantLogSequenceDataset(train_log_path, tid_mapping_path, tokenizer, min_sequence_length = min_sequence_length, max_sequence_length = max_sequence_length, step_size = step_size, time_mode = time_mode)
        val_dataset = VariantLogSequenceDataset(val_log_path, tid_mapping_path, tokenizer, min_sequence_length = min_sequence_length, max_sequence_length = max_sequence_length, step_size = step_size, time_mode = time_mode)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = None

    model = AnomalyModel(pt_embed_path, d_model, num_heads, ff_hidden_size, num_layers, se_dropout, tf_dropout, max_sequence_length = max_sequence_length, seq_enc_method = seq_enc_method, tid_mapping_file = tid_mapping_path)
    model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr = lr, epochs = n_epochs, steps_per_epoch = len(train_loader))

    criterion = torch.nn.CrossEntropyLoss()

    binary_anomaly_train = BinaryAnomalyTrain(model, device, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, freeze_embedding, es_patience, cpt_save_path)


    str_cfg = f'bin_d{dataset}_d_model{d_model}_l{num_layers}_h{num_heads}_{seq_enc_method}_len{min_sequence_length}-{max_sequence_length}_step{step_size}_mode{time_mode}_{pt_embed_filename}'
    binary_anomaly_train.train_model(n_epochs, tb_log_path, str_cfg)
    return

if __name__ == "__main__":
    args = parse_arguments()
    main(args)