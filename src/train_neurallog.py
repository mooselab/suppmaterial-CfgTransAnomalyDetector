import torch
from torch.utils.data import DataLoader
from anomaly_neurallog import NeuralLog
from utils import RawLogDataset, HDFSEmbeddedDataset, EarlyStopping
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
    parser = argparse.ArgumentParser(description="Argument parser for NeuralLog training.")
    
    parser.add_argument(
        '--dataset', 
        type=str,
        default='BGL', 
        help="Specify the dataset to be used."
    )
    
    parser.add_argument(
        '--env', 
        type=str,
        default='local',
        choices=['local', 'cluster'],
        help="Specify the environment (local or cluster)"
    )
    
    parser.add_argument(
        '--seq_len', 
        type=int, 
        default=128,
        help="Sequence length. Default is 128."
    )
    
    parser.add_argument(
        '--step_size', 
        type=int, 
        default=64,
        help="Step size. Default is 64."
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
    
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    
    return precision, recall, specificity, f1

class NeuralLogTrainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, es_patience=10, cpt_save_path='./checkpoint'):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.earlystopping = EarlyStopping(patience=es_patience, verbose=True, save_path=cpt_save_path)

    def train_model(self, n_epochs, tb_path='./runs', str_cfg=''):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(os.path.join(tb_path, str_cfg+'_'+timestamp))
        self.earlystopping.filename = str_cfg+'.cpt'

        scheduler_step_within_batch = isinstance(self.scheduler, OneCycleLR)
        print('Update scheduler in each batch:', scheduler_step_within_batch)

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            print(f'--- Training Epoch: {epoch:3d} ---')
            
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                log_seq = data['log_seq'].to(self.device)
                labels = data['label'].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(log_seq)
                loss = self.criterion(logits, labels)
                
                if math.isnan(loss.item()):
                    print('Loss nan!')
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                if self.scheduler and scheduler_step_within_batch:
                    self.scheduler.step()
                    
                total_loss += loss.item()

            if self.scheduler and not scheduler_step_within_batch:
                self.scheduler.step()
                
            avg_train_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{n_epochs}] - Avg. Training Loss: {avg_train_loss:.4f}")
            writer.add_scalar('Loss/Training', avg_train_loss, epoch)

            val_loss, precision, recall, specificity, f1 = self.validate_model()
            
            self.earlystopping(val_loss, self.model)
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
                log_seq = data['log_seq'].to(self.device)
                labels = data['label'].to(self.device)
                output_logits = self.model(log_seq)
                loss = self.criterion(output_logits, labels)
                if math.isnan(loss.item()):
                    print('Loss nan!')
                    continue
                total_val_loss += loss.item()
                predicted_probs = torch.argmax(output_logits, dim=1)
                all_true_labels.append(labels)
                all_predicted_probs.append(predicted_probs)

        all_true_labels = torch.cat(all_true_labels, dim=0)
        all_predicted_probs = torch.cat(all_predicted_probs, dim=0)

        precision, recall, specificity, f1 = calculate_metrics(all_true_labels, all_predicted_probs)
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
    
    # Set up paths
    dataset_prefix = os.path.join(env_paths['dataset_prefix'], dataset)
    cpt_save_path = os.path.join(env_paths['checkpoint'], 'neurallog')
    tb_log_path = os.path.join(env_paths['tensorboard'], 'neurallog')
    train_log_path = os.path.join(dataset_prefix, dataset_structure['train'])
    val_log_path = os.path.join(dataset_prefix, dataset_structure['test'])
    
    # HF_path only needed for cluster environment
    HF_path = env_paths.get('HF_HOME', '')
    
    # Create directories if they don't exist
    os.makedirs(cpt_save_path, exist_ok=True)
    os.makedirs(tb_log_path, exist_ok=True)

    # Model parameters
    embed_dim = 768  # BERT's default embedding dimension
    seq_len = args.seq_len
    step_size = args.step_size
    num_heads = 8
    ff_dim = 2048
    dropout = 0.1
    
    # Training parameters
    lr = 5e-4
    n_epochs = 100
    batch_size = 32
    es_patience = 100

    # Fix seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    if dataset == 'HDFS':
        seq_len = 300
        tid_mapping_path = os.path.join(dataset_prefix, dataset_structure['embeddings']['template_id_mapping'])
        embedding_file = os.path.join(dataset_prefix, dataset_structure['embeddings']['bert_embedding'])
        train_dataset = HDFSEmbeddedDataset(train_log_path, tid_mapping_path, embedding_file, seq_len=seq_len, shuffle_log_items=True)
        val_dataset = HDFSEmbeddedDataset(val_log_path, tid_mapping_path, embedding_file, seq_len=seq_len, shuffle_log_items=False)
    else:
        train_dataset = RawLogDataset(train_log_path, HF_path, seq_len=seq_len, step_size = step_size, cache_dir=env_paths['cache'])
        val_dataset = RawLogDataset(val_log_path, HF_path, seq_len=seq_len, step_size = step_size, cache_dir=env_paths['cache'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = NeuralLog(
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        dropout=dropout
    )
    model.to(device)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=n_epochs, steps_per_epoch=len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    # Create trainer and train
    trainer = NeuralLogTrainer(
        model, device, train_loader, val_loader, criterion, optimizer, 
        scheduler, es_patience, cpt_save_path
    )

    str_cfg = f'neurallog_{dataset}_emb{embed_dim}_len{seq_len}'
    trainer.train_model(n_epochs, tb_log_path, str_cfg)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)