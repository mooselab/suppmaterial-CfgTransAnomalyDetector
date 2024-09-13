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
    # precision = precision_score(true_labels_flat, predicted_labels_flat)
    # recall = recall_score(true_labels_flat, predicted_labels_flat)
    # f1 = f1_score(true_labels_flat, predicted_labels_flat)
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

        # eval_loss, eval_precision, eval_recall, eval_specificity, eval_f1 = self.evaluate_model()
        # print(f"Test Loss: {eval_loss:.4f}")
        # print(f"F1 score: {eval_f1:.3f}")
        # writer.add_scalar('Loss/Test', eval_loss, -1)
        # writer.add_scalar('F1/Test', eval_f1, -1)
        # writer.add_scalar('Specificity/Test', eval_specificity, -1)

        # val_loss, precison, recall, specificity, f1 = self.validate_model()
        # print(f"Validation Loss: {val_loss:.4f}")
        # print(f"F1 score: {f1:.3f}")
        # writer.add_scalar('Loss/Validation', val_loss, -1)
        # writer.add_scalar('F1/Validation', f1, -1)
        # writer.add_scalar('Specificity/Validation', specificity, -1)

        # print(self.scheduler)
        scheduler_exist = False if self.scheduler==None else True
        scheduler_step_within_batch = isinstance(self.scheduler, OneCycleLR)
        print('Update scheduler in each batch:', scheduler_step_within_batch)

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            # for name, param in self.model.named_parameters():
            #     print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
            print(f'--- Training Epoch: {epoch:3d} ---')
            for batch_idx, data in enumerate(tqdm(self.train_loader)):
                u_seq, u_interval ,u_label = data.values()
                if self.move_tensor:
                    u_seq = u_seq.to(self.device)
                    u_interval = u_interval.to(self.device)
                u_label = u_label.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(u_seq, u_interval)
                loss = self.criterion(logits, u_label)
                # batch_size = u_seq.size(0)
                # loss = loss / batch_size
                # print(loss.item())
                if math.isnan(loss.item()):
                    print('Loss nan!')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                if scheduler_exist and scheduler_step_within_batch: self.scheduler.step()
                total_loss += loss.item()

                # if batch_idx!= 0 and batch_idx%50 == 0:
                #     eval_loss, precision, recall, eval_specificity, eval_f1 = self.validate_model()
                #     writer.add_scalar('Loss/eval', eval_loss, batch_idx)
                #     writer.add_scalar('F1/eval', eval_f1, batch_idx)
                #     writer.add_scalar('Specificity/eval', eval_specificity, batch_idx)

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

            # if epoch % 5 ==0:
            #     eval_loss, eval_precision, eval_recall, eval_specificity, eval_f1 = self.evaluate_model()
            #     print(f"Testset Loss: {eval_loss:.4f}")
            #     print(f"F1 score: {eval_f1:.3f}")
            #     writer.add_scalar('Loss/Test', eval_loss, epoch)
            #     writer.add_scalar('F1/Test', eval_f1, epoch)
            #     writer.add_scalar('Specificity/Test', eval_specificity, epoch)
            writer.close()
            # os.makedirs('./checkpoint', exist_ok=True)
            # torch.save(self.model.state_dict(), './checkpoint/saved_model.pth')

    def validate_model(self):
        print('--- Validation ---')
        self.model.eval()
        total_val_loss = 0.0
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
                loss = self.criterion(output_logits, u_label)
                # batch_size = u_seq.size(0)
                # loss = loss / batch_size
                if math.isnan(loss.item()):
                    print('Loss nan!')
                total_val_loss += loss.item()
                predicted_probs = torch.argmax(output_logits, dim=1)
                all_true_labels.append(u_label)
                all_predicted_probs.append(predicted_probs)

            # for data in tqdm(self.val_loader):
            #     u_seq, u_interval ,u_label = data.values()
            #     output_logits = self.model(u_seq, u_interval)
            #     predicted_probs = torch.argmax(output_logits, dim=1)
            #     # predicted_probs = torch.where(output_logits >= 0.5, torch.tensor(1), torch.tensor(0))

            #     all_true_labels.append(u_label)
            #     all_predicted_probs.append(predicted_probs)

            #     output_logits = output_logits.detach().cpu()
            #     # output_logits = output_logits.view_as(u_label)
            #     loss = self.criterion(output_logits, u_label)
            #     batch_size = u_seq.size(0)
            #     loss = loss / batch_size
            #     total_val_loss += loss.item()

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
                log_seq, time_intervals ,labels = data.values()
                output_logits = self.model(log_seq, time_intervals, parsed = False)
                if self.move_tensor:
                    u_seq = u_seq.to(self.device)
                    u_interval = u_interval.to(self.device)
                log_seq = log_seq.to(self.device)
                time_intervals = time_intervals.to(self.device)
                labels = labels.to(self.device)
                # output_logits = output_logits.detach().cpu()
                loss = self.criterion(output_logits, labels)
                total_val_loss += loss.item()

                predicted_probs = torch.argmax(output_logits, dim=1)
                # predicted_probs = torch.where(output_logits >= 0.5, torch.tensor(1), torch.tensor(0))
                all_gt_labels.append(labels)
                all_predicted_probs.append(predicted_probs)
                # output_logits = output_logits.view_as(labels)

        all_gt_labels = torch.cat(all_gt_labels, dim=0)
        all_predicted_probs = torch.cat(all_predicted_probs, dim=0)

        # print(all_true_labels)
        # print(all_predicted_probs)

        precision, recall, specificity, f1 = calculate_metrics(all_gt_labels, all_predicted_probs)
        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss, precision, recall, specificity, f1

def main():
    test_log_path = './dataset/BGL/BGL_test.csv'
    cpt_save_path = './checkpoint'
    tb_log_path = './anomaly_runs'


    # Dataset
    dataset = 'HDFS'
    # pt_embed_file = 'random_384_embedding_params.pth'
    pt_embed_file = 'sentence_embedding_params.pth'
    train_log_path = f'./dataset/{dataset}/train.csv'
    val_log_path = f'./dataset/{dataset}/test.csv'
    tid_mapping_path = f'./dataset/{dataset}/template_id_mapping.json'
    pt_embed_path = f'./dataset/{dataset}/' + pt_embed_file



    # min_sequence_length = 64  # not applicable when doing anomaly detection
    min_sequence_length = 128
    max_sequence_length = 512
    step_size = 64
    # 'temporal'  'positional' 'time2vec', 'None','temporal_only', 'time2vec_only'
    seq_enc_method = 'None'

    time_mode = 'first'
    # time_mode = 'gap'
    num_heads = 8
    ff_hidden_size = 2048  # default
    num_layers = 2
    # lr = 1e-5
    lr = 5e-4
    n_epochs = 100

    gpu_cnt = torch.cuda.device_count()
    batch_factor = gpu_cnt if gpu_cnt>1 else 1
    batch_factor = 1
    train_batch_size = 32 * batch_factor
    val_batch_size = 32
    test_batch_size = 32

    # se_dropout = 0.1
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
    test_dataset = RawLogDataset(test_log_path, max_sequence_length = max_sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    model = AnomalyModel(pt_embed_path, num_heads, ff_hidden_size, num_layers, se_dropout, tf_dropout, max_sequence_length = max_sequence_length, seq_enc_method = seq_enc_method, tid_mapping_file = tid_mapping_path)
    model.to(device)
    # print('GPU Count:', torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)  # Adjust gamma as needed
    scheduler = OneCycleLR(optimizer, max_lr = lr, epochs = n_epochs, steps_per_epoch = len(train_loader))
    # scheduler = None

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    binary_anomaly_train = BinaryAnomalyTrain(model, device, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, freeze_embedding, es_patience, cpt_save_path)

    str_cfg = f'bin_d{dataset}_l{num_layers}_h{num_heads}_{seq_enc_method}_len{min_sequence_length}-{max_sequence_length}_step{step_size}_mode{time_mode}_{pt_embed_file}'
    binary_anomaly_train.train_model(n_epochs, tb_log_path, str_cfg)
    return

if __name__ == "__main__":
    main()