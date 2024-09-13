import torch
import torch.nn as nn
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from utils import LogSequenceDataset, LogSeqTokenizer, PositionalEncoding, TimeIntervalEncoding, Time2VecEncoding, summary
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class BiLSTM(nn.Module):
    def __init__(self, pt_embed_path : str, hidden_dim = 128, freeze_embedding = True):
        super(BiLSTM, self).__init__()
        pt_embed_param = torch.load(pt_embed_path)
        vocab_size = pt_embed_param['weight'].shape[0]
        self.input_dim = pt_embed_param['weight'].shape[1]
        self.embedding = nn.Embedding(vocab_size, self.input_dim)
        self.embedding.load_state_dict(pt_embed_param)
        self.embedding.weight.requires_grad = False
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 2)  # Multiply by 2 for bidirectional

        if freeze_embedding:
            if isinstance(self, torch.nn.DataParallel):
                underlying_model = self.module
                underlying_model.embedding.weight.requires_grad = False
            else:
                self.embedding.weight.requires_grad = False
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)  # 2 for bidirectional
        
        # Initialize cell state with zeros
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(x.device)
        
        embedded = self.embedding(x)

        # Forward propagate LSTM
        out, _ = self.lstm(embedded, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


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
    print(tn, fp, fn, tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall / (precision + recall)
    specificity = tn/(tn+fp)
    return precision, recall, specificity, f1


def main():
    train_log_path = './dataset/BGL/parsed_result/BGL_train.csv'
    val_log_path = './dataset/BGL/parsed_result/BGL_val.csv'
    tid_mapping_path = './dataset/BGL/model/template_id_mapping.json'
    # pt_embed_path = './dataset/BGL/model/embedding_params.pth'
    # pt_embed_path = './dataset/BGL/model/768_embedding_params.pth'
    pt_embed_path = './dataset/BGL/model/bert/sentence_embedding_params.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    time_mode = 'first'
    time_unit = 'millisecond'
    max_sequence_length = 22
    hidden_dim = 128
    train_batch_size = 64
    val_batch_size = 64
    lr = 1e-5

    tokenizer = LogSeqTokenizer(tid_mapping_path, max_sequence_length)
    train_dataset = LogSequenceDataset(train_log_path, tid_mapping_path, tokenizer, max_sequence_length = max_sequence_length, time_mode = time_mode, time_unit = time_unit, anomaly_detection = True, shuffle_log_items = False)
    val_dataset = LogSequenceDataset(val_log_path, tid_mapping_path, tokenizer, max_sequence_length = max_sequence_length, time_mode = time_mode, time_unit = time_unit, anomaly_detection = True)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model = BiLSTM(pt_embed_path, hidden_dim = hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()


    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        print(f'--- Training Epoch: {epoch:3d} ---')
        for data in tqdm(train_loader):
            u_seq, u_interval ,u_label = data.values()
            u_label = u_label.to(device)
            optimizer.zero_grad()
            logits = model(u_seq)
            loss = criterion(logits, u_label)
            # batch_size = u_seq.size(0)
            # loss = loss / batch_size
            # print(loss.item())
            if math.isnan(loss.item()):
                print('Loss nan!')
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}] - Avg. Training Loss: {avg_train_loss:.4f}")


        print('--- Validation ---')
        model.eval()
        total_val_loss = 0.0
        all_true_labels = []
        all_predicted_probs = []
        with torch.no_grad():
            for data in tqdm(val_loader):
                u_seq, u_interval ,u_label = data.values()
                u_label = u_label.to(device)
                output_logits = model(u_seq)
                loss = criterion(output_logits, u_label)
                # batch_size = u_seq.size(0)
                # loss = loss / batch_size
                if math.isnan(loss.item()):
                    print('Loss nan!')
                total_val_loss += loss.item()
                predicted_probs = torch.argmax(output_logits, dim=1)
                all_true_labels.append(u_label)
                all_predicted_probs.append(predicted_probs)

            all_true_labels = torch.cat(all_true_labels, dim=0)
            all_predicted_probs = torch.cat(all_predicted_probs, dim=0)

            precision, recall, specificity, f1 = calculate_metrics(all_true_labels, all_predicted_probs)
            avg_val_loss = total_val_loss / len(val_loader)
            print(f'--- Eval Epoch: {epoch:3d} ---')
            print(f'precision: {precision:3f}, recall: {recall:3f}, spec: {specificity:3f}, f1: {f1:3f}')

    return

if __name__ == "__main__":
    main()