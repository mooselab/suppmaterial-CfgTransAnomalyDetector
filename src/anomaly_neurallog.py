import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import PositionalEncoding, RawLogDataset

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ff_network(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class NeuralLog(nn.Module):
    def __init__(self, embed_dim=768, ff_dim=2048, seq_len=128, num_heads=8, dropout=0.1):
        super(NeuralLog, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout, max_len=seq_len)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # Change to (batch_size, embed_dim, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, embed_dim)
        
        # Classification
        return self.classifier(x)

def main():
    # Test the model
    log_file_path = './dataset/BGL/parsed_result/BGL_test_0.2.csv'
    dataset = RawLogDataset(log_file_path, seq_len=128, step_size = 128)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    
    model = NeuralLog()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for batch in data_loader:
        log_seq = batch['log_seq'].to(device)
        output = model(log_seq)
        print("Output shape:", output.shape)
        print("Output:", output)
        break

if __name__ == "__main__":
    main()
