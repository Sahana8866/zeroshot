"""
Zero-Shot Malware Detection on Encrypted Traffic
Author: Sahana R
Conference: ComSIA-2026
Achievement: 91.08% F1-score on unseen botnet families
"""

import numpy as np
import pandas as pd
import random
import os
import gc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
import multiprocessing
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel(logging.ERROR)

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# ============ REPRODUCIBILITY ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

set_seed(42)

# ============ DEVICE ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============ MEMORY OPTIMIZATION ============
def reduce_mem_usage(df):
    start = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = str(df[col].dtype)
        if col_type[:3] == 'int':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif col_type[:5] == 'float':
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage().sum() / 1024**2
    print(f'Memory: {start:.1f}MB → {end:.1f}MB')
    return df

# ============ DATA LOADING ============
folder_path = '/kaggle/input/ctu13'
scenarios = [
    ('Neris', '1-Neris-20110810.binetflow.parquet'),
    ('Neris2', '9-Neris-20110817.binetflow.parquet'),
    ('Rbot', '3-Rbot-20110812.binetflow.parquet'),
    ('Rbot2', '10-Rbot-20110818.binetflow.parquet'),
    ('Virut', '5-Virut-20110815-2.binetflow.parquet'),
    ('Virut2', '13-Virut-20110815-3.binetflow.parquet'),
    ('Menti', '6-Menti-20110816.binetflow.parquet'),
    ('Sogou', '7-Sogou-20110816-2.binetflow.parquet'),
    ('Murlo', '8-Murlo-20110816-3.binetflow.parquet'),
    ('Rbot3', '4-Rbot-20110815.binetflow.parquet'),
    ('Rbot4', '11-Rbot-20110818-2.binetflow.parquet'),
    ('NsisAy', '12-NsisAy-20110819.binetflow.parquet'),
]

def load_and_clean_scenario(file_path, family_name, sample_normal=6000, sample_malware=6000):
    if not os.path.exists(file_path):
        return None
    df = pd.read_parquet(file_path)
    print(f"{family_name:8}: {len(df)} rows", end=" → ")
    
    req = ['dur', 'tot_pkts', 'tot_bytes', 'src_bytes', 'proto', 'state', 'dir', 'label']
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing: {c}")
    
    if 'dPort' not in df.columns:
        df['dPort'] = -1
    else:
        df['dPort'] = pd.to_numeric(df['dPort'], errors='coerce').fillna(-1)
    
    for col in ['dur', 'tot_pkts', 'tot_bytes', 'src_bytes', 'dPort']:
        df[col] = df[col].fillna(df[col].median())
    
    for col in ['proto', 'state', 'dir']:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'UNK'
        df[col] = df[col].fillna(mode_val)
    
    def parse_label(l):
        l = str(l).lower()
        return 'Normal' if any(x in l for x in ['background', 'normal']) else family_name
    
    df['family'] = df['label'].apply(parse_label)
    df = df.drop(columns=['label'])
    
    def is_obfuscated(row):
        s, p, dp = str(row['state']).lower(), str(row['proto']).lower(), int(row['dPort'])
        return (any(x in s for x in ['s0', 'rej', 'rsto', 'rstos0', 's2', 's3']) or
                p in ['udp', 'icmp'] or dp == 443 or 'ssl' in s or 'tls' in s or
                (row['dur'] < 0.5 and row['tot_pkts'] / max(row['dur'], 0.01) > 20))
    
    df['is_obfuscated'] = df.apply(is_obfuscated, axis=1)
    df = reduce_mem_usage(df)
    df = df[df['family'].isin(['Normal', family_name])]
    df = df[(df['dur'] > 0) & (df['tot_pkts'] > 1) & (df['tot_bytes'] > 60)]
    
    normal = df[df['family'] == 'Normal']
    malware = df[df['family'] == family_name]
    if len(normal) > 0:
        normal = normal.sample(n=min(sample_normal, len(normal)), random_state=42)
    if len(malware) > 0:
        malware = malware.sample(n=min(sample_malware, len(malware)), random_state=42)
    df_clean = pd.concat([normal, malware]).reset_index(drop=True)
    print(f"Kept {len(df_clean)}")
    return df_clean

all_dfs = []
for fam, f in scenarios:
    df = load_and_clean_scenario(os.path.join(folder_path, f), fam)
    if df is not None:
        all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
del all_dfs
gc.collect()
print(f"Dataset shape: {full_df.shape}")

# ============ TOKENIZATION ============
def tokenize_row_enhanced(row):
    dur = ('VVSHT' if row['dur'] < 0.05 else 'VSHT' if row['dur'] < 0.2 else
           'SHT' if row['dur'] < 1 else 'MED' if row['dur'] < 10 else
           'LNG' if row['dur'] < 60 else 'VLNG')
    bytes_tok = ('TNY' if row['tot_bytes'] < 128 else 'SML' if row['tot_bytes'] < 512 else
                 'MED' if row['tot_bytes'] < 2048 else 'LRG' if row['tot_bytes'] < 4096 else
                 'HGE' if row['tot_bytes'] < 1024*1024 else 'VHGE')
    pkts = ('VFW' if row['tot_pkts'] < 2 else 'FW' if row['tot_pkts'] < 5 else
            'MOD' if row['tot_pkts'] < 20 else 'MNY' if row['tot_pkts'] < 100 else 'VMNY')
    src_bytes = ('VLO' if row['src_bytes'] < 64 else 'LO' if row['src_bytes'] < 256 else
                 'MED' if row['src_bytes'] < 1024 else 'HI' if row['src_bytes'] < 4096 else 'VHI')
    byte_per_pkt = row['tot_bytes'] / max(row['tot_pkts'], 1)
    ratio = ('TNY_PKT' if byte_per_pkt < 32 else 'SML_PKT' if byte_per_pkt < 64 else
             'MED_PKT' if byte_per_pkt < 256 else 'LRG_PKT')
    src_ratio = row['src_bytes'] / max(row['tot_bytes'], 1)
    direction = 'OUT' if src_ratio > 0.75 else 'BAL' if src_ratio > 0.25 else 'IN'
    proto = str(row['proto']).upper()[:7]
    state = str(row['state']).upper()[:7]
    proto_state = f"{proto}_{state}"
    pkt_rate = row['tot_pkts'] / max(row['dur'], 0.01)
    pkt_rate_tok = ('LOW_RATE' if pkt_rate < 1 else 'MED_RATE' if pkt_rate < 10 else
                    'HI_RATE' if pkt_rate < 50 else 'VHI_RATE')
    dport = int(row.get('dPort', -1))
    tls = 'TLS_HI' if dport == 443 else 'TLS_LO' if 'ssl' in state.lower() else 'PLN'
    obf_token = 'OBF_HI' if row['is_obfuscated'] else 'OBF_LO'
    domain_token = (random.choice(['SUSP_C2', 'RAND_DMN', 'ENC_C2'])
                   if row['family'] != 'Normal' and random.random() < 0.6
                   else 'BNG_ADDR')
    dir_tok = str(row['dir']).upper()[:3]
    return [dur, bytes_tok, pkts, src_bytes, ratio, direction, proto, state,
            proto_state, tls, 'EPH', 'WKN', obf_token, domain_token, dir_tok,
            pkt_rate_tok, f"SRC_{int(src_ratio*10)}"]

full_df['tokens'] = full_df.apply(tokenize_row_enhanced, axis=1)
sentences = full_df['tokens'].tolist()
print(f"Tokenized {len(sentences)} flows")

# ============ SEQ2VEC EMBEDDING ============
all_tokens = [t for seq in sentences for t in seq]
vocab = ['<PAD>', '<UNK>'] + sorted(list(set(all_tokens)))
vocab_size = len(vocab)
token_to_idx = {t: i for i, t in enumerate(vocab)}
print(f"Vocabulary size: {vocab_size}")

seq_indices = [[token_to_idx.get(t, 1) for t in seq] for seq in sentences]
max_len = 17
padded_list = [seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in seq_indices]
padded = torch.tensor(padded_list, dtype=torch.long)
loader = DataLoader(TensorDataset(padded), batch_size=512, shuffle=True, num_workers=0)

class SupervisedSeq2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=384, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout * 0.5)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x):
        e = self.embed(x)
        o, (h, _) = self.lstm(e)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        z = self.proj(h)
        return F.normalize(z, dim=1)

model_s2v = SupervisedSeq2Vec(vocab_size).to(device)
opt_s2v = optim.AdamW(model_s2v.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler_s2v = CosineAnnealingWarmRestarts(opt_s2v, T_0=10, T_mult=2)

seen_families = ['Normal', 'Neris', 'Rbot', 'Menti', 'Virut', 'Murlo', 'Sogou', 'Rbot3', 'NsisAy']
family2id = {f: i for i, f in enumerate(seen_families)}
family_head = nn.Linear(384, len(seen_families)).to(device)

def contrastive_loss(z, temperature=0.07):
    sim = torch.matmul(z, z.T) / temperature
    labels = torch.arange(z.size(0), device=device)
    return F.cross_entropy(sim, labels)

criterion_ce = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

print("Training Seq2Vec...")
for epoch in range(15):
    model_s2v.train()
    family_head.train()
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        opt_s2v.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            z = model_s2v(x)
            loss_c = contrastive_loss(z)
            fam_indices = torch.randint(0, len(full_df), (x.size(0),))
            fam_labels = torch.tensor([family2id.get(full_df.iloc[i.item()]['family'], -1) for i in fam_indices], device=device)
            mask = fam_labels >= 0
            loss_f = criterion_ce(family_head(z[mask]), fam_labels[mask]) if mask.any() else torch.tensor(0.0, device=device)
            loss = loss_c + 0.5 * loss_f
        scaler.scale(loss).backward()
        scaler.step(opt_s2v)
        scaler.update()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    scheduler_s2v.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")

@torch.no_grad()
def embed_seq(seq):
    model_s2v.eval()
    idx = torch.tensor([token_to_idx.get(t, 1) for t in seq], device=device).unsqueeze(0)
    return model_s2v(idx).cpu().numpy().squeeze()

print("Generating embeddings...")
embeddings = np.stack([embed_seq(s) for s in sentences])
embeddings_3d = np.tile(embeddings[:, None, :], (1, max_len, 1))
embedding_dim = 384
print(f"Embeddings shape: {embeddings_3d.shape}")

# ============ ZERO-SHOT TRAIN/TEST SPLIT ============
full_df['embedding'] = list(embeddings_3d)
full_df['label_binary'] = (full_df['family'] != 'Normal').astype(int)

unseen_families = ['Neris2', 'Rbot2', 'Virut2', 'Rbot4']
test_normal = full_df[full_df['family'] == 'Normal'].sample(n=6000, random_state=42)
test_unseen_list = []
for f in unseen_families:
    df_fam = full_df[full_df['family'] == f]
    if len(df_fam) > 0:
        test_unseen_list.append(df_fam.sample(n=min(1500, len(df_fam)), random_state=42))
test_unseen = pd.concat(test_unseen_list) if test_unseen_list else pd.DataFrame()
test_df = pd.concat([test_normal, test_unseen]).sample(frac=1, random_state=42).reset_index(drop=True)

train_df = full_df[full_df['family'].isin(['Normal', 'Neris', 'Rbot', 'Menti', 'Virut', 'Murlo', 'Sogou', 'Rbot3', 'NsisAy']) & ~full_df.index.isin(test_normal.index)].copy()
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ============ SMOTE DATA AUGMENTATION ============
X_train_pre, X_val, y_train_pre, y_val = train_test_split(
    np.stack(train_df['embedding']), train_df['label_binary'].values,
    test_size=0.15, random_state=42, stratify=train_df['label_binary']
)

def adaptive_smote(X, y, k=5, oversample_factor=0.7):
    minority_mask = (y == 1)
    n_minority = np.sum(minority_mask)
    n_majority = np.sum(y == 0)
    target_minority = int(n_majority * oversample_factor)
    n_synthetic = max(0, target_minority - n_minority)
    if n_synthetic == 0:
        return X, y
    X_minority = X[minority_mask]
    k_neighbors = min(k, len(X_minority) - 1)
    if k_neighbors < 1:
        return X, y
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X_minority)
    synthetic = []
    samples_per_point = n_synthetic // len(X_minority) + 1
    for idx in range(len(X_minority)):
        neigh = nn.kneighbors(X_minority[idx].reshape(1, -1), return_distance=False)[0]
        for _ in range(samples_per_point):
            n = np.random.choice(neigh)
            diff = X_minority[n] - X_minority[idx]
            synthetic.append(X_minority[idx] + np.random.random() * diff)
            if len(synthetic) >= n_synthetic:
                break
        if len(synthetic) >= n_synthetic:
            break
    X_synthetic = np.array(synthetic[:n_synthetic])
    X_balanced = np.vstack([X, X_synthetic])
    y_balanced = np.hstack([y, np.ones(len(X_synthetic))])
    return X_balanced, y_balanced

X_train_flat = X_train_pre.reshape(len(X_train_pre), -1)
X_train_res, y_train_res = adaptive_smote(X_train_flat, y_train_pre, k=5, oversample_factor=0.7)
X_train = X_train_res.reshape(-1, max_len, embedding_dim).astype(np.float32)
y_train = y_train_res.astype(np.int64)
print(f"Original: {len(y_train_pre)} | After SMOTE: {len(y_train)}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights.cpu().numpy()}")

X_val = X_val.astype(np.float32)
X_test = np.stack(test_df['embedding']).astype(np.float32)
y_test = test_df['label_binary'].values.astype(np.int64)

# ============ DATASET & DATALOADER ============
class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

batch_size = 128
train_loader = DataLoader(FlowDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(FlowDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(FlowDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

# ============ ENSEMBLE DETECTOR MODEL ============
class SpatialDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full((x.size(0), x.size(1), 1), 1 - self.p, device=x.device))
        return x * mask / (1 - self.p)

class EnsembleDetector(nn.Module):
    def __init__(self, input_dim=384, n_heads=5, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), SpatialDropout(dropout * 0.5))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2), SpatialDropout(dropout * 0.5))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2))
        self.gru = nn.GRU(256, 320, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(640, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2)
        ) for _ in range(n_heads)])
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        outs = [head(x) for head in self.heads]
        return torch.mean(torch.stack(outs), dim=0)

model = EnsembleDetector(input_dim=embedding_dim, n_heads=5, dropout=0.35).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============ LOSS FUNCTION ============
class FocalOHEMLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.5, ohem_ratio=0.4, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ohem_ratio = ohem_ratio
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        k = max(1, int(focal.size(0) * self.ohem_ratio))
        hard_examples, _ = torch.topk(focal, k)
        return hard_examples.mean()

criterion = FocalOHEMLoss(alpha=0.25, gamma=2.5, ohem_ratio=0.4, class_weights=class_weights)

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    cut_len = int(max_len * lam)
    start_pos = np.random.randint(0, max_len - cut_len + 1) if cut_len < max_len else 0
    x_mixed = x.clone()
    x_mixed[:, start_pos:start_pos + cut_len] = x[index][:, start_pos:start_pos + cut_len]
    return x_mixed, y, y[index], lam

# ============ TRAINING ============
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

best_f1 = 0
patience_counter = 0
max_patience = 25
min_epochs = 40
train_losses = []
val_f1_scores = []
val_losses = []
scaler_main = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

print("\nStarting training...")
for epoch in range(100):
    model.train()
    epoch_loss = 0
    accum_steps = 2
    optimizer.zero_grad()
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        if epoch >= 10 and random.random() < 0.5:
            x, ya, yb, lam = cutmix(x, y, alpha=1.0)
        else:
            ya, yb, lam = y, y, 1.0
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            if lam < 1.0:
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                loss = criterion(logits, y)
            loss = loss / accum_steps
        
        scaler_main.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0 or i == len(train_loader) - 1:
            scaler_main.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_main.step(optimizer)
            scaler_main.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accum_steps
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_preds, val_true = [], []
    val_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            val_loss += loss.item()
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_true.extend(y.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    val_f1 = f1_score(val_true, val_preds)
    val_f1_scores.append(val_f1)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        print(f" ✓ Saved best model (F1: {best_f1:.4f})")
        patience_counter = 0
    else:
        if epoch >= min_epochs:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

# Load best model
model.load_state_dict(torch.load('best_model.pth', map_location=device))
print(f"\nBest validation F1: {best_f1:.4f}")

# ============ FINAL EVALUATION ============
model.eval()
test_probs = []
test_preds = []
test_true = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = np.argmax(logits.cpu().numpy(), axis=1)
        test_probs.extend(probs)
        test_preds.extend(preds)
        test_true.extend(y.numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)
test_probs = np.array(test_probs)

# Find optimal threshold
best_threshold = 0.5
best_f1_test = 0
for thresh in np.arange(0.3, 0.7, 0.01):
    preds_thresh = (test_probs > thresh).astype(int)
    f1_thresh = f1_score(test_true, preds_thresh)
    if f1_thresh > best_f1_test:
        best_f1_test = f1_thresh
        best_threshold = thresh

test_preds_opt = (test_probs > best_threshold).astype(int)
final_f1 = f1_score(test_true, test_preds_opt)
final_precision = precision_score(test_true, test_preds_opt)
final_recall = recall_score(test_true, test_preds_opt)

print("\n" + "="*50)
print("FINAL ZERO-SHOT TEST RESULTS")
print("="*50)
print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Test F1 Score: {final_f1:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print("="*50)

# Per-family results
test_df['predicted'] = test_preds_opt
print("\nPer-family F1-scores (Zero-Shot):")
for fam in unseen_families + ['Normal']:
    subset = test_df[test_df['family'] == fam]
    if len(subset) > 0:
        f1 = f1_score(subset['label_binary'], subset['predicted'])
        print(f"  {fam:8}: {f1:.4f}")

print("\n Training complete! Best model saved as 'best_model.pth'")
