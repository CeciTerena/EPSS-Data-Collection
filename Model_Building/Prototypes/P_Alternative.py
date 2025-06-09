from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Sampler
from sentence_transformers import SentenceTransformer

# -----------------------------
# SEED = random.randint(0, 2**32 - 1) # actual random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------- paths and hyperparameters ------------------

DATA_PATH = Path("../Data_Files/merged_cve_text_with_epss.csv")
EMBED_FILE = Path("sbert_embeddings.npy")
SBERT_MODEL = "all-mpnet-base-v2"
BATCH_SIZE_EMBED = 32          # for SBERT encoding
BATCH_SIZE_TRAIN = 128         # for DataLoader
TEST_SPLIT = .20
EPOCHS = 200
LR = 1e-3
HIGH_THRESH = 0.02          # *raw* EPSS threshold – tune as you like

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Torch {torch.__version__} | Device: {DEVICE}")

ALPHA      = 20.0         # how much more the rare cases should count
RAW_THRESH = 0.02         # same threshold as above

# ============================================================

def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["epss_score"]).reset_index(drop=True) # drop rows with missing EPSS scores
    # df = df[df["source"] != "exploitdb"].copy() # drop exploitdb for now
    print("-> Data shape after cleaning:", df.shape)
    return df


def build_embeddings(texts: List[str], embed_file: Path, sbert_model: str = SBERT_MODEL, batch_size: int = BATCH_SIZE_EMBED, device: str = DEVICE,) -> np.ndarray:
    if embed_file.exists():
        try:
            X_cached = np.load(embed_file)
            expected_dim = X_cached.shape[1]                    # D from cache
            if X_cached.shape == (len(texts), expected_dim):
                print("-> Loading cached embeddings …")
                return X_cached
            else:
                print(
                    f"-> Cache shape {X_cached.shape} does not match "
                    f"(len(texts)={len(texts)}, dim={expected_dim}); re-computing."
                )
        except Exception as exc:
            print(f"-> Could not load cache ({exc}); re-computing.")

    print("-> Encoding texts with SBERT")
    sbert = SentenceTransformer(sbert_model, device=device)
    X_new = sbert.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        device=device,
        show_progress_bar=True,
    )
    np.save(embed_file, X_new)
    return X_new

# ============================================================
# class CVEDataset(Dataset):
#     def __init__(self, X: torch.Tensor, y: torch.Tensor):
#         self.X, self.y = X, y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
    
    
class WeightedDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        self.X, self.y, self.w = X, y, w
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

# ============================================================
# Model
# ============================================================
class EPSSPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128,   1),       nn.Sigmoid()  # outputs in [0, 1]
            # nn.Linear(input_dim, 4096), nn.LeakyReLU(),
            # nn.Linear(4096, 4096),     nn.LeakyReLU(),
            # nn.Linear(4096, 2048),     nn.LeakyReLU(),
            # nn.Linear(2048, 1024),     nn.LeakyReLU(),
            # nn.Linear(1024, 512),      nn.LeakyReLU(),
            # nn.Linear(512, 256),       nn.LeakyReLU(),
            # nn.Linear(256, 1)
            # nn.Sigmoid()  # outputs in [0, 1]         
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)

# ============================================================

class FoilBatchSampler(Sampler):
    """
    Yields indices so that each batch contains:
      • 1 randomly chosen 'high' sample  (foil)
      • (batch_size-1) randomly chosen 'low' samples
    """
    def __init__(self, high_idx, low_idx, batch_size, shuffle=True):
        assert batch_size > 1
        self.high_idx  = high_idx
        self.low_idx   = low_idx
        self.batch     = batch_size
        self.shuffle   = shuffle
        self.num_batches = int(np.ceil(len(low_idx) / (batch_size-1)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.high_idx)
            random.shuffle(self.low_idx)
        # cycle through the shorter list if necessary
        high_cycle = self.high_idx * ((self.num_batches // len(self.high_idx)) + 1)

        for b in range(self.num_batches):
            start = b * (self.batch - 1)
            end   = start + (self.batch - 1)
            batch = [high_cycle[b]] + self.low_idx[start:end]
            yield batch

    def __len__(self):
        return self.num_batches

# ============================================================

def train_epoch(model: nn.Module, loader: DataLoader, loss_fn, optim) -> float:
    model.train()
    total = 0.0 
    for xb, yb, wb in loader:               # <── get the weights
        xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
        optim.zero_grad()
        preds = model(xb)
        loss  = (loss_fn(preds, yb) * wb).mean()   # <── weight then mean
        loss.backward()
        optim.step()
        total += loss.item()   
    # for xb, yb in loader:
    #     xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    #     optim.zero_grad()
    #     preds = model(xb)
    #     loss = loss_fn(preds, yb)
    #     loss.backward()
    #     optim.step()
    #     total += loss.item()
    return total / len(loader)


def eval_epoch(model: nn.Module, loader: DataLoader, loss_fn) -> Tuple[float, float]:
    model.eval()
    total, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb, yb, _ in loader:         
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out  = model(xb)
            loss = loss_fn(out, yb).mean() 
            total += loss.item()
            preds.extend(out.cpu().numpy())
            trues.extend(yb.cpu().numpy())
        # for xb, yb in loader:
        #     xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        #     out = model(xb)
        #     loss = loss_fn(out, yb)
        #     total += loss.item()
        #     preds.extend(out.cpu().numpy())   
        #     # preds.extend(torch.clamp(out, 0, 1).cpu().numpy()) # clamp to [0, 1] --> Since sigmoid is not working, we are clamping afterwards.         
        #     trues.extend(yb.cpu().numpy())
    r2 = r2_score(trues, preds)
    return total / len(loader), r2, preds, trues


def show_sample_predictions(model: nn.Module, dataset: Dataset, n: int = 10):
    idxs = random.sample(range(len(dataset)), n)
    print("Sample predictions (pred | actual):")
    model.eval()
    for idx in idxs:
        # x, y_true = dataset[idx]
        x, y_true, _ = dataset[idx]          # ← ignore the weight
        with torch.no_grad():
            y_pred = model(x.unsqueeze(0).to(DEVICE)).item()
            # y_pred = torch.clamp(model(x.unsqueeze(0).to(DEVICE)), 0, 1).item() # --> For clamping
        print(f"{y_pred:.4f} | {y_true.item():.4f}")
            # z = model(x.unsqueeze(0).to(DEVICE))
            # pred = torch.exp(z).item()
        # print(f"{pred:.4f} | {y_true.item():.4f}")

# ============================================================

def main():
    # ----- data loading -----
    df = load_dataframe(DATA_PATH)
    texts = df["text"].fillna("").tolist()
    X_np = build_embeddings(texts, EMBED_FILE)
    
    # targets
    y_np = np.log1p(df["epss_score"].values)
    # y = torch.tensor(df["epss_score"].values, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_np, dtype=torch.float32, device=DEVICE)
    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SPLIT, random_state=SEED)
    
    
    # ----- sample wise loss weighting AND foil batch sampler -----
    raw_epss = torch.expm1(y_train).cpu().numpy()          # back-convert

    weights_np = 1 + ALPHA * (raw_epss >= RAW_THRESH).astype(float)
    w_train    = torch.tensor(weights_np, dtype=torch.float32, device=DEVICE)

    raw_epss_val = torch.expm1(y_val).cpu().numpy()
    w_val = torch.tensor(
        1 + ALPHA * (raw_epss_val >= RAW_THRESH).astype(float),
        dtype=torch.float32, device=DEVICE
    )
    
    train_ds = WeightedDataset(X_train, y_train, w_train)
    val_ds   = WeightedDataset(X_val,   y_val,   w_val)
    
    high_idx = np.where(raw_epss >= HIGH_THRESH)[0].tolist() #for the foil batch sampler
    low_idx  = np.where(raw_epss <  HIGH_THRESH)[0].tolist() #for the foil batch sampler


    foil_sampler = FoilBatchSampler(high_idx, low_idx, batch_size=BATCH_SIZE_TRAIN)
    train_loader = DataLoader(train_ds, batch_sampler=foil_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_TRAIN, shuffle=False)
    
    # --------------- no sample wise loss weighting and no foil batch sampler------------------------
        
    # train_loader = DataLoader(CVEDataset(X_train, y_train), batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    # val_loader   = DataLoader(CVEDataset(X_val,   y_val),   batch_size=BATCH_SIZE_TRAIN, shuffle=False)

    # ----- model / loss / optimiser -----
    model = EPSSPredictor(input_dim=X.shape[1]).to(DEVICE)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.HuberLoss(reduction="none")
    optim   = torch.optim.AdamW(model.parameters(), lr=LR)

    # ----- training loop -----
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, loss_fn, optim)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}")

    # ----- evaluation & stats -----
    test_loss, val_r2, preds_log, trues_log = eval_epoch(model, val_loader, loss_fn)
    print(f"\nFinal Test: loss = {test_loss:.4f}, R² = {val_r2:.4f}")

    actual = np.expm1(trues_log)
    predicted = np.expm1(preds_log)

    df_results = pd.DataFrame({"actual_epss": actual, "predicted_epss": predicted})
    print("\nDescriptive statistics for actual data:")
    print(df_results["actual_epss"].describe())
    print("\nDescriptive statistics for predictions:")
    print(df_results["predicted_epss"].describe())

    show_sample_predictions(model, WeightedDataset(X_val, y_val, w_val), n=10)
    
    print("=" * 60)


if __name__ == "__main__":
    main()