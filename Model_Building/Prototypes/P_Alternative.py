# Ok so my idea here is to create the model. 
# I was thinking i can actually put all the parts here together....
# Before tho i have to get something that puts every data file in the same format. 
# because we really fucked up by not having a standard format for the data files.
# Either way, not super hard. 
# and io fucked up by making things with json instead of csv. 
# well, anyways. 
#OK NICE NOW I ACTUALLY HAVE A FILE!!!!! AYYY


#THIS IS A SIMPLE INITIAL PROTOTYPE OF THE MODEL!!!!

import random
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score

print("Torch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)
torch.set_float32_matmul_precision("high")
# Load the CSV file
df = pd.read_csv('../merged_cve_text_with_epss.csv')
df = df.dropna(subset=['epss_score']).reset_index(drop=True) #THERE WERE MISSING EPSS SCORES. I NEED TO ADRESS THIS ON THE FILE THAT CREATES THE CSV.


sbert = SentenceTransformer("all-mpnet-base-v2", device=device)

#This is VERY VERY rudimentary. its just embedding each text using sbert. the batch size 32 control how man y texts are processes at once on the GPU for efficiency. 
texts = df['text'].fillna("").tolist()

if os.path.exists('sbert_embeddings.npy'):
    X = np.load('sbert_embeddings.npy')
else:
    X = sbert.encode(texts, batch_size=32, convert_to_numpy=True, device=device, show_progress_bar=True)
    np.save('sbert_embeddings.npy', X)

y = torch.tensor(df['epss_score'].values, dtype=torch.float32, device=device)
X = torch.tensor(X, dtype=torch.float32, device=device)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset class
class CVEDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CVEDataset(X_train, y_train)
val_dataset = CVEDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)


class EPPSPredictorPrototype(nn.Module):
    def __init__(self, input_dim):
        super(EPPSPredictorPrototype, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze()

model = EPPSPredictorPrototype(input_dim=X.shape[1])
model = model.to(device)

def show_sample_predictions(model, dataset, n=10):
    model.eval()
    device = next(model.parameters()).device  
    
    indices = random.sample(range(len(dataset)), n)
    for idx in indices:
        x, true_y = dataset[idx]
        x = x.unsqueeze(0).to(device)  
        true_y = true_y.to(device)       

        with torch.no_grad():
            pred_y = model(x)

        pred_y = pred_y.squeeze().item()
        true_y = true_y.item()

        print(f"Predicted: {pred_y:.4f} | Actual: {true_y:.4f}")

def eval_model(model, val_loader):
    model.eval()
    val_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    r2 = r2_score(all_targets, all_preds)
    return val_loss / len(val_loader), r2

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 200
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if epoch % 50 == 0:
        print("intermediate eval:")
        print(f"Epoch {epoch+1}, Val loss: {eval_model(model, val_loader)}")
    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}')

model.eval()
val_loss = 0
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        val_loss += loss.item()
        
print(f'Final Validation Loss: {eval_model(model, val_loader)}')

print("Validation predictions :")
show_sample_predictions(model, val_dataset, n=10)
print("--------------------------- :")
print("Training predictions :")
show_sample_predictions(model, train_dataset, n=10)