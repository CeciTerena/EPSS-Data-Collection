from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoModel, AutoTokenizer


class Config:
    SEED = 42
    DATA_PATH = Path("Data_Files/April/merged_cve_text_with_cvss.csv")
    PLOTS_DIR = Path("prototype_plots")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    EMBEDDING_TYPE = "MPNETBASE"  # Options: "SBERT", "MPNETBASE"
    SBERT_MODEL_NAME = "all-mpnet-base-v2"
    SBERT_EMBED_FILE = Path("sbert_embeddings.npy")

    HF_MODEL_NAME = "microsoft/mpnet-base"
    HF_EMBED_FILE = Path("microsoft_embeddings.npy")

    USE_CVSS_FEATURE = True
    SAMPLING_STRATEGY = "Foil"  # Options: "Standard", "Weighted", "Foil"
    ALPHA = 20.0  # Weight multiplier for higher percentile samples in "Weighted" and "Foil"
    RAW_THRESH = 0.02  # EPSS score threshold to be considered a "high" value sample -- here the choice is of anything over the 75% percentile of the training data.

    TEST_SPLIT = 0.20
    EPOCHS = 200
    LR = 1e-3
    BATCH_SIZE_EMBED = 32
    BATCH_SIZE_TRAIN = 128
    LOSS_FUNCTION = "Huber"  # Options: "Huber", "MSE"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config(config: type):
    print("=" * 60)
    print("Starting run with the following configuration:")
    for key, value in config.__dict__.items():
        if not key.startswith('__') and isinstance(value, (str, int, float, bool, Path)):
            print(f"{key}: {value}")
    print("=" * 60)

def initialize_run(config: type) -> None:
    set_seed(config.SEED)
    config.PLOTS_DIR.mkdir(exist_ok=True)
    print_config(config)
    print(f"Torch {torch.__version__} | Device: {config.DEVICE}")

def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["epss_score"]).reset_index(drop=True)
    print(f"Data shape after cleaning: {df.shape}")
    return df

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_or_build_embeddings(texts: List[str], config: type) -> np.ndarray:
    if config.EMBEDDING_TYPE == "SBERT":
        model_name, embed_file, encoder_func = config.SBERT_MODEL_NAME, config.SBERT_EMBED_FILE, encode_sbert
    elif config.EMBEDDING_TYPE == "MPNETBASE":
        model_name, embed_file, encoder_func = config.HF_MODEL_NAME, config.HF_EMBED_FILE, encode_mpnetbase
    else:
        raise ValueError(f"Unknown embedding type: {config.EMBEDDING_TYPE}")

    if embed_file.exists():
        try:
            cached_embeddings = np.load(embed_file)
            if cached_embeddings.shape[0] == len(texts):
                print(f"Loading cached embeddings from {embed_file}...")
                return cached_embeddings
            print(f"WARNING: Cache shape {cached_embeddings.shape} mismatch; re-computing.")
        except Exception as exc:
            print(f"WARNING: Could not load cache ({exc}); re-computing.")

    print(f"Encoding texts with {config.EMBEDDING_TYPE} model: {model_name}")
    new_embeddings = encoder_func(texts, model_name, config)
    np.save(embed_file, new_embeddings)
    print(f"Saved new embeddings to {embed_file}")
    return new_embeddings

def encode_sbert(texts: List[str], model_name: str, config: type) -> np.ndarray:
    model = SentenceTransformer(model_name, device=config.DEVICE)
    return model.encode(
        texts,
        batch_size=config.BATCH_SIZE_EMBED,
        convert_to_numpy=True,
        device=config.DEVICE,
        show_progress_bar=True
    )

def encode_mpnetbase(texts: List[str], model_name: str, config: type) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(config.DEVICE)
    model.eval()
    all_embeddings = []
    for i in range(0, len(texts), config.BATCH_SIZE_EMBED):
        batch_texts = texts[i: i + config.BATCH_SIZE_EMBED]
        encoded_input = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        ).to(config.DEVICE)

        with torch.no_grad():
            model_output = model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

        if (i // config.BATCH_SIZE_EMBED) % 10 == 0:
            print(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
    return np.concatenate(all_embeddings, axis=0)

class StandardDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features, self.labels = features, labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class WeightedDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
        self.features, self.labels, self.weights = features, labels, weights
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]

class FoilBatchSampler(Sampler):
    #Custom sampler that ensures each batch contains exactly one higher percentilesample and (batch_size - 1) lower percentile samples.
    # This is useful here due to the highly skewed distribution of EPSS scores -- targets.

    def __init__(self, high_value_indices: List[int], low_value_indices: List[int], batch_size: int, shuffle: bool = True):
        assert batch_size > 1, "FoilBatchSampler requires a batch size greater than 1."
        self.high_value_indices = high_value_indices
        self.low_value_indices = low_value_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = int(np.ceil(len(low_value_indices) / (batch_size - 1)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.high_value_indices)
            random.shuffle(self.low_value_indices)

        num_cycles = (self.num_batches // len(self.high_value_indices)) + 1
        cycled_high_indices = self.high_value_indices * num_cycles

        for batch_index in range(self.num_batches):
            high_sample_index = cycled_high_indices[batch_index]
            low_start = batch_index * (self.batch_size - 1)
            low_end = (batch_index + 1) * (self.batch_size - 1)
            low_sample_indices = self.low_value_indices[low_start:low_end]
            yield [high_sample_index] + low_sample_indices

    def __len__(self):
        return self.num_batches

class EPSSPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)

def prepare_data(config: type) -> Tuple[DataLoader, DataLoader, Dataset, int, bool]:
    df = load_dataframe(config.DATA_PATH)
    texts = df["text"].fillna("").tolist()
    feature_embeddings = get_or_build_embeddings(texts, config)

    if config.USE_CVSS_FEATURE:
        print("Adding CVSS score as a feature.")
        cvss_scores = df["cvss_score"].fillna(0.0).values.reshape(-1, 1) / 10.0
        feature_embeddings = np.hstack([feature_embeddings, cvss_scores])

    input_dim = feature_embeddings.shape[1]
    target_log_scores = torch.tensor(np.log1p(df["epss_score"].values), dtype=torch.float32)
    features_tensor = torch.tensor(feature_embeddings, dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        features_tensor, target_log_scores, test_size=config.TEST_SPLIT, random_state=config.SEED
    )

    is_weighted_strategy = config.SAMPLING_STRATEGY in ["Weighted", "Foil"]

    if config.SAMPLING_STRATEGY == "Standard":
        train_dataset = StandardDataset(X_train, y_train)
        validation_dataset = StandardDataset(X_val, y_val)
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    elif is_weighted_strategy:
        raw_epss_train = torch.expm1(y_train).cpu().numpy()
        weights_train = torch.tensor(1 + config.ALPHA * (raw_epss_train >= config.RAW_THRESH), dtype=torch.float32)
        weights_val = torch.tensor(1 + config.ALPHA * (torch.expm1(y_val).cpu().numpy() >= config.RAW_THRESH), dtype=torch.float32)

        train_dataset = WeightedDataset(X_train, y_train, weights_train)
        validation_dataset = WeightedDataset(X_val, y_val, weights_val)

        if config.SAMPLING_STRATEGY == "Weighted":
            train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
        else:
            high_idx = np.where(raw_epss_train >= config.RAW_THRESH)[0].tolist()
            low_idx = np.where(raw_epss_train < config.RAW_THRESH)[0].tolist()
            foil_sampler = FoilBatchSampler(high_idx, low_idx, batch_size=config.BATCH_SIZE_TRAIN)
            train_dataloader = DataLoader(train_dataset, batch_sampler=foil_sampler)
    else:
        raise ValueError(f"Unknown sampling strategy: {config.SAMPLING_STRATEGY}")

    validation_dataloader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False)
    return train_dataloader, validation_dataloader, validation_dataset, input_dim, is_weighted_strategy


def evaluate_epoch_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_features, batch_labels = batch[0].to(device), batch[1].to(device)
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_labels).mean()
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(train_dataloader: DataLoader, val_dataloader: DataLoader, input_dim: int, is_weighted: bool, config: type) -> Tuple[nn.Module, List[float], List[float]]:
    model = EPSSPredictor(input_dim=input_dim).to(config.DEVICE)
    loss_fn = {"Huber": nn.HuberLoss, "MSE": nn.MSELoss}[config.LOSS_FUNCTION](reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    train_losses, val_losses = [], []
    print("Starting training loop...")
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            if is_weighted:
                batch_features, batch_labels, batch_weights = (t.to(config.DEVICE) for t in batch)
                predictions = model(batch_features)
                loss = (loss_fn(predictions, batch_labels) * batch_weights).mean()
            else:
                batch_features, batch_labels = (t.to(config.DEVICE) for t in batch)
                predictions = model(batch_features)
                loss = loss_fn(predictions, batch_labels).mean()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        avg_val_loss = evaluate_epoch_loss(model, val_dataloader, loss_fn, config.DEVICE)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("Training complete.")
    return model, train_losses, val_losses

def evaluate_model_final(model: nn.Module, val_dataloader: DataLoader, val_dataset: Dataset, config: type) -> Tuple[float, float, np.ndarray, np.ndarray]:
    print("Performing final evaluation...")
    model.eval()
    loss_fn = {"Huber": nn.HuberLoss, "MSE": nn.MSELoss}[config.LOSS_FUNCTION](reduction="none")
    all_preds_log, all_trues_log, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            batch_features, batch_labels = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            predictions = model(batch_features)
            total_loss += loss_fn(predictions, batch_labels).mean().item()
            all_preds_log.extend(predictions.cpu().numpy())
            all_trues_log.extend(batch_labels.cpu().numpy())

    final_val_loss = total_loss / len(val_dataloader)
    final_val_r2 = r2_score(all_trues_log, all_preds_log)
    actual_epss, predicted_epss = np.expm1(all_trues_log), np.expm1(all_preds_log)

    print("\nSample predictions (predicted [log] | actual [log]):")
    sample_indices = random.sample(range(len(val_dataset)), 10)
    for idx in sample_indices:
        sample_features, sample_true_log = val_dataset[idx][:2]
        with torch.no_grad():
            sample_pred_log = model(sample_features.unsqueeze(0).to(config.DEVICE)).item()
        print(f"{sample_pred_log:.4f} | {sample_true_log.item():.4f}")

    return final_val_loss, final_val_r2, actual_epss, predicted_epss

def generate_plots_and_reports(train_losses, val_losses, actual_epss, predicted_epss, val_loss, val_r2, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.style.use('seaborn-v0_8-whitegrid')

    metrics_summary = (
        f"Final Metrics (log-space):\n"
        f"────────────────────────\n"
        f"Validation Loss: {val_loss:.4f}\n"
        f"Validation R²:   {val_r2:.4f}\n"
    )
    print(f"\n{metrics_summary}")

    df_results = pd.DataFrame({"actual_epss": actual_epss, "predicted_epss": predicted_epss})
    print("\nDescriptive statistics for actual data:\n" + df_results['actual_epss'].describe().to_string())
    print("\nDescriptive statistics for predictions:\n" + df_results['predicted_epss'].describe().to_string())
    df_results.to_csv(config.PLOTS_DIR / f"predictions_{timestamp}.csv", index=False)

    fig1, (ax1, ax_text1) = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax_text1.axis('off')
    ax_text1.text(0, 0.5, metrics_summary, va='center', ha='left', family='monospace', fontsize=12)
    loss_path = config.PLOTS_DIR / f"loss_{timestamp}.png"
    fig1.savefig(loss_path, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f"Saved loss plot to: {loss_path}")

    fig2, (ax2, ax_text2) = plt.subplots(ncols=2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    ax2.scatter(actual_epss, predicted_epss, alpha=0.5, s=15)
    if len(actual_epss[actual_epss > 0]) > 0:
        min_val, max_val = min(actual_epss[actual_epss > 0]), max(actual_epss)
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    ax2.set_xlabel("Actual EPSS Score")
    ax2.set_ylabel("Predicted EPSS Score")
    ax2.set_title("Actual vs. Predicted EPSS Scores (Log-Log Scale)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax_text2.axis('off')
    ax_text2.text(0, 0.5, metrics_summary, va='center', ha='left', family='monospace', fontsize=12)
    scatter_path = config.PLOTS_DIR / f"scatter_{timestamp}.png"
    fig2.savefig(scatter_path, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f"Saved scatter plot to: {scatter_path}")

def main():
    initialize_run(Config)
    train_loader, val_loader, val_dataset, input_dim, is_weighted = prepare_data(Config)
    model, train_losses, val_losses = train_model(train_loader, val_loader, input_dim, is_weighted, Config)
    val_loss, val_r2, actual, predicted = evaluate_model_final(model, val_loader, val_dataset, Config)
    generate_plots_and_reports(train_losses, val_losses, actual, predicted, val_loss, val_r2, Config)
    print("Run finished.")

if __name__ == "__main__":
    main()