from __future__ import annotations
import random
import copy
import logging
from pathlib import Path
from typing import Tuple, List, Union, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import r2_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================== Configuration ===========================
CONFIG = {
    'SEED': 42,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

    # Paths and models
    'DATA_PATH': Path('Data_Files/May/all_features_02_06.csv'),
    'SBERT_MODEL': 'all-mpnet-base-v2',
    'LONGFORMER_MODEL': 'allenai/longformer-base-4096',
    'MICROSOFT_MODEL': 'microsoft/mpnet-base',
    'EMBEDDINGS_DIR': Path('Embeddings'),
    'EMBED_FILE_SBERT': Path('Embeddings/sbert_embeddings.npy'),
    'EMBED_FILE_MICROSOFT': Path('Embeddings/microsoft_embeddings.npy'),
    'EMBED_FILE_LONGFORMER': Path('Embeddings/longformer_embeddings.npy'),
    'EMBED_FILE_HYBRID': Path('Embeddings/hybrid_sbert_longformer_embeddings.npy'),
    'PLOTS_DIR': Path('plots'),

    # Feature toggles
    'USE_SOURCES': {
        'mastodon': True,
        'reddit': True,
        'exploitdb': True,
        'bleepingcomputer': True,
        'telegram': True,
        'hackernews': True
    },

    'USE_FEATURES': {
        'text': True,
        'description': True,
        'cvss_score': True,
        'cvss_categorical': True,
    },

    # Training toggles
    'EMBEDDING_MODEL': 'sbert',  # options: 'sbert', 'microsoft', 'longformer', 'hybrid'
    'USE_BATCH_NORM': True,
    'USE_FOIL_BATCH_SAMPLING': False,
    'USE_LOG1P': False,
    'USE_LOGIT_TRANSFORM': False,
    'USE_CHUNKED_EMBEDDINGS': False, # For handling long texts with SBERT
    'LOSS_FUNCTION': 'huber', # options: 'huber', 'mse', 'mae'
    'USE_WEIGHT_DECAY': False,
    'WEIGHT_DECAY_RATE': 1e-5,
    'USE_COSINE_LR_SCHEDULER': True,

    # Hyperparameters
    'TEST_FRAC': 0.2,
    'BATCH_SIZE_EMBED': 32,
    'BATCH_SIZE_TRAIN': 128,
    'VAL_SPLIT': 0.10,
    'EPOCHS': 1000,
    'LR': 3e-5,
    'HIGH_THRESH': 0.02, # EPSS score threshold to be considered a higher sample in Foil Batch Sampling and PR Plots-- here the choice is of anything over roughly the 75% percentile of the data.

    # Weighted loss parameters
    'ALPHA': 1,
    'RAW_THRESH': 0.02, # Any sample with an EPSS score >= RAW_THRESH has its loss multiplied by a factor (ALPHA).

    # Early Stopping Parameters
    'EARLY_STOPPING_PATIENCE': 900,
    'EARLY_STOPPING_DELTA': 0.0001, # Minimum change that qualifies as an improvement
}

# =========================== Embedding Model Loading Utils ===========================
sbert_model_instance = None
longformer_tokenizer_instance = None
longformer_model_instance = None
microsoft_tokenizer_instance = None
microsoft_model_instance = None

def get_sbert_model():
    global sbert_model_instance
    if sbert_model_instance is None:
        print('-> Initializing SBERT model...')
        sbert_model_instance = SentenceTransformer(CONFIG['SBERT_MODEL'], device=str(CONFIG['DEVICE']))
    return sbert_model_instance

def get_longformer_tokenizer():
    global longformer_tokenizer_instance
    if longformer_tokenizer_instance is None:
        print('-> Initializing Longformer tokenizer...')
        longformer_tokenizer_instance = AutoTokenizer.from_pretrained(CONFIG['LONGFORMER_MODEL'])
    return longformer_tokenizer_instance

def get_longformer_model():
    global longformer_model_instance
    if longformer_model_instance is None:
        print('-> Initializing Longformer model...')
        longformer_model_instance = AutoModel.from_pretrained(CONFIG['LONGFORMER_MODEL']).to(CONFIG['DEVICE'])
    return longformer_model_instance

def get_microsoft_tokenizer():
    global microsoft_tokenizer_instance
    if microsoft_tokenizer_instance is None:
        microsoft_tokenizer_instance = AutoTokenizer.from_pretrained(CONFIG['MICROSOFT_MODEL'])
    return microsoft_tokenizer_instance

def get_microsoft_model():
    global microsoft_model_instance
    if microsoft_model_instance is None:
        microsoft_model_instance = AutoModel.from_pretrained(CONFIG['MICROSOFT_MODEL']).to(CONFIG['DEVICE'])
    return microsoft_model_instance

# =========================== Embedding Utils ===========================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_longformer_embeddings(texts: List[str], batch_size: int = CONFIG['BATCH_SIZE_EMBED']) -> np.ndarray:
    all_embeddings = [None] * len(texts)
    tokenizer = get_longformer_tokenizer()
    model = get_longformer_model()
    model.eval()

    print('-> Generating Longformer embeddings with smart batching...')
    lengths = [len(text) for text in texts]
    sorted_indices = sorted(range(len(texts)), key=lambda k: lengths[k])

    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding with Longformer'):
        batch_indices = sorted_indices[i : i + batch_size]
        batch_texts = [texts[j] for j in batch_indices]

        encoded_input = tokenizer(
            batch_texts,
            padding = 'longest',
            truncation = True,
            max_length = 4096,
            return_tensors = 'pt'
        ).to(CONFIG['DEVICE'])

        global_attention_mask = torch.zeros_like(encoded_input['input_ids'])
        global_attention_mask[:, 0] = 1

        with torch.no_grad():
            model_output = model(**encoded_input, global_attention_mask = global_attention_mask)

        pooled_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = F.normalize(pooled_embeddings, p = 2, dim = 1).cpu()

        for index, embedding in zip(batch_indices, normalized_embeddings):
            all_embeddings[index] = embedding

    final_embeddings = torch.stack(all_embeddings).numpy()
    print('-> Embedding generation complete.')
    return final_embeddings

def get_microsoft_embeddings(texts: List[str], batch_size: int = CONFIG['BATCH_SIZE_EMBED']) -> np.ndarray:
    embeddings_list = []
    tokenizer = get_microsoft_tokenizer()
    model = get_microsoft_model()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(batch_texts, padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(CONFIG['DEVICE'])

        with torch.no_grad():
            model_output = model(**encoded_input)
        pooled = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings_list.append(F.normalize(pooled, p = 2, dim = 1).cpu())
    return torch.cat(embeddings_list).numpy()

def get_sbert_embeddings(texts: List[str], batch_size: int = CONFIG['BATCH_SIZE_EMBED']) -> np.ndarray:
    sbert_model = get_sbert_model()

    if not CONFIG.get('USE_CHUNKED_EMBEDDINGS', False):
        print('-> Using standard SBERT encoding (truncating long texts).')
        return sbert_model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    print('-> Using unified chunking for all texts.')
    tokenizer = sbert_model.tokenizer
    max_seq_len = sbert_model.max_seq_length
    embedding_dim = sbert_model.get_sentence_embedding_dimension()
    device = sbert_model.device

    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    all_input_ids = tokenizer(texts, add_special_tokens = False, truncation = False, padding = False)['input_ids']
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.WARNING)

    all_chunks_ids = []
    doc_chunk_counts = []
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    window_size = max_seq_len - 2
    stride = window_size // 2

    for input_ids in tqdm(all_input_ids, desc='Creating chunks'):
        if not input_ids:
            doc_chunk_counts.append(1)
            all_chunks_ids.append([cls_id, sep_id])
            continue

        doc_chunks = [[cls_id] + input_ids[j:j + window_size] + [sep_id] for j in range(0, len(input_ids), stride)]
        all_chunks_ids.extend(doc_chunks)
        doc_chunk_counts.append(len(doc_chunks))

    final_embeddings = np.zeros((len(texts), embedding_dim))
    if not all_chunks_ids:
        return final_embeddings

    max_len = max(len(chunk) for chunk in all_chunks_ids)
    attention_masks, padded_chunks = [], []
    for chunk in all_chunks_ids:
        padding_len = max_len - len(chunk)
        padded_chunks.append(torch.tensor(chunk + [pad_id] * padding_len, dtype = torch.long))
        attention_masks.append(torch.tensor([1] * len(chunk) + [0] * padding_len, dtype = torch.long))

    input_ids_tensor = torch.stack(padded_chunks)
    attention_mask_tensor = torch.stack(attention_masks)

    chunk_dataset = torch.utils.data.TensorDataset(input_ids_tensor, attention_mask_tensor)
    chunk_loader = DataLoader(chunk_dataset, batch_size = batch_size, shuffle = False)

    chunk_embeddings_list = []
    sbert_model.eval()
    with torch.no_grad():
        for id_batch, mask_batch in tqdm(chunk_loader, desc='Encoding chunks'):
            features = {'input_ids': id_batch.to(device), 'attention_mask': mask_batch.to(device)}
            token_embeds = sbert_model[0](features)['token_embeddings']
            pooled_embeds = sbert_model[1]({'token_embeddings': token_embeds, 'attention_mask': mask_batch.to(device)})
            chunk_embeddings_list.append(pooled_embeds['sentence_embedding'].cpu())

    all_chunk_embeds = torch.cat(chunk_embeddings_list, dim=0).numpy()

    chunk_idx_start = 0
    for i, count in enumerate(tqdm(doc_chunk_counts, desc = 'Averaging embeddings')):
        if count > 0:
            doc_embeds = all_chunk_embeds[chunk_idx_start : chunk_idx_start + count]
            final_embeddings[i] = np.mean(doc_embeds, axis=0)
            chunk_idx_start += count
        else:
            final_embeddings[i] = np.zeros(embedding_dim)

    return final_embeddings

def get_hybrid_embeddings(texts: List[str], batch_size: int = CONFIG['BATCH_SIZE_EMBED']) -> np.ndarray:
    print('-> Using hybrid embedding strategy based on SBERT token limit.')
    sbert_model = get_sbert_model()
    tokenizer = sbert_model.tokenizer
    sbert_token_limit = sbert_model.max_seq_length - 2

    short_indices, short_texts = [], []
    long_indices, long_texts = [], []

    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    all_input_ids = tokenizer(texts, add_special_tokens = False, truncation = False, padding = False)['input_ids']
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.WARNING)

    for i, token_ids in enumerate(all_input_ids):
        if len(token_ids) <= sbert_token_limit:
            short_indices.append(i)
            short_texts.append(texts[i])
        else:
            long_indices.append(i)
            long_texts.append(texts[i])

    print(f'-> Found {len(short_texts)} texts within SBERT limit and {len(long_texts)} texts exceeding it.')
    embedding_dim = 768
    final_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)

    if short_texts:
        print('-> Generating embeddings for short texts using SBERT...')
        original_chunking_setting = CONFIG.get('USE_CHUNKED_EMBEDDINGS', False)
        CONFIG['USE_CHUNKED_EMBEDDINGS'] = False
        short_embs = get_sbert_embeddings(short_texts, batch_size = batch_size)
        CONFIG['USE_CHUNKED_EMBEDDINGS'] = original_chunking_setting
        if short_embs.shape[0] > 0:
            final_embeddings[short_indices] = short_embs

    if long_texts:
        print('-> Generating embeddings for long texts using Longformer...')
        longformer_batch_size = max(1, batch_size // 4)
        long_embs = get_longformer_embeddings(long_texts, batch_size = longformer_batch_size)
        if long_embs.shape[0] > 0: 
            final_embeddings[long_indices] = long_embs

    return final_embeddings

def load_or_compute_embeddings(texts: List[str], source_key: Union[str, None], model_type: str) -> np.ndarray:
    model_map = {
        'sbert': (get_sbert_embeddings, CONFIG['EMBED_FILE_SBERT']),
        'microsoft': (get_microsoft_embeddings, CONFIG['EMBED_FILE_MICROSOFT']),
        'longformer': (get_longformer_embeddings, CONFIG['EMBED_FILE_LONGFORMER']),
        'hybrid': (get_hybrid_embeddings, CONFIG['EMBED_FILE_HYBRID'])
    }
    if model_type not in model_map:
        raise ValueError(f"Unsupported embedding model type: '{model_type}'")

    get_fn, base_path = model_map[model_type]
    suffix = base_path.suffix
    base_filename = base_path.stem

    specific_fname = f"{base_filename}{f'_{source_key}' if source_key else ''}{suffix}"
    path = CONFIG['EMBEDDINGS_DIR'] / specific_fname

    if path.exists():
        try:
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                print(f'-> Loaded cached embeddings from {path}')
                return arr
            else:
                print(f'-> Cached embeddings at {path} have mismatching length. Recomputing.')
        except Exception as e:
            print(f'-> Error loading cached embeddings from {path}: {e}. Recomputing.')

    print(f'-> Computing embeddings for \'{source_key if source_key else "all"}\' using {model_type} model. Saving to {path}')
    embeddings = get_fn(texts)
    np.save(path, embeddings)
    return embeddings

# =========================== Features ===========================
def build_cvss_categorical(dataframe: pd.DataFrame) -> np.ndarray:
    mapping = {
        'attack_vector': 'av',
        'attack_complexity': 'ac',
        'privileges_required': 'pr',
        'user_interaction': 'ui',
        'scope': 'scope',
        'confidentiality_impact': 'ci',
        'integrity_impact': 'ii',
        'availability_impact': 'ai'
    }
    categorical_df = dataframe[list(mapping.keys())].fillna('NONE')
    return pd.get_dummies(categorical_df, prefix = mapping, dtype = float).values 

def assemble_feature_matrix(dataframe: pd.DataFrame) -> np.ndarray:
    feature_list: List[np.ndarray] = []
    num_samples = len(dataframe)

    num_enabled_sources = sum(CONFIG['USE_SOURCES'].values())
    total_sources = len(CONFIG['USE_SOURCES'])
    use_per_source_embeddings = num_enabled_sources > 0 and num_enabled_sources < total_sources

    if CONFIG['USE_FEATURES'].get('text', False):
        if use_per_source_embeddings:
            for source_name, is_enabled in CONFIG['USE_SOURCES'].items():
                if not is_enabled: continue
                mask_idx = dataframe['source'].astype(str).str.contains(source_name, case=False, na=False)
                texts = dataframe.loc[mask_idx, 'text'].fillna('').tolist()
                embs = load_or_compute_embeddings(texts, f'text_{source_name}', CONFIG['EMBEDDING_MODEL'])
                full_embs = np.zeros((num_samples, embs.shape[1]), dtype=float)
                full_embs[mask_idx.values] = embs
                feature_list.append(full_embs)
        else:
            texts = dataframe['text'].fillna('').tolist()
            feature_list.append(load_or_compute_embeddings(texts, 'all_text_full_dataset', CONFIG['EMBEDDING_MODEL']))

    if CONFIG['USE_FEATURES'].get('description', False):
        if use_per_source_embeddings:
            for source_name, is_enabled in CONFIG['USE_SOURCES'].items():
                if not is_enabled: continue
                mask_idx = dataframe['source'].astype(str).str.contains(source_name, case = False, na = False)
                descs = dataframe.loc[mask_idx, 'description'].fillna('').tolist()
                embs = load_or_compute_embeddings(descs, f'desc_{source_name}', CONFIG['EMBEDDING_MODEL'])
                full_embs = np.zeros((num_samples, embs.shape[1]), dtype=float)
                full_embs[mask_idx.values] = embs
                feature_list.append(full_embs)
        else:
            descs = dataframe['description'].fillna('').tolist()
            feature_list.append(load_or_compute_embeddings(descs, 'all_description_full_dataset', CONFIG['EMBEDDING_MODEL']))

    if CONFIG['USE_FEATURES'].get('cvss_score', False):
        feature_list.append(dataframe['cvss_score'].fillna(0).values.reshape(-1, 1) / 10.0)

    if CONFIG['USE_FEATURES'].get('cvss_categorical', False):
        feature_list.append(build_cvss_categorical(dataframe))

    source_one_hot_cols: List[np.ndarray] = []
    if any(CONFIG['USE_SOURCES'].values()):
        for source_name, is_enabled in CONFIG['USE_SOURCES'].items():
            if is_enabled:
                mask = dataframe['source'].astype(str).str.contains(source_name, case = False, na = False).astype(float).values.reshape(-1, 1)
                source_one_hot_cols.append(mask)
    if source_one_hot_cols:
        feature_list.append(np.hstack(source_one_hot_cols))

    if not feature_list:
        raise ValueError('No features selected in USE_FEATURES or USE_SOURCES.')
    return np.hstack(feature_list)

# =========================== Dataset and Sampler ===========================
class WeightedDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
        self.features = features
        self.targets = targets
        self.weights = weights
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.weights[idx]

class FoilBatchSampler(Sampler[int]):
    def __init__(self, high_value_indices, low_value_indices, batch_size, shuffle=True):
        self.high_indices = high_value_indices
        self.low_indices = low_value_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        if not self.high_indices:
            self.num_batches = int(np.ceil(len(self.low_indices) / self.batch_size))
            print('Warning: No high-value samples found for FoilBatchSampler. Using regular batching.')
        else:
            self.num_batches = int(np.ceil(len(self.low_indices) / (self.batch_size - 1)))
        if self.high_indices and len(self.low_indices) < (self.batch_size - 1):
            print('Warning: Not enough low-value samples to fill all batches.')

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.high_indices)
            random.shuffle(self.low_indices)

        if not self.high_indices:
            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(self.low_indices))
                yield self.low_indices[start_idx:end_idx]
            return

        cycled_high = self.high_indices * (self.num_batches // len(self.high_indices) + 1)
        for i in range(self.num_batches):
            high_sample = cycled_high[i]
            low_start = i * (self.batch_size - 1)
            low_end = min((i + 1) * (self.batch_size - 1), len(self.low_indices))
            low_samples = self.low_indices[low_start:low_end]
            yield [high_sample] + low_samples

    def __len__(self):
        return self.num_batches

# =========================== Model ===========================

class EPSSPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        NormLayer = lambda dim: nn.BatchNorm1d(dim) if CONFIG['USE_BATCH_NORM'] else nn.Identity()

        layers = [
            nn.Linear(input_dim, 512), NormLayer(512), nn.LeakyReLU(),
            nn.Linear(512, 512), NormLayer(512), nn.LeakyReLU(),
            nn.Linear(512, 128), NormLayer(128), nn.LeakyReLU(),
            nn.Linear(128, 1)
        ]
        if not (CONFIG['USE_LOGIT_TRANSFORM'] or CONFIG['USE_LOG1P']):
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)

# =========================== Training and Evaluation ===========================
def train_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss = 0
    for features, targets, weights in loader:
        features, targets, weights = features.to(device), targets.to(device), weights.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=str(device).split(':')[0], dtype=torch.bfloat16):
            predictions = model(features)
            loss = (loss_fn(predictions, targets) * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, loss_fn, device) -> Tuple[float, float, List[float], List[float]]:
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for features, targets, weights in loader:
            features, targets, weights = features.to(device), targets.to(device), weights.to(device)
            output = model(features)
            total_loss += (loss_fn(output, targets) * weights).mean().item()

            if CONFIG['USE_LOGIT_TRANSFORM']:
                pred_p = torch.sigmoid(output)
                true_p = torch.sigmoid(targets)
            elif CONFIG['USE_LOG1P']:
                pred_p = torch.expm1(output)
                true_p = torch.expm1(targets)
            else:
                pred_p = output
                true_p = targets

            all_predictions.extend(pred_p.cpu().tolist())
            all_targets.extend(true_p.cpu().tolist())

    avg_loss = total_loss / len(loader)
    r2 = r2_score(all_targets, all_predictions)
    return avg_loss, r2, all_predictions, all_targets

# =========================== Data Preparation ===========================
def load_and_clean_data() -> pd.DataFrame:
    active_path = CONFIG['DATA_PATH']
    print(f'-> Loading data from: {active_path}')
    dataframe = pd.read_csv(active_path).dropna(subset=['epss_score']).reset_index(drop=True)
    print(f'-> Loaded {dataframe.shape[0]} rows from {active_path}')
    return dataframe

def prepare_target_variable(dataframe: pd.DataFrame) -> torch.Tensor:
    epss_scores = dataframe['epss_score'].values
    if CONFIG['USE_LOGIT_TRANSFORM']:
        epss_clipped = np.clip(epss_scores, 1e-6, 1 - 1e-6)
        transformed_values = np.log(epss_clipped / (1 - epss_clipped))
    elif CONFIG['USE_LOG1P']:
        transformed_values = np.log1p(epss_scores)
    else:
        transformed_values = epss_scores
    return torch.tensor(transformed_values, dtype=torch.float32)

def calculate_sample_weights(targets_tensor: torch.Tensor) -> torch.Tensor:
    if CONFIG['USE_LOGIT_TRANSFORM']:
        raw_scores = torch.sigmoid(targets_tensor).numpy()
    elif CONFIG['USE_LOG1P']:
        raw_scores = np.expm1(targets_tensor.numpy())
    else:
        raw_scores = targets_tensor.numpy()
    weights = 1 + CONFIG['ALPHA'] * (raw_scores >= CONFIG['RAW_THRESH'])
    return torch.tensor(weights, dtype=torch.float32)

def create_dataloaders(X_train, y_train, X_val, y_val) -> Tuple[DataLoader, DataLoader]:
    w_train = calculate_sample_weights(y_train)
    w_val = calculate_sample_weights(y_val)
    train_dataset = WeightedDataset(X_train, y_train, w_train)
    val_dataset = WeightedDataset(X_val, y_val, w_val)

    if CONFIG['USE_FOIL_BATCH_SAMPLING']:
        raw_train_scores = torch.sigmoid(y_train).numpy() if CONFIG['USE_LOGIT_TRANSFORM'] else y_train.numpy()
        high_indices = np.where(raw_train_scores >= CONFIG['HIGH_THRESH'])[0].tolist()
        low_indices = np.where(raw_train_scores < CONFIG['HIGH_THRESH'])[0].tolist()
        train_sampler = FoilBatchSampler(high_indices, low_indices, CONFIG['BATCH_SIZE_TRAIN'])
        train_loader = DataLoader(train_dataset, batch_sampler = train_sampler) 
    else:
        train_loader = DataLoader(train_dataset, batch_size = CONFIG['BATCH_SIZE_TRAIN'], shuffle = True)

    val_loader = DataLoader(val_dataset, batch_size = CONFIG['BATCH_SIZE_TRAIN'], shuffle = False)
    return train_loader, val_loader

def create_test_loader(X_test, y_test) -> DataLoader:
    weights = calculate_sample_weights(y_test)
    test_dataset = WeightedDataset(X_test, y_test, weights)
    return DataLoader(test_dataset, batch_size = CONFIG['BATCH_SIZE_TRAIN'], shuffle = False)

def split_data(
    features: torch.Tensor,
    targets: torch.Tensor,
    dataframe: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_indices = dataframe[dataframe['epss_status'] == 'original'].index.tolist()
    non_orig_indices = dataframe[dataframe['epss_status'] != 'original'].index.tolist()

    orig_trainval_indices, test_indices = train_test_split(
        orig_indices,
        test_size = CONFIG['TEST_FRAC'],
        random_state = CONFIG['SEED'],
        shuffle = True
    )
    trainval_indices = non_orig_indices + orig_trainval_indices

    X_trainval = features[trainval_indices]
    y_trainval = targets[trainval_indices]
    X_test = features[test_indices]
    y_test = targets[test_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size = CONFIG['VAL_SPLIT'], random_state = CONFIG['SEED']
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# =========================== Plotting ===========================

def create_report_text(settings: Dict, metrics: Dict) -> str:
    active_feats = [name for name, on in settings['USE_FEATURES'].items() if on]
    feats_text = 'None' if not active_feats else '\n  • '.join(active_feats)

    active_sources = [s for s, on in settings['USE_SOURCES'].items() if on]
    if not active_sources or len(active_sources) == len(settings['USE_SOURCES']):
        src_text = 'All'
    else:
        src_text = '\n  • '.join(active_sources)

    config_details_list = []
    for key, value in settings.items():
        if isinstance(value, dict):
            detail = f'  • {key}:\n'
            for sub_key, sub_value in value.items():
                detail += f'    - {sub_key}: {str(sub_value)}\n'
            config_details_list.append(detail.strip())
        else:
            config_details_list.append(f'  • {key}: {str(value)}')
    config_details_text = '\n'.join(config_details_list)

    cv_tag = ' (Last Fold)' if metrics.get('cv_active') else ''
    cv_avg_tag = ' (Avg over folds)' if metrics.get('cv_active') else ''

    metrics_summary = (
        '\n\nMetrics:' +
        (f'\n  • Log-Log Fit: {metrics["log_log_eq"]}' if metrics.get("log_log_eq") else '') +
        f'\n  • Final validation loss: {metrics["val_loss"]:.4f}{cv_tag}' +
        f'\n  • Final test loss: {metrics["test_loss"]:.4f}{cv_avg_tag}' +
        f'\n  • Test R²: {metrics["test_r2"]:.4f}{cv_avg_tag}' +
        ((f'\n  • Test PR-AUC: {metrics["pr_auc"]:.4f}{cv_avg_tag}' if not np.isnan(metrics["pr_auc"]) else '') if metrics.get('pr_auc') is not None else '')
    )

    return (
        'Active features:\n  • ' + feats_text +
        '\n\nSources:\n  • ' + src_text +
        '\n\nFull Configuration:\n' + config_details_text +
        metrics_summary
    )

def plot_loss_curve(ax, train_losses, val_losses, cv_active):
    ax.plot(train_losses, label = 'Train Loss')
    ax.plot(val_losses, label = 'Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    title = 'Training and Validation Loss' + (' (Last Fold)' if cv_active else '')
    ax.set_title(title)
    ax.legend()

def plot_scatter(ax, trues, preds, cv_active):
    ax.scatter(trues, preds, alpha = 0.5)
    mn, mx = min(trues), max(trues)
    ax.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
    ax.set_xlabel('Actual EPSS')
    ax.set_ylabel('Predicted EPSS')
    title = 'Actual vs Predicted EPSS' + (' (Last Fold Test Set)' if cv_active else '')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')

    regression_line_eq = None
    if len(trues) > 1 and len(preds) > 1:
        positive_mask = (np.array(trues) > 0) & (np.array(preds) > 0)
        if np.any(positive_mask):
            active_trues = np.array(trues)[positive_mask]
            active_preds = np.array(preds)[positive_mask]
            if len(active_trues) >= 2:
                log_trues, log_preds = np.log10(active_trues), np.log10(active_preds)
                m_log, c_log = np.polyfit(log_trues, log_preds, 1)
                K = 10**c_log
                x_fit = np.array([np.min(active_trues), np.max(active_trues)])
                y_fit = K * (x_fit**m_log)
                ax.plot(x_fit, y_fit, color = 'purple', linestyle = '-.', label = 'Log-Log Fit')
                regression_line_eq = f'y ≈ {K:.2e}x^{{{m_log:.2f}}}'
    ax.legend()
    return regression_line_eq

def plot_pr_curve(ax, trues, preds, cv_active):
    binary_threshold = 0.02
    true_binary = np.array(trues) >= binary_threshold
    pred_scores = np.array(preds)
    precision, recall, _ = precision_recall_curve(true_binary, pred_scores)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    title = f'Precision-Recall Curve (Threshold: {binary_threshold})' + (' (Last Fold Test Set)' if cv_active else '')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def generate_plots_and_report(train_losses, val_losses, preds, trues, settings, test_loss, test_r2, test_pr_auc, cv_active = False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = settings['PLOTS_DIR']

    pd.DataFrame({'actual': trues, 'predicted': preds}).to_csv(plots_dir / f'actual_vs_pred_{timestamp}.csv', index = False)

    fig_loss, (ax_loss, ax_text_loss) = plt.subplots(ncols = 2, figsize = (12, 5), gridspec_kw = {'width_ratios': [3, 1]})
    plot_loss_curve(ax_loss, train_losses, val_losses, cv_active)

    fig_scatter, (ax_scatter, ax_text_scatter) = plt.subplots(ncols = 2, figsize = (12, 5), gridspec_kw = {'width_ratios': [3, 1]})
    log_log_eq = plot_scatter(ax_scatter, trues, preds, cv_active)

    fig_pr = None
    if test_pr_auc is not None:
        fig_pr, (ax_pr, ax_text_pr) = plt.subplots(ncols = 2, figsize = (12, 5), gridspec_kw = {'width_ratios': [3, 1]})
        plot_pr_curve(ax_pr, trues, preds, cv_active)

    metrics = {
        'val_loss': val_losses[-1], 'test_loss': test_loss, 'test_r2': test_r2,
        'pr_auc': test_pr_auc, 'log_log_eq': log_log_eq, 'cv_active': cv_active
    }
    report_text = create_report_text(settings, metrics)

    for ax_text in [ax_text_loss, ax_text_scatter, ax_text_pr if fig_pr else None]:
        if ax_text:
            ax_text.axis('off')
            ax_text.text(0, 1, report_text, va='top', ha='left', family='monospace', fontsize=9)

    loss_path = plots_dir / f'loss_{timestamp}.png'
    fig_loss.savefig(loss_path, bbox_inches='tight')
    plt.close(fig_loss)

    scatter_path = plots_dir / f'scatter_{timestamp}.png'
    fig_scatter.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig_scatter)

    if fig_pr:
        pr_path = plots_dir / f'pr_curve_{timestamp}.png'
        fig_pr.savefig(pr_path, bbox_inches='tight')
        plt.close(fig_pr)

# =========================== Main Execution Pipeline ===========================

def execute_run():
    global CONFIG
    random.seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    torch.manual_seed(CONFIG['SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['SEED'])

    CONFIG['EMBEDDINGS_DIR'].mkdir(exist_ok = True) 
    CONFIG['PLOTS_DIR'].mkdir(exist_ok = True, parents = True)
    print(f'Using device: {CONFIG["DEVICE"]}')

    dataframe = load_and_clean_data()

    print('-> Assembling features for the FULL dataset...')
    features_full = torch.tensor(assemble_feature_matrix(dataframe), dtype = torch.float32)
    targets_full = prepare_target_variable(dataframe)
    input_dim = features_full.shape[1]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features_full, targets_full, dataframe)

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)
    test_loader = create_test_loader(X_test, y_test)

    print(f'Train size: {len(train_loader.dataset)}')
    print(f'Validation size: {len(val_loader.dataset)}')
    print(f'Test size: {len(test_loader.dataset)}')

    model = EPSSPredictor(input_dim=input_dim).to(CONFIG['DEVICE'])

    if CONFIG['USE_WEIGHT_DECAY']:
        optimizer = torch.optim.AdamW(model.parameters(), lr = CONFIG['LR'], weight_decay = CONFIG['WEIGHT_DECAY_RATE'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = CONFIG['LR'])

    scheduler = None
    if CONFIG.get('USE_COSINE_LR_SCHEDULER', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = CONFIG['EPOCHS'])
        print('-> Using Cosine Annealing LR Scheduler.')

    loss_fn = {
        'huber': nn.HuberLoss(reduction='none'),
        'mse':   nn.MSELoss(reduction='none'),
        'mae':   nn.L1Loss(reduction='none')
    }[CONFIG['LOSS_FUNCTION']]

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    for epoch in range(CONFIG['EPOCHS']):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, CONFIG['DEVICE'])
        val_loss, val_r2, _, _ = evaluate_model(model, val_loader, loss_fn, CONFIG['DEVICE'])
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{CONFIG["EPOCHS"]} — LR {current_lr:.2e} | Train {train_loss:.4f} | Val {val_loss:.4f} | R² {val_r2:.4f}')

        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss - CONFIG['EARLY_STOPPING_DELTA']:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG['EARLY_STOPPING_PATIENCE']:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    test_loss, test_r2, test_preds, test_trues = evaluate_model(model, test_loader, loss_fn, CONFIG['DEVICE'])

    test_pr_auc = None
    if len(np.unique(np.array(test_trues) >= 0.02)) > 1:
        precision, recall, _ = precision_recall_curve(np.array(test_trues) >= 0.02, np.array(test_preds))
        test_pr_auc = auc(recall, precision)
        print(f'Final Test — loss {test_loss:.4f}, R² {test_r2:.4f}, PR-AUC {test_pr_auc:.4f}')
    else:
        print(f'Final Test — loss {test_loss:.4f}, R² {test_r2:.4f} (PR-AUC not calculable)')

    generate_plots_and_report(
        train_losses, val_losses,
        test_preds, test_trues,
        CONFIG, test_loss, test_r2, test_pr_auc
    )

def main():
    global CONFIG
    print(f'PyTorch Version: {torch.__version__}')
    base_config = copy.deepcopy(CONFIG)
    sources = list(base_config['USE_SOURCES'].keys())

    final_plots_dir = Path('final_plots_longform')
    final_plots_dir.mkdir(exist_ok=True)

    print(f'\n\n{"="*30} RUNNING EXPERIMENT: ALL SOURCES {"="*30}\n')
    CONFIG = copy.deepcopy(base_config)
    CONFIG['USE_SOURCES'] = {s: True for s in sources}
    CONFIG['PLOTS_DIR'] = final_plots_dir / 'all_sources'
    execute_run()

    for source_to_remove in sources:
        print(f'\n\n{"="*30} RUNNING EXPERIMENT: LEAVING OUT {source_to_remove.upper()} {"="*30}\n')
        CONFIG = copy.deepcopy(base_config)
        CONFIG['USE_SOURCES'] = {s: True for s in sources}
        CONFIG['USE_SOURCES'][source_to_remove] = False
        CONFIG['PLOTS_DIR'] = final_plots_dir / f'leave_out_{source_to_remove}'
        execute_run()

    for source_to_keep in sources:
        print(f'\n\n{"="*30} RUNNING EXPERIMENT: ONLY {source_to_keep.upper()} {"="*30}\n')
        CONFIG = copy.deepcopy(base_config)
        CONFIG['USE_SOURCES'] = {s: False for s in sources}
        CONFIG['USE_SOURCES'][source_to_keep] = True
        CONFIG['PLOTS_DIR'] = final_plots_dir / f'only_{source_to_keep}'
        execute_run()

if __name__ == '__main__':
    main()