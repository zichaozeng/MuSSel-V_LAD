"""K-means clustering for VLAD visual vocabulary generation.

This script computes cluster centres (visual vocabulary) used for VLAD encoding.
It samples features from reference images, normalises them, and performs K-means
clustering (typically K=32 or K=64). The resulting cluster centres are cached
for use during descriptor extraction.

Process:
1. Load pre-extracted DINOv2 features from HDF5
2. Sample features (spatial subsampling for large datasets)
3. Normalise feature descriptors
4. Perform K-means clustering with cosine distance
5. Save cluster centres to cache file

Usage:
    python cluster_centre.py <dataset>
    
Example:
    python cluster_centre.py laurel
"""

import os
import torch
from typing import Optional
import numpy as np
import fast_pytorch_kmeans as fpk
import h5py
from natsort import natsorted
from tqdm import tqdm
import torch.nn.functional as F
import random
import einops as ein
from config import paths, datasets
from MuSSel import fit_cluster_centers
import argparse

parser = argparse.ArgumentParser(description="Visual place recognition with multi-scale superpixel clustering.")
parser.add_argument('dataset', type=str, help='Dataset name (e.g., 17places).')
parser.add_argument('--device', type=str, default='cuda', help='Device to run computations.')

args = parser.parse_args()
dataset_config = datasets.get(args.dataset, {})

images_path = paths["images"]
features_path = paths["features"]
c_center_path = paths["cluster_centers"]

# Load DINO features
dino_feature_path = os.path.join(features_path, dataset_config["dino_h5_filename_r"])
dino_file = h5py.File(dino_feature_path, "r")
keys = list(dino_file.keys())

db_descriptors = []
sample_threshold = 2000
sample_percentage = 0.3

if len(keys) > sample_threshold:
    print(f"Applying sampling for large dataset: {args.dataset}")
    random.seed(42)
    sampled_keys = random.sample(keys, k=int(len(keys) * sample_percentage))
else:
    sampled_keys = keys

for key in tqdm(sampled_keys, desc="Processing keys"):
    original_data = dino_file[key]['features'][()]
    subsampled_data = original_data[:, :, ::2, ::2] if len(keys) > sample_threshold else original_data

    desc_tensor = torch.from_numpy(subsampled_data.reshape(1, 1536, -1)).to(args.device)
    normalized_desc = F.normalize(desc_tensor, dim=1)
    db_descriptors.append(normalized_desc.permute(0, 2, 1).cpu())

db_descriptors = torch.cat(db_descriptors, dim=0)

# Perform K-means clustering
num_clusters = 32
cache_file_path = os.path.join(c_center_path, f"{args.dataset}_c_centers.pt")
print(f"Writing cluster centers to: {cache_file_path}")

fit_cluster_centers(
    args.dataset,
    num_clusters,
    cache_file_path,
    desc_dim=None,
    normalize_descs=True,
    train_descs=ein.rearrange(db_descriptors, "n k d -> (n k) d")
)

print("VLAD clustering completed successfully.")
