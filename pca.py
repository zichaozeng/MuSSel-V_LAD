"""PCA dimensionality reduction for segment VLAD descriptors.

This script fits a PCA model on reference image descriptors to reduce the
dimensionality of VLAD representations. The fitted model is saved and later
applied to both reference and query descriptors during VPR evaluation.

PCA Benefits:
- Reduces computational cost for similarity search
- Removes noise and redundant dimensions
- Often improves retrieval performance through decorrelation

Usage:
    python pca.py <dataset> <experiment> <seg_method> <adj_mode> <adj_hop>
    
Example:
    python pca.py laurel Sp128_ao3_pca slic rs3 3
"""

import os
import torch
import numpy as np
import h5py
from natsort import natsorted
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle
from config import paths, datasets, experiments
from MuSSel import countNumMasksInDataset, load_superpixels_lb, seg_vlad_gpu_single_lb, adj_generator_labels, preload_masks
import argparse

parser = argparse.ArgumentParser(description="PCA for reference images")
parser.add_argument('dataset', type=str, help='Dataset name (e.g., 17places).')
parser.add_argument('experiment', type=str, help='Experiment setting.')
parser.add_argument('seg_method', type=str, help='Segmentation method (e.g., slic, seeds, sam, and fastsam).')
parser.add_argument('adj_mode', type=str, help='Adjacency matrix mode.')
parser.add_argument('adj_hop', type=float, help='Adjacency matrix hop.')
parser.add_argument('--feature_model', type=str, default='dino', help='Feature model to use: dino_ft, dino and clip.')
parser.add_argument('--device', type=str, default='cuda', help='Device to run computations.')

args = parser.parse_args()

# Load dataset and paths configuration
dataset_name = args.dataset
dataset_config = datasets.get(dataset_name, {})
experiment_config = experiments.get(args.experiment, {})

segment_mode = experiment_config["segment_model"]

images_path = paths["images"]
features_path = paths["features"]
segments_path = paths["segments"]
c_center_path = paths["cluster_centers"]
pca_path = paths["pca"]

reference_data_path = os.path.join(images_path, dataset_name, dataset_config["data_subpath1_r"])
query_data_path = os.path.join(images_path, dataset_name, dataset_config["data_subpath2_q"])

c_centers_file = os.path.join(c_center_path, f"{dataset_name}_c_centers.pt")
c_centers = torch.load(c_centers_file)

# Load DINO features
print("Loading DINO features...")
dino_feature_ref_path = os.path.join(features_path, dataset_config[f"{args.feature_model}_h5_filename_r"])

dino_file_ref = h5py.File(dino_feature_ref_path, "r")

# Image indices
ims_sidx, ims_eidx, ims_step = 0, None, 1
ims1_r = natsorted(os.listdir(f'{reference_data_path}'))
ims1_r = ims1_r[ims_sidx:ims_eidx][::ims_step]


dh = dataset_config['cfg']['desired_height'] // 14
dw = dataset_config['cfg']['desired_width'] // 14
idx_matrix = np.empty((dataset_config['cfg']['desired_height'], dataset_config['cfg']['desired_width'], 2)).astype('int32')
for i in range(dataset_config['cfg']['desired_height']):
    for j in range(dataset_config['cfg']['desired_width']):
        idx_matrix[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
ind_matrix = torch.tensor(ind_matrix, device='cuda')

print("Loading masks...")
if segment_mode == "SAM":
    segment_r_path = os.path.join(segments_path, f"{dataset_name}_r_{args.seg_method}_{int(dataset_config['cfg']['desired_width']/2)}.h5")
else:
    segment_r_path = os.path.join(segments_path, f"{dataset_name}_r_{args.seg_method}_{dataset_config['cfg']['desired_width']}.h5")


segment_r_h5 = h5py.File(segment_r_path, "r")

print("Counting number of masks in dataset...")
num_masks_r = countNumMasksInDataset(ims1_r, segment_r_h5, segment_mode)
print("num_masks_r", num_masks_r)

# PCA parameters
accumulated_segments = 0
max_segments = 30000 #50000
global_sampling_rate = min(1, max_segments / num_masks_r)
pca_lower_dim = 1024 # 512
pca_whiten = True
svd_solver = 'arpack'

# PCA model
pca = PCA(n_components=pca_lower_dim, whiten=pca_whiten, svd_solver=svd_solver)

adj_mode = args.adj_mode
order = experiment_config['order']
print("nbr agg order number: ", order)

desc_dim = 1536

segFtVLAD1_list = [] 
imInds1 = np.array([], dtype=int)

print("Processing for reference images...")
for r_id, r_img in tqdm(enumerate(ims1_r), total=len(ims1_r), desc="Processing for reference images..."):
    if segment_mode == "SAM":
        segment_r = preload_masks(segment_r_h5, r_img)
        segment_r = [mask for mask in segment_r if np.any(mask)]
    elif segment_mode.startswith("segments"):
        segment_r = load_superpixels_lb(segment_r_h5, r_img, segment_mode)
    else:
        raise ValueError(f"Unknown segment model: {segment_mode}")
    
    if order: 
        adjMat_r_ind = adj_generator_labels(segment_r, args.adj_hop, segment_mode, adj_mode)
    else:
        adjMat_r_ind = None

    gd = seg_vlad_gpu_single_lb(
        ind_matrix, 
        idx_matrix, 
        dino_file_ref, 
        r_img, 
        segment_r, 
        c_centers, 
        dataset_config['cfg'], 
        segment_mode, 
        desc_dim=desc_dim, 
        adj_mat=adjMat_r_ind
        )

    #
    gd = gd.to(dtype=torch.float32) # Convert to float32 for PCA to keep RAM in check

    current_batch_size = gd.shape[0]
    sample_size = int(current_batch_size * global_sampling_rate)

    if experiment_config["pca"]:
        if sample_size > 0:
            torch.manual_seed(42)
            sample_indices = torch.randperm(current_batch_size)[:sample_size]
            sampled_gd = gd[sample_indices]
            segFtVLAD1_list.append(sampled_gd)
            accumulated_segments += sampled_gd.shape[0]
        else:
            segFtVLAD1_list.append(gd)

    if accumulated_segments >= max_segments:
        break

print("Before cat")
segFtVLAD1 = torch.cat(segFtVLAD1_list, dim=0)
print("After cat") 
del segFtVLAD1_list
print("After del")

if experiment_config["pca"]:
    print("svd solver using : ", svd_solver)
    print("NOTE: This process may take some time depending on the size of the data. \n Please do not exit...")

    pca.fit(segFtVLAD1.numpy())
    pca_model_path = os.path.join(pca_path, f"{dataset_name}_{args.experiment}_{args.seg_method}_{adj_mode}{args.adj_hop}.pkl")

    print("saving PCA.")
    with open(pca_model_path, "wb") as file:
        # pickle.dump(ipca, file)
        pickle.dump(pca, file)

    print("DONE: PCA for reference images (50k randomly sampled segments) and saving to pickle file")
