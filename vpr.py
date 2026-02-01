"""Visual Place Recognition using MuSSel-V.

This script performs visual place recognition by:
1. Loading pre-extracted features and segmentation masks
2. Computing segment-based VLAD descriptors with optional neighbour aggregation
3. Applying optional PCA dimensionality reduction
4. Evaluating recall performance against ground truth

Usage:
    python vpr.py <dataset> <experiment> <seg_method> <adj_mode> <adj_hop> <save_name> \
                  [--feature_model <model>] [--device <device>]
"""

import os
import torch
import numpy as np
import h5py
from natsort import natsorted
from tqdm import tqdm
from config import paths, datasets, experiments
from MuSSel import get_gt, apply_pca_transform_from_pkl, recall_segloc, load_superpixels_lb, getIdxSingleFast_lb, seg_vlad_gpu_single_lb, adj_generator_labels, preload_masks
import argparse

# Command-line argument parser
parser = argparse.ArgumentParser(description="Visual place recognition with multi-scale superpixel clustering.")
parser.add_argument('dataset', type=str, help='Dataset name (e.g., 17places, VPAir, laurel, hawkins).')
parser.add_argument('experiment', type=str, help='Experiment configuration key from config.py.')
parser.add_argument('seg_method', type=str, help='Segmentation method (e.g., slic, seeds, sam, fastsam).')
parser.add_argument('adj_mode', type=str, help='Adjacency matrix generation mode (e.g., cc, rs2).')
parser.add_argument('adj_hop', type=float, help='Adjacency matrix hop distance for neighbour aggregation.')
parser.add_argument('save_name', type=str, help='Name identifier to save the results.')
parser.add_argument('--feature_model', type=str, default='dino', help='Feature model: dino, dino_ft, or clip.')
parser.add_argument('--device', type=str, default='cuda', help='Computation device (cuda or cpu).')

args = parser.parse_args()

# Load dataset and experiment configurations
dataset_name = args.dataset
dataset_config = datasets.get(dataset_name, {})
experiment_config = experiments.get(args.experiment, {})

# Determine segmentation model from experiment configuration
segment_mode = experiment_config["segment_model"]

# Initialise paths from configuration
images_path = paths["images"]
features_path = paths["features"]
segments_path = paths["segments"]
c_center_path = paths["cluster_centers"]
pca_path = paths["pca"]
results_path = paths["results"]

# Construct dataset-specific paths
reference_data_path = os.path.join(images_path, dataset_name, dataset_config["data_subpath1_r"])
query_data_path = os.path.join(images_path, dataset_name, dataset_config["data_subpath2_q"])

# Load pre-computed cluster centres for VLAD encoding
c_centers_file = os.path.join(c_center_path, f"{dataset_name}_c_centers.pt")
c_centers = torch.load(c_centers_file)

# Load pre-extracted DINO feature files (HDF5 format)
dino_feature_ref_path = os.path.join(features_path, dataset_config["dino_h5_filename_r"])
dino_feature_query_path = os.path.join(features_path, dataset_config["dino_h5_filename_q"])

dino_file_ref = h5py.File(dino_feature_ref_path, "r")
dino_file_query = h5py.File(dino_feature_query_path, "r")

# Image selection parameters (for subset processing if needed)
ims_sidx, ims_eidx, ims_step = 0, None, 1

# Load and sort image lists for reference and query sets
ims1_r = natsorted(os.listdir(f'{reference_data_path}'))
ims1_r = ims1_r[ims_sidx:ims_eidx][::ims_step]
ims2_q = natsorted(os.listdir(f'{query_data_path}'))
ims2_q = ims2_q[ims_sidx:ims_eidx][::ims_step]

# Create index matrices for mapping pixel coordinates to DINO feature map positions
# DINOv2 uses a patch size of 14×14 pixels
dh = dataset_config['cfg']['desired_height'] // 14
dw = dataset_config['cfg']['desired_width'] // 14

# Build coordinate mapping matrix (H×W×2) storing feature map indices for each pixel
idx_matrix = np.empty((dataset_config['cfg']['desired_height'], dataset_config['cfg']['desired_width'], 2)).astype('int32')
for i in range(dataset_config['cfg']['desired_height']):
    for j in range(dataset_config['cfg']['desired_width']):
        idx_matrix[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])

# Flatten to linear indices for efficient lookup
ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
ind_matrix = torch.tensor(ind_matrix, device='cuda')

print("Loading masks...")

# Load segmentation masks from HDF5 files
# SAM masks are stored at half resolution, superpixel masks at full resolution
if segment_mode == "SAM":
    segment_r_path = os.path.join(segments_path, f"{dataset_name}_r_{args.seg_method}_{int(dataset_config['cfg']['desired_width']/2)}.h5")
    segment_q_path = os.path.join(segments_path, f"{dataset_name}_q_{args.seg_method}_{int(dataset_config['cfg']['desired_width']/2)}.h5")
else:
    segment_r_path = os.path.join(segments_path, f"{dataset_name}_r_{args.seg_method}_{dataset_config['cfg']['desired_width']}.h5")
    segment_q_path = os.path.join(segments_path, f"{dataset_name}_q_{args.seg_method}_{dataset_config['cfg']['desired_width']}.h5")

segment_r_h5 = h5py.File(segment_r_path, "r")
segment_q_h5 = h5py.File(segment_q_path, "r")

# Neighbour aggregation order from experiment configuration
order = experiment_config['order']
print("nbr agg order number: ", order)

# Adjacency matrix generation mode
adj_mode = args.adj_mode

# Storage for segment ranges and indices
segRange1 = []  # Segment ranges for reference images
segRange2 = []  # Segment ranges for query images

# Descriptor dimensions
desc_dim = 1536  # DINOv2-g feature dimension
vlad_dim = 32 * desc_dim  # VLAD dimension (32 clusters × 1536)

total_segments = 0  # Counter for sampled segments
max_segments = 50000  # Maximum segments to sample in total

batch_size = 100  # Number of images to process before applying PCA

# Descriptor storage lists
segFtVLAD1_list = []  # Reference segment VLAD descriptors (no PCA)
segFtVLAD1Pca_list = []  # Reference segment VLAD descriptors (with PCA)
batch_descriptors_r = []  # Batch accumulator for reference descriptors

segFtVLAD2_list = []  # Query segment VLAD descriptors (no PCA)
segFtVLAD2Pca_list = []  # Query segment VLAD descriptors (with PCA)
batch_descriptors_q = []  # Batch accumulator for query descriptors 

# Initialise image index arrays
imInds1 = np.array([], dtype=int)
imInds2 = np.array([], dtype=int)

# Load ground truth correspondences for evaluation
gt = get_gt(
    dataset=dataset_name,
    cfg=dataset_config,
    workdir_data=images_path,
    ims1_r=ims1_r,
    ims2_q=ims2_q,
    func_vpr_module=None
)

# Path to PCA transformation model
pca_model_path = os.path.join(pca_path, f"{dataset_name}_{args.experiment}_{args.seg_method}_{adj_mode}{args.adj_hop}.pkl")

# Process reference images to extract segment-based VLAD descriptors
for r_id, r_img in tqdm(enumerate(ims1_r), total=len(ims1_r), desc="Processing for reference images..."):
    # Load segmentation masks for current image
    if segment_mode == "SAM":
        # SAM produces variable number of masks per image
        segment_r = preload_masks(segment_r_h5, r_img)
        segment_r = [mask for mask in segment_r if np.any(mask)]  # Filter empty masks
    elif segment_mode.startswith("segments"):
        # Superpixel segmentation (SLIC/SEEDS) produces fixed grid
        segment_r = load_superpixels_lb(segment_r_h5, r_img, segment_mode)
    else:
        raise ValueError(f"Unknown segment model: {segment_mode}")
    
    # Get segment indices and image indices for this image
    imInds1_ind, regInds1_ind = getIdxSingleFast_lb(r_id, segment_r, segment_mode, minArea=experiment_config['minArea'])
    imInds1 = np.concatenate((imInds1, imInds1_ind))

    # Generate adjacency matrix for neighbour aggregation if enabled
    if order:
        adjMat_r_ind = adj_generator_labels(segment_r, args.adj_hop, segment_mode, adj_mode)
    else:
        adjMat_r_ind = None

    # Compute segment-based VLAD descriptor with optional neighbour aggregation
    gd = seg_vlad_gpu_single_lb(
        ind_matrix,  # Pixel to feature map index
        idx_matrix,  # 2D coordinate matrix
        dino_file_ref,  # Pre-extracted DINO features
        r_img,  # Image key
        segment_r,  # Segmentation masks
        c_centers,  # VLAD cluster centres
        dataset_config['cfg'],  # Dataset configuration
        segment_mode,  # Segmentation type
        desc_dim=1536,  # Feature dimension
        adj_mat=adjMat_r_ind  # Adjacency matrix
    )
    
    if experiment_config["pca"]:
        batch_descriptors_r.append(gd)
        if (r_id + 1) % batch_size == 0 or (r_id + 1) == len(ims1_r): ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
            segFtVLAD1_batch = torch.cat(batch_descriptors_r, dim=0)
            # Reset batch descriptors for the next batch
            batch_descriptors_r = []

            print("Applying PCA to batch descriptors... at image ", r_id)
            segFtVLAD1Pca_batch  = apply_pca_transform_from_pkl(segFtVLAD1_batch, pca_model_path)
            del segFtVLAD1_batch

            segFtVLAD1Pca_list.append(segFtVLAD1Pca_batch)
            

    else:
        segFtVLAD1_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time


if experiment_config["pca"]:
    segFtVLAD1 = torch.cat(segFtVLAD1Pca_list, dim=0)
    print("Shape of segment descriptors with PCA:", segFtVLAD1.shape)
    del segFtVLAD1Pca_list
else:
    segFtVLAD1 = torch.cat(segFtVLAD1_list, dim=0)
    print("Shape of segment descriptors without PCA:", segFtVLAD1.shape)
    del segFtVLAD1_list

for i in range(imInds1[-1]+1):
    segRange1.append(np.where(imInds1==i)[0])

for q_id, q_img in tqdm(enumerate(ims2_q), total=len(ims2_q), desc="Processing for query images..."):
    if segment_mode == "SAM":
        segment_q = preload_masks(segment_q_h5, q_img)
        segment_q = [mask for mask in segment_q if np.any(mask)]
    elif segment_mode.startswith("segments"):
        segment_q = load_superpixels_lb(segment_q_h5, q_img, segment_mode)
    else:
        raise ValueError(f"Unknown segment model: {segment_mode}")

    imInds2_ind, regInds2_ind = getIdxSingleFast_lb(q_id,segment_q,segment_mode,minArea=experiment_config['minArea'])
    imInds2 = np.concatenate((imInds2, imInds2_ind))

    if order: 
        adjMat_q_ind = adj_generator_labels(segment_q, args.adj_hop, segment_mode, adj_mode)
    else:
        adjMat_q_ind = None

    gd = seg_vlad_gpu_single_lb(
        ind_matrix, 
        idx_matrix, 
        dino_file_query, 
        q_img, 
        segment_q, 
        c_centers, 
        dataset_config['cfg'], 
        segment_mode, 
        desc_dim=1536, 
        adj_mat=adjMat_q_ind
        )
        
    if experiment_config["pca"]:
        batch_descriptors_q.append(gd)
        if (q_id + 1) % batch_size == 0 or (q_id + 1) == len(ims2_q): ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
            segFtVLAD2_batch = torch.cat(batch_descriptors_q, dim=0)
            # Reset batch descriptors for the next batch
            batch_descriptors_q = []

            print("query: Applying PCA to batch descriptors... at image ", q_id)
            segFtVLAD2Pca_batch  = apply_pca_transform_from_pkl(segFtVLAD2_batch, pca_model_path)
            del segFtVLAD2_batch

            segFtVLAD2Pca_list.append(segFtVLAD2Pca_batch)
            

    else:
        segFtVLAD2_list.append(gd) #imfts_batch same as gd here, in the full image function, it is for 100 images at a time


if experiment_config["pca"]:
    segFtVLAD2 = torch.cat(segFtVLAD2Pca_list, dim=0)
    print("Shape of q segment descriptors with PCA:", segFtVLAD2.shape)
    del segFtVLAD2Pca_list
else:
    segFtVLAD2 = torch.cat(segFtVLAD2_list, dim=0)
    print("Shape of q segment descriptors without PCA:", segFtVLAD2.shape)
    del segFtVLAD2_list

for i in range(imInds2[-1]+1):
    segRange2.append(np.where(imInds2==i)[0])

recall_results = recall_segloc(results_path, dataset_name, experiment_config, args.experiment, segFtVLAD1, segFtVLAD2, gt, segRange2, imInds1, False)


# save recall results to txt file
recall_results_file = os.path.join(results_path, f"{dataset_name}_{args.experiment}_{args.seg_method}_{adj_mode}{args.adj_hop}_{args.save_name}_pt_recall.txt")
with open(recall_results_file, "w") as f:
    for v in recall_results:
        f.write(f"{v}\n")
    
# print("VLAD + PCA RESULTS for segloc for dataset config: ", dataset_config, " ::: experiment config ::: ", experiment_config, " using pca file : ", pca_model_path, "experiment_name: ", experiment_name)
