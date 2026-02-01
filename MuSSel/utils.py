"""Utility functions for segmentation, clustering, and feature processing.

This module provides essential utilities for MuSSel-V:

Clustering and PCA:
- fit_cluster_centers: K-means clustering with caching
- apply_pca_transform_from_pkl: Apply pre-fitted PCA transformation

Segmentation Loading:
- load_superpixels_lb: Load superpixel masks (single or multi-scale)
- preload_masks: Load SAM masks
- countNumMasksInDataset: Count total segments in dataset

Adjacency Matrix Generation:
- adj_generator_labels: Generate adjacency matrices for neighbour aggregation
- nbrMasksAGGFastSingle_cc: Contact-based adjacency (dilation)
- nbrMasksAGGFastSingle_rs2: Radius-based adjacency (centroid distance)

Index Management:
- getIdxSingleFast_lb: Generate image and segment indices
"""

import os
import torch
import numpy as np
import fast_pytorch_kmeans as fpk
from natsort import natsorted
from tqdm import tqdm
from scipy.spatial import Delaunay, KDTree
from math import sqrt, pi
import torch.nn.functional as F
import pickle
from typing import Optional
import cv2
import random
from scipy.ndimage import center_of_mass


def is_valid_cache(dataset_name, cache_file_path: Optional[str]) -> bool:
    cache_dir = os.path.dirname(cache_file_path)
    if not cache_dir:
        print("No cache directory set")
        return False
    if not os.path.exists(cache_dir):
        print("Cache directory doesn't exist")
        return False
    if os.path.exists(cache_file_path):
        return True
    print("Cache directory doesn't contain cluster centers")
    return False

def fit_cluster_centers(dataset_name, num_clusters: int, cache_file_path: Optional[str], desc_dim: Optional[int], 
                        normalize_descs: bool = True, train_descs: Optional[np.ndarray] = None):
    """
    Fit K-means cluster centres for VLAD encoding, with caching support.
    
    If cached cluster centres exist, they are loaded. Otherwise, K-means is fitted
    on the provided training descriptors and cached for future use.
    
    Args:
        dataset_name (str): Dataset identifier for logging.
        num_clusters (int): Number of K-means clusters (typically 32 or 64).
        cache_file_path (Optional[str]): Path to save/load cluster centres (.pt file).
        desc_dim (Optional[int]): Descriptor dimension (inferred if None).
        normalize_descs (bool): Whether to L2-normalise descriptors before clustering.
        train_descs (Optional[np.ndarray]): Training descriptors (N × D) if fitting.
        
    Returns:
        fpk.KMeans: Fitted K-means object with centroids.
        
    Raises:
        ValueError: If no cached centres exist and train_descs is None.
    """
    kmeans = fpk.KMeans(num_clusters, mode="cosine")
    cache_dir = os.path.dirname(cache_file_path)
    if is_valid_cache(dataset_name, cache_file_path):
        print("Using cached cluster centers")
        cluster_centers = torch.load(cache_file_path)
        kmeans.centroids = cluster_centers
        if desc_dim is None:
            desc_dim = cluster_centers.shape[1]
            print(f"Descriptor dimension set to {desc_dim}")
    else:
        if train_descs is None:
            raise ValueError("Training descriptors required if no valid cache available")

        train_descs = torch.tensor(train_descs, dtype=torch.float32) if isinstance(train_descs, np.ndarray) else train_descs
        desc_dim = desc_dim or train_descs.shape[1]

        if normalize_descs:
            train_descs = F.normalize(train_descs)

        kmeans.fit(train_descs)
        cluster_centers = kmeans.centroids

        if cache_dir:
            print("Saving cluster centers to cache")
            torch.save(cluster_centers, cache_file_path)

    return kmeans

def apply_pca_transform_from_pkl(data_tensor, pca_model_path):
    """
    Load a pre-fitted PCA model and apply transformation to data.
    
    This function is used to apply dimensionality reduction to VLAD descriptors,
    typically reducing from 49,152D (32 clusters × 1536D) to a lower dimension
    for more efficient retrieval.

    Args:
        data_tensor (torch.Tensor): Input data to transform (N × D).
        pca_model_path (str): Path to pickled sklearn PCA model.

    Returns:
        torch.Tensor: Transformed data with reduced dimensionality.
    """
    # Ensure data is on CPU and converted to a NumPy array for PCA transformation
    data_np = data_tensor.cpu().numpy()

    # Load the fitted PCA model from disk
    print(pca_model_path)
    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file)
        
    # Apply the PCA transform to the data
    transformed_data_np = pca.transform(data_np)

    # Convert the transformed data back to a PyTorch tensor
    transformed_data_tensor = torch.from_numpy(transformed_data_np)

    return transformed_data_tensor
#######################################################################

#######################################################################
def normalizeFeat(rfts):
    rfts = np.array(rfts).reshape([len(rfts),-1])
    rfts /= np.linalg.norm(rfts,axis=1)[:,None]
    return rfts

def load_lb_n_segment(segment_h5, image_key, n_segment):
    segments = segment_h5[image_key][n_segment][:]
    return segments

def load_superpixels_lb(segment_h5, image_key, n_segment):
    """
    Load superpixel segmentation masks from HDF5 file.
    
    Supports both single-scale and multi-scale (mixed) segmentations.
    For multi-scale, loads multiple resolutions and returns as dictionary.
    
    Args:
        segment_h5 (h5py.File): HDF5 file containing segmentation masks.
        image_key (str): Image identifier within the HDF5 file.
        n_segment (str): Segmentation scale identifier:
            - 'segments_16', 'segments_32', etc. for single scale
            - 'segments_mixed' for multi-scale (64, 128, 256)
            
    Returns:
        np.ndarray or dict: Segmentation mask(s).
            - Single scale: (H × W) array with segment labels
            - Multi-scale: dict mapping scale names to mask arrays
    """
    if n_segment == 'segments_mixed':
        # n_segment = ["segments_64", "segments_128"]
        # n_segment = ["segments_128", "segments_256"]
        n_segment = ["segments_64", "segments_128", "segments_256"]
        # n_segment = ["segments_16", "segments_32", "segments_64"]
        mask_segs = {}
        for n_seg in n_segment:
            mask_segs[n_seg] = load_lb_n_segment(segment_h5, image_key, n_seg)
    else:
        mask_segs = load_lb_n_segment(segment_h5, image_key, n_segment)
    return mask_segs

def countNumMasks(ims, masks_in, n_segment):
    count = 0
    # print(masks_in.keys())
    for im_name in tqdm(ims, desc="Counting num of masks in dataset"):
        # Directly constructing the path to the masks for the current image
        mask_path = f"{im_name}"
        count += len(np.unique(masks_in[mask_path][n_segment][:]))
    return count

def countNumMasksInDataset(ims, masks_in, n_segment):
    if n_segment == 'segments_mixed':
        # n_segment = ["segments_64", "segments_128"]
        # n_segment = ["segments_128", "segments_256"]
        n_segment = ["segments_64", "segments_128", "segments_256"]
        # n_segment = ["segments_16", "segments_32", "segments_64"]
        count = 0
        for n_seg in n_segment:
            count += countNumMasks(ims, masks_in, n_seg)
    elif n_segment.startswith("segments"):
        count = countNumMasks(ims, masks_in, n_segment)
    elif n_segment == "SAM":
        count = countNumMasks_sam(ims, masks_in)
    else:
        raise ValueError(f"Unknown segment model: {n_segment}")
    return count

def countNumMasks_sam(ims, masks_in):
    count = 0
    for im_name in tqdm(ims, desc="Counting num of masks in dataset"):
        # Directly constructing the path to the masks for the current image
        mask_path = f"{im_name}/masks/"
        if mask_path in masks_in:
            mask_keys = natsorted(masks_in[mask_path].keys())
            count += len(mask_keys)
    return count

def preload_masks(masks_in, image_key):
    masks_path = f"{image_key}/masks/"
    mask_keys = natsorted(masks_in[masks_path].keys())
    masks_seg = [masks_in[masks_path + k]['segmentation'][()] for k in mask_keys]
    return masks_seg

#############################################################
def nbrMasksAGGFastSingle_cc(masks_seg, order):
    """
    Generate adjacency matrix using contact-based connectivity (dilation).
    
    Two segments are considered neighbours if they touch after morphological dilation.
    The order parameter controls multi-hop connectivity via matrix power.
    
    Args:
        masks_seg (list): List of binary segmentation masks.
        order (int): Number of hops for neighbour aggregation (1 = direct neighbours).
        
    Returns:
        torch.Tensor: Binary adjacency matrix (N × N) on CUDA, dtype double.
    """
    num_masks = len(masks_seg)
    adj_matrix = np.zeros((num_masks, num_masks), dtype=int)

    kernel = np.ones((3, 3), np.uint8)

    for i in range(num_masks):
        mask_i = masks_seg[i].astype(np.uint8)
        dilated_mask_i = cv2.dilate(mask_i, kernel, iterations=1) 

        for j in range(num_masks):
            if i != j:
                mask_j = masks_seg[j].astype(np.uint8)
                if np.any(dilated_mask_i & mask_j): 
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1 

    distance = order 
    A = torch.tensor(adj_matrix, dtype=torch.int32)
    A_d = (A.matrix_power(distance) > 0)  
    A_d.fill_diagonal_(False)  
    
    adj_tensor = A_d
    return adj_tensor.to("cuda").double()

def nbrMasksAGGFastMixed_cc(mixed_masks_seg, order, segment_mode):
    if segment_mode == 'segments_mixed':
        adj_mats = {}

        for n_segment, masks_seg in mixed_masks_seg.items():
            adj_mats[n_segment] = nbrMasksAGGFastSingle_cc(masks_seg, order)
    else:
        adj_mats = nbrMasksAGGFastSingle_cc(mixed_masks_seg, order)

    return adj_mats

def nbrMasksAGGFastSingle_rs2(masks_seg, order):
    """
    Generate adjacency matrix using radius-based connectivity (centroid distance).
    
    Two segments are neighbours if their centroids are within a radius determined by
    the order parameter and estimated patch size. Uses KD-tree for efficient lookup.
    
    Args:
        masks_seg (list): List of binary segmentation masks.
        order (float): Radius multiplier (larger = more neighbours).
        
    Returns:
        torch.Tensor: Binary adjacency matrix (N × N) on CUDA, dtype double.
    """
    path_size = sqrt(len(masks_seg[0])*len(masks_seg[0][0])/len(masks_seg))
    basic_radius = path_size*sqrt(2)
    radius = basic_radius*order

    mask_cords = np.array([np.array(np.nonzero(mask_seg)).mean(1)[::-1] for mask_seg in masks_seg])  
    tree = KDTree(mask_cords)

    num_masks = len(mask_cords)
    adj_matrix = np.zeros((num_masks, num_masks), dtype=int)

    for i in range(num_masks):
        neighbors = tree.query_ball_point(mask_cords[i], r=radius)  

        for j in neighbors:
            if i != j:
                adj_matrix[i, j] = 1

    A = torch.tensor(adj_matrix, dtype=torch.int32)
    A_d = (A.matrix_power(1) > 0)
    A_d.fill_diagonal_(False)

    return A_d.to("cuda").double()

def nbrMasksAGGFastMixed_rs2(mixed_masks_seg, order, segment_mode):
    if segment_mode == 'segments_mixed':
        adj_mats = {}

        for n_segment, masks_seg in mixed_masks_seg.items():
            adj_mats[n_segment] = nbrMasksAGGFastSingle_rs2(masks_seg, order)
    else:
        adj_mats = nbrMasksAGGFastSingle_rs2(mixed_masks_seg, order)

    return adj_mats

def nbrMasksAGGFastSingle_rs3(masks_seg, order):
    radius = sqrt((len(masks_seg[0][0])*len(masks_seg[0])*order)/pi)

    mask_cords = np.array([np.array(np.nonzero(mask_seg)).mean(1)[::-1] for mask_seg in masks_seg])  
    tree = KDTree(mask_cords)

    num_masks = len(mask_cords)
    adj_matrix = np.zeros((num_masks, num_masks), dtype=int)

    for i in range(num_masks):
        neighbors = tree.query_ball_point(mask_cords[i], r=radius)  

        for j in neighbors:
            if i != j:
                adj_matrix[i, j] = 1

    A = torch.tensor(adj_matrix, dtype=torch.int32)
    A_d = (A.matrix_power(1) > 0)
    A_d.fill_diagonal_(False)

    return A_d.to("cuda").double()

def nbrMasksAGGFastMixed_rs3(mixed_masks_seg, order, segment_mode):
    if segment_mode == 'segments_mixed':
        adj_mats = {}

        for n_segment, masks_seg in mixed_masks_seg.items():
            adj_mats[n_segment] = nbrMasksAGGFastSingle_rs3(masks_seg, order)
    else:
        adj_mats = nbrMasksAGGFastSingle_rs3(mixed_masks_seg, order)

    return adj_mats

def nbrMasksAGGFastSingle_rd(masks_seg, num_neighbors):
    num_masks = len(masks_seg)
    random_adj_matrix = np.zeros((num_masks, num_masks), dtype=bool)  

    for i in range(num_masks):
        neighbors = random.sample([j for j in range(num_masks) if j != i], num_neighbors) 
        for j in neighbors:
            random_adj_matrix[i, j] = True
    random_adj_tensor = torch.tensor(random_adj_matrix, dtype=torch.bool)

    return random_adj_tensor.to("cuda").double()

def nbrMasksAGGFastMixed_rd(mixed_masks_seg, order, segment_mode):
    nums_neighbors = {'segments_256': 36, 'segments_128': 32, 'segments_64': 24}
    if segment_mode == 'segments_mixed':
        adj_mats = {}

        for n_segment, masks_seg in mixed_masks_seg.items():
            num_neighbors = nums_neighbors[n_segment]
            adj_mats[n_segment] = nbrMasksAGGFastSingle_rd(masks_seg, num_neighbors)
    else:
        adj_mats = nbrMasksAGGFastSingle_rd(mixed_masks_seg, nums_neighbors[segment_mode])

    return adj_mats

#############################################################
def getIdxSingleFast_lb(img_idx, label, segment_mode, minArea=400):
    imInds = []
    regIndsIm = []

    if segment_mode == "segments_mixed":
        for n_segment, lb in label.items():
            unique_labels = np.unique(lb)

            for idx in unique_labels:
                imInds.append(img_idx) 
                regIndsIm.append(idx)  
    else:
        unique_labels = np.unique(label)

        for idx in unique_labels:
            imInds.append(img_idx) 
            regIndsIm.append(idx)  

    return np.array(imInds), regIndsIm


def nbrMasksAGGFastSingle_rs3_labels(masks_seg, order):
    unique_labels = np.unique(masks_seg)
    mask_cords = np.array([center_of_mass(masks_seg == label) for label in unique_labels])
    mask_cords = mask_cords[:, ::-1]

    radius = sqrt((masks_seg.shape[0] * masks_seg.shape[1] *order)/pi)

    tree = KDTree(mask_cords)
    num_masks = len(mask_cords)
    adj_matrix = np.zeros((num_masks, num_masks), dtype=int)

    for i in range(num_masks):
        neighbors = tree.query_ball_point(mask_cords[i], r=radius)  
        for j in neighbors:
            if i != j:
                adj_matrix[i, j] = 1

    A = torch.tensor(adj_matrix, dtype=torch.int32)
    A_d = (A.matrix_power(1) > 0)
    A_d.fill_diagonal_(False)

    return A_d.to("cuda").double()

def nbrMasksAGGFastMixed_rs3_labels(mixed_masks_seg, order, segment_mode):
    if segment_mode == 'segments_mixed':
        adj_mats = {}

        for n_segment, masks_seg in mixed_masks_seg.items():
            adj_mats[n_segment] = nbrMasksAGGFastSingle_rs3_labels(masks_seg, order)
    else:
        adj_mats = nbrMasksAGGFastSingle_rs3_labels(mixed_masks_seg, order)

    return adj_mats

#############################################################
def getNbrsDelaunay_lb(tri,v):
    indptr, indices = tri.vertex_neighbor_vertices
    v_nbrs = indices[indptr[v]:indptr[v+1]]
    return v_nbrs

def nbrMasksAGGFastSingle_tri_labels(labels, order):
    n = labels.max() + 1  
    h, w = labels.shape

    unique_labels = np.unique(labels)

    mask_cords = np.array([center_of_mass(labels == label) for label in unique_labels])

    adj_mat = torch.zeros((len(mask_cords), len(mask_cords)))

    if n > 3:
        tri = Delaunay(mask_cords)
        for v in range(n):
            nbrsList = getNbrsDelaunay_lb(tri, v)
            adj_mat[v, [v] + nbrsList.tolist()] = 1

        adj_mat_power = adj_mat.clone()
        for _ in range(order - 1):
            adj_mat_power = adj_mat_power @ adj_mat
        adj_mat = adj_mat_power.bool()

    else:
        nbr_list = list(range(min(n, 2)))
        for v in range(n):
            adj_mat[v, nbr_list] = 1
        adj_mat = adj_mat.bool()
        
    return adj_mat.to("cuda").double()


def nbrMasksAGGFastMixed_lb_tri(mixed_masks_seg, order, segment_mode):
    if segment_mode == 'segments_mixed':
        adj_mats = {}

        for n_segment, masks_seg in mixed_masks_seg.items():
            adj_mats[n_segment] = nbrMasksAGGFastSingle_tri_labels(masks_seg, order)
    else:
        adj_mats = nbrMasksAGGFastSingle_tri_labels(mixed_masks_seg, order)

    return adj_mats

def adj_generator_labels(segment_r, order, segment_mode, adj_mode):
    \"\"\"\n    Generate adjacency matrix for segment-based neighbour aggregation.\n    \n    Dispatches to appropriate adjacency generation method based on mode.\n    Supports both single-scale and multi-scale segmentations.\n    \n    Args:\n        segment_r (np.ndarray or dict): Segmentation label map(s).\n        order (int or float): Neighbour connectivity order.\n        segment_mode (str): Segmentation type ('segments_N' or 'segments_mixed').\n        adj_mode (str): Adjacency mode:\n            - 'rs3': Radius-based with fixed scaling\n            - 'mes': Mean Euclidean shift-based connectivity\n            - 'tri': Triangulation-based connectivity\n            \n    Returns:\n        torch.Tensor or dict: Adjacency matrix/matrices on CUDA.\n    \"\"\"\n    if segment_mode.startswith(\"segments\"):\n        if adj_mode == \"rs3\":\n            adjMat_r_ind = nbrMasksAGGFastMixed_rs3_labels(segment_r, order, segment_mode)\n        elif adj_mode == \"mes\":\n            adjMat_r_ind = nbrMasksAGGFastMixed_rs3_mes(segment_r, order, segment_mode)\n        elif adj_mode == \"tri\":\n            order = int(order)\n            adjMat_r_ind = nbrMasksAGGFastMixed_lb_tri(segment_r, order, segment_mode)\n        # elif adj_mode == \"rs\":\n        else:\n            raise ValueError(f\"Unknown adj_mode: {adj_mode}\")\n    return adjMat_r_ind