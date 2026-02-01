"""VLAD encoding with segment-based aggregation and neighbour pooling.

This module implements Vector of Locally Aggregated Descriptors (VLAD) encoding
specifically designed for segment-based visual place recognition. Key features:

- Segment-aware VLAD encoding using pre-computed cluster centres
- Multi-scale superpixel support (single or mixed scales)
- Optional neighbour aggregation via adjacency matrices
- GPU-accelerated computation using PyTorch

The VLAD residuals are computed per segment and aggregated across K-means clusters,
producing compact yet discriminative representations for place recognition.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
import time


def seg_vlad_gpu_single_lb(ind, idx, desc_path_in, img_key, segMask, c_centers, cfg, segment_mode, desc_dim=1536, adj_mat=None):
    """
    Compute segment-based VLAD descriptors for a single image with label-based segmentation.
    
    This function:
    1. Loads pre-extracted features from HDF5
    2. Maps features to segments using the segmentation mask
    3. Assigns features to nearest cluster centres
    4. Computes residuals and aggregates per segment
    5. Applies optional neighbour aggregation via adjacency matrix
    
    Args:
        ind (torch.Tensor): Flattened pixel indices for feature map lookup.
        idx (np.ndarray): 2D coordinate matrix for pixel-to-feature mapping.
        desc_path_in (h5py.File): HDF5 file containing pre-extracted features.
        img_key (str): Image identifier key in the HDF5 file.
        segMask (np.ndarray or dict): Segmentation mask(s) - array for single scale,
                                       dict for multi-scale.
        c_centers (torch.Tensor): Cluster centres for VLAD (K × D).
        cfg (dict): Dataset configuration containing image dimensions.
        segment_mode (str): Segmentation type ('segments_N', 'segments_mixed', 'SAM').
        desc_dim (int): Feature descriptor dimension (default: 1536 for DINOv2-g).
        adj_mat (torch.Tensor or dict, optional): Adjacency matrix for neighbour aggregation.
        
    Returns:
        torch.Tensor: Segment VLAD descriptors (N_segments × (K * desc_dim)).
    """
    dh = cfg['desired_height'] // 14
    dw = cfg['desired_width'] // 14

    dino_desc = torch.from_numpy(desc_path_in[img_key]['features'][()]) 
    total_elements = dino_desc.shape[2] * dino_desc.shape[3]
    dino_desc = dino_desc.reshape(1, desc_dim, total_elements).to('cuda')
    dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)

    if segment_mode == "segments_mixed":
        mask_idx = {}
        for n_segment, masks_seg in segMask.items():
            N = masks_seg.max().item() + 1
            segLabel = torch.from_numpy(masks_seg).to('cuda')
            segLabel_flat = segLabel.flatten()
            mask_idx[n_segment] = torch.zeros((N, dh * dw), device='cuda').bool()
            mask_idx[n_segment][segLabel_flat, ind] = True
            mask_idx[n_segment] = mask_idx[n_segment].double()
    else:
        N = segMask.max().item() + 1
        segLabel = torch.from_numpy(segMask).to('cuda')
        segLabel_flat = segLabel.flatten()
        mask_idx = torch.zeros((N, dh * dw), device='cuda').bool()
        mask_idx[segLabel_flat, ind] = True
        mask_idx = mask_idx.double()

    reg_feat_per = dino_desc_norm.permute(0, 2, 1)
    # if segment_mode == "SAM":
    #     gd = vlad_single_SAM(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx, adj_mat.to("cuda"))
    # elif segment_mode == "Multi-SP":
    #     gd = vlad_single(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx)
    # else:
    #     gd = vlad_single(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx)

    gd = vlad_single_lb(reg_feat_per.squeeze(), c_centers.to('cuda'), idx, mask_idx, adj_mat, segment_mode)

    return gd.cpu()

def vlad_single_lb(query_descs, c_centers, idx, masks, adj_mat, segment_mode):
    """
    Compute VLAD encoding for segments with optional neighbour aggregation.
    
    Args:
        query_descs (torch.Tensor): Feature descriptors (N_features × D).
        c_centers (torch.Tensor): Cluster centres (K × D).
        idx: Index matrix (not used in current implementation).
        masks (torch.Tensor or dict): Binary masks for segments.
        adj_mat (torch.Tensor or dict, optional): Adjacency matrix for aggregation.
        segment_mode (str): Segmentation type identifier.
        
    Returns:
        torch.Tensor: VLAD descriptors per segment (N_segments × (K * D)).
    """
    num_clusters = 32  # Number of VLAD clusters
    
    # Normalise cluster centres and assign features to nearest cluster
    c_centers_norm = F.normalize(c_centers, dim=1).to('cuda')
    labels = torch.argmax(query_descs @ c_centers_norm.T, dim=1)
    
    # Compute residuals (feature - assigned cluster centre)
    residuals = query_descs - c_centers[labels]

    if adj_mat is not None:
        n_vlad = vlad_matmuls_per_cluster_lb(num_clusters,masks,residuals.double(),labels,segment_mode,adj_mat)
    else:
        n_vlad = vlad_matmuls_per_cluster_lb(num_clusters,masks,residuals.double(),labels,segment_mode)

    return n_vlad

def vlad_matmuls_per_cluster_lb(num_c, masks, res, clus_labels, segment_mode, adjMat=None, device='cuda'):
    """
    Aggregate VLAD residuals per cluster with optional neighbour aggregation.
    
    This function:
    1. Creates adjacency matrix (identity if not provided)
    2. For each cluster, aggregates residuals within segments
    3. Applies optional neighbour pooling via adjacency matrix
    4. Normalises per-cluster and final descriptors
    
    Args:
        num_c (int): Number of clusters (K).
        masks (torch.Tensor or dict): Segment masks (N_segments × N_features).
        res (torch.Tensor): Feature residuals (N_features × D).
        clus_labels (torch.Tensor): Cluster assignment per feature (N_features,).
        segment_mode (str): 'segments_mixed' for multi-scale, else single scale.
        adjMat (torch.Tensor or dict, optional): Adjacency matrix (N_segments × N_segments).
        device (str): Computation device.
        
    Returns:
        torch.Tensor: Normalised VLAD descriptors (N_segments × (K * D)).
    """
    num_m = len(masks)

    # if adjMat is None:
    #     adjMat = torch.eye(num_m,dtype=masks.dtype,device=masks.device)

    if adjMat is None:
        if segment_mode == "segments_mixed":
            adjMat = {}
            for n_segment, masks_seg in masks.items():
                num_m = len(masks_seg)
                adjMat[n_segment] = torch.eye(num_m,dtype=masks_seg.dtype,device=masks_seg.device)
        else:
            num_m = len(masks)
            adjMat = torch.eye(num_m,dtype=masks.dtype,device=masks.device)


    if segment_mode == "segments_mixed":
        vlads_mixed = []
        for n_segment, masks_seg in masks.items():
            mask = masks_seg
            vlads = []
            for li in range(num_c):
                inds_li = torch.where(clus_labels==li)[0].to(device)
                masks_nbrAgg = (adjMat[n_segment] @ mask[:,inds_li])
                vlad = masks_nbrAgg.bool().to(mask.dtype) @ res[inds_li,:]
                vlad = F.normalize(vlad, dim=1)
                vlads.append(vlad)
            vlads = torch.stack(vlads).permute(1,0,2).reshape(len(mask),-1)
            vlads = F.normalize(vlads, dim=1)
            vlads_mixed.append(vlads)
        vlads_final = torch.cat(vlads_mixed, dim=0)
        return vlads_final
    else:
        vlads = []
        for li in range(num_c):
            inds_li = torch.where(clus_labels==li)[0].to(device)
            masks_nbrAgg = (adjMat @ masks[:,inds_li])
            vlad = masks_nbrAgg.bool().to(masks.dtype) @ res[inds_li,:]
            vlad = F.normalize(vlad, dim=1)
            vlads.append(vlad)
        vlads = torch.stack(vlads).permute(1,0,2).reshape(len(masks),-1)
        vlads = F.normalize(vlads, dim=1)
        return vlads
