"""MuSSel-V: Multi-Scale Superpixel-based Visual Place Recognition.

This module provides the core functionality for segment-based visual place recognition:
- Image segmentation (SLIC, SEEDS, SAM)
- Feature extraction (DINOv2, CLIP)
- VLAD encoding with neighbour aggregation
- Ground truth generation and evaluation
- Utility functions for PCA, clustering, and adjacency matrix generation
"""

from .segmenters import process_slic_to_h5, process_seeds_to_h5
from .extractors import DinoFeatureExtractor, save_features_to_h5
from .utils import fit_cluster_centers, load_superpixels_lb, countNumMasksInDataset, getIdxSingleFast_lb, apply_pca_transform_from_pkl, preload_masks, adj_generator_labels
from .vlads import seg_vlad_gpu_single_lb
from .ground_truth import get_gt
from .evaluators import recall_segloc