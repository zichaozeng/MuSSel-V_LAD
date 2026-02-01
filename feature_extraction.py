"""Feature extraction script for visual place recognition.

This script extracts pixel-level features from images using pre-trained vision models.
Currently supported models:
- DINOv2 (dinov2_vitg14): Self-supervised vision transformer
- CLIP (commented out): Contrastive language-image pre-training

Features are saved to HDF5 files for efficient access during VPR experiments.

Usage:
    python feature_extraction.py <dataset> --dino_extract
    
Example:
    python feature_extraction.py laurel --dino_extract
"""

import argparse
import os
from natsort import natsorted
from config import paths, datasets
from MuSSel import DinoFeatureExtractor, save_features_to_h5

parser = argparse.ArgumentParser(
    description="Pixel-level feature extraction using DINOv2 or CLIP."
)

parser.add_argument(
    'dataset', 
    type=str, 
    help='Dataset name (e.g., 17places).'
)

parser.add_argument(
    '--dino_extract', 
    action='store_true', 
    help='Extract DINO features (default: False).'
)
parser.add_argument(
    '--clip_extract', 
    action='store_true', 
    help='Extract CLIP features (default: False).'
)

# Parse arguments
args = parser.parse_args()
dataset = args.dataset
dino_extract = args.dino_extract
clip_extract = args.clip_extract

# Get dataset configuration
dataset_config = datasets.get(dataset, {})
dataset_cfg = dataset_config["cfg"]

# Get paths
images_path = paths["images"]
datapath_r = os.path.join(images_path, dataset, dataset_config["data_subpath1_r"])
datapath_q = os.path.join(images_path, dataset, dataset_config["data_subpath2_q"])

# Target resolution
desired_width = dataset_cfg["desired_width"]
desired_height = dataset_cfg["desired_height"]
dino_width, dino_height = desired_width, desired_height
clip_width, clip_height = desired_width, desired_height

# Extraction configuration
ims_sidx, ims_eidx, ims_step = 0, None, 1
list_all = [
    {
        "datapath": datapath_r,
        "dino_h5_savepath": os.path.join(paths["features"], f"{dataset}_r_dino_{dino_width}.h5"),
        "clip_h5_savepath": os.path.join(paths["features"], f"{dataset}_r_clip_{clip_width}.h5"),
    },
    {
        "datapath": datapath_q,
        "dino_h5_savepath": os.path.join(paths["features"], f"{dataset}_q_dino_{dino_width}.h5"),
        "clip_h5_savepath": os.path.join(paths["features"], f"{dataset}_q_clip_{clip_width}.h5"),
    }
]

# Extract DINO features
if dino_extract:
    for iter_dict in list_all:
        datapath = iter_dict["datapath"]
        dino_h5_savepath = iter_dict["dino_h5_savepath"]
        ims = natsorted(os.listdir(f'{datapath}'))  
        ims = ims[ims_sidx:ims_eidx][::ims_step]

        # DINO model configuration
        dino_pt_model = "dinov2_vitg14"
        cfg_dino = { "desired_width": dino_width, "desired_height": dino_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
                    "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True, \
                    "dino_strides": 4, "use_traced_model": False, "rmin": 0, "DAStoreFull":False, "dinov2": True, "wrap":False, \
                    "resize": True}

        print("DINO extraction started...")
        dino_exractor = DinoFeatureExtractor(dino_pt_model, 31, 'value', device='cuda',norm_descs=False)
        save_features_to_h5(dino_h5_savepath, cfg_dino, ims, dino_exractor, image_folder=datapath)
    print("\n \n DINO EXTRACTED DONE \n \n ")

# Extract CLIP features
# if clip_extract:
#     for iter_dict in list_all:
#         datapath = iter_dict["datapath"]
#         clip_h5_savepath = iter_dict["clip_h5_savepath"]
#         ims = natsorted(os.listdir(f'{datapath}'))
#         ims = ims[ims_sidx:ims_eidx][::ims_step]
#         clip_pt_model = "ViT-L/14"

#         # CLIP model configuration
#         cfg_clip = {
#             'rmin': 0 
#         }
#         print("CLIP extraction started...")
#         clip_extractor =ClipFeatureExtractor(clip_pt_model, layer=11, device="cuda")
#         save_features_to_h5(clip_h5_savepath, cfg_clip, ims, clip_extractor, model_type="CLIP", image_folder=datapath)
#     print("\n \n CLIP EXTRACTED DONE \n \n ")