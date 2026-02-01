"""Image segmentation script for visual place recognition.

This script generates segmentation masks using various algorithms:
- SLIC (Simple Linear Iterative Clustering)
- SEEDS (Superpixels Extracted via Energy-Driven Sampling)
- SAM (Segment Anything Model) - requires additional setup
- FastSAM - requires additional setup

Masks are saved to HDF5 files with configurable resolutions.

Usage:
    python image_segmentation.py <dataset> --slic_extract [--full_resolution]
    python image_segmentation.py <dataset> --seeds_extract
    
Examples:
    python image_segmentation.py laurel --slic_extract
    python image_segmentation.py hawkins --seeds_extract --full_resolution
"""

import argparse
import os
from natsort import natsorted
from config import paths, datasets
from MuSSel import process_slic_to_h5, process_seeds_to_h5

parser = argparse.ArgumentParser(
    description="Image segmentation using SAM, FastSAM, SLIC or SEEDS."
)

parser.add_argument(
    'dataset', 
    type=str, 
    help='Dataset name (e.g., 17places).'
)
# parser.add_argument(
#     '--sam_extract', 
#     action='store_true', 
#     help='Extract SAM segments (default: False).'
# )
# parser.add_argument(
#     '--fastsam_extract', 
#     action='store_true', 
#     help='Extract Fasy-SAM segments (default: False).'
# )
parser.add_argument(
    '--slic_extract', 
    action='store_true', 
    help='Extract SLIC segments (default: False).'
)
parser.add_argument(
    '--seeds_extract', 
    action='store_true', 
    help='Extract SEEDS segments (default: False).'
)
parser.add_argument(
    '--full_resolution', 
    action='store_true', 
    help='Use full resolution images for mask extraction (default: False).'
)

# Parse arguments
args = parser.parse_args()
dataset = args.dataset
# sam_extract = args.sam_extract
# fastsam_extract = args.fastsam_extract
slic_extract = args.slic_extract
seeds_extract = args.seeds_extract
full_resolution = args.full_resolution

# Get dataset configuration
dataset_config = datasets.get(dataset, {})
dataset_cfg = dataset_config["cfg"]

# Get paths
images_path = paths["images"]
# models_path = paths["pretrained_models"]
datapath_r = os.path.join(images_path, dataset, dataset_config["data_subpath1_r"])
datapath_q = os.path.join(images_path, dataset, dataset_config["data_subpath2_q"])
# sam_checkpoint_path = os.path.join(models_path, "models", "segment-anything", "sam_vit_h_4b8939.pth")

# Target resolution
desired_width = dataset_cfg["desired_width"]
desired_height = dataset_cfg["desired_height"]

# Resolution for SAM
if full_resolution:
    sam_width, sam_height = desired_width, desired_height
    print(f"Using full resolution images for mask extraction: {sam_width}x{sam_height}")
else:
    sam_width, sam_height = desired_width // 2, desired_height // 2
    print(f"Using half resolution images for mask extraction: {sam_width}x{sam_height}")

# Segmentation configuration
ims_sidx, ims_eidx, ims_step = 0, None, 1
list_all = [
        {
            "datapath": datapath_r,
            "sam_h5_savepath": os.path.join(paths["segments"], f"{dataset}_r_sam_{sam_width}.h5"),
            "fastsam_h5_savepath": os.path.join(paths["segments"], f"{dataset}_r_fastsam_{sam_width}.h5"),
            "slic_h5_savepath": os.path.join(paths["segments"], f"{dataset}_r_slic_{desired_width}.h5"),  
            "seeds_h5_savepath": os.path.join(paths["segments"], f"{dataset}_r_seeds_{desired_width}.h5"),  
            "patch_h5_savepath": os.path.join(paths["segments"], f"{dataset}_r_patch_{desired_width}.h5"),
        },
        {
            "datapath": datapath_q,
            "sam_h5_savepath": os.path.join(paths["segments"], f"{dataset}_q_sam_{sam_width}.h5"),
            "fastsam_h5_savepath": os.path.join(paths["segments"], f"{dataset}_q_fastsam_{sam_width}.h5"),
            "slic_h5_savepath": os.path.join(paths["segments"], f"{dataset}_q_slic_{desired_width}.h5"),
            "seeds_h5_savepath": os.path.join(paths["segments"], f"{dataset}_q_seeds_{desired_width}.h5"),
            "patch_h5_savepath": os.path.join(paths["segments"], f"{dataset}_q_patch_{desired_width}.h5"),
        }
    ]

# if sam_extract:
#     for iter_dict in list_all:
#         datapath = iter_dict["datapath"]
#         ims = natsorted(os.listdir(f'{datapath}'))
#         ims = ims[ims_sidx:ims_eidx][::ims_step]
#         sam_h5_savepath = iter_dict["sam_h5_savepath"]
#         SAM_checkpoint_path = os.path.join(models_path, "segment-anything", "sam_vit_h_4b8939.pth")

#         # SAM model configuration
#         cfg_sam = { "desired_width": sam_width, "desired_height": sam_height, "detect": 'dino', "use_sam": True, "class_threshold": 0.9, \
#                     "desired_feature": 0, "query_type": 'text', "sort_by": 'area', "use_16bit": False, "use_cuda": True,\
#                     "dino_strides": 4, "use_traced_model": False, 
#                     "rmin":0, "DAStoreFull":False, "dinov2": True, "wrap":False, "resize": True} # robohop specifc params

#         print("SAM extraction started...")
#         SAM = loadSAM(SAM_checkpoint_path,cfg_sam, device="cuda:1")
#         process_SAM_to_h5(sam_h5_savepath, cfg_sam, ims, SAM, image_folder=datapath)
#     print("\n \n SAM EXTRACTED DONE \n \n ")

# if fastsam_extract:
#     for iter_dict in list_all:
#         datapath = iter_dict["datapath"]
#         ims = natsorted(os.listdir(f'{datapath}'))
#         ims = ims[ims_sidx:ims_eidx][::ims_step]
#         fast_sam_h5_savepath = iter_dict["fastsam_h5_savepath"]
#         FastSAM_checkpoint_path = os.path.join(models_path, "FastSAM", "FastSAM-x.pt")

#         # FastSAM model configuration
#         cfg_fastsam = { "desired_width": sam_width, "desired_height": sam_height, "retina_masks": True, "rmin":0, "imgsz": 1024, "conf": 0.4, "iou":0.9} # robohop specifc params

#         print("FastSAM extraction started...")
#         FastSAM = loadFastSAM(FastSAM_checkpoint_path, device="cuda")
#         process_FastSAM_to_h5(fast_sam_h5_savepath, cfg_fastsam, ims, FastSAM, image_folder=datapath)
#     print("\n \n SAM EXTRACTED DONE \n \n ")

if slic_extract:
    for iter_dict in list_all:
        datapath = iter_dict["datapath"]
        ims = natsorted(os.listdir(f'{datapath}'))
        ims = ims[ims_sidx:ims_eidx][::ims_step]
        slic_h5_savepath = iter_dict["slic_h5_savepath"]

        # SLIC configuration
        cfg_slic = { "desired_width": desired_width, "desired_height": desired_height, "segment_scale": [64, 128, 256]}

        print("SLIC superpixel extraction started...")
        process_slic_to_h5(ims, slic_h5_savepath, cfg_slic, image_folder=datapath)
    print("\n \n SLIC EXTRACTED DONE \n \n ")

if seeds_extract:
    for iter_dict in list_all:
        datapath = iter_dict["datapath"]
        ims = natsorted(os.listdir(f'{datapath}'))
        ims = ims[ims_sidx:ims_eidx][::ims_step]
        seeds_h5_savepath = iter_dict["seeds_h5_savepath"]
        
        # SEEDS configuration
        cfg_seeds = { "desired_width": desired_width, "desired_height": desired_height, "segment_scale": [64, 128, 256]} 

        print("SEEDS superpixel extraction started...")
        process_seeds_to_h5(ims, seeds_h5_savepath, cfg_seeds, image_folder=datapath)
    print("\n \n SEEDS EXTRACTED DONE \n \n ")