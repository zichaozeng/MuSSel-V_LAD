"""Image segmentation methods for visual place recognition.

This module provides implementations for various segmentation algorithms:
- SLIC (Simple Linear Iterative Clustering) superpixels
- SEEDS (Superpixels Extracted via Energy-Driven Sampling)
- SAM (Segment Anything Model) - commented out, requires additional dependencies
- FastSAM - commented out, requires additional dependencies

Segmentation masks are saved to HDF5 files for efficient storage and retrieval.
"""

import os
import numpy as np
import h5py
from tqdm import tqdm
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.transform import resize
import imageio
# import sys
# sys.path.append(f'{your_FastSAM}/FastSAM')
# from fastsam import FastSAM, FastSAMPrompt
from PIL import Image
import torch

# #######################################################################
# def loadFastSAM(FastSAM_checkpoint, device = 'cuda'):
#     fastsam_model = FastSAM(FastSAM_checkpoint)
#     fastsam_model.to("cuda") 
#     return fastsam_model

# def process_single_FastSAM(cfg, img, models, device):
#     img = img.convert('RGB')
#     img_np = np.array(img)
#     img_np = cv2.resize(img_np, (cfg['desired_width'],cfg['desired_height']))
#     img = Image.fromarray(img_np)

#     everything_results = models(
#         img,
#         device="cuda",
#         retina_masks=cfg['retina_masks'],
#         imgsz=cfg['desired_width'],
#         conf=cfg['conf'],
#         iou=cfg['iou'],    
#     )

#     prompt_process = FastSAMPrompt(img, everything_results, device="cuda")
#     ann = prompt_process.everything_prompt()
#     if isinstance(ann, torch.Tensor):
#         ann_numpy = ann.cpu().numpy().astype(bool)
#     elif isinstance(ann, list) and len(ann) == 0:
#         print("Warning: ann is an empty list!")
#         ann_numpy = np.zeros((2, 256, 256), dtype=bool)
#         ann_numpy[0] = np.ones((256, 256), dtype=bool)
#     else:
#         ann_numpy = np.zeros((2, 256, 256), dtype=bool)
#         ann_numpy[0] = np.ones((256, 256), dtype=bool)

#     ann_numpy = (ann_numpy.astype(np.uint8) * 255)
#     ann_resized = cv2.resize(ann_numpy, (cfg['desired_width'], cfg['desired_height']), interpolation=cv2.INTER_NEAREST)
#     ann_resized = torch.from_numpy((ann_resized > 127).astype(bool)) 
#     masks = [{"segmentation": ann_numpy[i]} for i in range(ann_numpy.shape[0])]
#     return img, masks

# def process_FastSAM_to_h5(h5FullPath,cfg,ims,models,device="cuda",image_folder="./"):
#     rmin = cfg['rmin']
#     with h5py.File(h5FullPath, "w") as f:
#         for i, _ in enumerate(tqdm(ims)):
#             if isinstance(ims[0],str):
#                 imname = ims[i] 
#                 im = Image.open(f'{image_folder}/{imname}')
#             else:
#                 imname, im = i, ims[i][rmin:,:,:]
#             im_p, masks = process_single_FastSAM(cfg,im,models,device)
#             grp = f.create_group(f"{imname}")
#             grp.create_group("masks")
#             for j, m in enumerate(masks):
#                 for k in m.keys():
#                     grp["masks"].create_dataset(f"{j}/{k}", data=m[k])    
# #######################################################################


# #######################################################################
# def loadSAM(sam_checkpoint, cfg, device = 'cuda'):
#     model_type = "vit_h"
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     mask_generator = SamAutomaticMaskGenerator(sam)
#     return mask_generator

# def process_single_SAM(cfg, img, models, device):
#     mask_generator = models
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if cfg['resize']:
#         img_p = cv2.resize(img, (cfg['desired_width'],cfg['desired_height']))
 
#     else : img_p = img
#     masks = mask_generator.generate(img_p)

#     return img_p, masks

# def process_SAM_to_h5(h5FullPath,cfg,ims,models,device="cuda",image_folder="./"):
#     rmin = cfg['rmin']
#     with h5py.File(h5FullPath, "w") as f:
#         for i, _ in enumerate(tqdm(ims)):
#             if isinstance(ims[0],str):
#                 imname = ims[i] 
#                 im = cv2.imread(f'{image_folder}/{imname}')[rmin:,:,:]
#             else:
#                 imname, im = i, ims[i][rmin:,:,:]
#             im_p, masks = process_single_SAM(cfg,im,models,device)
#             grp = f.create_group(f"{imname}")
#             grp.create_group("masks")
#             for j, m in enumerate(masks):
#                 for k in m.keys():
#                     grp["masks"].create_dataset(f"{j}/{k}", data=m[k])    
# #######################################################################

#######################################################################
def load_superpixels(segment_h5, image_key):
    segments = segment_h5[image_key][:]
    unique_superpixels = np.unique(segments)
    mask_segs = []
    for superpixel_id in unique_superpixels:
        mask_segs.append((segments == superpixel_id)) 
    return mask_segs

def getIdxSingleFast(img_idx, masks_seg, minArea=400, returnMask=True):
    imInds = []
    regIndsIm = []
    segmask = []
    count = 0
    for mask in masks_seg:
        if returnMask:
            segmask.append(mask)
        regIndsIm.append(count)
        imInds.append(img_idx)
        count += 1
    return np.array(imInds), regIndsIm, segmask

def countNumMasksInDataset(ims, masks_in):
    count = 0
    # print(masks_in.keys())
    for im_name in tqdm(ims, desc="Counting num of masks in dataset"):
        # Directly constructing the path to the masks for the current image
        mask_path = f"{im_name}"
        count += len(np.unique(masks_in[mask_path][:]))
    return count
#######################################################################

#######################################################################
def process_slic_to_h5(images, output_file, cfg_sp, compactness=10, image_folder="./"):
    """
    Perform SLIC superpixel segmentation and save results to HDF5.
    
    SLIC (Simple Linear Iterative Clustering) generates compact, approximately
    uniform superpixels. Multiple scales can be generated per image.

    Args:
        images (list): List of image filenames to process.
        output_file (str): Path to output HDF5 file.
        cfg_sp (dict): Segmentation configuration containing:
            - segment_scale (list): List of superpixel counts (e.g., [64, 128, 256])
            - desired_height (int): Target image height
            - desired_width (int): Target image width
        compactness (float): SLIC compactness parameter (higher = more regular shapes).
        image_folder (str): Directory containing input images.
        
    HDF5 Structure:
        /<image_filename>/segments_<N>: (H × W) array with segment labels [0, N-1]
    """

    with h5py.File(output_file, 'w') as f:
        for image_file in tqdm(images, desc="Processing images"):
            image_path = os.path.join(image_folder, image_file)
            image = img_as_float(imageio.imread(image_path))

            if image.shape[-1] == 4:
                image = image[..., :3]

            # Resize (optional)
            # image = resize(image, (cfg_sp["desired_height"], cfg_sp["desired_width"]), anti_aliasing=True)

            channel_axis = None if len(image.shape) == 2 else -1
            segment_scale = cfg_sp["segment_scale"]
            desired_height = cfg_sp["desired_height"]
            desired_width = cfg_sp["desired_width"]

            group = f.create_group(f"{image_file}")
            for n_segments in segment_scale:
                segments = slic(image, n_segments=n_segments, compactness=compactness, 
                                max_num_iter=10, channel_axis=channel_axis, start_label=0)
                
                if segments.shape[0] != desired_height or segments.shape[1] != desired_width:
                    segments_resized = resize(
                        segments, 
                        (desired_height, desired_width), 
                        order=0,
                        preserve_range=True, 
                        anti_aliasing=False
                    ).astype(int)
                else:
                    segments_resized = segments
                group.create_dataset(f"segments_{n_segments}", data=segments_resized)
#######################################################################

#######################################################################
def process_seeds_to_h5(images, output_file, cfg_sp, image_folder="./"):
    """
    Perform SEEDS superpixel segmentation and save results to HDF5.
    
    SEEDS (Superpixels Extracted via Energy-Driven Sampling) generates boundary-
    preserving superpixels using histogram-based energy minimisation.

    Args:
        images (list): List of image filenames to process.
        output_file (str): Path to output HDF5 file.
        cfg_sp (dict): Segmentation configuration containing:
            - segment_scale (list): List of superpixel counts
            - desired_height (int): Target image height
            - desired_width (int): Target image width
        image_folder (str): Directory containing input images.
        
    HDF5 Structure:
        /<image_filename>/segments_<N>: (H × W) array with segment labels
    """

    with h5py.File(output_file, 'w') as f:

        for image_file in tqdm(images, desc="Processing images"):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)

            # Resize (optional)
            # image = resize(image, (cfg_sp["desired_height"], cfg_sp["desired_width"]), anti_aliasing=True)
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

            segment_scale = cfg_sp["segment_scale"]
            desired_height = cfg_sp["desired_height"]
            desired_width = cfg_sp["desired_width"]

            group = f.create_group(f"{image_file}")
            for n_segments in segment_scale:
                seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2],
                                            num_superpixels=n_segments, num_levels=4, prior=2)

                seeds.iterate(lab_image, 10)
                segments = seeds.getLabels()
                
                if segments.shape[0] != desired_height or segments.shape[1] != desired_width:
                    segments_resized = resize(
                        segments, 
                        (desired_height, desired_width), 
                        order=0,
                        preserve_range=True, 
                        anti_aliasing=False
                    ).astype(int)
                else:
                    segments_resized = segments

                group.create_dataset(f"segments_{n_segments}", data=segments_resized)
#######################################################################
