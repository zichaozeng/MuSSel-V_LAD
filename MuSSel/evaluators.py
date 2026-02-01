"""Evaluation functions for visual place recognition.

This module provides functions for:
- Computing recall@N metrics
- Matching query segments to database segments
- Aggregating segment-level matches to image-level predictions
- Weighted Borda count for rank aggregation

The main evaluation metric is recall@N, which measures the percentage of queries
for which the correct match appears in the top-N retrieved database images.
"""

import os
import torch
from typing import Literal
import numpy as np
import fast_pytorch_kmeans as fpk
import h5py
from natsort import natsorted
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import torch.nn.functional as F
import time
import pickle
import faiss
from math import sqrt

def normalizeFeat(rfts):
    rfts = np.array(rfts).reshape([len(rfts),-1])
    rfts /= np.linalg.norm(rfts,axis=1)[:,None]
    return rfts

def weighted_borda_count(*ranked_lists_with_scores):
    """
    Merge ranked lists using weighted Borda count based on similarity scores.
    
    This method aggregates multiple ranked lists by summing similarity scores
    for each candidate. Unlike traditional Borda count (position-based), this
    uses the actual similarity scores, giving more weight to higher-confidence
    matches.
    
    Args:
        *ranked_lists_with_scores: Variable number of lists containing (index, score) tuples.
            Each list represents ranked candidates from a different source (e.g., different
            segments or different scales).
    
    Returns:
        list: Indices sorted by aggregated scores (highest first).
    """
    scores = {}
    for ranked_list in ranked_lists_with_scores:
        for index, score in ranked_list:
            if index in scores:
                scores[index] += score
            else:
                scores[index] = score
    sorted_indices = sorted(scores.keys(), key=lambda index: scores[index], reverse=True)
    return sorted_indices

def get_matches(matches,gt,sims,segRangeQuery,imIndsRef,n=1,method="max_sim"):
    """
    final version seems to be max_seg_topk_wt_borda_Im (need to confirm)
    
    """
    preds=[]
    for i in range(len(gt)):
        if method=="max_seg_topk_wt_borda_Im":
            match_patch = matches[segRangeQuery[i]].T.tolist()
            #TODO min max norm
            # sims_patch = sims[segRangeQuery[i]].T.tolist()
            sims_patch = sims[segRangeQuery[i]].T
            sims_max = np.max(sims)
            sims_min = np.min(sims)
            sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
            sims_patch = sims_patch.tolist()
            pair_patch = [list(zip(imIndsRef[match_patch[k]],sims_patch[k])) for k in range(len(sims_patch))]
            # pair_patch = [list(zip(match_patch[k],sims_patch[k])) for k in range(len(sims_patch))]
            match_patch = weighted_borda_count(*pair_patch)
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(match_patch[:n])
        elif method=="max_seg_topk":
            match_patch = matches[segRangeQuery[i]].flatten()
            segIdx = np.where(np.bincount(imIndsRef[match_patch])>0)[0]
            pred = segIdx[np.flip(np.argsort(np.bincount(imIndsRef[match_patch])[segIdx])[-n:])]
            # sim_score_t = sim_img.T[i][pred]
            
            # preds.append(pred[np.flip(np.argsort(sim_score_t))])
            preds.append(pred)
    return preds

def calc_recall(pred,gt,n,analysis=False):
    recall=[0]*n
    recall_per_query=[0]*len(gt)
    num_eval = 0
    for i in range(len(gt)):
        if len(gt[i])==0:
                continue
        num_eval+=1
        for j in range(len(pred[i])):
            # print(len(max_seg_preds[i]))
            # print(i)

            if n==1:
                if pred[i] in gt[i]:
                    recall[j]+=1
                    recall_per_query[i]=1
                    break
            else:
                if pred[i][j] in gt[i]:
                    recall[j]+=1
                    break

    recalls = np.cumsum(recall)/float(num_eval)
    print("POSITIVES/TOTAL segVLAD for this dataset: ", np.cumsum(recall),"/", num_eval)
    if analysis:
        return recalls.tolist(), recall_per_query
    return recalls.tolist()

def convert_to_queries_results_for_map(max_seg_preds, gt):
    queries_results = []
    for query_idx, refs in enumerate(max_seg_preds):
        query_results = [ref in gt[query_idx] for ref in refs]
        queries_results.append(query_results)
    return queries_results

def recall_segloc(workdir, dataset_name, experiment_config,experiment_name, segFtVLAD1, segFtVLAD2, gt, segRange2, imInds1, map_calculate):

    # RECALL CALCULATION
    # if pca then d = 512 else d = 49152
    if experiment_config["pca"]:
        d = 1024 #512 #PCA Dimension
        print("POTENTIAL CAUSE for error: Using d in pca before index faiss as", d, "\n 1024 or 512, check properly")
    else:
        d = 49152 #VLAD Dimension

    # nlist = sqrt(segFtVLAD1.shape[0]) 
    # quantizer = faiss.IndexFlatL2(d) 
    # index = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss.METRIC_L2)  
    index = faiss.IndexFlatL2(d)

    if experiment_config["pca"]:

        if isinstance(segFtVLAD1, torch.Tensor):
            segFtVLAD1_np = segFtVLAD1.detach().cpu().numpy().astype(np.float32)
        else:
            segFtVLAD1_np = segFtVLAD1.astype(np.float32)

        del segFtVLAD1

        if isinstance(segFtVLAD2, torch.Tensor):
            segFtVLAD2_np = segFtVLAD2.detach().cpu().numpy().astype(np.float32)
        else:
            segFtVLAD2_np = segFtVLAD2.astype(np.float32)

        del segFtVLAD2

        # index.train(normalizeFeat(segFtVLAD1_np))
        # index.add(normalizeFeat(segFtVLAD1_np))
        # sims, matches = index.search(normalizeFeat(segFtVLAD2_np), 200)

        index.add(normalizeFeat(segFtVLAD1_np))
        sims, matches = index.search(normalizeFeat(segFtVLAD2_np),200)

    else:
        index.add(segFtVLAD1.detach().cpu().numpy().astype(np.float32))
        # Should you normalize here?
        # sims, matches = index.search(segFtVLAD2.numpy(), 100)
        sims, matches = index.search(segFtVLAD2.detach().cpu().numpy().astype(np.float32), 200)
    
    
    # For  now just take 50 of those 100 matches
    sims_50 = sims[:, :50]
    matches_50 = matches[:, :50]

    sims_50 =2-sims_50#.T[0]
    # matches_justfirstone = matches.T[0]
    # sims =2-sims.T[0]
    max_seg_preds = get_matches(matches_50,gt,sims_50,segRange2,imInds1,n=20,method="max_seg_topk_wt_borda_Im")

    # max_seg_preds = func_vpr.get_matches_old(matches,gt,sims,segRange2,imInds1,n=5,method="max_seg")
    max_seg_recalls = calc_recall(max_seg_preds, gt, 20)

    print("VLAD + PCA Results \n ")
    # if map_calculate:
    #     # mAP calculation
    #     queries_results = convert_to_queries_results_for_map(max_seg_preds, gt)
    #     map_value = calculate_map(queries_results)
    #     print(f"Mean Average Precision (mAP): {map_value}")

    print("Max Seg Logs: ", max_seg_recalls)
    
    return max_seg_recalls