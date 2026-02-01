"""Ground truth generation for visual place recognition evaluation.

This module provides functions to generate ground truth correspondences between
query and database images based on:
- GPS/UTM coordinates (for outdoor datasets)
- Pose information (for indoor/underground datasets)
- Spatial proximity thresholds

Ground truth is used during evaluation to determine whether retrieved images
are correct matches (true positives) or incorrect (false positives).
"""

import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# from dataloaders.baidu_dataloader import Baidu_Dataset
from dataloaders.hawkins_dataloader import Hawkins
from dataloaders.laurel_dataloader import Laurel
from dataloaders.vpair_dataloader import VPAir

def get_utm(paths):
    coords =[]
    for path in paths:
        
        gps_coords = float(path.split('@')[1]),float(path.split('@')[2])
        coords.append(gps_coords)
    return coords


def get_positives(utmDb, utmQ, posDistThr, retDists=False):
    """
    Find positive matches within a spatial distance threshold.
    
    Uses nearest neighbour search to find all database locations within
    a specified radius of each query location. This defines the ground truth
    positive set for evaluation.
    
    Args:
        utmDb (np.ndarray): Database UTM coordinates (N_db × 2).
        utmQ (np.ndarray): Query UTM coordinates (N_q × 2).
        posDistThr (float): Distance threshold in metres for positive matches.
        retDists (bool): Whether to return distances (default: False).
        
    Returns:
        positives: List of arrays, where positives[i] contains indices of
                   database images within threshold of query i.
        distances (optional): Corresponding distances if retDists=True.
    """
    
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(utmDb)

    print("Using Localization Radius: ", posDistThr)
    distances, positives = knn.radius_neighbors(utmQ, radius=posDistThr)

    if retDists:
        return positives, distances
    else:
        return positives


def get_gt(dataset, cfg, workdir_data, ims1_r=None, ims2_q=None, func_vpr_module=None):
    """
    Retrieves the ground truth (gt) based on the specified dataset.

    Parameters:
        dataset (str): The name of the dataset.
        cfg (dict): Configuration settings.
        workdir_data (str): Path to the working directory data.
        ims1_r (list, optional): List of reference image paths (required for some datasets).
        ims2_q (list, optional): List of query image paths (required for some datasets).
        func_vpr_module (module, optional): Module containing VPR-related functions.

    Returns:
        gt: Ground truth data structure appropriate for the dataset.
    """
    if dataset == "baidu":
        vpr_dl = Baidu_Dataset(cfg["cfg"], workdir_data, 'baidu') 
        gt = vpr_dl.soft_positives_per_query

    elif dataset in ["mslsSF", "mslsCPH"]:
        GT_ROOT = './dataloaders/msls_npy_files/'
        city_name = "sf" if dataset == "mslsSF" else "cph"
        vpr_dl = MSLS(city_name=city_name, GT_ROOT=GT_ROOT)
        gt = vpr_dl.soft_positives_per_query

    elif dataset == "pitts":
        npy_pitts_path = f"{workdir_data}/{dataset}/test/"
        db = np.load(f"{npy_pitts_path}database.npy")
        q = np.load(f"{npy_pitts_path}queries.npy")
        utmDb = get_utm(db)
        utmQ = get_utm(q)
        gt = get_positives(utmDb, utmQ, 25)

    elif dataset == "SFXL":
        if ims1_r is None or ims2_q is None:
            raise ValueError("ims1_r and ims2_q must be provided for the SFXL dataset.")
        database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims1_r]).astype(float)
        queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in ims2_q]).astype(float)
        positive_dist_threshold = 25
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(database_utms)
        gt = knn.radius_neighbors(queries_utms, radius=positive_dist_threshold, return_distance=False)

    elif dataset == "17places":
        if ims2_q is None:
            raise ValueError("ims2_q must be provided for the 17places dataset.")
        loc_rad = 5
        gt = [list(np.arange(i - loc_rad, i + loc_rad + 1)) for i in range(len(ims2_q))]

    elif dataset == "AmsterTime":
        if ims1_r is None:
            raise ValueError("ims1_r must be provided for the AmsterTime dataset.")
        gt = [[i] for i in range(len(ims1_r))]

    elif dataset == "VPAir":
        vpr_dl = VPAir(cfg, workdir_data, 'VPAir')
        gt = vpr_dl.soft_positives_per_query

    elif dataset == "hawkins":
        vpr_dl = Hawkins(cfg["cfg"], workdir_data, 'hawkins_long_corridor') 
        gt = vpr_dl.soft_positives_per_query
    
    elif dataset == "laurel":
        vpr_dl = Laurel(cfg["cfg"], workdir_data, 'laurel')
        gt = vpr_dl.soft_positives_per_query
    
    elif dataset == "SPEDTEST":
        gt = []
        for i in range(607):
            gt.append([i])

    # elif dataset == "Nordland":
    #     data = np.load('{your_dataset}/ground_truth_new.npy', allow_pickle=True)
    #     gt = []
    #     for i in range(len(data)):
    #         gt.append(data[i][1])
    else:
        print("Dataset not found but saving descriptors, calculate recall later")
        gt = None  # Ensures descriptors are saved; recall can be calculated later.

    return gt
