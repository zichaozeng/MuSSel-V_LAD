"""Configuration file for MuSSel-V datasets and paths.

This module contains the configuration dictionaries for various datasets used in
visual place recognition experiments, including paths, feature extraction settings,
and dataset-specific parameters.
"""

# Workspace directory containing all data and outputs
working_space = './workspace'

# Directory paths for different data types and outputs
paths = {
    "images": f"{working_space}",                    # Root directory for dataset images
    "features": f"{working_space}/features",         # Extracted features (DINOv2, CLIP, etc.)
    "segments": f"{working_space}/segments",         # Image segmentation masks
    "pca": f"{working_space}/pca",                   # PCA transformation models
    "cluster_centers": f"{working_space}/cache",     # K-means cluster centres cache
    # "pretrained_models": f"/{working_space}/models",  # Pre-trained model weights
    "results": f"{working_space}/results",           # Evaluation results and outputs
}

# Dataset configurations for various VPR benchmarks
# Each dataset includes:
#   - Feature file names for reference (r) and query (q) images
#   - Data subdirectory paths
#   - Image processing configuration (resolution, cropping)
datasets = {
    # Baidu dataset configuration
    "baidu": {
        "dino_h5_filename_r":   "baidu_r_dino_640.h5",       # DINO features for reference images
        "dino_h5_filename_q":   "baidu_q_dino_640.h5",       # DINO features for query images
        "dinoft_h5_filename_r": "baidu_r_dinoft_640.h5",     # Fine-tuned DINO features for reference
        "dinoft_h5_filename_q": "baidu_q_dinoft_640.h5",     # Fine-tuned DINO features for query
        "data_subpath1_r": "training_images_undistort",      # Reference images subdirectory
        "data_subpath2_q": "query_images_undistort",         # Query images subdirectory
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480},  # Image processing config
    },
    "17places": {
        "dino_h5_filename_r":        "17places_r_dino_640.h5",
        "dino_h5_filename_q":        "17places_q_dino_640.h5",
        "dinoft_h5_filename_r":      "17places_r_dinoft_640.h5",
        "dinoft_h5_filename_q":      "17places_q_dinoft_640.h5",
        "data_subpath1_r": "ref",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
    },
    "SFXL": {
        "dino_h5_filename_r":   "SFXL_r_dino_512.h5",
        "dino_h5_filename_q":   "SFXL_q_dino_512.h5",
        "dinoft_h5_filename_r":      "SFXL_r_dinoft_512.h5",
        "dinoft_h5_filename_q":      "SFXL_q_dinoft_512.h5",
        "data_subpath1_r": "database",
        "data_subpath2_q": "queries",
        "cfg": {'rmin': 0, 'desired_width': 512, 'desired_height': 512}, 
    },
    "mslsCPH": {
        "dino_h5_filename_r":   "mslsCPH_r_dino_640.h5",
        "dino_h5_filename_q":   "mslsCPH_q_dino_640.h5",
        "dinoft_h5_filename_r":   "mslsCPH_r_dinoft_640.h5",
        "dinoft_h5_filename_q":   "mslsCPH_q_dinoft_640.h5",
        "data_subpath1_r": "database",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
    },
    "VPAir": {
        "dino_h5_filename_r": "VPAir_r_dino_800.h5",
        "dino_h5_filename_q": "VPAir_q_dino_800.h5",
        "dinoft_h5_filename_r":      "VPAir_r_dinoft_800.h5",
        "dinoft_h5_filename_q":      "VPAir_q_dinoft_800.h5",
        "data_subpath1_r": "reference_views",
        "data_subpath2_q": "queries",
        "cfg": {'rmin': 0, 'desired_width': 800, 'desired_height': 600}, 
    },
    "pitts": {
        "dino_h5_filename_r": "pitts_r_dino_640.h5",
        "dino_h5_filename_q": "pitts_q_dino_640.h5",
        "dinoft_h5_filename_r": "pitts_r_dinoft_640.h5",
        "dinoft_h5_filename_q": "pitts_q_dinoft_640.h5",
        "data_subpath1_r": "test/database",
        "data_subpath2_q": "test/queries",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
    },
    "AmsterTime": {
        "dino_h5_filename_r": "AmsterTime_r_dino_256.h5",
        "dino_h5_filename_q": "AmsterTime_q_dino_256.h5",
        "dinoft_h5_filename_r":      "AmsterTime_r_dinoft_256.h5",
        "dinoft_h5_filename_q":      "AmsterTime_q_dinoft_256.h5",
        "data_subpath1_r": "new",
        "data_subpath2_q": "old",
        "cfg": {'rmin': 0, 'desired_width': 256, 'desired_height': 256},
    },
    "hawkins": {
        "dino_h5_filename_r": "hawkins_r_dino_640.h5",
        "dino_h5_filename_q": "hawkins_q_dino_640.h5",
        "dinoft_h5_filename_r":      "hawkins_r_dinoft_640.h5",
        "dinoft_h5_filename_q":      "hawkins_q_dinoft_640.h5",
        "data_subpath1_r": "db_images",
        "data_subpath2_q": "q_images",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
    },
    "laurel": {
        "dino_h5_filename_r": "laurel_r_dino_640.h5",
        "dino_h5_filename_q": "laurel_q_dino_640.h5",
        "dinoft_h5_filename_r":      "laurel_r_dinoft_640.h5",
        "dinoft_h5_filename_q":      "laurel_q_dinoft_640.h5",
        "data_subpath1_r": "db_images",
        "data_subpath2_q": "q_images",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 480}, 
    },
    "SPEDTEST": {
        "dino_h5_filename_r": "SPEDTEST_r_dino_256.h5",
        "dino_h5_filename_q": "SPEDTEST_q_dino_256.h5",
        "dinoft_h5_filename_r":      "SPEDTEST_r_dinoft_256.h5",
        "dinoft_h5_filename_q":      "SPEDTEST_q_dinoft_256.h5",
        "data_subpath1_r": "ref",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 256, 'desired_height': 256}, 
    },
    "Nordland": {
        "dino_h5_filename_r": "Nordland_r_dino_640.h5",
        "dino_h5_filename_q": "Nordland_q_dino_640.h5",
        "dinoft_h5_filename_r":      "Nordland_r_dinoft_640.h5",
        "dinoft_h5_filename_q":      "Nordland_q_dinoft_640.h5",
        "data_subpath1_r": "ref",
        "data_subpath2_q": "query",
        "cfg": {'rmin': 0, 'desired_width': 640, 'desired_height': 360}, 
    }
}

# Experiment configurations for different segmentation and aggregation methods
# Each configuration specifies:
#   - segment_model: Type of segmentation (SAM, segments_16/32/64/128/256, segments_mixed)
#   - minArea: Minimum segment area threshold (0 = no filtering)
#   - order: Neighbour aggregation order (None = no aggregation, 3 = 3-hop neighbours)
#   - pca: Whether to apply PCA dimensionality reduction
experiments = {
    # SAM-based experiments with 3-hop neighbour aggregation and PCA
    \"SAM_ao3_pca\": {
        \"segment_model\": \"SAM\",                    # Segment Anything Model
        \"minArea\": 0,                             # No minimum area filtering
        \"order\": 3,                               # 3-hop neighbour aggregation
        \"pca\": True,                              # Apply PCA reduction
    },
    "FastSAM_ao3_pca": {
        "segment_model": "SAM",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },
    "SAM_na_pca": {
        "segment_model": "SAM",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "FastSAM_na_pca": {
        "segment_model": "SAM",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "Sp16_na_pca": {
        "segment_model": "segments_16",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "Sp32_na_pca": {
        "segment_model": "segments_32",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "Sp64_na_pca": {
        "segment_model": "segments_64",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "Sp128_na_pca": {
        "segment_model": "segments_128",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "Sp256_na_pca": {
        "segment_model": "segments_256",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "SpMixed_na_pca": {
        "segment_model": "segments_mixed",
        "minArea": 0,
        "order": None,
        "pca": True,
    },
    "Sp16_ao3_pca": {
        "segment_model": "segments_16",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },
    "Sp32_ao3_pca": {
        "segment_model": "segments_32",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },
    "Sp64_ao3_pca": {
        "segment_model": "segments_64",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },
    "Sp128_ao3_pca": {
        "segment_model": "segments_128",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },
    "Sp256_ao3_pca": {
        "segment_model": "segments_256",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },
    "SpMixed_ao3_pca": {
        "segment_model": "segments_mixed",
        "minArea": 0,
        "order": 3,
        "pca": True,
    },

    "Sp64_ao3_np": {
        "segment_model": "segments_64",
        "minArea": 0,
        "order": 3,
        "pca": False,
    },
    "Sp128_ao3_np": {
        "segment_model": "segments_128",
        "minArea": 0,
        "order": 3,
        "pca": False,
    },
    "Sp256_ao3_np": {
        "segment_model": "segments_256",
        "minArea": 0,
        "order": 3,
        "pca": False,
    },
    "SpMixed_ao3_np": {
        "segment_model": "segments_mixed",
        "minArea": 0,
        "order": 3,
        "pca": False,
    },
}
