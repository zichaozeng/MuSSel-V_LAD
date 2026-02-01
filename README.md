# **(MuS)Sel-V: Dilated Superpixel Aggregation for Visual Place Recognition**



[![Static Badge](https://img.shields.io/badge/Home_Page-purple)](https://zichaozeng.github.io/MuSSel-V) &nbsp;
[![Static Badge](https://img.shields.io/badge/-RA--L-blue)](https://ieeexplore.ieee.org/document/11302767) &nbsp;

# ⭐ **Abstract**
Visual Place Recognition (VPR) is a fundamental task in robotics and computer vision, enabling systems to identify locations seen in the past using visual information. Previous state-of-the-art approaches focus on encoding and retrieving semantically meaningful supersegment representations of images to significantly enhance recognition recall rates. However, we find that they struggle to cope with significant variations in viewpoint and scale, as well as scenes with sparse or limited information. Furthermore, these semantic-driven supersegment representations often exclude semantically meaningless yet valuable pixel information.  In this paper, we propose dilated superpixels to aggregate local descriptors, named **Sel-V**. This visually compact and complete representation substantially improves the robustness of segment-based methods and enhances the recognition of images with large variations. To further improve robustness, we introduce a multi-scale superpixel adaptive method - **MuSSel-V**, designed to accommodate a wide range of tasks across different domains. Extensive experiments conducted on benchmark datasets demonstrate that our method significantly outperforms existing approaches in recall, in diverse and complex environments characterised by dynamic changes or minimal scene information. Moreover, compared to existing supersegment representations, our approach achieves a notable advantage in processing speed. 

**Note: *Currently, we only provide the standard Sel-V and MuSSel-V with pre-trained DINOv2 for feature extraction and SEEDS or SLIC for segmentation. The complete code will be released in the future.***

# 🏙️ **Dataset**
## **Dataset Downloading**
For quick testing, we recommend downloading 17Places, VPAir, Laurel, and Hawkins from [AnyLoc](https://github.com/AnyLoc/AnyLoc/issues/34#issuecomment-2162492086). 

## **Dataset Preparation**

After downloading, place the datasets into the workspace. If you encounter path errors, please refer to the structure below and `config.py`.

```bash
wokspace/
├── 17places/
│   ├── query/
│   ├── ref/
│   ├── ...
├── laurel/
│   ├── db_images/
│   ├── q_images/
│   ├── ...
├── your_custom_data/
│   ├── query/
│   ├── ref/
│   ├── ...
├── features/
├── segments/
├── cache/
├── pca/
├── results/
```

# 🛠 Environment Setup
We implement our experiments using **Python 3.10** and **PyTorch 2.4.1+cu121**.

To set up the environment, run the following commands:
```bash
conda env create -f mussel.yaml
conda activate mussel
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
# 🔥 Inference
## Feature Extraction 
To perform feature extraction using **laurel** as an example, run:
```bash
python feature_extraction.py laurel --dino_extract
```
Or for custom data:
```bash
python feature_extraction.py <dataset> [--dino_extract]
```
**Parameters:**
- `<dataset>`: Specify the name of the dataset (e.g., `laurel` or `your_custom_data`).
- `[--dino_extract]`: Use pre-trained DINOv2 for feature extraction.

***Note**: Codes for extraction using other backbones like CLIP and fine-tuned DINO will be released soon.*
##  Segmentation. 

Choose between `--seeds_extract` or `--slic_extract` for segmentation:
```bash
python image_segmentation.py laurel --seeds_extract
```
Or for custom data:
```bash
python image_segmentation.py <dataset> [--seeds_extract | --slic_extract]
```
- `<dataset>`: Specify the dataset name.
- `[--seeds_extract]`: Use the SEEDS algorithm for segmentation.
- `[--slic_extract]`: Use the SLIC algorithm for segmentation.


##  Clustering

Run clustering on the extracted features:
```bash
python cluster_centre.py laurel
```
Or for custom data:
```bash
python cluster_centre.py <dataset> 
```

## PCA Reduction 

**Options**:
- Use `Sp64_ao3_pca`, `Sp128_ao3_pca`, or `Sp256_ao3_pca` for **Sel-V** with scales of 64, 128, or 256.
- Use `SpMixed_ao3_pca` for **MuSSel-V**.
- More dilation functions will be released soon.

Run PCA reduction:
```bash
python pca.py laurel Sp128_ao3_pca seeds tri 3
```
Or for custom data:
```bash
python pca.py <dataset> <experiment> <segmentation_method> <dilation_methods> <hop/order>
```
**Parameters**:
- `<dataset>`: Dataset name.
- `<experiment>`: Experiment setting (e.g., `Sp128_ao3_pca`).
- `<segmentation_method>`: `seeds` or `slic` (Support for `sam` and `fastsam` coming soon).
- `<dilation_methods>`: Dilation function (`tri` for Delaunay Triangulation, more coming soon).
- `<hop/order>`: Neighborhood matrix order.

## Building VLAD and evaluation
Run the following command to build VLAD descriptors and perform evaluation:
```bash
python vpr.py laurel Sp128_ao3_pca seeds tri 3 test_name
```
Or for custom data:
```bash
python pca.py <dataset> <experiment> <segmentation_method> <dilation_methods> <hop/order> <save_name>
```
**Parameters**:
- `<save_name>`: Name for the result file.

After running the command, you can:
- **View the results directly in the terminal**, or
- Find them saved at: `./workspace/results/<save_name>.txt`


## Potential Troubleshooting
**Issue**: `ModuleNotFoundError: No module named 'cv2.ximgproc'` **when running** `image_segmentation.py`

If you encounter the following error:
```vbnet
ModuleNotFoundError: No module named 'cv2.ximgproc'
```

**Solution**:

1. Uninstall existing OpenCV packages: 
```bash
pip uninstall opencv-python
pip uninstall opencv-contrib-python
```
2. Reinstall **opencv-contrib-python** (this package includes `cv2.ximgproc` and other extra modules):
```bash
pip install opencv-contrib-python
```
**After reinstalling, try running** `image_segmentation.py` **again**. If the error persists, make sure to check your Python environment and that the correct package version is installed.


# ❤️ Acknowledgements
We borrow some of the code from **AnyLoc** and **SegVLAD**. We thank the authors of AnyLoc and SegVLAD for making their code publicly available.


# 📜 Citation

Should our work offer you even the slightest inspiration, we would be most honoured if you chose to cite our paper.

```bibtex
@ARTICLE{zeng2026dailated,
        author={Zeng, Zichao and Goo, June Moh and Boehm, Jan},
        journal={IEEE Robotics and Automation Letters}, 
        title={Dilated Superpixel Aggregation for Visual Place Recognition}, 
        year={2026},
        volume={11},
        number={2},
        pages={2002-2009},
        doi={10.1109/LRA.2025.3645658}
}
```

**Note: *The code is still under development. More alternative methods will be released soon. Stay tuned!***
