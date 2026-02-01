"""Feature extraction classes and methods for visual place recognition.

This module provides:
- DINOv2 feature extraction with configurable layers and facets
- Image preprocessing and transformation pipelines
- Efficient feature saving to HDF5 format
- Support for multiple vision models (DINO, CLIP)
"""

import torch
import torchvision.transforms as tvf
import h5py
from tqdm import tqdm
from PIL import Image
from typing import Literal
# import clip
import cv2


def get_image_transform(model_type: str):
    """
    Returns the appropriate image transformation pipeline based on the model type.
    
    Args:
        model_type (str): Type of model ('DINO' or 'CLIP').

    Returns:
        torchvision.transforms.Compose: Transformation pipeline with normalisation.
        
    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == "DINO":
        return tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type == "CLIP":
        return tvf.Compose([
            tvf.Resize((224, 224)),
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_model_and_preprocessor(model_name: str, device: str = "cuda"):
    """
    Loads a specified vision model and its associated preprocessor.
    
    Currently supports:
    - DINOv2 variants (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
    
    Args:
        model_name (str): Name of the model to load (e.g., 'dinov2_vitg14').
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: (model, preprocessor)
            - model (torch.nn.Module): The loaded model in evaluation mode.
            - preprocessor (Callable): The associated image preprocessing function.
            
    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name.startswith("dinov2"):
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        preprocess = get_image_transform("DINO")
    # elif model_name.startswith("ViT"):
    #     model, preprocess = clip.load(model_name, device=device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.eval().to(device), preprocess

class DinoFeatureExtractor:
    """
    Feature extractor for DINOv2 models with configurable layer and facet extraction.
    
    This class uses forward hooks to extract intermediate features from specific
    transformer layers. Features can be extracted from queries, keys, values, or
    the full token representations.

    Args:
        model_name (str): Name of the DINOv2 model (e.g., 'dinov2_vitg14').
        layer (int): Transformer layer index to extract features from (0-indexed).
        facet (Literal["query", "key", "value", "token"]): 
            Which component to extract:
            - "query": Query vectors from attention
            - "key": Key vectors from attention  
            - "value": Value vectors from attention
            - "token": Full token representations
        use_cls (bool): Whether to include the CLS token (default: False).
        norm_descs (bool): Whether to L2-normalise descriptors (default: True).
        device (str): Device to run the model on ('cuda' or 'cpu').
        
    Attributes:
        dino_model: The loaded DINOv2 model.
        hook_handle: Forward hook registration handle.
        _hook_out: Storage for hooked layer output.
    """
    def __init__(self, model_name: str, layer: int, facet: Literal["query", "key", "value", "token"],
                 use_cls=False, norm_descs=True, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.dino_model, _ = load_model_and_preprocessor(model_name, device)
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self._hook_out = None
        self.hook_handle = self._register_hook()

    def _register_hook(self):
        """Registers a forward hook to capture features from the specified layer."""
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        if self.facet == "token":
            return self.dino_model.blocks[self.layer].register_forward_hook(_forward_hook)
        else:
            return self.dino_model.blocks[self.layer].attn.qkv.register_forward_hook(_forward_hook)

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from an input image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Extracted features.
        """
        with torch.no_grad():
            _ = self.dino_model(image_tensor)
            if not self.use_cls:
                self._hook_out = self._hook_out[:, 1:, :]

            if self.facet in ["query", "key", "value"]:
                d_len = self._hook_out.shape[2] // 3
                if self.facet == "query":
                    self._hook_out = self._hook_out[:, :, :d_len]
                elif self.facet == "key":
                    self._hook_out = self._hook_out[:, :, d_len:2 * d_len]
                else:
                    self._hook_out = self._hook_out[:, :, 2 * d_len:]

            if self.norm_descs:
                self._hook_out = torch.nn.functional.normalize(self._hook_out, dim=-1)
        return self._hook_out

    def __del__(self):
        self.hook_handle.remove()


# # CLIP feature extractor class
# class ClipFeatureExtractor:
#     """
#     Feature extractor for CLIP models.

#     Args:
#         model_name (str): Name of the CLIP model.
#         layer (int): Layer to extract features from.
#         use_cls (bool): Whether to include the CLS token (default: False).
#         norm_descs (bool): Whether to normalise descriptors (default: True).
#         device (str): Device to run the model on (default: "cuda").
#     """
#     def __init__(self, model_name: str, layer: int, use_cls=False, norm_descs=True, device: str = "cuda") -> None:
#         self.device = torch.device(device)
#         self.clip_model, _ = load_model_and_preprocessor(model_name, device)
#         self.layer = layer
#         self.use_cls = use_cls
#         self.norm_descs = norm_descs
#         self._hook_out = None
#         self.hook_handle = self._register_hook()

#     def _register_hook(self):
#         """Registers a forward hook to capture features from the specified layer."""
#         def _forward_hook(module, inputs, output):
#             if output is not None:
#                 self._hook_out = output.permute(1, 0, 2)
#         return self.clip_model.visual.transformer.resblocks[self.layer].register_forward_hook(_forward_hook)

#     def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         Extracts features from an input image tensor.

#         Args:
#             image_tensor (torch.Tensor): Input image tensor.

#         Returns:
#             torch.Tensor: Extracted features.
#         """
#         with torch.no_grad():
#             _ = self.clip_model.encode_image(image_tensor)
#             if not self.use_cls:
#                 self._hook_out = self._hook_out[:, 1:, :]

#             if self.norm_descs:
#                 self._hook_out = torch.nn.functional.normalize(self._hook_out, dim=-1)
#         return self._hook_out

#     def __del__(self):
#         self.hook_handle.remove()

# General feature extraction method
def extract_features(image, extractor, device="cuda", upsample=True, model_type="DINO"):
    """
    Extracts features from an image using a specified extractor.

    Args:
        image (numpy.ndarray): Input image.
        extractor: Feature extractor object.
        device (str): Device to run the model on (default: "cuda").
        upsample (bool): Whether to upsample the feature map to the image size (default: True).
        model_type (str): Model type ("DINO" or "CLIP").

    Returns:
        torch.Tensor: Extracted feature map.
    """
    preprocess = get_image_transform(model_type)
    # image_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    image_tensor = preprocess(Image.fromarray(image)).to(device)

    if model_type == "DINO":
        c, h, w = image_tensor.shape
        h_reduced, w_reduced = h // 14, w // 14
        h_new, w_new = h_reduced * 14, w_reduced * 14
        image_tensor = tvf.CenterCrop((h_new, w_new))(image_tensor)[None, ...]
        features = extractor(image_tensor)
        features = features.reshape(1, h_reduced, w_reduced, -1)
        features = features.permute(0, 3, 1, 2)
    elif model_type == "CLIP":
        image_tensor = image_tensor.unsqueeze(0)
        features = extractor(image_tensor)
        features = features.reshape(1, 16, 16, 1024)
        features = features.permute(0, 3, 1, 2)

    # Extracting features
    return features

# Save features to h5 
def save_features_to_h5(save_path, cfg, images, extractor, device="cuda", model_type="DINO", image_folder="./"):
    """
    Saves extracted features to an h5 file.

    Args:
        save_path (str): Path to save the h5 file.
        cfg (dict): Configuration dictionary containing additional parameters.
        images (list): List of image paths or arrays.
        extractor: Feature extractor object.
        device (str): Device to run the model on (default: "cuda").
        model_type (str): Model type ("DINO" or "CLIP").
    """
    rmin = cfg['rmin']

    with h5py.File(save_path, "w") as f:
        for i, image_path in enumerate(tqdm(images)):
            if isinstance(image_path, str):
                image = cv2.imread(f'{image_folder}/{image_path}')[rmin:, :, :]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path[rmin:, :, :]

            if cfg['resize']:
                image = cv2.resize(image, (cfg['desired_width'],cfg['desired_height']))
            else : 
                image =image

            features = extract_features(image, extractor, device, upsample=False, model_type=model_type)

            group = f.create_group(f"{image_path}")
            group.create_dataset("features", data=features.detach().cpu().numpy())
