"""
FastFit Inference Module for Footwear & Bags Virtual Try-On.

Uses the FastFit model which supports 5 categories:
tops, bottoms, dresses, shoes, and bags.

Requirements:
    - GPU with >= 10GB VRAM (T4 16GB works)
    - FastFit repo cloned at FASTFIT_REPO_PATH
    - Model weights auto-downloaded from HuggingFace
"""

import os
import sys
import traceback
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from logging_config import logger
from vton_errors import VTONProcessingError
from aws_utils import (
    download_pil_image,
    get_pil_images_from_s3_folder,
)

# FastFit configuration
FASTFIT_REPO_PATH = os.getenv("fastfit_repo_path", "/srv/FastFit")
FASTFIT_MODEL_PATH = os.getenv("fastfit_model_path", "Models/FastFit-MR-1024")
FASTFIT_UTIL_MODEL_PATH = os.getenv("fastfit_util_model_path", "Models/Human-Toolkit")
FASTFIT_DEVICE = os.getenv("fastfit_device", "cuda")
FASTFIT_NUM_STEPS = int(os.getenv("fastfit_num_steps", "50"))
FASTFIT_GUIDANCE_SCALE = float(os.getenv("fastfit_guidance_scale", "2.5"))
FASTFIT_MIXED_PRECISION = os.getenv("fastfit_mixed_precision", "fp16")

# Add FastFit repo to path
if FASTFIT_REPO_PATH not in sys.path:
    sys.path.insert(0, FASTFIT_REPO_PATH)

PERSON_SIZE = (768, 1024)


def center_crop_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """Center crop image to target aspect ratio."""
    width, height = img.size
    current_ratio = width / height
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
        left = (width - new_width) // 2
        top = 0
    else:
        new_width = width
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) // 2
    return img.crop((left, top, left + new_width, top + new_height))


class FastFitWrapper:
    """Wrapper for FastFit virtual try-on inference pipeline."""

    def __init__(self):
        self.pipeline = None
        self.dwpose_detector = None
        self.densepose_detector = None
        self.schp_lip_detector = None
        self.schp_atr_detector = None
        self.device = FASTFIT_DEVICE
        self.is_loaded = False

    def load_model(self):
        """Load FastFit model and utility models into GPU memory."""
        if self.is_loaded:
            logger.info("FastFit model already loaded.")
            return

        logger.info("Loading FastFit model from: %s", FASTFIT_MODEL_PATH)

        try:
            from huggingface_hub import snapshot_download
            from module.pipeline_fastfit import FastFitPipeline
            from parse_utils import (
                DWposeDetector,
                DensePose,
                SCHP,
            )

            # Download models if not present
            if not os.path.exists(FASTFIT_MODEL_PATH):
                os.makedirs(FASTFIT_MODEL_PATH, exist_ok=True)
                logger.info("Downloading FastFit model weights...")
                snapshot_download(
                    repo_id="zhengchong/FastFit-MR-1024",
                    local_dir=FASTFIT_MODEL_PATH,
                    local_dir_use_symlinks=False,
                )

            if not os.path.exists(FASTFIT_UTIL_MODEL_PATH):
                os.makedirs(FASTFIT_UTIL_MODEL_PATH, exist_ok=True)
                logger.info("Downloading Human-Toolkit models...")
                snapshot_download(
                    repo_id="zhengchong/Human-Toolkit",
                    local_dir=FASTFIT_UTIL_MODEL_PATH,
                    local_dir_use_symlinks=False,
                )

            # Load utility models
            logger.info("Loading DWPose detector...")
            self.dwpose_detector = DWposeDetector(
                pretrained_model_name_or_path=os.path.join(
                    FASTFIT_UTIL_MODEL_PATH, "DWPose"
                ),
                device="cpu",
            )

            logger.info("Loading DensePose detector...")
            self.densepose_detector = DensePose(
                model_path=os.path.join(
                    FASTFIT_UTIL_MODEL_PATH, "DensePose"
                ),
                device=self.device,
            )

            logger.info("Loading SCHP LIP detector...")
            self.schp_lip_detector = SCHP(
                ckpt_path=os.path.join(
                    FASTFIT_UTIL_MODEL_PATH, "SCHP", "schp-lip.pth"
                ),
                device=self.device,
            )

            logger.info("Loading SCHP ATR detector...")
            self.schp_atr_detector = SCHP(
                ckpt_path=os.path.join(
                    FASTFIT_UTIL_MODEL_PATH, "SCHP", "schp-atr.pth"
                ),
                device=self.device,
            )

            # Load FastFit pipeline
            logger.info("Loading FastFit pipeline...")
            self.pipeline = FastFitPipeline(
                base_model_path=FASTFIT_MODEL_PATH,
                device=self.device,
                mixed_precision=FASTFIT_MIXED_PRECISION,
                allow_tf32=True,
            )

            self.is_loaded = True
            logger.info("FastFit model loaded successfully.")

        except Exception as e:
            logger.error("Failed to load FastFit model: %s", e)
            logger.error(traceback.format_exc())
            raise VTONProcessingError(f"FastFit model load failed: {e}")

    def unload_model(self):
        """Unload FastFit model from GPU memory."""
        self.pipeline = None
        self.dwpose_detector = None
        self.densepose_detector = None
        self.schp_lip_detector = None
        self.schp_atr_detector = None
        torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("FastFit model unloaded from GPU.")

    def preprocess_person(
        self, person_image: Image.Image
    ) -> Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess person image: crop, resize, detect pose/parsing."""
        person_image = person_image.convert("RGB")
        person_image = center_crop_to_aspect_ratio(person_image, 3 / 4)
        person_image = person_image.resize(PERSON_SIZE, Image.LANCZOS)

        # Pose estimation
        pose_img = self.dwpose_detector(person_image)
        if not isinstance(pose_img, Image.Image):
            raise VTONProcessingError("Pose estimation failed")

        # DensePose and human parsing
        densepose_arr = np.array(self.densepose_detector(person_image))
        lip_arr = np.array(self.schp_lip_detector(person_image))
        atr_arr = np.array(self.schp_atr_detector(person_image))

        return person_image, pose_img, densepose_arr, lip_arr, atr_arr

    def generate_mask(
        self,
        densepose_arr: np.ndarray,
        lip_arr: np.ndarray,
        atr_arr: np.ndarray,
        category: str = "shoes",
    ) -> Image.Image:
        """Generate mask for the specific category only.
        
        For shoes: only mask the foot/shoe region
        For bags: only mask the bag region
        For clothing: mask the full outfit area
        """
        from parse_utils.automasker import (
            part_mask_of,
            hull_mask,
            DENSE_INDEX_MAP,
            LIP_MAPPING,
            ATR_MAPPING,
        )
        import cv2

        w, h = densepose_arr.shape[:2]
        dilate_kernel = max(w, h) // 500
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)

        if category in ("shoes", "footwear", "shoe"):
            # Only mask the shoe/foot region
            shoe_mask = (
                part_mask_of(["Left-shoe", "Right-shoe"], lip_arr, LIP_MAPPING)
                | part_mask_of(["Left-shoe", "Right-shoe"], atr_arr, ATR_MAPPING)
            )
            # Also include feet from densepose
            feet_mask = part_mask_of(["feet"], densepose_arr, DENSE_INDEX_MAP)
            mask_area = shoe_mask | feet_mask
            # Dilate to cover edges
            mask_area = cv2.dilate(mask_area.astype(np.uint8), dilate_kernel, iterations=3)
            return Image.fromarray(mask_area * 255)

        elif category in ("bags", "bag"):
            # Only mask the bag region
            bag_mask = (
                part_mask_of(["Bag"], lip_arr, LIP_MAPPING)
                | part_mask_of(["Bag"], atr_arr, ATR_MAPPING)
            )
            mask_area = cv2.dilate(bag_mask.astype(np.uint8), dilate_kernel, iterations=3)
            return Image.fromarray(mask_area * 255)

        else:
            # For clothing, use the full multi-ref mask
            from parse_utils import multi_ref_cloth_agnostic_mask
            return multi_ref_cloth_agnostic_mask(
                densepose_arr, lip_arr, atr_arr,
                square_cloth_mask=False, horizon_expand=True,
            )

    def prepare_reference_images(
        self,
        garment_image: Image.Image,
        category: str,
        ref_height: int = 512,
    ) -> Tuple[List[Image.Image], List[str], List[int]]:
        """Prepare reference images in the format FastFit expects.

        FastFit expects 5 slots: upper, lower, overall, shoe, bag.
        We fill the relevant slot and leave others empty.
        """
        clothing_ref_size = (int(ref_height * 3 / 4), ref_height)
        accessory_ref_size = (384, 512)

        ref_images = []
        ref_labels = []
        ref_attention_masks = []

        categories = ["upper", "lower", "overall", "shoe", "bag"]

        # Map input category to FastFit slot
        category_slot_map = {
            "shoes": "shoe",
            "footwear": "shoe",
            "shoe": "shoe",
            "bags": "bag",
            "bag": "bag",
        }
        target_slot = category_slot_map.get(category.lower(), category.lower())

        for slot in categories:
            target_size = (
                accessory_ref_size if slot in ["shoe", "bag"]
                else clothing_ref_size
            )

            if slot == target_slot:
                img = garment_image.convert("RGB").resize(
                    target_size, Image.LANCZOS
                )
                ref_images.append(img)
                ref_labels.append(slot)
                ref_attention_masks.append(1)
            else:
                ref_images.append(
                    Image.new("RGB", target_size, color=(0, 0, 0))
                )
                ref_labels.append(slot)
                ref_attention_masks.append(0)

        return ref_images, ref_labels, ref_attention_masks

    def run_inference(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: str = "shoes",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: int = 42,
    ) -> Image.Image:
        """Run FastFit inference for a single item try-on."""
        if not self.is_loaded:
            self.load_model()

        num_inference_steps = num_inference_steps or FASTFIT_NUM_STEPS
        guidance_scale = guidance_scale or FASTFIT_GUIDANCE_SCALE

        logger.info(
            "Running FastFit inference: category=%s, steps=%d, guidance=%.1f",
            category, num_inference_steps, guidance_scale,
        )

        try:
            # Preprocess person image
            processed_person, pose_img, densepose_arr, lip_arr, atr_arr = (
                self.preprocess_person(person_image)
            )

            # Generate mask — category-specific (shoes only masks feet, etc.)
            mask_img = self.generate_mask(densepose_arr, lip_arr, atr_arr, category)

            # Prepare reference images
            ref_images, ref_labels, ref_attention_masks = (
                self.prepare_reference_images(garment_image, category)
            )

            # Run pipeline
            generator = torch.Generator(device=self.device).manual_seed(seed)

            with torch.no_grad():
                result = self.pipeline(
                    person=processed_person,
                    mask=mask_img,
                    ref_images=ref_images,
                    ref_labels=ref_labels,
                    ref_attention_masks=ref_attention_masks,
                    pose=pose_img,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    return_pil=True,
                )

            if isinstance(result, list) and len(result) > 0:
                logger.info("FastFit inference completed successfully.")
                return result[0]

            raise VTONProcessingError("FastFit returned no valid image")

        except VTONProcessingError:
            raise
        except Exception as e:
            logger.error("FastFit inference failed: %s", e)
            logger.error(traceback.format_exc())
            raise VTONProcessingError(f"FastFit inference failed: {e}")


# Global pipeline instance (singleton)
_fastfit_wrapper = None


def get_fastfit_pipeline() -> FastFitWrapper:
    """Get or create the global FastFit pipeline instance."""
    global _fastfit_wrapper
    if _fastfit_wrapper is None:
        _fastfit_wrapper = FastFitWrapper()
    return _fastfit_wrapper


def run_fastfit(
    data: Dict,
    category: str,
    cloth_index: int = 0,
) -> List[Dict]:
    """Run FastFit try-on for footwear/bags.

    This function mirrors the interface of `run_vton` from inference.py
    so it can be used as a drop-in replacement for supported categories.

    Args:
        data: Dictionary containing S3 paths for person and garment images
        category: Item category ('shoes', 'bags')
        cloth_index: Index of the garment image to use

    Returns:
        List of dicts with 'pil_image' and 'file_name' keys
    """
    logger.info("Starting FastFit processing for category: %s", category)

    try:
        # Load person image from S3
        # For FastFit, we use the raw uploaded image directly
        # (FastFit does its own preprocessing - pose, parsing, masking)
        bucket_name = data['human_bucket']
        folder = data['human_folder']
        logger.info(
            "Loading person images from s3://%s/%s", bucket_name, folder
        )

        # Try to get from preprocessed folder first
        person_image = None
        try:
            customer_images = get_pil_images_from_s3_folder(bucket_name, folder)
            person_image = customer_images['pil_img_dict'].get('image')
        except Exception:
            pass

        # If not found in preprocessed, load raw image directly
        if person_image is None:
            # The raw image is at: {raw_bucket}/{user_id}/{request_id}.png
            raw_bucket = os.getenv("raw_images_bucket", "product-images-groome-1")
            # folder is like "user_id/request_id", raw image is "user_id/request_id.png"
            raw_key = f"{folder}.png"
            logger.info(
                "Trying raw image from s3://%s/%s", raw_bucket, raw_key
            )
            try:
                person_image = download_pil_image(raw_bucket, raw_key)
            except Exception:
                pass

        if person_image is None:
            raise VTONProcessingError(
                "Person image not found in S3 folder"
            )

        # Load garment/shoe/bag image from S3
        cloth_bucket = data['cloth_bucket'][cloth_index]
        cloth_path = data['cloth_path'][cloth_index]
        logger.info(
            "Loading garment from s3://%s/%s", cloth_bucket, cloth_path
        )
        garment_image = download_pil_image(cloth_bucket, cloth_path)
        garment_image = garment_image.convert("RGB")

        # Get pipeline and run inference
        wrapper = get_fastfit_pipeline()
        result_image = wrapper.run_inference(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
        )

        return [{
            "pil_image": result_image,
            "file_name": f"fastfit_{category}",
        }]

    except VTONProcessingError:
        raise
    except Exception as e:
        logger.error("FastFit processing error: %s", e)
        logger.error(traceback.format_exc())
        raise VTONProcessingError(f"FastFit processing failed: {e}")
