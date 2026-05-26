"""
FastFit Inference Module for Footwear & Bags Virtual Try-On.

This module handles inference for categories not supported by IDM-VTON:
- shoes/footwear
- bags

Uses the FastFit model (Stable Diffusion v1.5 Inpainting based) which supports
5 categories: tops, bottoms, dresses, shoes, and bags.

Requirements:
    - GPU with >= 10GB VRAM (T4 16GB works)
    - FastFit repo cloned at FASTFIT_REPO_PATH
    - Model weights auto-downloaded from HuggingFace
"""

import os
import sys
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
FASTFIT_MODEL_ID = os.getenv(
    "fastfit_model_id", "zhengchong/FastFit-SR-1024"
)
FASTFIT_DEVICE = os.getenv("fastfit_device", "cuda")
FASTFIT_NUM_STEPS = int(os.getenv("fastfit_num_steps", "50"))
FASTFIT_GUIDANCE_SCALE = float(os.getenv("fastfit_guidance_scale", "2.5"))

# Add FastFit repo to path
if FASTFIT_REPO_PATH not in sys.path:
    sys.path.insert(0, FASTFIT_REPO_PATH)


class FastFitPipeline:
    """Wrapper for FastFit virtual try-on inference pipeline."""

    def __init__(self):
        self.pipe = None
        self.auto_masker = None
        self.device = FASTFIT_DEVICE
        self.is_loaded = False

    def load_model(self):
        """Load FastFit model components into GPU memory."""
        if self.is_loaded:
            logger.info("FastFit model already loaded.")
            return

        logger.info("Loading FastFit model from: %s", FASTFIT_MODEL_ID)

        try:
            from module.pipeline import FastFitPipeline as FFPipeline
            from module.automasker import AutoMasker

            # Load the pipeline
            self.pipe = FFPipeline.from_pretrained(
                FASTFIT_MODEL_ID,
                torch_dtype=torch.float16,
            ).to(self.device)

            # Load AutoMasker for automatic mask generation
            self.auto_masker = AutoMasker(
                densepose_path=os.path.join(
                    FASTFIT_REPO_PATH, "parse_utils"
                ),
                device=self.device,
            )

            self.is_loaded = True
            logger.info("FastFit model loaded successfully.")

        except Exception as e:
            logger.error("Failed to load FastFit model: %s", e)
            raise VTONProcessingError(f"FastFit model load failed: {e}")

    def unload_model(self):
        """Unload FastFit model from GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.auto_masker is not None:
            del self.auto_masker
            self.auto_masker = None
        torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("FastFit model unloaded from GPU.")

    def generate_mask(
        self,
        person_image: Image.Image,
        category: str,
    ) -> Image.Image:
        """Generate mask for the target region using AutoMasker.

        Args:
            person_image: PIL image of the person
            category: Category of item ('shoes' or 'bags')

        Returns:
            PIL Image mask (white = region to replace)
        """
        if self.auto_masker is None:
            raise VTONProcessingError("AutoMasker not loaded")

        # Map categories to AutoMasker labels
        category_map = {
            "shoes": "shoes",
            "footwear": "shoes",
            "bags": "bags",
            "bag": "bags",
        }

        mask_category = category_map.get(category.lower(), category.lower())
        logger.info("Generating mask for category: %s", mask_category)

        try:
            mask = self.auto_masker(person_image, mask_category)
            return mask
        except Exception as e:
            logger.error("Mask generation failed: %s", e)
            raise VTONProcessingError(f"Mask generation failed: {e}")

    def run_inference(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        mask_image: Optional[Image.Image] = None,
        category: str = "shoes",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: int = 42,
    ) -> Image.Image:
        """Run FastFit inference for a single item try-on.

        Args:
            person_image: PIL image of the person
            garment_image: PIL image of the garment/shoe/bag
            mask_image: Optional pre-computed mask. If None, auto-generated.
            category: Item category ('shoes', 'bags', etc.)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility

        Returns:
            PIL Image of the try-on result
        """
        if not self.is_loaded:
            self.load_model()

        num_inference_steps = num_inference_steps or FASTFIT_NUM_STEPS
        guidance_scale = guidance_scale or FASTFIT_GUIDANCE_SCALE

        # Resize images to model's expected resolution
        target_size = (768, 1024)  # width x height
        person_image = person_image.resize(target_size, Image.LANCZOS)
        garment_image = garment_image.resize(target_size, Image.LANCZOS)

        # Generate mask if not provided
        if mask_image is None:
            mask_image = self.generate_mask(person_image, category)
        else:
            mask_image = mask_image.resize(target_size, Image.LANCZOS)

        logger.info(
            "Running FastFit inference: steps=%d, guidance=%.1f, seed=%d",
            num_inference_steps,
            guidance_scale,
            seed,
        )

        generator = torch.Generator(self.device).manual_seed(seed)

        try:
            with torch.inference_mode():
                result = self.pipe(
                    image=person_image,
                    mask_image=mask_image,
                    garment_image=garment_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

            logger.info("FastFit inference completed successfully.")
            return result

        except Exception as e:
            logger.error("FastFit inference failed: %s", e)
            raise VTONProcessingError(f"FastFit inference failed: {e}")


# Global pipeline instance (singleton)
_fastfit_pipeline = None


def get_fastfit_pipeline() -> FastFitPipeline:
    """Get or create the global FastFit pipeline instance."""
    global _fastfit_pipeline
    if _fastfit_pipeline is None:
        _fastfit_pipeline = FastFitPipeline()
    return _fastfit_pipeline


def run_fastfit(
    data: Dict,
    category: str,
    cloth_index: int = 0,
) -> List[Dict]:
    """Run FastFit try-on for footwear/bags.

    This function mirrors the interface of `run_vton` from inference.py
    so it can be used as a drop-in replacement for supported categories.

    Args:
        data: Dictionary containing:
            - human_bucket: S3 bucket for person images
            - human_folder: S3 folder for person images
            - cloth_bucket: List of S3 buckets for garment images
            - cloth_path: List of S3 paths for garment images
        category: Item category ('shoes', 'bags')
        cloth_index: Index of the garment image to use

    Returns:
        List of dicts with 'pil_image' and 'file_name' keys
    """
    logger.info("Starting FastFit processing for category: %s", category)

    try:
        # Load person image from S3
        bucket_name = data['human_bucket']
        folder = data['human_folder']
        logger.info(
            "Loading person images from s3://%s/%s", bucket_name, folder
        )
        customer_images = get_pil_images_from_s3_folder(bucket_name, folder)
        person_image = customer_images['pil_img_dict'].get('image')

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

        # Check if a pre-computed mask exists (from preprocessing)
        mask_image = customer_images['pil_img_dict'].get('shoes-mask')
        if mask_image is None:
            mask_image = customer_images['pil_img_dict'].get('foot-mask')
        # If no mask found, AutoMasker will generate one

        # Get pipeline and run inference
        pipeline = get_fastfit_pipeline()
        result_image = pipeline.run_inference(
            person_image=person_image,
            garment_image=garment_image,
            mask_image=mask_image,
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
