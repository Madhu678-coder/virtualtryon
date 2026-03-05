import numpy as np
import torch
from typing import List
import random
import os
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTextModel,
)

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import (
    UNet2DConditionModel as UNet2DConditionModel_ref
)
from src.tryon_pipeline import (
    StableDiffusionXLInpaintPipeline as TryonPipeline
)
from data_loader import process_data_s3
from logging_config import logger
from vton_errors import DataLoadingError, VTONProcessingError

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access environment variables
pretrained_model_name_or_path = os.getenv('pretrained_model_name_or_path', "./IDM-VTON")
output_dir = os.getenv('output_dir', "./output_sample_images")
data_dir = os.getenv('data_dir', "./sample_images")


class Args:
    """Class to hold model configuration and arguments."""
    
    # Model hyperparameters
    output_dir = output_dir
    seed = 42
    guidance_scale = 2.0
    mixed_precision = 'fp16'  # Options: no | fp16 | bf16
    num_inference_steps = 30

    # Data paths and configurations
    data_dir = data_dir
    height = 1024
    width = 768
    _output_filename = None


def pil_to_tensor(images):
    """
    Convert a PIL image to a PyTorch tensor.

    Args:
        images: PIL Image to be converted.

    Returns:
        PyTorch tensor representation of the image.
    """
    logger.info("Converting PIL images to PyTorch tensors.")
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


def initialize_model_components():
    """Initialize and return all model components."""
    logger.info(f"Initializing accelerator with mixed precision: {Args.mixed_precision}")
    
    accelerator_project_config = ProjectConfiguration(project_dir=Args.output_dir)
    accelerator = Accelerator(
        mixed_precision=Args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Configure logging verbosity based on process rank
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed for reproducibility
    if Args.seed is not None:
        logger.info(f"Setting seed for reproducibility: {Args.seed}")
        set_seed(Args.seed)

    # Create output directory if it doesn't exist
    if accelerator.is_main_process:
        if Args.output_dir is not None:
            os.makedirs(Args.output_dir, exist_ok=True)
            logger.info(f"Created output directory at {Args.output_dir}.")

    # Define weight type based on mixed precision
    weight_dtype = (
        torch.float16 if Args.mixed_precision == "fp16" else torch.float32
    )

    return accelerator, weight_dtype


def load_model_components(weight_dtype):
    """Load and return all model components."""
    logger.info("Loading model components.")
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="image_encoder",
        torch_dtype=weight_dtype,
    )
    
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet_encoder",
        torch_dtype=weight_dtype,
    )
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    )
    
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    return (
        noise_scheduler, vae, unet, image_encoder, unet_encoder,
        text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    )


def setup_pipeline(
    components,
    weight_dtype,
    accelerator
):
    """Set up and return the TryOnPipeline."""
    (
        noise_scheduler, vae, unet, image_encoder, unet_encoder,
        text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    ) = components

    # Freeze models
    for model in [unet, vae, image_encoder, unet_encoder,
                 text_encoder_one, text_encoder_two]:
        model.requires_grad_(False)

    unet_encoder.to(accelerator.device, weight_dtype)
    unet.eval()
    unet_encoder.eval()

    # Initialize pipeline
    pipe = TryonPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        unet_encoder=unet_encoder,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    return pipe


def process_sample(pipe, sample, args):
    """Process a single sample through the pipeline."""
    img_emb_list = [sample['cloth'][i] for i in range(sample['cloth'].shape[0])]
    
    # Prepare prompts
    num_prompts = sample['cloth'].shape[0]
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    # prompt = [sample["caption"]] * num_prompts if not isinstance(
    #     sample["caption"], List) else sample["caption"]
    prompt = sample["caption"]    
    
    # Ensure prompts are lists
    if not isinstance(prompt, List):
        prompt = [prompt] * num_prompts
    if not isinstance(negative_prompt, List):
        negative_prompt = [negative_prompt] * num_prompts

    # Concatenate image embeddings
    image_embeds = torch.cat(img_emb_list, dim=0)

    with torch.inference_mode():
        # Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        prompt = sample["caption_cloth"]
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        (prompt_embeds_c, _, _, _,) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=negative_prompt,
        )
        
        # Set generator
        generator = None
        if args.seed is not None:
            generator = torch.Generator(pipe.device).manual_seed(args.seed)

        try:
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=sample['pose_img'],
                text_embeds_cloth=prompt_embeds_c,
                cloth=sample["cloth_pure"].to(pipe.device),
                mask_image=sample['inpaint_mask'],
                image=(sample['image'] + 1.0) / 2.0,
                sampleheight=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=image_embeds,
            )[0]
            return images
        except Exception as err:
            logger.error("Failed to generate images: %s", str(err))
            raise VTONProcessingError("Failed to generate images") from err


def run_inference(torch_dataloader):
    """Run inference on the provided DataLoader of images and cloth."""
    pil_res_images = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for sample in torch_dataloader:
                images = process_sample(pipe, sample, Args)
                
                for image in images:
                    output_file_name = str(random.randint(0, 1000000))
                    pil_res_images.append({
                        "file_name": output_file_name,
                        'pil_image': image
                    })

    logger.info("Completed inference for all samples.")
    return pil_res_images


def run_vton(data, category, cloth_index=0, model_image_pil=None):
    """Process input data and run the virtual try-on model for inference.

    Args:
        data: Input data containing cloth and model images.
        category: Category of the clothing.
        cloth_index: Index of the selected cloth from the input.
        model_image_pil: PIL image of the model to be used for try-on.

    Returns:
        List of dictionaries containing generated images and their filenames.

    Raises:
        DataLoadingError: If there is an error in data processing.
        VTONProcessingError: If there is an error during inference.
    """
    try:
        torch_dataloader = process_data_s3(
            data,
            category,
            cloth_index,
            model_image_pil
        )
        logger.info("Data processing completed successfully.")
    except DataLoadingError as err:
        logger.error("Error loading data: %s", str(err))
        raise

    try:
        res_pil_images = run_inference(torch_dataloader)
        logger.info(
            "Inference completed successfully, generated %d images.",
            len(res_pil_images)
        )
    except VTONProcessingError as err:
        logger.error("Error during inference: %s", str(err))
        raise

    return res_pil_images


# Initialize model components and pipeline
accelerator, weight_dtype = initialize_model_components()
model_components = load_model_components(weight_dtype)
pipe = setup_pipeline(model_components, weight_dtype, accelerator)