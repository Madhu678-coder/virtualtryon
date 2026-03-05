import os
import torch.utils.data as data
import torch
from typing import Tuple, List, Dict
from torchvision import transforms
from transformers import CLIPImageProcessor
from logging_config import logger
from vton_errors import DataLoadingError
from aws_utils import (
    query_dynamodb,
    parse_s3_uri,
    get_pil_images_from_s3_folder,
    download_pil_image
)


from dotenv import load_dotenv
load_dotenv()

# Global Constants and Environment Variables
VTON_TABLE = os.getenv('vton_table', None) or "vton_processing"
VTON_PAR_KEY_NAME = os.getenv('vton_par_key_name', None) or 'user_id'
VTON_SORT_KEY_NAME = os.getenv('vton_sort_key_name', None) or 'request_id'

logger.info("VTON_TABLE: %s", VTON_TABLE)
logger.info("VTON_PAR_KEY_NAME: %s", VTON_PAR_KEY_NAME)


class LoadData(data.Dataset):
    """
    A custom dataset class for loading and processing images and masks for
    Virtual Try-On (VTON) tasks.

    Attributes:
        size (Tuple[int, int]): Target image dimensions (height, width).
        transform (Compose): Transformation pipeline for images.
        toTensor (ToTensor): Tensor conversion for masks.
        clip_processor (CLIPImageProcessor): CLIP processor for encoding images.
    """
    def __init__(self, size: Tuple[int, int] = (512, 384)):
        super(LoadData, self).__init__()
        
        self.height = size[0]
        self.width = size[1]
        self.size = size

        # Define transformations for input data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        self.toTensor = transforms.ToTensor()
        self.clip_processor = CLIPImageProcessor()
        
        logger.info("LoadData initialized with image size: %s", self.size)

    def get_processed_data(
        self,
        image_pil: Dict,
        mask_pil: Dict,
        pose_pil: Dict,
        cloth_pil: Dict,
        cloth_annotation: str,
        model_annotation: str,
        return_dataloader: bool = True
    ):
        """
        Processes input images and masks, and prepares data for model inference.

        Args:
            image_pil (Dict): Customer image (filename and PIL image object).
            mask_pil (Dict): Segmentation mask (filename and PIL image object).
            pose_pil (Dict): Pose image (filename and PIL image object).
            cloth_pil (Dict): Clothing image (filename and PIL image object).
            cloth_annotation (str): Caption describing the clothing item.
            model_annotation (str): Caption describing the model's outfit.
            return_dataloader (bool): Whether to return data as DataLoader.

        Returns:
            DataLoader or List[Dict]: Processed data ready for inference.
        """
        c_name = cloth_pil['file_name']
        im_name = image_pil['file_name']
        
        logger.info("Processing data for clothing: %s, image: %s", c_name, im_name)

        # Convert and resize images
        cloth_pil_img = cloth_pil['pil_image'].convert("RGB")
        image_pil_img = image_pil['pil_image'].convert("RGB")
        mask_pil_img = mask_pil['pil_image']
        pose_pil_img = pose_pil['pil_image']
        
        im_pil_big = image_pil_img.resize((self.width, self.height))
        image = self.transform(im_pil_big)
        mask = mask_pil_img.resize((self.width, self.height))
        mask = self.toTensor(mask)[:1]  # Keep only the first channel
        mask = 1 - mask  # Invert mask for processing
        im_mask = image * mask

        # Process pose image
        pose_img = self.transform(pose_pil_img)
        
        # Prepare the data dictionary
        data_dict = {
            "c_name": c_name,
            "im_name": im_name,
            "image": image,
            "cloth_pure": self.transform(cloth_pil_img),
            "cloth": self.clip_processor(
                images=cloth_pil_img,
                return_tensors="pt"
            ).pixel_values,
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption_cloth": "a photo of " + cloth_annotation,
            "caption": "model is wearing a " + model_annotation,
            "pose_img": pose_img,
        }
        
        logger.info("Processed data for image: %s", im_name)

        if return_dataloader:
            logger.info("Creating DataLoader for processed data.")
            torch_dataloader = torch.utils.data.DataLoader(
                [data_dict],
                shuffle=False,
                batch_size=1
            )
            return torch_dataloader
        return [data_dict]

    def get_processed_batch_data(
        self,
        pil_data_list: List[Dict],
        batch_size: int
    ):
        """
        Processes a batch of input PIL data and returns a DataLoader object.

        Args:
            pil_data_list (List[Dict]): List of dictionaries containing PIL data.
            batch_size (int): Batch size for DataLoader.

        Returns:
            DataLoader: DataLoader object for processed batch data.
        """
        logger.info("Processing batch data with batch size: %d", batch_size)
        
        result_list = []
        for pil_data in pil_data_list:
            temp_res = self.get_processed_data(
                pil_data['image'],
                pil_data['mask'],
                pil_data['pose'],
                pil_data['cloth'],
                cloth_annotation="",
                model_annotation="",
                return_dataloader=False
            )
            result_list.append(temp_res[0])
        
        torch_dataloader = torch.utils.data.DataLoader(
            result_list,
            shuffle=False,
            batch_size=batch_size
        )
        logger.info("Batch data processing complete.")
        return torch_dataloader


def fetch_data_from_db(user_id: str, request_id: str) -> Dict:
    """
    Fetches data from DynamoDB based on user ID and request ID.

    Args:
        user_id (str): Partition key value.
        request_id (str): Sort key value.

    Returns:
        Dict: Fetched item with parsed S3 URIs.

    Raises:
        Exception: If no data found or database error occurs.
    """
    try:
        columns = [
            "category",
            "customer_images",
            "product_images",
            "output_image",
            "pre_processing_status"
        ]
        items = query_dynamodb(
            table_name=VTON_TABLE,
            partition_key=VTON_PAR_KEY_NAME,
            partition_value=user_id,
            sort_key=VTON_SORT_KEY_NAME,
            sort_value=request_id,
            columns=columns
        )

        if not items:
            logger.info("No data found in database.")
            raise Exception("No data found in database.")

        item = items[0]
        logger.info("Fetched data from database: %s", item)

        # Parse S3 URIs
        item['human_bucket'], item['human_folder'] = parse_s3_uri(
            item['customer_images']
        )
        item['cloth_bucket'], item['cloth_path'] = [], []

        for product in item['product_images']:
            temp_bucket, temp_path = parse_s3_uri(product)
            item['cloth_bucket'].append(temp_bucket)
            item['cloth_path'].append(temp_path)

        if item.get('output_image'):
            item['output_bucket'], item['output_folder'] = parse_s3_uri(
                item['output_image']
            )

        return item

    except Exception as e:
        logger.error("Error fetching data from database: %s", e)
        raise


def process_data_s3(
    data: Dict,
    category: str,
    cloth_index: int = 0,
    model_image_pil = None,
    height: int = 1024,
    width: int = 768
) -> data.DataLoader:
    """
    Load and prepare data from S3 for inference.

    Args:
        data (Dict): Dictionary containing input parameters.
        category (str): Category of the clothing.
        cloth_index (int): Index of the product image to use.
        model_image_pil: Preprocessed model image if provided.
        height (int): Height of the processed image.
        width (int): Width of the processed image.

    Returns:
        data.DataLoader: Processed data ready for inference.

    Raises:
        ValueError: If no mask is found for the category.
        DataLoadingError: For errors during data processing.
    """
    logger.info("Starting data processing for S3 with inputs: %s", data)

    try:
        # Load customer images from S3
        bucket_name = data['human_bucket']
        folder = data['human_folder']
        logger.info(
            "Loading customer images from s3://%s/%s",
            bucket_name,
            folder
        )
        customer_processed_images = get_pil_images_from_s3_folder(
            bucket_name,
            folder
        )
        customer_s3_images = customer_processed_images['pil_img_dict']

        # Download the cloth image from S3
        cloth_file_name = "cloth-1.jpg"
        cloth_bucket = data['cloth_bucket'][cloth_index]
        cloth_path = data['cloth_path'][cloth_index]
        logger.info(
            "Downloading cloth image from s3://%s/%s",
            cloth_bucket,
            cloth_path
        )
        res_pil = download_pil_image(cloth_bucket, cloth_path)
        res_pil = res_pil.convert("RGB")
        cloth_pil_image = {
            "file_name": cloth_file_name,
            "pil_image": res_pil
        }

        # Prepare annotations
        cloth_anno = category
        model_anno = category

        # Get appropriate mask
        category_to_mask = {
            "upper": "upper-mask",
            "lower": "lower-mask",
            "dress": "dress-mask"
        }
        mask_key = category_to_mask.get(category.lower(), "upper-mask")
        mask_image = customer_s3_images.get(mask_key)

        if mask_image is None:
            raise ValueError(f"No mask found for category '{category}'")

        # Initialize and process data
        loadDataobj = LoadData(size=(height, width))

        if model_image_pil:
            customer_s3_images["image"] = {
                "file_name": "temp",
                "pil_image": model_image_pil
            }

        processed_data = loadDataobj.get_processed_data(
            customer_s3_images["image"],
            mask_image,
            customer_s3_images["pose"],
            cloth_pil_image,
            cloth_anno,
            model_anno
        )
        return processed_data

    except Exception as e:
        logger.error("Error during data processing: %s", e)
        raise DataLoadingError(f"Error creating processed data: {str(e)}")