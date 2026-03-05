import json
import os
import traceback
from time import sleep
from typing import Dict, List, Optional

import boto3
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from aws_utils import (
    update_dynamodb_item,
    upload_pil_image_to_s3,
)
from data_loader import fetch_data_from_db
from inference import run_vton
from logging_config import logger
from vton_errors import (
    DataLoadingError,
    VTONProcessingError,
    MissingDataError,
)

# Initialize environment variables
queue_url = os.getenv('queue_url', "https://sqs.ap-southeast-2.amazonaws.com/730335611421/VTON")
VTON_TABLE = os.getenv('vton_table', "vton-collection")
VTON_PAR_KEY_NAME = os.getenv('vton_par_key_name', 'user_id')
VTON_SORT_KEY_NAME = os.getenv('vton_sort_key_name', 'request_id')

DEFAULT_VTON_OUTPUT_BUCKET = os.getenv("default_vton_output_bucket", "groome-results")
DEFAULT_VTON_OUTPUT_FOLDER = os.getenv("default_vton_output_folder", "vton_api_outputs")

# Initialize SQS client
sqs = boto3.client('sqs')


class PostData(BaseModel):
    """Model for processing input data from SQS."""
    user_id: str
    request_id: str


def update_vton_status(
    user_id: str,
    request_id: str,
    s3_uri: str = "",
    status: str = "FAILED"
) -> None:
    """Update VTON status in DynamoDB.

    Args:
        user_id: User identifier
        request_id: Request identifier
        s3_uri: S3 URI of processed image
        status: Processing status
    """
    update_dynamodb_item(
        table_name=VTON_TABLE,
        partition_key_name=VTON_PAR_KEY_NAME,
        partition_key=user_id,
        sort_key_name=VTON_SORT_KEY_NAME,
        sort_key=request_id,
        update_attributes={
            "vton_s3_uri": s3_uri,
            "vton_process_status": status,
        }
    )


def handle_sqs_error(
    error: Exception,
    post_data: Dict,
    error_msg: str
) -> None:
    """Handle SQS processing errors.

    Args:
        error: Exception that occurred
        post_data: Message data
        error_msg: Error message to log
    """
    logger.error(error_msg, error)
    logger.error(traceback.format_exc())
    
    if post_data.get("request_id"):
        update_vton_status(
            post_data.get("user_id"),
            post_data.get("request_id")
        )


def upload_to_s3(
    res_pil_images: List,
    upload_bucket_name: str,
    upload_prefix: str,
    filename: Optional[List[str]] = None
) -> List[Dict]:
    """Upload processed images to S3.

    Args:
        res_pil_images: List of processed PIL images
        upload_bucket_name: Target S3 bucket name
        upload_prefix: S3 prefix/folder path
        filename: Optional list of filenames

    Returns:
        List of dictionaries with upload details
    """
    extn = '.png'
    res_images = []

    for idx, pil_image_obj in enumerate(res_pil_images):
        file_name_ = filename[idx] if filename else pil_image_obj['file_name']
        pil_image = pil_image_obj['pil_image']
        upload_path = f"{upload_prefix}/{file_name_}{extn}"

        try:
            ress = upload_pil_image_to_s3(
                pil_image,
                upload_bucket_name,
                upload_path
            )
        except Exception as err:
            logger.error("Failed to upload image: %s", upload_path)
            logger.error(traceback.format_exc())
            raise err

        res_images.append({
            "s3_uri": ress,
            "pil_image": pil_image,
            "file_name": file_name_
        })
        logger.info(
            "Uploaded image to s3://%s/%s",
            upload_bucket_name,
            upload_path
        )

    logger.info("All images uploaded successfully.")
    return res_images


def process_upper_lower_garment(post_data: Dict) -> List:
    """Process both upper and lower garments.

    Args:
        post_data: Processing data dictionary

    Returns:
        List of processed images
    """
    # Process upper garment
    data1 = {
        "human_bucket": post_data["human_bucket"],
        "human_folder": post_data["human_folder"],
        "category": "upper",
        "cloth_bucket": post_data["cloth_bucket"],
        "cloth_path": post_data["cloth_path"],
    }
    result1 = run_vton(data1, "upper", 0)
    logger.info("Processed upper garment successfully")
    res_pil_img = result1[0]['pil_image']

    # Process lower garment with upper result
    data2 = {
        "human_bucket": post_data["human_bucket"],
        "human_folder": post_data["human_folder"],
        "category": "lower",
        "cloth_bucket": post_data["cloth_bucket"],
        "cloth_path": post_data["cloth_path"],
    }
    result2 = run_vton(data2, "lower", 1, res_pil_img)
    logger.info("Processed lower garment successfully")

    return result2 if result2 else "Processing failed"


def process_single_garment(post_data: Dict) -> List:
    """Process a single garment category.

    Args:
        post_data: Processing data dictionary

    Returns:
        List of processed images
    """
    data = {
        "human_bucket": post_data["human_bucket"],
        "human_folder": post_data["human_folder"],
        "category": post_data["category"],
        "cloth_bucket": post_data["cloth_bucket"],
        "cloth_path": post_data["cloth_path"],
    }
    return run_vton(data, post_data["category"])


def process_sqs_message(data: Dict) -> bool:
    """Process a single SQS message.

    Args:
        data: Message data dictionary

    Returns:
        bool: True if processed successfully

    Raises:
        Various exceptions based on processing status
    """
    try:
        sqs_msg = PostData(**data)
    except Exception as err:
        msg = "Error processing message, Some fields are missing"
        logger.error("%s: %s", msg, err)
        raise MissingDataError(f"{msg}: {err}")

    try:
        post_data = fetch_data_from_db(sqs_msg.user_id, sqs_msg.request_id)
        filename = sqs_msg.request_id
    except Exception as err:
        msg = "Error fetching data from database"
        logger.error("%s: %s", msg, err)
        raise DataLoadingError(f"{msg}: {err}")

    if post_data['pre_processing_status'] in ["PENDING", ""]:
        logger.info("Pre-processing is pending")
        return False
    elif post_data['pre_processing_status'] == "FAILED":
        raise DataLoadingError("Pre-processing failed")

    try:
        if post_data["category"] == 'upper_lower':
            result = process_upper_lower_garment(post_data)
        else:
            result = process_single_garment(post_data)

        output_bucket = (post_data.get('output_bucket') or
                        DEFAULT_VTON_OUTPUT_BUCKET)
        output_folder = (post_data.get("output_folder") or
                        DEFAULT_VTON_OUTPUT_FOLDER)
        
        result_s3 = upload_to_s3(
            result,
            output_bucket,
            output_folder,
            filename=[filename]
        )
        logger.info("Uploaded processed result to S3: %s", result_s3)

        update_vton_status(
            sqs_msg.user_id,
            sqs_msg.request_id,
            result_s3[0]['s3_uri'],
            "COMPLETED"
        )
        logger.info("Updated DynamoDB with VTON results")

    except (DataLoadingError, VTONProcessingError) as err:
        logger.error("Processing error: %s", err)
        logger.error(traceback.format_exc())
        raise

    except Exception as err:
        logger.error("Unexpected error: %s", err)
        logger.error(traceback.format_exc())
        raise

    return True


def poll_sqs(queue_url: str) -> None:
    """Poll SQS queue for messages.

    Args:
        queue_url: SQS queue URL
    """
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10
            )
        except Exception as err:
            logger.error("Error receiving SQS messages: %s", err)
            logger.error(traceback.format_exc())
            sleep(5)
            continue

        messages = response.get('Messages', [])
        
        for message in messages:
            receipt_handle = message['ReceiptHandle']
            try:
                post_data = json.loads(message['Body'])
                logger.info(
                    "Processing message ID: %s",
                    message['MessageId']
                )
                logger.info(post_data)

                if not process_sqs_message(post_data):
                    logger.error(
                        "Pre-processing pending | Message ID: %s",
                        message['MessageId']
                    )
                    continue

            except (MissingDataError, DataLoadingError,
                    VTONProcessingError) as err:
                handle_sqs_error(
                    err,
                    post_data,
                    "Error processing message"
                )
            except Exception as err:
                handle_sqs_error(
                    err,
                    post_data,
                    f"Error processing message {message['MessageId']}"
                )

            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info(
                "Processed and deleted message: %s",
                message['MessageId']
            )

        if not messages:
            logger.info("Queue empty, waiting...")
        sleep(5)


def start_polling(queue_url: str) -> None:
    """Start SQS polling process.

    Args:
        queue_url: SQS queue URL
    """
    logger.info("Starting to poll SQS queue: %s", queue_url)
    poll_sqs(queue_url)


if __name__ == "__main__":
    start_polling(queue_url)