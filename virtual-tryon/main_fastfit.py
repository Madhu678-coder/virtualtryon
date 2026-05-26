"""
FastFit SQS Poller - Dedicated footwear/bags try-on worker.

This is a standalone worker designed to run on a separate EC2 instance
(e.g., g4dn.xlarge with T4 16GB) specifically for footwear and bags
try-on using the FastFit model.

It polls the same SQS queue as main.py but only processes messages
where category is 'shoes', 'footwear', or 'bags'. Other categories
are skipped (left for the IDM-VTON worker to handle).

Usage:
    python main_fastfit.py

Environment Variables (in .env):
    queue_url: SQS queue URL to poll
    vton_table: DynamoDB table name
    fastfit_repo_path: Path to cloned FastFit repo
    fastfit_model_id: HuggingFace model ID for FastFit
"""

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
from fastfit_inference import run_fastfit, get_fastfit_pipeline
from logging_config import logger
from vton_errors import (
    DataLoadingError,
    VTONProcessingError,
    MissingDataError,
)

# Environment variables
queue_url = os.getenv(
    'queue_url',
    "https://sqs.ap-southeast-2.amazonaws.com/730335611421/VTON"
)
VTON_TABLE = os.getenv('vton_table', "vton-collection")
VTON_PAR_KEY_NAME = os.getenv('vton_par_key_name', 'user_id')
VTON_SORT_KEY_NAME = os.getenv('vton_sort_key_name', 'request_id')
DEFAULT_VTON_OUTPUT_BUCKET = os.getenv(
    "default_vton_output_bucket", "groome-results"
)
DEFAULT_VTON_OUTPUT_FOLDER = os.getenv(
    "default_vton_output_folder", "vton_api_outputs"
)

# Categories handled by this worker
FASTFIT_CATEGORIES = {"shoes", "footwear", "bags", "bag"}

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
    """Update VTON status in DynamoDB."""
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
    """Handle SQS processing errors."""
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
    """Upload processed images to S3."""
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


def process_footwear_bags(post_data: Dict) -> List:
    """Process footwear or bags try-on using FastFit.

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
    return run_fastfit(data, post_data["category"])


def process_sqs_message(data: Dict) -> Optional[bool]:
    """Process a single SQS message.

    Args:
        data: Message data dictionary

    Returns:
        True if processed successfully
        False if preprocessing is pending
        None if category is not handled by this worker (skip)
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

    # Check if this category is handled by FastFit
    category = post_data.get("category", "").lower()
    if category not in FASTFIT_CATEGORIES:
        logger.info(
            "Category '%s' not handled by FastFit worker, skipping.",
            category
        )
        return None  # Signal to not delete message

    if post_data['pre_processing_status'] in ["PENDING", ""]:
        logger.info("Pre-processing is pending")
        return False
    elif post_data['pre_processing_status'] == "FAILED":
        raise DataLoadingError("Pre-processing failed")

    try:
        result = process_footwear_bags(post_data)

        output_bucket = (
            post_data.get('output_bucket') or DEFAULT_VTON_OUTPUT_BUCKET
        )
        output_folder = (
            post_data.get("output_folder") or DEFAULT_VTON_OUTPUT_FOLDER
        )

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
        logger.info("Updated DynamoDB with FastFit results")

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
    """Poll SQS queue for footwear/bags messages.

    Only processes messages with categories in FASTFIT_CATEGORIES.
    Messages with other categories are left in the queue for the
    IDM-VTON worker to handle.
    """
    # Pre-load the model on startup
    logger.info("Pre-loading FastFit model...")
    pipeline = get_fastfit_pipeline()
    pipeline.load_model()
    logger.info("FastFit model ready. Starting SQS polling...")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,
                VisibilityTimeout=300,  # 5 min timeout for processing
            )
        except Exception as err:
            logger.error("Error receiving SQS messages: %s", err)
            logger.error(traceback.format_exc())
            sleep(5)
            continue

        messages = response.get('Messages', [])

        for message in messages:
            receipt_handle = message['ReceiptHandle']
            post_data = {}

            try:
                post_data = json.loads(message['Body'])
                logger.info(
                    "Received message ID: %s", message['MessageId']
                )
                logger.info(post_data)

                result = process_sqs_message(post_data)

                if result is None:
                    # Not our category — don't delete, let it return
                    # to queue for IDM-VTON worker
                    logger.info(
                        "Skipping non-footwear message: %s",
                        message['MessageId']
                    )
                    continue

                if result is False:
                    # Preprocessing pending — don't delete
                    logger.info(
                        "Pre-processing pending | Message ID: %s",
                        message['MessageId']
                    )
                    continue

            except (MissingDataError, DataLoadingError,
                    VTONProcessingError) as err:
                handle_sqs_error(
                    err, post_data, "Error processing message"
                )
            except Exception as err:
                handle_sqs_error(
                    err,
                    post_data,
                    f"Error processing message {message['MessageId']}"
                )

            # Delete message after successful processing or error
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


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FastFit Worker Starting")
    logger.info("Supported categories: %s", FASTFIT_CATEGORIES)
    logger.info("Queue URL: %s", queue_url)
    logger.info("=" * 60)
    poll_sqs(queue_url)
