import boto3
import logging
from botocore.exceptions import ClientError
from PIL import Image
from io import BytesIO
import os
from logging_config import logger


# Initialize the DynamoDB client
dynamodb = boto3.resource('dynamodb')

# Initialize S3 client
s3 = boto3.client('s3')

from dotenv import load_dotenv
# Load environment variables
load_dotenv()


# Environment variables with default values
VTON_TABLE = os.getenv('vton_table', "vton_processing")
VTON_PAR_KEY_NAME = os.getenv('vton_par_key_name', 'user_id')


def download_pil_image(bucket_name, file_key):
    """
    Downloads an image from S3 and loads it as a PIL image.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the S3 bucket.

    Returns:
        PIL.Image: The downloaded image as a PIL object.

    Raises:
        Exception: If the download or image processing fails.
    """
    try:
        logger.info(
            "Downloading image from S3 bucket %s, key: %s",
            bucket_name,
            file_key
        )
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        img_data = response['Body'].read()

        # Load the image using PIL
        img = Image.open(BytesIO(img_data))
        img.verify()  # Verify the file is a valid image
        img = Image.open(BytesIO(img_data))  # Reload after verification
        logger.info("Image successfully downloaded and loaded as a PIL object")
        return img
    except ClientError as e:
        logger.error(
            "Failed to download image from S3. Bucket: %s, Key: %s, Error: %s",
            bucket_name,
            file_key,
            e
        )
        raise e
    except IOError as e:
        logger.error(
            "Error loading image from bytes. File might be corrupted. Key: %s, Error: %s",
            file_key,
            e
        )
        raise e
    except Exception as e:
        logger.error("Unexpected error occurred: %s", e)
        raise e


def process_image_file(bucket_name, file_key, pil_img_list, pil_img_dict):
    """Helper function to process individual image files."""
    file_name = file_key.split('/')[-1].split('.')[0]
    file_extension = file_key.split('.')[-1].lower()

    if file_extension not in ['jpg', 'jpeg', 'png']:
        logger.warning("Unsupported file format for file: %s", file_key)
        return

    pil_img = download_pil_image(bucket_name, file_key)
    if not pil_img:
        logger.warning(
            "Skipping file %s due to download or loading error",
            file_key
        )
        return

    pil_img_list.append({"file_name": file_name, "pil_image": pil_img})
    pil_img_dict[file_name.split('_')[-1]] = {
        "file_name": file_name,
        "pil_image": pil_img
    }
    logger.info("Added image %s to list and dictionary", file_name)


def get_pil_images_from_s3_folder(bucket_name, folder):
    """
    Fetches a list of images from an S3 folder and loads them as PIL images.

    Args:
        bucket_name (str): Name of the S3 bucket.
        folder (str): Folder prefix in the S3 bucket.

    Returns:
        dict: A dictionary with lists of PIL images and their metadata.

    Raises:
        Exception: If fetching or loading images fails.
    """
    try:
        logger.info("Fetching list of image objects from S3 folder: %s", folder)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)
        pil_img_list = []
        pil_img_dict = {}

        for obj in response.get('Contents', []):
            process_image_file(
                bucket_name,
                obj['Key'],
                pil_img_list,
                pil_img_dict
            )

        return {"pil_img_list": pil_img_list, "pil_img_dict": pil_img_dict}

    except ClientError as e:
        logger.error("Failed to list objects in S3 folder: %s. Error: %s", folder, e)
        raise e
    except Exception as e:
        logger.error("Unexpected error occurred while fetching images from S3: %s", e)
        raise e


def upload_pil_image_to_s3(pil_image: Image, bucket_name: str, prefix: str):
    """
    Uploads a PIL image to an S3 bucket.

    Args:
        pil_image (Image): The PIL image to upload.
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The S3 key prefix for the uploaded image.

    Returns:
        str: The S3 URL of the uploaded image.

    Raises:
        Exception: If the upload fails.
    """
    try:
        buffer = BytesIO()
        logger.info("Created a bytes buffer for the image")

        pil_image.save(buffer, format="JPEG")
        logger.info("Image saved to buffer in JPEG format")

        buffer.seek(0)
        logger.info("Buffer position reset to the beginning")

        s3.put_object(
            Bucket=bucket_name,
            Key=prefix,
            Body=buffer,
            ContentType='image/jpeg'
        )
        logger.info("Image successfully uploaded to s3://%s/%s", bucket_name, prefix)
        return f"s3://{bucket_name}/{prefix}"

    except Exception as e:
        logger.error("Error uploading image: %s", e)
        raise e


def build_update_expressions(update_attributes):
    """Helper function to build DynamoDB update expressions."""
    update_expression = "SET "
    expression_attribute_values = {}
    
    for idx, (attr, value) in enumerate(update_attributes.items()):
        update_expression += f"{attr} = :val{idx}, "
        expression_attribute_values[f":val{idx}"] = value
    
    return update_expression.rstrip(", "), expression_attribute_values


def update_dynamodb_item(
    table_name: str,
    partition_key_name: str,
    partition_key: str,
    sort_key_name: str = None,
    sort_key: str = None,
    update_attributes: dict = None
):
    """
    Updates an item in DynamoDB with specified attributes.

    Args:
        table_name (str): Name of the DynamoDB table.
        partition_key_name (str): Partition key name.
        partition_key (str): Partition key value.
        sort_key_name (str, optional): Sort key name.
        sort_key (str, optional): Sort key value.
        update_attributes (dict): Dictionary of attributes to update.

    Returns:
        dict: Response from DynamoDB.

    Raises:
        Exception: If the update fails.
    """
    if not update_attributes:
        logger.warning("No attributes provided for update")
        return None

    table = dynamodb.Table(table_name)
    key = {partition_key_name: partition_key}
    
    if sort_key_name and sort_key:
        key[sort_key_name] = sort_key

    update_expression, expression_attribute_values = build_update_expressions(
        update_attributes
    )

    try:
        logger.info("Updating DynamoDB item with key: %s", key)
        response = table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ReturnValues="UPDATED_NEW"
        )
        logger.info("Update successful for key: %s", key)
        return response
    except ClientError as e:
        logger.error(
            "Failed to update item in DynamoDB: %s",
            e.response['Error']['Message']
        )
        raise e
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise e


def query_dynamodb(
    table_name,
    partition_key,
    partition_value,
    sort_key=None,
    sort_value=None,
    columns=None
):
    """
    Queries DynamoDB for items matching the specified keys and columns.

    Args:
        table_name (str): Name of the DynamoDB table.
        partition_key (str): Partition key name.
        partition_value (str): Partition key value.
        sort_key (str, optional): Sort key name.
        sort_value (str, optional): Sort key value.
        columns (list, optional): List of columns to retrieve.

    Returns:
        list: List of items matching the query.

    Raises:
        Exception: If the query fails.
    """
    table = dynamodb.Table(table_name)

    key_condition_expression = f"{partition_key} = :pk"
    expression_attribute_values = {":pk": partition_value}
    
    if sort_key and sort_value:
        key_condition_expression += f" AND {sort_key} = :sk"
        expression_attribute_values[":sk"] = sort_value

    query_args = {
        "KeyConditionExpression": key_condition_expression,
        "ExpressionAttributeValues": expression_attribute_values
    }
    
    if columns:
        query_args["ProjectionExpression"] = ", ".join(columns)

    try:
        logger.info("Querying DynamoDB with keys: %s", query_args)
        response = table.query(**query_args)
        return response.get('Items', [])
    except ClientError as e:
        logger.error("Error querying DynamoDB: %s", e.response['Error']['Message'])
        raise e
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise e


def parse_s3_uri(s3_uri):
    """
    Parses an S3 URI into bucket name and key.

    Args:
        s3_uri (str): The S3 URI to parse.

    Returns:
        tuple: A tuple containing bucket name and key.

    Raises:
        ValueError: If the URI is invalid.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI. It should start with 's3://'")
    
    uri = s3_uri[5:]
    parts = uri.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# Setting logging level to DEBUG for detailed logs
logger.setLevel(logging.DEBUG)