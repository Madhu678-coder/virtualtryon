import os
import sys
import uuid
import json
from datetime import datetime, timezone
from io import BytesIO

import boto3
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from PIL import Image

# Add parent directory to path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

load_dotenv(os.path.join(PARENT_DIR, ".env"))

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

app = Flask(__name__)

# AWS config
REGION = os.getenv("aws_region", "us-east-1")
QUEUE_URL = os.getenv("queue_url", "")
VTON_TABLE = os.getenv("vton_table", "vton-collection")
OUTPUT_BUCKET = os.getenv("default_vton_output_bucket", "groome-results-1")
OUTPUT_FOLDER = os.getenv("default_vton_output_folder", "vton_api_outputs")
PREPROCESSED_BUCKET = os.getenv("preprocessed_bucket", "vton-preprocessed-1")
PRODUCT_BUCKET = os.getenv("product_bucket", "product-images-groome-1")

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(VTON_TABLE)

# Initialize preprocessing models
GPU_ID = int(os.getenv("gpu_id", "0"))
parsing_model = Parsing(GPU_ID)
openpose_model = OpenPose(GPU_ID)


def upload_pil_to_s3(pil_image, bucket, key, fmt="PNG"):
    """Upload a PIL image to S3."""
    buf = BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType=f"image/{fmt.lower()}")


def generate_mask_from_parsing(parsed_image, category):
    """Generate clothing mask from parsed segmentation image.
    
    Parsing labels (ATR dataset):
    0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes,
    5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe,
    11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm,
    16: Bag, 17: Scarf, 18: Neck
    """
    parsing_array = np.array(parsed_image)

    if category == "upper":
        mask = np.isin(parsing_array, [4]).astype(np.uint8) * 255
    elif category == "lower":
        mask = np.isin(parsing_array, [5, 6]).astype(np.uint8) * 255
    elif category == "dress":
        mask = np.isin(parsing_array, [7]).astype(np.uint8) * 255
    else:
        mask = np.isin(parsing_array, [4, 5, 6]).astype(np.uint8) * 255

    return Image.fromarray(mask, mode="L")


def preprocess_person_image(pil_image, user_id, request_id):
    """Run human parsing and openpose, upload all results to S3."""
    # Resize to expected dimensions
    pil_image = pil_image.resize((768, 1024))
    prefix = f"{user_id}/{request_id}"

    # Save temp image for parsing (it expects a directory)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "input.jpg")
        pil_image.save(img_path)

        # Run human parsing
        parsed_image, face_mask = parsing_model(tmpdir)

    # Run openpose
    keypoints = openpose_model(pil_image, resolution=384)

    # Generate pose image (simple keypoint visualization)
    pose_img = draw_pose_image(keypoints, 768, 1024)

    # Generate masks for all categories
    upper_mask = generate_mask_from_parsing(parsed_image, "upper")
    lower_mask = generate_mask_from_parsing(parsed_image, "lower")
    dress_mask = generate_mask_from_parsing(parsed_image, "dress")

    # Upload all to S3 with naming convention: prefix_key.png
    # aws_utils.py splits filename on '_' and uses last part as dict key
    upload_pil_to_s3(pil_image, PREPROCESSED_BUCKET, f"{prefix}/0_image.png")
    upload_pil_to_s3(pose_img, PREPROCESSED_BUCKET, f"{prefix}/1_pose.png")
    upload_pil_to_s3(upper_mask, PREPROCESSED_BUCKET, f"{prefix}/2_upper-mask.png")
    upload_pil_to_s3(lower_mask, PREPROCESSED_BUCKET, f"{prefix}/3_lower-mask.png")
    upload_pil_to_s3(dress_mask, PREPROCESSED_BUCKET, f"{prefix}/4_dress-mask.png")


def draw_pose_image(keypoints, width, height):
    """Draw a simple pose image from openpose keypoints."""
    import cv2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    points = keypoints["pose_keypoints_2d"]

    # Scale keypoints to image size
    scaled = []
    for pt in points:
        x = int(pt[0] * width / 384)
        y = int(pt[1] * height / 512)
        scaled.append((x, y))

    # Draw keypoints
    for pt in scaled:
        if pt[0] > 0 or pt[1] > 0:
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1)

    # Draw limb connections
    limbs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (0, 14), (0, 15), (14, 16), (15, 17)
    ]
    for a, b in limbs:
        if a < len(scaled) and b < len(scaled):
            pa, pb = scaled[a], scaled[b]
            if (pa[0] > 0 or pa[1] > 0) and (pb[0] > 0 or pb[1] > 0):
                cv2.line(canvas, pa, pb, (255, 255, 255), 2)

    return Image.fromarray(canvas)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tryon", methods=["POST"])
def submit_tryon():
    person_file = request.files.get("person_image")
    garment_file = request.files.get("garment_image")
    category = request.form.get("category", "upper")

    if not person_file or not garment_file:
        return jsonify({"error": "Both person and garment images are required"}), 400

    user_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    try:
        # Process person image (parsing + openpose + masks + upload)
        person_img = Image.open(person_file).convert("RGB")
        preprocess_person_image(person_img, user_id, request_id)

        # Upload garment image
        garment_key = f"validated_product_images/products/{request_id}/image_1.png"
        garment_img = Image.open(garment_file).convert("RGB")
        upload_pil_to_s3(garment_img, PRODUCT_BUCKET, garment_key)

        # Create DynamoDB record
        table.put_item(Item={
            "user_id": user_id,
            "request_id": request_id,
            "category": category,
            "customer_images": f"s3://{PREPROCESSED_BUCKET}/{user_id}/{request_id}",
            "product_images": [f"s3://{PRODUCT_BUCKET}/{garment_key}"],
            "output_image": f"s3://{OUTPUT_BUCKET}/{OUTPUT_FOLDER}",
            "pre_processing_status": "COMPLETED",
            "vton_process_status": "PENDING",
            "vton_s3_uri": "",
            "timestamp": str(datetime.now(timezone.utc)),
        })

        # Send SQS message
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps({
                "user_id": user_id,
                "request_id": request_id,
            }),
        )

        return jsonify({"user_id": user_id, "request_id": request_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<user_id>/<request_id>")
def check_status(user_id, request_id):
    resp = table.get_item(Key={"user_id": user_id, "request_id": request_id})
    item = resp.get("Item")
    if not item:
        return jsonify({"error": "Not found"}), 404

    result = {
        "status": item.get("vton_process_status", "PENDING"),
        "result_url": "",
    }

    if result["status"] == "COMPLETED" and item.get("vton_s3_uri"):
        s3_uri = item["vton_s3_uri"]
        bucket = s3_uri.replace("s3://", "").split("/")[0]
        key = "/".join(s3_uri.replace("s3://", "").split("/")[1:])
        result["result_url"] = s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
        )

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
