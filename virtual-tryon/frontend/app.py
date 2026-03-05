import os
import sys
import uuid
import json
import threading
from datetime import datetime, timezone
from io import BytesIO

import boto3
import numpy as np
import torch
import requests as http_requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from PIL import Image

# Add parent directory to path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

load_dotenv(os.path.join(PARENT_DIR, ".env"))

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from preprocess.openpose.annotator.util import resize_image, HWC3

app = Flask(__name__)

# AWS config
REGION = os.getenv("aws_region", "us-east-1")
QUEUE_URL = os.getenv("queue_url", "")
VTON_TABLE = os.getenv("vton_table", "vton-collection")
OUTPUT_BUCKET = os.getenv("default_vton_output_bucket", "groome-results-1")
OUTPUT_FOLDER = os.getenv("default_vton_output_folder", "vton_api_outputs")
PREPROCESSED_BUCKET = os.getenv("preprocessed_bucket", "vton-preprocessed-1")
PRODUCT_BUCKET = os.getenv("product_bucket", "product-images-groome-1")

# Telegram config
TELEGRAM_TOKEN = os.getenv("telegram_token", "8660907511:AAHYlE-vXQ4_F2Yk2jXUHqZILfHrCJHOigM")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(VTON_TABLE)

# Initialize preprocessing models
GPU_ID = int(os.getenv("gpu_id", "0"))
parsing_model = Parsing(GPU_ID)
openpose_model = OpenPose(GPU_ID)

# Store telegram photos in memory and on disk
TELEGRAM_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "telegram_photos")
os.makedirs(TELEGRAM_PHOTOS_DIR, exist_ok=True)


# --- Telegram Bot ---

def telegram_download_file(file_id):
    """Download a file from Telegram by file_id."""
    resp = http_requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
    file_path = resp.json()["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    return http_requests.get(file_url).content


def telegram_send_message(chat_id, text):
    """Send a text message to a Telegram chat."""
    http_requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text})


def telegram_send_photo(chat_id, photo_bytes, caption=""):
    """Send a photo to a Telegram chat."""
    http_requests.post(
        f"{TELEGRAM_API}/sendPhoto",
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": ("result.png", photo_bytes, "image/png")},
    )


def process_telegram_update(update):
    """Process a single Telegram update."""
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")

    if not chat_id:
        return

    # Handle photo messages
    if "photo" in message:
        # Get the largest photo
        photo = message["photo"][-1]
        file_id = photo["file_id"]

        try:
            photo_bytes = telegram_download_file(file_id)
            img = Image.open(BytesIO(photo_bytes)).convert("RGB")

            # Save with timestamp-based name
            photo_id = str(uuid.uuid4())[:8]
            username = message.get("from", {}).get("first_name", "User")
            filename = f"{username}_{photo_id}.png"
            filepath = os.path.join(TELEGRAM_PHOTOS_DIR, filename)
            img.save(filepath)

            telegram_send_message(chat_id, f"Photo received! You can now select it in the Virtual Try-On UI as '{filename}'.")
        except Exception as e:
            telegram_send_message(chat_id, f"Error processing photo: {str(e)}")

    elif "text" in message:
        text = message["text"]
        if text == "/start":
            telegram_send_message(chat_id, "Welcome to Virtual Try-On Bot!\n\nSend me your photo and it will appear in the web UI for try-on.")
        else:
            telegram_send_message(chat_id, "Send me a photo to use in the Virtual Try-On!")


def telegram_polling():
    """Poll Telegram for updates."""
    offset = 0
    while True:
        try:
            resp = http_requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            updates = resp.json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                process_telegram_update(update)
        except Exception as e:
            print(f"Telegram polling error: {e}")
            import time
            time.sleep(5)


# --- Helper Functions ---

def upload_pil_to_s3(pil_image, bucket, key, fmt="PNG"):
    """Upload a PIL image to S3."""
    buf = BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType=f"image/{fmt.lower()}")


def generate_mask_from_parsing(parsed_image, category):
    """Generate clothing mask from parsed segmentation image.
    
    ATR label scheme from parsing_api.py:
    0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes,
    5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
    11=Head, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
    16=Bag, 17=Scarf, 18=Neck (added by LIP model)
    """
    parsing_array = np.array(parsed_image)
    if category == "upper":
        # Include upper clothes (4) + arms (14,15) to match the full pipeline's
        # hole_fill + arm_mask approach from parsing_api.py
        mask = np.isin(parsing_array, [4, 14, 15]).astype(np.uint8) * 255
    elif category == "lower":
        mask = np.isin(parsing_array, [5, 6]).astype(np.uint8) * 255
    elif category == "dress":
        mask = np.isin(parsing_array, [7, 4, 5, 6, 14, 15]).astype(np.uint8) * 255
    else:
        mask = np.isin(parsing_array, [4, 5, 6, 7, 14, 15]).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def preprocess_person_image(pil_image, user_id, request_id):
    """Run human parsing and openpose using the preprocess folder functions, upload all results to S3."""
    import cv2
    import tempfile

    pil_image = pil_image.resize((768, 1024))
    prefix = f"{user_id}/{request_id}"

    # --- Human Parsing via Parsing class (calls onnx_inference from parsing_api.py) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "input.jpg")
        pil_image.save(img_path)
        parsed_image, face_mask = parsing_model(tmpdir)

    # --- OpenPose via the preprocessor (OpenposeDetector) directly ---
    # This gives us both keypoints AND the detected_map (proper colored pose image)
    input_np = np.asarray(pil_image)
    input_np = HWC3(input_np)
    input_np = resize_image(input_np, 384)  # resizes to 384x512
    H, W, C = input_np.shape
    with torch.no_grad():
        pose, detected_map = openpose_model.preprocessor(input_np, hand_and_face=False)
    # detected_map is the proper colored OpenPose visualization at 384x512
    # Resize to 768x1024 to match the person image
    pose_img_np = cv2.resize(detected_map, (768, 1024), interpolation=cv2.INTER_LANCZOS4)
    pose_img = Image.fromarray(pose_img_np)

    # --- Generate masks from the refined parsing result ---
    upper_mask = generate_mask_from_parsing(parsed_image, "upper")
    lower_mask = generate_mask_from_parsing(parsed_image, "lower")
    dress_mask = generate_mask_from_parsing(parsed_image, "dress")

    # --- Upload all to S3 ---
    upload_pil_to_s3(pil_image, PREPROCESSED_BUCKET, f"{prefix}/0_image.png")
    upload_pil_to_s3(pose_img, PREPROCESSED_BUCKET, f"{prefix}/1_pose.png")
    upload_pil_to_s3(upper_mask, PREPROCESSED_BUCKET, f"{prefix}/2_upper-mask.png")
    upload_pil_to_s3(lower_mask, PREPROCESSED_BUCKET, f"{prefix}/3_lower-mask.png")
    upload_pil_to_s3(dress_mask, PREPROCESSED_BUCKET, f"{prefix}/4_dress-mask.png")




# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/telegram_photos/<filename>")
def serve_telegram_photo(filename):
    return send_from_directory(TELEGRAM_PHOTOS_DIR, filename)


@app.route("/api/telegram-photos")
def list_telegram_photos():
    """List all photos received from Telegram."""
    photos = []
    for f in sorted(os.listdir(TELEGRAM_PHOTOS_DIR), reverse=True):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            photos.append({"filename": f, "url": f"/telegram_photos/{f}"})
    return jsonify(photos)


@app.route("/api/tryon", methods=["POST"])
def submit_tryon():
    garment_file = request.files.get("garment_image")
    category = request.form.get("category", "upper")
    telegram_photo = request.form.get("telegram_photo", "")

    # Person image can come from upload or telegram
    person_file = request.files.get("person_image")

    if not garment_file:
        return jsonify({"error": "Garment image is required"}), 400

    if not person_file and not telegram_photo:
        return jsonify({"error": "Please upload a photo or select one from Telegram"}), 400

    user_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    try:
        # Load person image
        if telegram_photo:
            photo_path = os.path.join(TELEGRAM_PHOTOS_DIR, telegram_photo)
            if not os.path.exists(photo_path):
                return jsonify({"error": "Telegram photo not found"}), 404
            person_img = Image.open(photo_path).convert("RGB")
        else:
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

        return jsonify({
            "user_id": user_id,
            "request_id": request_id,
            "preprocessed": get_preprocessed_urls(user_id, request_id),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_preprocessed_urls(user_id, request_id):
    """Generate presigned URLs for all preprocessed images."""
    prefix = f"{user_id}/{request_id}"
    files = {
        "image": f"{prefix}/0_image.png",
        "pose": f"{prefix}/1_pose.png",
        "upper_mask": f"{prefix}/2_upper-mask.png",
        "lower_mask": f"{prefix}/3_lower-mask.png",
        "dress_mask": f"{prefix}/4_dress-mask.png",
    }
    urls = {}
    for name, key in files.items():
        urls[name] = s3.generate_presigned_url(
            "get_object", Params={"Bucket": PREPROCESSED_BUCKET, "Key": key}, ExpiresIn=3600
        )
    return urls


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
    # Start Telegram bot polling in background thread
    bot_thread = threading.Thread(target=telegram_polling, daemon=True)
    bot_thread.start()
    print("Telegram bot started!")

    app.run(host="0.0.0.0", port=5000, debug=False)
