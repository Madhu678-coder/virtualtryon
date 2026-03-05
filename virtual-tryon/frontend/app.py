import os
import sys
import uuid
import json
import cv2
import tempfile
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

# Import preprocessing classes and utilities from the preprocess folder
# Parsing: runs onnx_inference() from parsing_api.py which does:
#   - ATR model (512x512) for body parsing with 18 labels
#   - hole_fill + refine_hole + arm_mask refinement on upper cloth region
#   - LIP model (473x473) for neck parsing (adds label 18)
#   - Returns palette image (labels 0-18) + face_mask
from preprocess.humanparsing.run_parsing import Parsing

# OpenPose: runs body pose estimation
# OpenPose.__call__ returns only keypoints (discards detected_map)
# OpenPose.preprocessor is the OpenposeDetector which returns (pose, detected_map)
# detected_map is the proper colored pose visualization from draw_pose()
from preprocess.openpose.run_openpose import OpenPose

# Utilities used by OpenPose internally to prepare input images
# HWC3: ensures 3-channel uint8
# resize_image: resizes to target resolution maintaining 64-pixel alignment
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
TELEGRAM_TOKEN = os.getenv("telegram_token", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(VTON_TABLE)

# Initialize preprocessing models (same as what the IDM-VTON pipeline uses)
GPU_ID = int(os.getenv("gpu_id", "0"))
parsing_model = Parsing(GPU_ID)
openpose_model = OpenPose(GPU_ID)

# Store telegram photos on disk
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
    http_requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text})


def telegram_send_photo(chat_id, photo_bytes, caption=""):
    http_requests.post(
        f"{TELEGRAM_API}/sendPhoto",
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": ("result.png", photo_bytes, "image/png")},
    )


def process_telegram_update(update):
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    if not chat_id:
        return

    if "photo" in message:
        photo = message["photo"][-1]
        file_id = photo["file_id"]
        try:
            photo_bytes = telegram_download_file(file_id)
            img = Image.open(BytesIO(photo_bytes)).convert("RGB")
            photo_id = str(uuid.uuid4())[:8]
            username = message.get("from", {}).get("first_name", "User")
            filename = f"{username}_{photo_id}.png"
            filepath = os.path.join(TELEGRAM_PHOTOS_DIR, filename)
            img.save(filepath)
            telegram_send_message(chat_id, f"Photo received! Select it in the Virtual Try-On UI as '{filename}'.")
        except Exception as e:
            telegram_send_message(chat_id, f"Error processing photo: {str(e)}")
    elif "text" in message:
        text = message["text"]
        if text == "/start":
            telegram_send_message(chat_id, "Welcome to Virtual Try-On Bot!\n\nSend me your photo and it will appear in the web UI for try-on.")
        else:
            telegram_send_message(chat_id, "Send me a photo to use in the Virtual Try-On!")


def telegram_polling():
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


# --- Preprocessing Helper Functions ---

def upload_pil_to_s3(pil_image, bucket, key, fmt="PNG"):
    """Upload a PIL image to S3."""
    buf = BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType=f"image/{fmt.lower()}")


def generate_mask_from_parsing(parsed_image, category):
    """
    Generate a binary clothing mask from the refined parsing result.

    The parsed_image is a palette-mode PIL image from onnx_inference() in parsing_api.py.
    np.array() on a palette image returns label indices (0-18), NOT RGB values.

    ATR label scheme (after full onnx_inference refinement):
      0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes,
      5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
      11=Head, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
      16=Bag, 17=Scarf, 18=Neck (added by LIP model)

    The backend data_loader.py processes the mask as:
      mask = toTensor(mask)[:1]   -> first channel, 0-255 mapped to 0.0-1.0
      mask = 1 - mask             -> invert: clothing=0, rest=1
      im_mask = image * mask      -> blacks out clothing region (agnostic image)

    So the mask must be: white (255) where clothing is, black (0) elsewhere.
    Including arms (14,15) in upper/dress masks matches parsing_api.py's
    hole_fill + arm_mask approach which treats arms as part of the upper region.
    """
    parsing_array = np.array(parsed_image)

    if category == "upper":
        mask = np.isin(parsing_array, [4, 14, 15]).astype(np.uint8) * 255
    elif category == "lower":
        mask = np.isin(parsing_array, [5, 6]).astype(np.uint8) * 255
    elif category == "dress":
        mask = np.isin(parsing_array, [7, 4, 5, 6, 14, 15]).astype(np.uint8) * 255
    else:
        mask = np.isin(parsing_array, [4, 5, 6, 7, 14, 15]).astype(np.uint8) * 255

    return Image.fromarray(mask, mode="L")


def preprocess_person_image(pil_image, user_id, request_id):
    """
    Run the full IDM-VTON preprocessing pipeline on a person image.

    This replicates exactly what the preprocess folder does:
    1. Human Parsing via Parsing class -> onnx_inference() from parsing_api.py
       - ATR model at 512x512 with hole_fill, refine_hole, arm_mask refinement
       - LIP model at 473x473 for neck parsing
       - Returns refined palette image with labels 0-18
    2. OpenPose via OpenposeDetector -> body estimation + draw_pose()
       - Resizes input to 384x512 (via resize_image with 64-pixel alignment)
       - Runs body pose estimation (body.py)
       - draw_pose() -> draw_bodypose() creates colored pose visualization
       - detected_map is in BGR (cv2 convention), must convert to RGB for PIL

    Uploads to S3 with naming convention that matches aws_utils.py's
    process_image_file() which extracts dict keys via file_name.split('_')[-1]:
      0_image.png   -> key "image"
      1_pose.png    -> key "pose"
      2_upper-mask.png -> key "upper-mask"
      3_lower-mask.png -> key "lower-mask"
      4_dress-mask.png -> key "dress-mask"

    data_loader.py's process_data_s3() then looks up:
      customer_s3_images["image"], ["pose"], ["upper-mask"], etc.
    """
    # Resize to 768x1024 (width x height) - the standard IDM-VTON input size
    pil_image = pil_image.resize((768, 1024))
    prefix = f"{user_id}/{request_id}"

    # --- Step 1: Human Parsing ---
    # Parsing class expects a directory path containing image files.
    # SimpleFolderDataset in datasets/simple_extractor_dataset.py reads all
    # images from the directory, applies affine transform (center/scale),
    # and feeds them through the ONNX models.
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "input.jpg")
        pil_image.save(img_path)
        parsed_image, face_mask = parsing_model(tmpdir)

    # --- Step 2: OpenPose ---
    # We call openpose_model.preprocessor (OpenposeDetector) directly instead of
    # openpose_model() because __call__ discards the detected_map and only
    # returns keypoints. We need the detected_map (colored pose visualization).
    #
    # Replicate the same input preparation that OpenPose.__call__ does:
    #   input_image = HWC3(input_image)        -> ensure 3-channel uint8
    #   input_image = resize_image(input_image, 384)  -> resize to 384x512
    input_np = np.asarray(pil_image).copy()  # copy to make it writable
    input_np = HWC3(input_np)
    input_np = resize_image(input_np, 384)

    with torch.no_grad():
        pose, detected_map = openpose_model.preprocessor(input_np, hand_and_face=False)

    # detected_map is in BGR (OpenposeDetector flips to BGR before body estimation,
    # and draw_pose uses cv2 drawing functions which work in BGR).
    # The commented code in run_openpose.py confirms this:
    #   output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
    # Convert BGR -> RGB for PIL, then resize to 768x1024
    pose_img_rgb = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
    pose_img_resized = cv2.resize(pose_img_rgb, (768, 1024), interpolation=cv2.INTER_LANCZOS4)
    pose_img = Image.fromarray(pose_img_resized)

    # --- Step 3: Generate masks from refined parsing result ---
    upper_mask = generate_mask_from_parsing(parsed_image, "upper")
    lower_mask = generate_mask_from_parsing(parsed_image, "lower")
    dress_mask = generate_mask_from_parsing(parsed_image, "dress")

    # --- Step 4: Upload all preprocessed images to S3 ---
    # File naming must match what aws_utils.process_image_file expects:
    # It splits filename by '_' and takes the last part as the dict key.
    upload_pil_to_s3(pil_image, PREPROCESSED_BUCKET, f"{prefix}/0_image.png")
    upload_pil_to_s3(pose_img, PREPROCESSED_BUCKET, f"{prefix}/1_pose.png")
    upload_pil_to_s3(upper_mask, PREPROCESSED_BUCKET, f"{prefix}/2_upper-mask.png")
    upload_pil_to_s3(lower_mask, PREPROCESSED_BUCKET, f"{prefix}/3_lower-mask.png")
    upload_pil_to_s3(dress_mask, PREPROCESSED_BUCKET, f"{prefix}/4_dress-mask.png")


# --- S3 Presigned URL Helper ---

def get_preprocessed_urls(user_id, request_id):
    """Generate presigned URLs for all preprocessed images so the UI can display them."""
    prefix = f"{user_id}/{request_id}"
    urls = {}
    key_map = {
        "image": "0_image.png",
        "pose": "1_pose.png",
        "upper_mask": "2_upper-mask.png",
        "lower_mask": "3_lower-mask.png",
        "dress_mask": "4_dress-mask.png",
    }
    for name, filename in key_map.items():
        try:
            urls[name] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": PREPROCESSED_BUCKET, "Key": f"{prefix}/{filename}"},
                ExpiresIn=3600,
            )
        except Exception:
            urls[name] = ""
    return urls


# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/telegram_photos/<filename>")
def serve_telegram_photo(filename):
    return send_from_directory(TELEGRAM_PHOTOS_DIR, filename)


@app.route("/api/telegram-photos")
def list_telegram_photos():
    photos = []
    if os.path.isdir(TELEGRAM_PHOTOS_DIR):
        for f in sorted(os.listdir(TELEGRAM_PHOTOS_DIR), reverse=True):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                photos.append({"filename": f, "url": f"/telegram_photos/{f}"})
    return jsonify(photos)


@app.route("/api/tryon", methods=["POST"])
def submit_tryon():
    try:
        category = request.form.get("category", "upper")
        user_id = str(uuid.uuid4())[:8]
        request_id = str(uuid.uuid4())[:8]

        # Get person image from upload or telegram
        telegram_photo = request.form.get("telegram_photo")
        if telegram_photo:
            photo_path = os.path.join(TELEGRAM_PHOTOS_DIR, telegram_photo)
            if not os.path.exists(photo_path):
                return jsonify({"error": "Telegram photo not found"}), 400
            person_img = Image.open(photo_path).convert("RGB")
        else:
            person_file = request.files.get("person_image")
            if not person_file:
                return jsonify({"error": "No person image provided"}), 400
            person_img = Image.open(person_file.stream).convert("RGB")

        garment_file = request.files.get("garment_image")
        if not garment_file:
            return jsonify({"error": "No garment image provided"}), 400
        garment_img = Image.open(garment_file.stream).convert("RGB")

        # Run full IDM-VTON preprocessing pipeline on person image
        preprocess_person_image(person_img, user_id, request_id)

        # Upload garment to product bucket
        garment_key = f"{user_id}/{request_id}/garment.jpg"
        buf = BytesIO()
        garment_img.save(buf, format="JPEG")
        buf.seek(0)
        s3.put_object(Bucket=PRODUCT_BUCKET, Key=garment_key, Body=buf, ContentType="image/jpeg")

        # Create DynamoDB record matching what main.py expects
        table.put_item(Item={
            "user_id": user_id,
            "request_id": request_id,
            "category": category,
            "customer_images": f"s3://{PREPROCESSED_BUCKET}/{user_id}/{request_id}",
            "product_images": [f"s3://{PRODUCT_BUCKET}/{garment_key}"],
            "output_image": f"s3://{OUTPUT_BUCKET}/{OUTPUT_FOLDER}",
            "pre_processing_status": "COMPLETED",
            "vton_process_status": "PENDING",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        # Send SQS message to trigger backend processing
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps({"user_id": user_id, "request_id": request_id}),
        )

        # Generate presigned URLs for preprocessed images
        preprocessed = get_preprocessed_urls(user_id, request_id)

        return jsonify({
            "user_id": user_id,
            "request_id": request_id,
            "status": "PENDING",
            "preprocessed": preprocessed,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<user_id>/<request_id>")
def check_status(user_id, request_id):
    try:
        resp = table.get_item(Key={"user_id": user_id, "request_id": request_id})
        item = resp.get("Item", {})
        status = item.get("vton_process_status", "PENDING")
        result_url = ""

        if status == "COMPLETED" and item.get("vton_s3_uri"):
            # Generate presigned URL for the result image
            s3_uri = item["vton_s3_uri"]
            if s3_uri.startswith("s3://"):
                parts = s3_uri[5:].split("/", 1)
                result_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": parts[0], "Key": parts[1]},
                    ExpiresIn=3600,
                )

        return jsonify({"status": status, "result_url": result_url})
    except Exception as e:
        return jsonify({"status": "PENDING", "result_url": ""})


# --- Main ---

if __name__ == "__main__":
    # Start Telegram bot polling in background thread
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=telegram_polling, daemon=True)
        t.start()
        print("Telegram bot polling started.")

    print("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
