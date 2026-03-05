import os
import sys
import uuid
import json
import time
import threading
from datetime import datetime, timezone
from io import BytesIO

import boto3
import requests as http_requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from PIL import Image

# Add parent directory to path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

load_dotenv(os.path.join(PARENT_DIR, ".env"))

app = Flask(__name__)

# AWS config
REGION = os.getenv("aws_region", "us-east-1")
QUEUE_URL = os.getenv("queue_url", "")                          # VTON SQS queue
PREPROCESS_QUEUE_URL = os.getenv("preprocess_queue_url", "")    # Preprocessing SQS queue
VTON_TABLE = os.getenv("vton_table", "vton-collection")
OUTPUT_BUCKET = os.getenv("default_vton_output_bucket", "groome-results-1")
OUTPUT_FOLDER = os.getenv("default_vton_output_folder", "vton_api_outputs")
PREPROCESSED_BUCKET = os.getenv("preprocessed_bucket", "vton-preprocessed-1")
PRODUCT_BUCKET = os.getenv("product_bucket", "product-images-groome-1")
RAW_IMAGES_BUCKET = os.getenv("raw_images_bucket", "product-images-groome-1")  # bucket for raw person uploads

# Telegram config
TELEGRAM_TOKEN = os.getenv("telegram_token", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(VTON_TABLE)

# Store telegram photos on disk
TELEGRAM_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "telegram_photos")
os.makedirs(TELEGRAM_PHOTOS_DIR, exist_ok=True)


# ============================================================
# Telegram Bot
# ============================================================

def telegram_download_file(file_id):
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
            time.sleep(5)


# ============================================================
# S3 Upload Helper
# ============================================================

def upload_pil_to_s3(pil_image, bucket, key, fmt="PNG"):
    buf = BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType=f"image/{fmt.lower()}")


# ============================================================
# S3 Presigned URL Helper — matches preprocessing server output naming
# Preprocessing server outputs: cm_image.png, cm_upper-mask.png,
#   cm_lower-mask.png, cm_dress-mask.png, cm_dense_pose.png
# ============================================================

def get_preprocessed_urls(user_id, request_id):
    prefix = f"{user_id}/{request_id}"
    urls = {}
    key_map = {
        "image": "cm_image.png",
        "pose": "cm_dense_pose.png",
        "upper_mask": "cm_upper-mask.png",
        "lower_mask": "cm_lower-mask.png",
        "dress_mask": "cm_dress-mask.png",
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


# ============================================================
# Flask Routes
# ============================================================

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
    """
    New flow — no local preprocessing:
    1. Upload raw person image to S3
    2. Upload garment image to S3
    3. Create DynamoDB record with pre_processing_status=PENDING
    4. Send SQS message to preprocessing queue (S3 event format)
    5. Return immediately — frontend polls /api/status for results
    """
    try:
        category = request.form.get("category", "upper")
        user_id = str(uuid.uuid4())[:8]
        request_id = str(uuid.uuid4())[:8]

        # Get person image
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

        # Step 1: Upload raw person image to S3
        # The preprocessing server extracts request_id from the image key:
        #   request_id = image_key.split('/')[-1].split('.')[0]
        # So we use: {user_id}/{request_id}.png
        person_key = f"{user_id}/{request_id}.png"
        upload_pil_to_s3(person_img, RAW_IMAGES_BUCKET, person_key)

        # Step 2: Upload garment image to S3
        garment_key = f"{user_id}/{request_id}/garment.jpg"
        buf = BytesIO()
        garment_img.save(buf, format="JPEG")
        buf.seek(0)
        s3.put_object(Bucket=PRODUCT_BUCKET, Key=garment_key, Body=buf, ContentType="image/jpeg")

        # Step 3: Create DynamoDB record with PENDING preprocessing status
        table.put_item(Item={
            "user_id": user_id,
            "request_id": request_id,
            "category": category,
            "customer_images": "",  # will be set by preprocessing server
            "product_images": [f"s3://{PRODUCT_BUCKET}/{garment_key}"],
            "output_image": f"s3://{OUTPUT_BUCKET}/{OUTPUT_FOLDER}",
            "pre_processing_status": "PENDING",
            "vton_process_status": "PENDING",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        # Step 4: Send SQS message to preprocessing queue
        # The preprocessing server expects S3 event format:
        #   payload['detail']['bucket']['name'] and payload['detail']['object']['key']
        preprocess_message = {
            "detail": {
                "bucket": {"name": RAW_IMAGES_BUCKET},
                "object": {"key": person_key}
            }
        }
        sqs.send_message(
            QueueUrl=PREPROCESS_QUEUE_URL,
            MessageBody=json.dumps(preprocess_message),
        )

        return jsonify({
            "user_id": user_id,
            "request_id": request_id,
            "status": "PENDING",
            "message": "Image submitted for preprocessing. Poll /api/status for updates.",
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
        pre_status = item.get("pre_processing_status", "PENDING")
        vton_status = item.get("vton_process_status", "PENDING")
        result_url = ""
        preprocessed = {}

        # If preprocessing is done, return preprocessed image URLs
        if pre_status == "COMPLETED":
            preprocessed = get_preprocessed_urls(user_id, request_id)

        # If VTON is done, return result URL
        if vton_status == "COMPLETED" and item.get("vton_s3_uri"):
            s3_uri = item["vton_s3_uri"]
            if s3_uri.startswith("s3://"):
                parts = s3_uri[5:].split("/", 1)
                result_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": parts[0], "Key": parts[1]},
                    ExpiresIn=3600,
                )

        return jsonify({
            "status": vton_status,
            "pre_processing_status": pre_status,
            "result_url": result_url,
            "preprocessed": preprocessed,
        })
    except Exception:
        return jsonify({"status": "PENDING", "pre_processing_status": "PENDING", "result_url": "", "preprocessed": {}})


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=telegram_polling, daemon=True)
        t.start()
        print("Telegram bot polling started.")

    print("Starting Flask server on port 5000...")
    print(f"Preprocessing queue: {PREPROCESS_QUEUE_URL}")
    print(f"VTON queue: {QUEUE_URL}")
    app.run(host="0.0.0.0", port=5000, debug=False)
