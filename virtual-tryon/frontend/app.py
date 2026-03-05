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
from PIL import Image, ImageDraw

# Add parent directory to path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PARENT_DIR)

load_dotenv(os.path.join(PARENT_DIR, ".env"))

# Import preprocessing classes from the preprocess folder
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from preprocess.openpose.annotator.util import resize_image, HWC3

# DensePose: used by official IDM-VTON for pose image generation
# Falls back to OpenPose skeleton if detectron2/densepose not available
DENSEPOSE_CONFIG = os.path.join(PARENT_DIR, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
DENSEPOSE_MODEL = os.path.join(PARENT_DIR, "ckpt", "densepose", "model_final_162be9.pkl")
USE_DENSEPOSE = False

try:
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    if os.path.isfile(DENSEPOSE_MODEL) and os.path.isfile(DENSEPOSE_CONFIG):
        USE_DENSEPOSE = True
        print("[OK] DensePose available — using official IDM-VTON pose pipeline")
    else:
        print(f"[WARN] DensePose checkpoint not found at {DENSEPOSE_MODEL}")
        print("[WARN] Falling back to OpenPose skeleton for pose image")
except ImportError as e:
    print(f"[WARN] detectron2/densepose not installed: {e}")
    print("[WARN] Falling back to OpenPose skeleton for pose image")

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

# Initialize preprocessing models
GPU_ID = int(os.getenv("gpu_id", "0"))
parsing_model = Parsing(GPU_ID)
openpose_model = OpenPose(GPU_ID)

# Store telegram photos on disk
TELEGRAM_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "telegram_photos")
os.makedirs(TELEGRAM_PHOTOS_DIR, exist_ok=True)


# ============================================================
# Mask generation — exact copy from IDM-VTON utils_mask.py
# https://github.com/yisol/IDM-VTON/blob/main/gradio_demo/utils_mask.py
# ============================================================

label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3,
    "upper_clothes": 4, "skirt": 5, "pants": 6, "dress": 7,
    "belt": 8, "left_shoe": 9, "right_shoe": 10, "head": 11,
    "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17,
}


def extend_arm_mask(wrist, elbow, scale):
    wrist = elbow + scale * (wrist - elbow)
    return wrist


def mask_hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


def mask_refine(mask):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine, contours, i, color=255, thickness=-1)
    return refine


def get_mask_location(model_type, category, model_parse, keypoint, width=384, height=512):
    """Exact copy of IDM-VTON gradio_demo/utils_mask.py get_mask_location()"""
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + \
                     (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data).reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)

    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2,
                     shoulder_left[1] - ARM_LINE_WIDTH // 2,
                     shoulder_left[0] + ARM_LINE_WIDTH // 2,
                     shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2,
                      shoulder_right[1] - ARM_LINE_WIDTH // 2,
                      shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(
                np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(),
                'white', ARM_LINE_WIDTH, 'curve'
            )
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(
                np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(),
                'white', ARM_LINE_WIDTH, 'curve'
            )
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)

    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(
            np.logical_or(im_arms_left, im_arms_right).astype('float32'),
            np.ones((5, 5), np.uint16), iterations=4
        )
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total

    img = np.where(inpaint_mask, 255, 0)
    dst = mask_hole_fill(img.astype(np.uint8))
    dst = mask_refine(dst)
    inpaint_mask = dst / 255 * 1

    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray


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
            import time
            time.sleep(5)


# ============================================================
# Pose image generation
# ============================================================

def generate_densepose_image(pil_image):
    """
    Generate DensePose dp_segm visualization — exact match of IDM-VTON Gradio demo.
    From gradio_demo/app.py:
        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        args = apply_net.create_argument_parser().parse_args(
            ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
             './ckpt/densepose/model_final_162be9.pkl', 'dp_segm',
             '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:,:,::-1]
        pose_img = Image.fromarray(pose_img).resize((768,1024))
    """
    human_img_arg = _apply_exif_orientation(pil_image.resize((384, 512)))
    human_img_bgr = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        'show', DENSEPOSE_CONFIG, DENSEPOSE_MODEL, 'dp_segm',
        '-v', '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_result = args.func(args, human_img_bgr)
    pose_img_rgb = pose_result[:, :, ::-1]  # BGR to RGB
    return Image.fromarray(pose_img_rgb).resize((768, 1024))


def generate_openpose_image(pil_image):
    """
    Fallback: generate OpenPose skeleton visualization.
    Uses OpenposeDetector directly to get detected_map (colored skeleton).
    """
    input_np = np.asarray(pil_image).copy()
    input_np = HWC3(input_np)
    input_np = resize_image(input_np, 384)

    with torch.no_grad():
        _, detected_map = openpose_model.preprocessor(input_np, hand_and_face=False)

    # detected_map is BGR, convert to RGB
    pose_img_rgb = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
    pose_img_resized = cv2.resize(pose_img_rgb, (768, 1024), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(pose_img_resized)


# ============================================================
# Preprocessing — matches IDM-VTON Gradio demo start_tryon()
# ============================================================

def upload_pil_to_s3(pil_image, bucket, key, fmt="PNG"):
    buf = BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf, ContentType=f"image/{fmt.lower()}")


def preprocess_person_image(pil_image, user_id, request_id):
    """
    Exact replication of IDM-VTON gradio_demo/app.py start_tryon() preprocessing:

    Step 1: human_img = human_img_orig.resize((768,1024))
    Step 2: keypoints = openpose_model(human_img.resize((384,512)))
    Step 3: model_parse, _ = parsing_model(human_img.resize((384,512)))
    Step 4: mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
            mask = mask.resize((768,1024))
    Step 5: DensePose at 384x512 for pose image (or OpenPose fallback)
    Step 6: Upload to S3
    """
    # Step 1: Resize to 768x1024
    pil_image = pil_image.resize((768, 1024))
    prefix = f"{user_id}/{request_id}"

    # Step 2: OpenPose keypoints at 384x512
    # Gradio: keypoints = openpose_model(human_img.resize((384,512)))
    keypoints = openpose_model(pil_image.resize((384, 512)))

    # Step 3: Human Parsing at 384x512
    # Gradio: model_parse, _ = parsing_model(human_img.resize((384,512)))
    # Parsing expects a directory path (SimpleFolderDataset), so save to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "input.jpg")
        pil_image.resize((384, 512)).save(img_path)
        parsed_image, face_mask = parsing_model(tmpdir)

    # Step 4: Generate masks using get_mask_location (from utils_mask.py)
    # Gradio: mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
    #         mask = mask.resize((768,1024))
    upper_mask, _ = get_mask_location('hd', 'upper_body', parsed_image, keypoints)
    lower_mask, _ = get_mask_location('hd', 'lower_body', parsed_image, keypoints)
    dress_mask, _ = get_mask_location('hd', 'dresses', parsed_image, keypoints)

    upper_mask = upper_mask.resize((768, 1024), Image.NEAREST)
    lower_mask = lower_mask.resize((768, 1024), Image.NEAREST)
    dress_mask = dress_mask.resize((768, 1024), Image.NEAREST)

    # Step 5: Pose image — DensePose (official) or OpenPose (fallback)
    if USE_DENSEPOSE:
        pose_img = generate_densepose_image(pil_image)
    else:
        pose_img = generate_openpose_image(pil_image)

    # Step 6: Upload all preprocessed images to S3
    upload_pil_to_s3(pil_image, PREPROCESSED_BUCKET, f"{prefix}/0_image.png")
    upload_pil_to_s3(pose_img, PREPROCESSED_BUCKET, f"{prefix}/1_pose.png")
    upload_pil_to_s3(upper_mask, PREPROCESSED_BUCKET, f"{prefix}/2_upper-mask.png")
    upload_pil_to_s3(lower_mask, PREPROCESSED_BUCKET, f"{prefix}/3_lower-mask.png")
    upload_pil_to_s3(dress_mask, PREPROCESSED_BUCKET, f"{prefix}/4_dress-mask.png")


# ============================================================
# S3 Presigned URL Helper
# ============================================================

def get_preprocessed_urls(user_id, request_id):
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
    try:
        category = request.form.get("category", "upper")
        user_id = str(uuid.uuid4())[:8]
        request_id = str(uuid.uuid4())[:8]

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

        preprocess_person_image(person_img, user_id, request_id)

        garment_key = f"{user_id}/{request_id}/garment.jpg"
        buf = BytesIO()
        garment_img.save(buf, format="JPEG")
        buf.seek(0)
        s3.put_object(Bucket=PRODUCT_BUCKET, Key=garment_key, Body=buf, ContentType="image/jpeg")

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

        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps({"user_id": user_id, "request_id": request_id}),
        )

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
            s3_uri = item["vton_s3_uri"]
            if s3_uri.startswith("s3://"):
                parts = s3_uri[5:].split("/", 1)
                result_url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": parts[0], "Key": parts[1]},
                    ExpiresIn=3600,
                )

        return jsonify({"status": status, "result_url": result_url})
    except Exception:
        return jsonify({"status": "PENDING", "result_url": ""})


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if TELEGRAM_TOKEN:
        t = threading.Thread(target=telegram_polling, daemon=True)
        t.start()
        print("Telegram bot polling started.")

    print(f"DensePose enabled: {USE_DENSEPOSE}")
    print("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
