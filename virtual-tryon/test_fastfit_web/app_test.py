"""
Simple Flask test server for FastFit footwear try-on.
Uses EC2 IAM role for AWS access — no credentials needed in browser.

Usage:
    python app_test.py

Access at: http://<ec2-public-ip>:8080
"""

import os
import sys
import uuid
import json
import boto3
from io import BytesIO
from datetime import datetime, timezone
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = Flask(__name__)

# AWS config from environment / IAM role
REGION = os.getenv("aws_region", "us-east-1")
QUEUE_URL = os.getenv("queue_url", "")
VTON_TABLE = os.getenv("vton_table", "vton-collection")
OUTPUT_BUCKET = os.getenv("default_vton_output_bucket", "groome-results-1")
OUTPUT_FOLDER = os.getenv("default_vton_output_folder", "vton_api_outputs")
PREPROCESSED_BUCKET = os.getenv("preprocessed_bucket", "vton-preprocessed-1")
PRODUCT_BUCKET = os.getenv("product_bucket", "product-images-groome-1")
RAW_IMAGES_BUCKET = os.getenv("raw_images_bucket", "product-images-groome-1")

# AWS clients (use IAM role automatically)
sqs = boto3.client("sqs", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(VTON_TABLE)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FastFit Footwear Try-On Test</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 1100px; margin: 0 auto; }
        h1 { text-align: center; color: #1a365d; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; color: #718096; margin-bottom: 2rem; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .upload-box {
            border: 2px dashed #cbd5e0;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .upload-box:hover { border-color: #4299e1; background: #ebf8ff; }
        .upload-box.has-image { border-color: #48bb78; border-style: solid; padding: 0.5rem; }
        .upload-box img { max-width: 100%; max-height: 250px; border-radius: 4px; object-fit: contain; }
        .upload-box .label { font-weight: 600; color: #4a5568; margin-bottom: 0.5rem; font-size: 1.1rem; }
        .upload-box p { color: #a0aec0; font-size: 0.85rem; }
        input[type="file"] { display: none; }
        .controls {
            display: flex; align-items: center; gap: 1rem;
            justify-content: center; margin-top: 1.5rem; flex-wrap: wrap;
        }
        .controls label { font-weight: 600; color: #4a5568; }
        select {
            padding: 0.6rem 1.2rem; border: 1px solid #e2e8f0;
            border-radius: 6px; font-size: 1rem;
        }
        .btn {
            background: #2b6cb0; color: white; border: none;
            padding: 0.8rem 2.5rem; border-radius: 8px;
            font-size: 1.05rem; font-weight: 600; cursor: pointer;
        }
        .btn:hover { background: #2c5282; }
        .btn:disabled { background: #a0aec0; cursor: not-allowed; }
        .result-box {
            border: 2px solid #e2e8f0; border-radius: 8px;
            min-height: 280px; display: flex; align-items: center;
            justify-content: center; overflow: hidden;
        }
        .result-box img { max-width: 100%; max-height: 400px; object-fit: contain; }
        .status {
            text-align: center; padding: 1rem; border-radius: 8px;
            margin-top: 1rem; font-weight: 500;
        }
        .status.pending { background: #fefcbf; color: #975a16; }
        .status.success { background: #c6f6d5; color: #276749; }
        .status.error { background: #fed7d7; color: #9b2c2c; }
        .info { background: #ebf8ff; border: 1px solid #bee3f8; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }
        .info p { color: #2c5282; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>👟 FastFit Footwear Try-On</h1>
        <p class="subtitle">Test footwear and bags virtual try-on</p>

        <div class="info">
            <p>🔒 Using EC2 IAM role for AWS access. No credentials needed.</p>
        </div>

        <div class="card">
            <div class="grid">
                <div>
                    <div class="upload-box" id="personBox" onclick="document.getElementById('personInput').click()">
                        <div class="label">👤 Person Image</div>
                        <p>Upload full-body photo</p>
                    </div>
                    <input type="file" id="personInput" accept="image/*" onchange="previewImage(this, 'personBox')">
                </div>
                <div>
                    <div class="upload-box" id="garmentBox" onclick="document.getElementById('garmentInput').click()">
                        <div class="label">👟 Shoe / Bag</div>
                        <p>Upload product image</p>
                    </div>
                    <input type="file" id="garmentInput" accept="image/*" onchange="previewImage(this, 'garmentBox')">
                </div>
                <div>
                    <div class="result-box" id="resultBox">
                        <p style="color:#a0aec0">Result will appear here</p>
                    </div>
                </div>
            </div>

            <div class="controls">
                <label>Category:</label>
                <select id="category">
                    <option value="shoes">Shoes / Footwear</option>
                    <option value="bags">Bags</option>
                </select>
                <button class="btn" id="submitBtn" onclick="submitTryOn()">🚀 Try On</button>
            </div>
            <div id="statusDiv"></div>
        </div>
    </div>

    <script>
        function previewImage(input, boxId) {
            const box = document.getElementById(boxId);
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    box.innerHTML = '<img src="' + e.target.result + '">';
                    box.classList.add('has-image');
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        function setStatus(msg, type) {
            document.getElementById('statusDiv').innerHTML =
                '<div class="status ' + type + '">' + msg + '</div>';
        }

        async function submitTryOn() {
            var personInput = document.getElementById('personInput');
            var garmentInput = document.getElementById('garmentInput');
            var category = document.getElementById('category').value;

            if (!personInput.files || !personInput.files[0] || !garmentInput.files || !garmentInput.files[0]) {
                setStatus('❌ Please upload both images', 'error');
                return;
            }

            document.getElementById('submitBtn').disabled = true;
            setStatus('⏳ Uploading and processing...', 'pending');

            const formData = new FormData();
            formData.append('person_image', personInput.files[0]);
            formData.append('garment_image', garmentInput.files[0]);
            formData.append('category', category);

            try {
                const resp = await fetch('/api/tryon', { method: 'POST', body: formData });
                const data = await resp.json();

                if (data.error) {
                    setStatus('❌ ' + data.error, 'error');
                    document.getElementById('submitBtn').disabled = false;
                    return;
                }

                setStatus('⏳ Processing... (user: ' + data.user_id + ', request: ' + data.request_id + ')', 'pending');
                pollResult(data.user_id, data.request_id);

            } catch (err) {
                setStatus('❌ Error: ' + err.message, 'error');
                document.getElementById('submitBtn').disabled = false;
            }
        }

        function pollResult(userId, requestId) {
            let attempts = 0;
            const interval = setInterval(async () => {
                attempts++;
                if (attempts > 60) {
                    clearInterval(interval);
                    setStatus('❌ Timeout', 'error');
                    document.getElementById('submitBtn').disabled = false;
                    return;
                }

                try {
                    const resp = await fetch('/api/status/' + userId + '/' + requestId);
                    const data = await resp.json();

                    if (data.status === 'COMPLETED' && data.result_url) {
                        clearInterval(interval);
                        document.getElementById('resultBox').innerHTML =
                            '<img src="' + data.result_url + '">';
                        setStatus('✅ Done!', 'success');
                        document.getElementById('submitBtn').disabled = false;
                    } else if (data.status === 'FAILED') {
                        clearInterval(interval);
                        setStatus('❌ Processing failed', 'error');
                        document.getElementById('submitBtn').disabled = false;
                    } else {
                        setStatus('⏳ Processing... (' + attempts + '/60)', 'pending');
                    }
                } catch (e) {}
            }, 5000);
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE, queue_url=QUEUE_URL)


@app.route("/api/tryon", methods=["POST"])
def submit_tryon():
    try:
        category = request.form.get("category", "shoes")
        user_id = str(uuid.uuid4())[:8]
        request_id = str(uuid.uuid4())[:8]

        person_file = request.files.get("person_image")
        garment_file = request.files.get("garment_image")

        if not person_file or not garment_file:
            return jsonify({"error": "Both images required"}), 400

        person_img = Image.open(person_file.stream).convert("RGB")
        garment_img = Image.open(garment_file.stream).convert("RGB")

        # Upload person image to S3
        person_key = f"{user_id}/{request_id}.png"
        buf = BytesIO()
        person_img.save(buf, format="PNG")
        buf.seek(0)
        s3.put_object(Bucket=RAW_IMAGES_BUCKET, Key=person_key, Body=buf, ContentType="image/png")

        # Upload garment image to S3
        garment_key = f"{user_id}/{request_id}/garment.jpg"
        buf = BytesIO()
        garment_img.save(buf, format="JPEG")
        buf.seek(0)
        s3.put_object(Bucket=PRODUCT_BUCKET, Key=garment_key, Body=buf, ContentType="image/jpeg")

        # Create DynamoDB record — set pre_processing_status to COMPLETED
        # since FastFit does its own preprocessing
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
            "status": "PENDING",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status/<user_id>/<request_id>")
def check_status(user_id, request_id):
    try:
        resp = table.get_item(Key={"user_id": user_id, "request_id": request_id})
        item = resp.get("Item", {})
        vton_status = item.get("vton_process_status", "PENDING")
        result_url = ""

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
            "result_url": result_url,
        })
    except Exception:
        return jsonify({"status": "PENDING", "result_url": ""})


if __name__ == "__main__":
    print(f"Queue URL: {QUEUE_URL}")
    print(f"Region: {REGION}")
    print(f"Starting test server on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=False)
