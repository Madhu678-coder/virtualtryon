# **Setup/Installation and Inference**
This repo has the code to run virtual try-on inference.

### **EC2 Instance Setup ([confluence link](https://minfyhelpdesk.atlassian.net/wiki/x/BwDut))**

For optimal performance, it is highly recommended to launch your EC2 instance using the **NVIDIA GPU-Optimized AMI** from the AWS Marketplace (AMI ID: `ami-02e56cd85e65e275d`). This AMI comes pre-installed with all necessary NVIDIA drivers, ensuring your environment is ready for GPU-accelerated workloads.

- For more details, refer to the [official AWS documentation on NVIDIA drivers for EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#preinstalled-nvidia-driver).

# **Requirements**
```
# Clone the repository into /srv (recommended root location)
cd /srv
git clone https://github.com/groomi-ai/VTON.git
cd VTON
```
##### **This will be the structure of the repo**
```
VTON
|-- configs/
|-- IDM-VTON/
|-- ip_adaptor/
|-- preprocess/
|-- src/
|-- .env 
|-- aws_utils.py
|-- data_loader.py
|-- inference.py
|-- logging_config.py
|-- main.py
|-- requirements.txt
|-- setup_sqs_poller.sh
|-- stop_sqs_poller.sh
|-- vton_errors.py
```


### **Model Artifacts:** 

The IDM-VTON([IDM-VTON](https://github.com/yisol/IDM-VTON)) model artifacts are required for inference. These are originally from Hugging Face ([source link](https://huggingface.co/yisol/IDM-VTON/tree/main)) and have been uploaded to S3 for easier access.

#### **Download IDM-VTON Artifacts from S3**

1. **Install AWS CLI** (if not already installed):
   ```
   pip install awscli
   ```
2. **Configure AWS CLI** (if not already configured):
   ```
   aws configure
   # Enter your AWS Access Key, Secret Key, region, and output format
   ```
3. **Download the model artifacts:**
   ```
   aws s3 sync s3://groom-vton/model_artifacts/IDM-VTON/ IDM-VTON/
   ```
   This will download all necessary subfolders and files into the `IDM-VTON/` directory.

> **Note:** The original model parameters are from Hugging Face: [https://huggingface.co/yisol/IDM-VTON/tree/main](https://huggingface.co/yisol/IDM-VTON/tree/main). They have been uploaded to S3 for convenience and faster access.

##### **After download, your IDM-VTON folder should look like:**
```
|-- IDM-VTON/
    |-- densepose
    |-- humanparsing
    |-- image_encoder
    |-- openpose
    |-- scheduler
    |-- text_encoder
    |-- text_encoder_2
    |-- tokenizer
    |-- tokenizer_2
    |-- unet
    |-- unet_encoder
    |-- vae
    |-- model_index.json
```


### **Creating Python environment and Installing the requirements**
1. Create a virtual environment
```
python3 -m venv vton_env
```
2. Activate the environment
```
source vton_env/bin/activate
```
3. Install the dependencies from requirements.txt
```
pip install -r requirements.txt
```

## **Launching SQS poller with -systemd**
1. Go to ```.env``` file and update the path variables (*example below*)
```
SERVICE_FILE="/etc/systemd/system/sqs_poller.service"
WORKING_DIRECTORY="/srv/fashchat-ai-vton" #pointing to root folder of your project
VENV_PATH="/srv/fashchat-ai-vton/vton_venv" # pointing to venv folder
PYTHON_SCRIPT="${WORKING_DIRECTORY}/main.py"
LOG_FILE="${WORKING_DIRECTORY}/sqs_poller_logs.txt"
GPU_RESET_COMMAND="sudo /usr/bin/nvidia-smi --gpu-reset -i 0"
# EXEC_START_COMMAND="$VENV_PATH/python3 $PYTHON_SCRIPT"
pretrained_model_name_or_path="${WORKING_DIRECTORY}/IDM-VTON"
output_dir="${WORKING_DIRECTORY}/output_sample_images"
data_dir="${WORKING_DIRECTORY}/sample_images"
queue_url="https://sqs.ap-southeast-2.amazonaws.com/730335611421/VTON" # The sqs url from where we can poll messages

vton_table="vton-collection"
vton_par_key_name="user_id"
vton_sort_key_name="request_id"
default_vton_output_bucket="groome-results"
default_vton_output_folder="vton_api_outputs"
```
2. Save the file.
3. In the bash terminal run the bash script
```
   bash ./setup_sqs_poller.sh
```
4. The bash script will create systemd file and start polling.
5. Your machine will start polling the SQS for new messages.
6. To view the logs open the sqs_poller
```
    tail -f -s 1 ./sqs_poller_logs.txt
```
7. To stop the polling run the script
```
    bash ./stop_sqs_poller.sh
```

## Model Architecture and Parameters

*The IDM-VTON model's architecture and parameters are based on the research outlined in*  
*[this paper](https://idm-vton.github.io/).*  
```
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```
*For more details on the model implementation, you can refer to the*  
*[IDM-VTON GitHub Repository](https://github.com/yisol/IDM-VTON).*

## **Setting up AWS SQS and Getting the queue_url**

To enable message polling, you need an AWS SQS queue. Here's how to set it up and get the `queue_url` for your `.env` file using the AWS CLI:

1. **Create a new SQS queue using AWS CLI:**
   ```bash
   aws sqs create-queue --queue-name vton_test
   ```
   - Replace `vton_test` with your desired queue name.
   - This will output a JSON object containing the `QueueUrl`.

2. **Get the Queue URL:**
   ```bash
   aws sqs get-queue-url --queue-name vton_test
   ```
   - Copy the `QueueUrl` value from the output (e.g., `https://sqs.<region>.amazonaws.com/<account-id>/<queue-name>`)

3. **Update your `.env` file:**
   - Set the `queue_url` variable in your `.env` file to the copied URL:
     ```
     queue_url="https://sqs.<region>.amazonaws.com/<account-id>/<queue-name>"
     ```

For more details, see the [AWS SQS CLI documentation](https://docs.aws.amazon.com/cli/latest/reference/sqs/index.html).

## **Setting up AWS S3 Buckets and DynamoDB Table**

The application requires several AWS S3 buckets and a DynamoDB table for storing images and metadata. Please create the following resources in your AWS account using the AWS CLI:

### **S3 Buckets**
Create these buckets in your preferred AWS region:
- `groome-results` (for VTON output images)
- `vton-preprocessed` (for preprocessed customer images)
- `product-images-groome` (for product images)

You can create a bucket using the AWS CLI (if the name is already taken, use a different name and update the .env):
```bash
aws s3 mb s3://groome-results
aws s3 mb s3://vton-preprocessed
aws s3 mb s3://product-images-groome
```

### **DynamoDB Table**
Create a DynamoDB table named `vton-collection` with the following keys using the AWS CLI:
- Partition key: `user_id` (String)
- Sort key: `request_id` (String)

```bash
aws dynamodb create-table \
  --table-name vton-collection \
  --attribute-definitions AttributeName=user_id,AttributeType=S AttributeName=request_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH AttributeName=request_id,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST
```

#### **Example Item Schema**
A sample item in the `vton-collection` table:
```json
{
  "user_id": { "S": "e790-e7f8-494f-c3d0-19ed-aaa6-a23e-905d" },
  "request_id": { "S": "17b361f7-2be9-4386-a38f-56fa9f2cb98a" },
  "category": { "S": "upper" },
  "customer_images": { "S": "s3://vton-preprocessed/e790-e7f8-494f-c3d0-19ed-aaa6-a23e-905d/17b361f7-2be9-4386-a38f-56fa9f2cb98a" },
  "output_image": { "S": "s3://groome-results/vton_api_outputs" },
  "pre_processing_status": { "S": "COMPLETED" },
  "product_images": { "L": [ { "S": "s3://product-images-groome/validated_product_images/products/2c12eaa7-63c5-4adc-84cc-2c370b55957d/image_1.png" } ] },
  "timestamp": { "S": "2025-06-12 07:51:59.029206+00:00" },
  "validation_status": { "M": { "message": { "S": "Image validated successfully" }, "status": { "S": "Success" } } },
  "vton_process_status": { "S": "COMPLETED" },
  "vton_s3_uri": { "S": "s3://groome-results/vton_api_outputs/17b361f7-2be9-4386-a38f-56fa9f2cb98a.png" }
}
```

### **.env Variables Reference**
Add the following variables to your `.env` file to match your AWS resources:
```bash
vton_table="vton-collection"
vton_par_key_name="user_id"
vton_sort_key_name="request_id"
default_vton_output_bucket="groome-results"
default_vton_output_folder="vton_api_outputs"
```
- `vton_table`: Name of the DynamoDB table
- `vton_par_key_name`: Partition key name
- `vton_sort_key_name`: Sort key name
- `default_vton_output_bucket`: S3 bucket for VTON output images
- `default_vton_output_folder`: Folder inside the output bucket for storing results


