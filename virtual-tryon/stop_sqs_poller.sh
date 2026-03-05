#!/bin/bash

# Define the service name
SERVICE_NAME="sqs_poller.service"
LOG_FILE="/home/ec2-user/SageMaker/IDM_VTON/IDM-VTON/sqs_poller_logs.txt"

# Function to stop the service
stop_service() {
    echo "Stopping the $SERVICE_NAME..."
    sudo systemctl stop "$SERVICE_NAME"
    echo "$SERVICE_NAME stopped."
}

# Function to stop logging
stop_logging() {
    echo "Stopping logging for $SERVICE_NAME..."
    # Use pkill to stop the logging process (journalctl)
    pkill -f "journalctl -u $SERVICE_NAME"
    echo "Logging stopped."
}

# Call the functions
stop_service
stop_logging

