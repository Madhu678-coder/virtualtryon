#!/bin/bash

# Load environment variables from .env file
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
else
    echo ".env file not found!"
    exit 1
fi

EXEC_START_COMMAND="$VENV_PATH/bin/python3 $PYTHON_SCRIPT"

# Function to update the service file
update_service_file() {
    sudo bash -c "cat > $SERVICE_FILE << 'EOF'
[Unit]
Description=SQS Poller Service
After=network.target

[Service]
Type=idle
WorkingDirectory=$WORKING_DIRECTORY

Environment='PATH=$VENV_PATH/bin'

ExecStartPre=$GPU_RESET_COMMAND

ExecStart=$EXEC_START_COMMAND

Restart=always
RestartSec=5
StartLimitInterval=0

StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=multi-user.target

EOF"
}

# Function to set up the virtual environment if not present
setup_virtualenv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo "Virtual environment not found. Creating one..."
        python3 -m venv "$VENV_PATH"  # Create virtual environment
        echo "Virtual environment created at $VENV_PATH."
    else
        echo "Virtual environment found at $VENV_PATH."
    fi
}

# Function to install dependencies
install_dependencies() {
    echo "Installing Python dependencies..."
    source "$VENV_PATH/bin/activate"  # Activate the virtual environment
    pip install -r "$WORKING_DIRECTORY/requirements.txt"  # Install dependencies
    deactivate  # Deactivate the virtual environment after installation
}

# Function to restart the service
restart_service() {
    sudo systemctl daemon-reload
    sudo systemctl restart sqs_poller.service
    sudo systemctl enable sqs_poller.service
    echo "Service sqs_poller.service restarted."
}

# Function to log service status
log_service_status() {
    sudo systemctl status sqs_poller.service
    echo "Logging service output to $LOG_FILE"
}

# Update the service file
update_service_file

# Set up virtual environment if necessary
setup_virtualenv

# Install dependencies
install_dependencies

# Restart the service
restart_service

# Log the service status
log_service_status

