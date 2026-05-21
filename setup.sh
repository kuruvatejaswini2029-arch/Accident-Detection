#!/bin/bash
# Streamlit deployment setup script

# Create models directory
mkdir -p models

# Install dependencies
pip install -r requirements.txt

echo "Setup complete!"
