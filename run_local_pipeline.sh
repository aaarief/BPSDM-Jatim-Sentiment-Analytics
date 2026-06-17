#!/bin/bash

# Define credentials path
CREDS_PATH="/home/arief/Downloads/dibimbing-484913-be75c3271d7e.json"

if [ -f "$CREDS_PATH" ]; then
    echo "Using credentials from $CREDS_PATH"
    export GOOGLE_SHEETS_CREDENTIALS=$(cat "$CREDS_PATH")
else
    echo "ERROR: Service account key not found at $CREDS_PATH. Google Sheets upload will fail."
fi

# Get Gemini API key if not set
if [ -z "$GEMINI_API_KEY" ]; then
    read -p "Enter your Gemini API Key: " input_key
    export GEMINI_API_KEY="$input_key"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is required to run the pipeline."
    exit 1
fi

# Run the pipeline
python pipeline.py
