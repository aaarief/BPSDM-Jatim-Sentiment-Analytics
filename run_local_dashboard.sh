#!/bin/bash

# Define credentials path
CREDS_PATH="/home/arief/Downloads/dibimbing-484913-be75c3271d7e.json"

if [ -f "$CREDS_PATH" ]; then
    echo "Using credentials from $CREDS_PATH"
    export GOOGLE_SHEETS_CREDENTIALS=$(cat "$CREDS_PATH")
else
    echo "Warning: Service account key not found at $CREDS_PATH. Falling back to local offline CSV data."
fi

# Run Streamlit
streamlit run app.py
