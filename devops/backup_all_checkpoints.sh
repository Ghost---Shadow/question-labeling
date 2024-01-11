#!/bin/bash

# Source directory
SOURCE_DIR=/mnt/checkpoints/checkpoints

# Google Cloud Storage bucket name
GCS_BUCKET=q-labeling

# Recursively upload the files to the bucket
gsutil -m cp -r $SOURCE_DIR/* gs://$GCS_BUCKET/

# Notification
echo "Upload completed. Files from $SOURCE_DIR uploaded to gs://$GCS_BUCKET/"
