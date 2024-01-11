#!/bin/bash

# Directory to be zipped
REMOTE_DIR=/mnt/checkpoints/artifacts
ZIP_NAME=artifacts.zip

# Google Cloud Storage bucket name
GCS_BUCKET=q-labeling

# Zip the directory
zip -r $ZIP_NAME $REMOTE_DIR

# Upload the zip file to Google Cloud Storage bucket
gsutil cp $ZIP_NAME gs://$GCS_BUCKET/

# Remove the zip file from the remote server
rm $ZIP_NAME

# Notification
echo "Upload completed. File $ZIP_NAME uploaded to gs://$GCS_BUCKET/"
