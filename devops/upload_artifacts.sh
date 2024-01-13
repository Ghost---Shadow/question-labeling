#!/bin/bash

DIR=./artifacts
GCS_BUCKET=q-labeling

gsutil cp -r $DIR gs://$GCS_BUCKET/artifacts
