#!/bin/bash

# DIR=./artifacts
# GCS_BUCKET=q-labeling

gsutil -m cp -r ./artifacts gs://q-labeling/artifacts
