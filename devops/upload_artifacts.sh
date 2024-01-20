#!/bin/bash

# DIR=./artifacts
# GCS_BUCKET=q-labeling

gsutil -m cp -n -r ./artifacts gs://q-labeling/artifacts
