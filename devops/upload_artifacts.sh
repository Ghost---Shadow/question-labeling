#!/bin/bash

DIR=./artifacts
GCS_BUCKET=q-labeling

gsutil -m cp -r $DIR gs://$GCS_BUCKET/artifacts
