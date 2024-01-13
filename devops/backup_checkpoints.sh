#!/bin/bash

python devops/plan_checkpoint_upload.py

source upload_checkpoints.sh

source devops/stop_current_gcp_instance.sh
