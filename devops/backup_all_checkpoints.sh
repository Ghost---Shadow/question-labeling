#!/bin/bash

python devops/find_most_recent_checkpoints.py

source upload_checkpoints.sh

source devops/stop_current_gcp_instance.sh
