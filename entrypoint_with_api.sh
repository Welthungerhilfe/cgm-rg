#!/bin/bash

set -euo pipefail

printenv | grep -v "no_proxy" >> /etc/environment

python3 src/download_model.py
python3 src/get_and_post_workflow.py

#python3 src/result_gen_with_api.py --url http://localhost:5001 --scan_parent_dir data/scans/ --blur_workflow_path src/schema/blur-workflow-post.json

cron -f