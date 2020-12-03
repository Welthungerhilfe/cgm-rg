#!/bin/bash

python3 src/get_and_post_worflow.py

#python3 src/result_gen_with_api2.py --url http://localhost:5001 --scan_parent_dir data/scans/ --blur_workflow_path src/schema/blur-workflow-post.json

cron -f