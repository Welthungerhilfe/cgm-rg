#!/bin/bash

set -euo pipefail

printenv | grep -v "no_proxy" >> /etc/environment

python3 src/download_model.py
python3 src/get_and_post_workflow.py

#python3 src/result_gen_with_api.py

cron -f