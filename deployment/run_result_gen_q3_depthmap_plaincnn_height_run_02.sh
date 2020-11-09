#!/bin/bash

env=$(jq -r '.Environment' ~/PythonCode/dbconnection.json)
folder="cgminbmz$env"

logfile=~/log/q3_depthmap_plaincnn_height_run_02_result_gen.log
exec >> $logfile 2>&1

echo "start time: $(date)"

/usr/bin/python3 ~/PythonCode/result_gen.py ~/"$folder"/db/measure_result/ ~/PythonCode/dbconnection.json "$folder" q3_depthmap_height_run_02 q3-depthmap-height-run-02-ci ~/PythonCode/camera_calibration.txt

echo "end time: $(date)"
