#!/bin/bash

env=$(jq -r '.Environment' ~/PythonCode/dbconnection.json)
folder="cgminbmz$env"

logfile=~/"$folder"/log/gapnet_s1_result_gen.log
exec >> $logfile 2>&1

echo "start time: $(date)"

/usr/bin/python3 ~/PythonCode/result_gen.py ~/"$folder"/db/measure_result/ ~/PythonCode/dbconnection.json "$folder" GAPNet_height_s1 aci-gapnet-height-s1-4-ci ~/PythonCode/camera_calibration.txt

echo "end time: $(date)"
