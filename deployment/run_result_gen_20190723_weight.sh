#!/bin/bash

env=$(jq -r '.Environment' ~/PythonCode/dbconnection.json)
folder="cgminbmz$env"

logfile=~/"$folder"/log/weight_20190723_result_gen.log
exec >> $logfile 2>&1

echo "start time: $(date)"

/usr/bin/python3 ~/PythonCode/result_gen.py ~/"$folder"/db/measure_result/ ~/PythonCode/dbconnection.json "$folder" 20190723-1119_2550-638weight aci-pointnet-weight-20190723-ci ~/PythonCode/camera_calibration.txt

echo "end time: $(date)"
