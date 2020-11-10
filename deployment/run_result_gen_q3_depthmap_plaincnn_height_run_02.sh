#!/bin/bash


logfile=~/log/q3_depthmap_plaincnn_height_run_02_result_gen.log
exec >> $logfile 2>&1

echo "start time: $(date)"

/usr/bin/python3 ~/src/result_gen.py q3_depthmap_height_run_02 q3-depthmap-height-run-02-ci

echo "end time: $(date)"
