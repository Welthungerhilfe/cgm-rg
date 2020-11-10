#!/bin/bash


logfile=~/log/gapnet_s1_result_gen.log
exec >> $logfile 2>&1

echo "start time: $(date)"

/usr/bin/python3 ~/src/result_gen.py GAPNet_height_s1 aci-gapnet-height-s1-4-ci

echo "end time: $(date)"
