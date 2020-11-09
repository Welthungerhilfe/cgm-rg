#!/bin/bash

logfile=~/log/weight_20190723_result_gen.log
exec >> $logfile 2>&1

echo "start time: $(date)"

/usr/bin/python3 ~/src/result_gen.py 20190723-1119_2550-638weight aci-pointnet-weight-20190723-ci

echo "end time: $(date)"
