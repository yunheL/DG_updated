#!/bin/bash
REPORT=report
ABS_LOC=/home/ubuntu
DATE=./log/`date +'%I_%M_%S_%p'`i
REGEX_PRECI='\d+(\.\d{1,4})?'
 
cd $ABS_LOC
touch $REPORT
echo "Experiment Starting... @ $DATE">$REPORT


for j in {1..4}
do
        rm -rf /tmp/train_logs
        echo "keep $(($j*25))% data">>$REPORT
        for i in {1..12}
        do
        echo "train @ $i/2 epoch, $(($j*25))%"
        echo "train @ $i/2 epoch, $(($j*25))%">>$REPORT
        { time python3 ./dist/example.py --ps_hosts=54.214.227.27:4000 --worker_hosts=54.214.227.27:4444 --job_name=worker --task_index=0 --num_partition=4 --num_batch=$j --remove_oldlogs=0 --train_steps=$((275*$i)); } 2>&1 | grep -e real -e user -e sys >>$REPORT

        echo "eval @ $i/2 epoch, $(($j*25))%"
        echo -n "eval @ $i/2 epoch, $(($j*25))%">>$REPORT
	echo -n ", precision = ">>$REPORT
        python3 ./dist/example_eval.py --eval_steps=$((275*$i+1)) 2>&1 | grep -v Extracting >>$REPORT
	echo >>$REPORT
        done
done
