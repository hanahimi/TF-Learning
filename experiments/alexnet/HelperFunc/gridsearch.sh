#!/bin/bash
###########################################
# File Name: gridsearch.sh
# Grid search for the hyper-parameters,
# such as the learning rate and the weight
# decay, for training a deep network
# Author: ying chenlu
# Email: ychenlu92@hotmail.com
# Created Time: 2016/11/13,12:45:47
###########################################
# helping message
if [ $# -eq 0 ]
then
    echo "run ./gridsearch.sh -h|--help"
    exit 0
fi
if [ $# -eq 1 ]
then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]
    then
        echo "usage: ./gridsearch workdir netprototxt max_iter display weights"
        echo "parameters:"
        echo -e "\tworkdir: working directory of the network"
        echo -e "\tnetprototxt: architucture file of the network"
        echo -e "\tmax_iter: max iteration"
        echo -e "\tdisplay: display interval"
        echo -e "\tweights (optional): full path of weight file"
    else
        echo "run ./gridsearch.sh -h|--help"
    fi
    exit 0
fi
# checking parameters
if [ $# -lt 4 ]
then
    echo "Error: no enough parameters"
    exit 1
fi
# assigning parameters
workdir=$1
if [ ! -d $workdir ]
then
    echo "Error: $workdir doen't exist"
    exit 1
fi
netprototxt=${workdir}/${2}
if [ ! -f $netprototxt ]
then
    echo "Error: $netprototxt doesn't exist"
    exit 1
fi
max_iter=$3
display=$4
if [ $# -gt 4 ]
then
    weightfile=$5
    if [ ! -f "${weightfile}" ];
    then
        echo "Error: $weightfile doesn't exist"
        exit 1;
    fi
fi
#preparing other parameters and temp files
cafferoot="/data/sys/caffe/"
if [ ! -d $cafferoot ]
then
    echo "Error: $cafferoot doesn't exist"
    exit 1
fi

cd $workdir
echo "pwd: `pwd`"
tmpsolver=tmpsolver.prototxt
echo "" >  $tmpsolver
tmpresult=tmpresult.out
echo "" > $tmpresult
searchresult=searchresult.txt
echo "" > $searchresult

for i in `seq 3 5`;
do
    # generating the prefix of learning rates
    lr_prefix="0."
    for k in `seq 2 $i`;
    do
        lr_prefix="${lr_prefix}0"
    done
    for j in 5 1;
    do
        # learning rate
        lr="${lr_prefix}$j"
        # generating the temp prototxt file for a solver
        echo -e "net: \"${netprototxt}\"" > ${tmpsolver}
        echo -e "base_lr: ${lr}" >> ${tmpsolver}
        echo -e "lr_policy: \"fixed\"" >> ${tmpsolver}
        echo -e "momentum: 0.9" >> ${tmpsolver}
        echo -e "momentum2: 0.999" >> ${tmpsolver}
        echo -e "weight_decay: 0.0005" >> ${tmpsolver}
        echo -e "max_iter: ${max_iter}" >> ${tmpsolver}
        echo -e "display: ${display}" >> ${tmpsolver} 
        echo -e "snapshot_prefix: \"tmp\"" >> ${tmpsolver}
        echo -e "type: \"Adam\"" >> ${tmpsolver}
        echo -e "solver_mode: GPU" >> ${tmpsolver}
        echo -e "-----------------------------" >> ${searchresult}
        echo -e "learning rate: ${lr}" >> ${searchresult}
        echo -e "\n" >> ${searchresult}
        # training the network
        if [ $# -eq 4 ];
        then
            echo "Training from scratch"
            ${cafferoot}/build/tools/caffe train --solver=${tmpsolver} \
                --gpu=1 >${tmpresult} 2>&1
        else
            echo "Fine tune with $weightfile"
            ${cafferoot}/build/tools/caffe train --solver=${tmpsolver} \
                --weights=${weightfile} --gpu=1 >${tmpresult} 2>&1
        fi
        # parsing the training lob
        ${cafferoot}/tools/extra/parse_log.py ${tmpresult} ${workdir}
        # collecting the training message
        cat ${tmpsolver} >> ${searchresult}
        echo -e "\n****training result****" >> ${searchresult}
        cat "${tmpresult}.train" >> ${searchresult}
        echo -e "\n\n" >> ${searchresult}
    done
done
rm tmp*
