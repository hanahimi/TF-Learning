#!/usr/bin/env python
#-*-coding:utf-8-*
#########################################
  # File Name: parseresult.py
"""
    Parse the training log of Caffe.
    Draw graphes of losses and accuracy rates.
"""
  # Author: ying chenlu
  # Mail: ychenlu92@hotmail.com 
  # Created Time: 2016/11/13,15:19:23
  # Usage: ./parseresult.py -h
########################################
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas
import matplotlib.pyplot as plt
from subprocess import call

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--inputfile", type=str, help="training log of Caffe")
parser.add_argument("--outputdir", type=str, default='./', help="output directory")
parser.add_argument("--Xaxis", type=str, default="NumIters",
        help="X axis of image")
parser.add_argument("--Yaxis", type=str, default="loss",
        help="y axis of image")
args = parser.parse_args()

# cafferoot = raw_input("Enter your caffe_root: ")
cafferoot = "/data/sys/caffe"
assert os.path.exists(os.path.join(cafferoot, "tools/extra/parse_log.py"))
# extract message for training and testing from the whole log
call("python {:s} {:s} {:s}".format(os.path.join(cafferoot, "tools/extra/parse_log.py"),
    args.inputfile, args.outputdir), shell=True)

train_log = args.inputfile + ".train"
test_log = args.inputfile + ".test"
train_df = pandas.read_csv(train_log, header=0)
test_df = pandas.read_csv(test_log, header=0)
assert args.Xaxis in train_df.columns
assert args.Xaxis in test_df.columns
assert args.Yaxis in train_df.columns
assert args.Yaxis in test_df.columns

train_x = np.asarray(train_df[args.Xaxis])
train_y = np.asarray(train_df[args.Yaxis])
test_x = np.asarray(test_df[args.Xaxis])
test_y = np.asarray(test_df[args.Yaxis])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_x, train_y, "r", linewidth=1.5, label="train")
ax.plot(test_x, test_y, "b", linewidth=1.5, label="test")
ax.legend(loc="upper right")
plt.xlabel(args.Xaxis)
plt.ylabel(args.Yaxis)
fig.savefig(os.path.join(args.outputdir, "%s.jpg"%args.Yaxis))

print("training %s"%args.Yaxis)
print(train_y[-100:])
print("testing %s"%args.Yaxis)
print(test_y[-100:])
idx = np.argmin(test_y)
print("minimum testing %s is %s at %s %s"%(args.Yaxis,
        test_y[idx], args.Xaxis, test_x[idx]))