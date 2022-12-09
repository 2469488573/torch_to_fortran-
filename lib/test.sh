#!/bin/bash

#移植程序后需要修改路径
bridge_root='/data/chengxl/pblh_deeplearning/torch_bridge_fortran'
deep_learning=$bridge_root/python

cd $deep_learning
pwd

echo "运行深度学习训练程序"

./deep_learning_regression.py

echo "传递参数到bridge文件夹内"

cp *.txt $bridge_root

cd $bridge_root
pwd

echo "编译"
./ifort_make.sh

echo "运行torch_bridge_fortran"
./bridge


