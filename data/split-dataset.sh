#!/bin/sh

N_valid=3000
N_test=3000
N=`wc -l wiki.txt | awk '{print $1}'`
N_train=$(($N-$N_valid-$N_test))

DIR=wiki/original

mkdir -p wiki 
mkdir -p wiki/original 
mkdir -p wiki/tokenized 

echo create $N_train wiki.train
echo create $N_valid wiki.valid
echo create $N_test wiki.test

head -n $N_train wiki.txt > $DIR/wiki.train
tail -n $(($N_valid+$N_test)) wiki.txt | head -n $N_valid > $DIR/wiki.valid
tail -n $(($N_valid+$N_test)) wiki.txt | tail -n $N_test > $DIR/wiki.test

