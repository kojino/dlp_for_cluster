#!/bin/bash
declare -a arr=(10 50 100)

for j in ${arr[@]}
do
for i in $(seq 0.3 0.1 10)
do
python cluster_test.py --sigma $i --num_iter $j
done
done
exit 0
