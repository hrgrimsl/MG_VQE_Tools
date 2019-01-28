#!/usr/bin/env bash

python LiHPes.py
for i in `seq 5 24`;
    do
        python Driver.py LiH_"$i"_1_PT2 LiH_"$i"_1_PT2.out &
    done
for i in `seq 5 24`;
    do
        python Driver.py LiH_"$i"_2_PT2 LiH_"$i"_2_PT2.out &
    done

wait
exit;


