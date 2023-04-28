#!/bin/bash
log=../log/SDGC/tab3_fig6.txt

printf "Log File - " > $log
date >> $log

for n in 1024 4096 16384;
    do for l in 120 480 1920; 
        do info=`./SDGC -m BF -n $n -l $l | grep -oP '(?<=BF info:).*'`
        echo == BF on $n-$l ==, $info >> $log
        info=`./SDGC -m SNIG -n $n -l $l | grep -oP '(?<=SNIG info:).*'`
        echo == SNIG on $n-$l ==, $info >> $log
        info=`./SDGC -m XY -n $n -l $l | grep -oP '(?<=XY info:).*'`
        echo == XY on $n-$l ==, $info >> $log
        info=`./SDGC -m SNICIT -n $n -l $l | grep -oP '(?<=SNICIT info:).*'`
        echo == SNICIT on $n-$l ==, $info >> $log
    done
done


for l in 120 480 1920; 
    do info=`./SDGC -m BF -n 65536 -l $l | grep -oP '(?<=BF info:).*'`
    echo == BF on 65536-$l ==, $info >> $log
    info=`./SDGC -m SNIG -n 65536 -l $l | grep -oP '(?<=SNIG info:).*'`
    echo == SNIG on 65536-$l ==, $info >> $log
done
