#!/bin/bash
log=../log/SDGC/fig89.txt

printf "Log File - " > $log
date >> $log

for t in 12 14 16 18 30 60 90 118
    do for n in 1024 4096 16384;
        do info=`./SDGC -m XY -n $n -l 120 -t $t | grep -oP '(?<=XY info:).*'`
        echo == XY on $n-120 with t=$t ==, $info >> $log
        info=`./SDGC -m SNICIT -n $n -l 120 -t $t | grep -oP '(?<=SNICIT info:).*'`
        echo == SNICIT on $n-120 with t=$t ==, $info >> $log
    done
done

for b in 10000 15000 20000 30000 60000
    do for n in 1024 4096 16384;
        do info=`./SDGC -m XY -n $n -l 1920 -b $b | grep -oP '(?<=XY info:).*'`
        echo == XY on $n-1920 with B=$b ==, $info >> $log
        info=`./SDGC -m SNICIT -n $n -l 1920 -b $b | grep -oP '(?<=SNICIT info:).*'`
        echo == SNICIT on $n-1920 with B=$b ==, $info >> $log
    done
done