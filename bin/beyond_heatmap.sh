#!/bin/bash
# SNICIT-A
log=../log/beyond/snicit_a.txt
echo "Running SNICIT on benchmark A..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do for t in {0..16..2}; 
        do info=`./beyond -t $t -b $B -m SNICIT -k A | grep -oP '(?<=SNICIT info:).*'`
        echo ==t is $t, B is $B==, $info >> $log
    done
done

log=../log/beyond/snig_a.txt
echo "Running SNIG on benchmark A..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do info=`./beyond -b $B -m SNIG -k A | grep -oP '(?<=SNIG info:).*'`
    echo ==B is $B==, $info >> $log
done

log=../log/beyond/snicit_b.txt
echo "Running SNICIT on benchmark B..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do for t in {0..16..2}; 
        do info=`./beyond -t $t -b $B -m SNICIT -k B | grep -oP '(?<=SNICIT info:).*'`
        echo ==t is $t, B is $B==, $info >> $log
    done
done

log=../log/beyond/snig_b.txt
echo "Running SNIG on benchmark B..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do info=`./beyond -b $B -m SNIG -k B | grep -oP '(?<=SNIG info:).*'`
    echo ==B is $B==, $info >> $log
done

log=../log/beyond/snicit_c.txt
echo "Running SNICIT on benchmark C..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do for t in {0..10..2}; 
        do info=`./beyond -t $t -b $B -m SNICIT -k C | grep -oP '(?<=SNICIT info:).*'`
        echo ==t is $t, B is $B==, $info >> $log
    done
done

log=../log/beyond/snig_c.txt
echo "Running SNIG on benchmark C..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do info=`./beyond -b $B -m SNIG -k C | grep -oP '(?<=SNIG info:).*'`
    echo ==B is $B==, $info >> $log
done

log=../log/beyond/snicit_d.txt
echo "Running SNICIT on benchmark D..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do for t in {0..10..2}; 
        do info=`./beyond -t $t -b $B -m SNICIT -k D | grep -oP '(?<=SNICIT info:).*'`
        echo ==t is $t, B is $B==, $info >> $log
    done
done

log=../log/beyond/snig_d.txt
echo "Running SNIG on benchmark D..."
printf "Log File - " > $log
date >> $log
for B in 1000 2000 2500 5000 10000;
    do info=`./beyond -b $B -m SNIG -k D | grep -oP '(?<=SNIG info:).*'`
    echo ==B is $B==, $info >> $log
done