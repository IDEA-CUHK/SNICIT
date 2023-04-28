#!/bin/bash
# SNICIT-A
log=../log/beyond/tab4_fig11.txt

printf "Log File - " > $log
date >> $log

echo "Running BF on benchmark A..."
info=`./beyond -m BF -k A | grep -oP '(?<=BF info:).*'`
echo == BF on A ==, $info >> $log

echo "Running SNIG on benchmark A..."
info=`./beyond -m SNIG -k A | grep -oP '(?<=SNIG info:).*'`
echo == SNIG on A ==, $info >> $log

echo "Running SNICIT on benchmark A..."
info=`./beyond -m SNICIT -k A | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on A ==, $info >> $log
# ////////////////////////////////////////////////////////////////
echo "Running BF on benchmark B..."
info=`./beyond -m BF -k B | grep -oP '(?<=BF info:).*'`
echo == BF on B ==, $info >> $log

echo "Running SNIG on benchmark B..."
info=`./beyond -m SNIG -k B | grep -oP '(?<=SNIG info:).*'`
echo == SNIG on B ==, $info >> $log

echo "Running SNICIT on benchmark B..."
info=`./beyond -m SNICIT -k B | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on B ==, $info >> $log
# ////////////////////////////////////////////////////////////////////
echo "Running BF on benchmark C..."
info=`./beyond -m BF -k C | grep -oP '(?<=BF info:).*'`
echo == BF on C ==, $info >> $log

echo "Running SNIG on benchmark C..."
info=`./beyond -m SNIG -k C | grep -oP '(?<=SNIG info:).*'`
echo == SNIG on C ==, $info >> $log

echo "Running SNICIT on benchmark C..."
info=`./beyond -m SNICIT -k C | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on C ==, $info >> $log
# ///////////////////////////////////////////////////////////////////
echo "Running BF on benchmark D..."
info=`./beyond -m BF -k D | grep -oP '(?<=BF info:).*'`
echo == BF on D ==, $info >> $log

echo "Running SNIG on benchmark D..."
info=`./beyond -m SNIG -k D | grep -oP '(?<=SNIG info:).*'`
echo == SNIG on D ==, $info >> $log

echo "Running SNICIT on benchmark D..."
info=`./beyond -m SNICIT -k D | grep -oP '(?<=SNICIT info:).*'`
echo == SNICIT on D ==, $info >> $log
