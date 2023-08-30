#!/bin/bash

# enter practice
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'

# choose one track randomly
xte 'key Up'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'

rand_num=$(( RANDOM % 10 ))
for (( i=0; i<$rand_num; i++ )); do
    xte 'key Right'
    xte 'usleep 100000'
done

xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'

# start new race
xte 'key Up'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'