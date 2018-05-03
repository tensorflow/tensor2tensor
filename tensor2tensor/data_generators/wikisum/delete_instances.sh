#!/bin/bash

# Delete Google Compute Engine instances with naming structure $NAME-$INDEX
# (e.g. machines created with parallel_launch.py).
# Example usage:
# delete_instances.sh fetch-ref-urls 1000

NAME=$1
MAX=$2
MIN=${3:-0}

LOG_F=/tmp/delete-$NAME-logs.txt

echo "Deleting $MAX instances starting with $NAME-$MIN"

for i in $(seq $MIN $MAX)
do
  gcloud compute instances delete --quiet $NAME-$i > $LOG_F 2>&1 &
  if [[ $(( i % 100 )) == 0 ]]
  then
    # Give it some room to breathe every 100
    sleep 30
  fi
done

echo "Delete commands launched. Logs redirected to $LOG_F"
