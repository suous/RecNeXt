#!/bin/bash

while pgrep python > /dev/null; do
  echo "sleep 60 second"
  sleep 60
done

# wait until another training process stop.
sh dist_train_cifar.sh