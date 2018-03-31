#!/bin/bash

for dir in [ dataset, checkpoint, log, output ]
do
  rm -r $dir
  mkdir $dir
done
