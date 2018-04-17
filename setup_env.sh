#!/bin/bash

aws s3 cp s3://nebula-poc/enamel/Anaconda3-4.3.0-Linux-x86_64.sh /home/ubuntu
bash /home/ubuntu/Anaconda3-4.3.0-Linux-x86_64.sh -b
echo "export PATH=/home/ubuntu/anaconda3/bin:$PATH" >> /home/ubuntu/.bash_profile

source /home/ubuntu/.bash_profile
conda create -n udacity_ml
pip install -e .
