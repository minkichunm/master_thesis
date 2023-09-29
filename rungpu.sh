#!/bin/sh
### General options
### â€“- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Thesis
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


#Load preinstalled modules
module load python3/3.8.0
module load gcc

#Create a virtual environment for Python3
#python3 -m venv hello_keras

#Activate virtual environment
#source hello_keras/bin/activate


nvidia-smi 

#If pip3 fails, use: which pip3, to make sure it is the one in the virutal environment.
#which pip3
#pip3 -m install --user --upgrade pip
#pip3 -m install --user keras
#pip3 install tensorflow
python3 -m pip install --user tensorflow

module load tensorrt/6.0.1.5-cuda-10.1
module load cudnn/v7.6.5.32-prod-cuda-10.1
#Load remaining modules for Python 3.6.2 (Tensorflow Backend for Keras)
#Panda for handling data
#module load tensorflow/1.5-gpu-python-3.6.2

python3 NN_param_compression_cifar.py  -d "results_test" -e 75 -m 2 -type entropy -coeff 0.0 1.0
python3 NN_param_decompression_cifar.py -d "results_test"

