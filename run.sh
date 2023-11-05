### referred to /appl/cuda/gpucourse2016/gpujobscript.sh
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
#BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=5GB]"
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
#BSUB -o %J.out
#BSUB -e %J.err
# -- end of LSF options --

# Create a virtual environment
python3 -m venv tf_env

# Activate the virtual environment
source tf_env/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install tensorflow==2.4 pandas dahuffman scipy

module load tensorrt/7.2.1.6-cuda-11.0
module load cudnn/v8.0.5.39-prod-cuda-11.0

MYGPUS=$(awk -Fgpu '{print $2}' $PBS_GPUFILE | sort -n | tr '\n' ' ' | sed 's/ /,/')
RESERVEDGPUS=$(cat $PBS_GPUFILE | wc -l)
ALLGPUS=$(/sbin/lspci  | grep -c "3D controller: NVIDIA" )

echo "my cuda device(s):                      " $MYGPUS
echo "hostname:                               " $(hostname)
echo "# number of GPUs in the system:         " $ALLGPUS
echo "# number of GPUs reserved for this job: " $RESERVEDGPUS

# we have to respect $MYGPUS here
# to make bandwidthtest happy: use 'all', if we are using all GPUs:
if [[ $RESERVEDGPUS -eq $ALLGPUS ]]; then
        MYGPUS=all
fi

echo /appl/cuda/10.1/samples/bin/x86_64/linux/release/bandwidthTest --device=$MYGPUS
/appl/cuda/10.1/samples/bin/x86_64/linux/release/bandwidthTest --device=$MYGPUS

nvidia-smi
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,name --format=csv,noheader,nounits

python3 main.py -dir "mtest2" -ds "cifar" -e 500 -m 3 -type entropy -so 3 -coeff 0.0 10.0
#python3 NN_param_compression_cifar.py  -dir "cifar_new3" -ds "cifar" -e 90 -m 13 -type entropy -so 3 -coeff 0.0 1e-3 1e-2 1e-1 0.5 1.0 5.0 10.0 100.0 1000.0
#python3 NN_param_compression_cifar.py  -dir "cifar_new4" -ds "cifar" -e 150 -m 13 -type entropy -so 3 -coeff 0.0 1000.0 10000.0 1000000.0 100000000.0
#python3 NN_param_decompression_cifar.py -dir "cifar_mobilnetv2" -ds "cifar"


