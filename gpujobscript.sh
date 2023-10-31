### referred to /appl/cuda/gpucourse2016/gpujobscript.sh
#!/bin/bash
# **** You have to submit this jobscript from gpulogin
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:15:00
#PBS -q gpuqueue

###  just make sure, that we are alone on the machine for benchmarking....
###  (remove one # from the next line)
##PBS -l naccesspolicy=singlejob

cd $PBS_O_WORKDIR

module load cuda/7.5

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

echo /appl/cuda/7.5/samples/bin/x86_64/linux/release/bandwidthTest --device=$MYGPUS
/appl/cuda/7.5/samples/bin/x86_64/linux/release/bandwidthTest --device=$MYGPUS

