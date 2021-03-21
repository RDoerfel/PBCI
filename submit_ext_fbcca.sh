#!/bin/sh
### get parmeters
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            length)                     length=${VALUE} ;;
            subjects)                   subjects=${VALUE} ;;
            tag)                        tag=${VALUE} ;;
            *)
    esac

done
echo "$0 executed with: "
echo "subjects = $subjects, length = $length, tag = $tag"

### General options
### –- specify queue: gpuv100, gputitanxpascal, gpuk40, gpum2050 --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ext_fbcca
### -- ask for number of cores (default: 1) --
#BSUB -n 24
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 2:00
### -- request 10GB of memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s202286@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o log/efbcca_gpu-%J.out
#BSUB -e log/efbcca_gpu_%J.err
# -- end of LSF options --

# Exits if any errors occur at any point (non-zero exit code)
set -e

# Load modules 
module load python3/3.8.4
# module load mne
# module load sklearn
# module load seaborn

# Load virtual Python environment
source pbci/bin/activate

##################################################################
# Execute your own code by replacing the sanity check code below #
##################################################################
python3 advanced_fbcca.py --length $length --subjects $subjects --tag $tag
