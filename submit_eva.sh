#!/bin/sh
### Submit task
### General options
### –- specify queue: gpuv100, gputitanxpascal, gpuk40, gpum2050 --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J eval_cl
### -- ask for number of cores (default: 1) --
#BSUB -n 8
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
### -- request 5GB of memory
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
#BSUB -o log/cca_gpu-%J.out
#BSUB -e log/cca_gpu_%J.err
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
python3 bayes.py --method cca
python3 bayes.py --method fbcca
python3 bayes.py --method ext_cca
python3 bayes.py --method ext_fbcca
