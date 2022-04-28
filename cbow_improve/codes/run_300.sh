#!/bin/bash

#SBATCH --gpus=1


module load anaconda/2021.05
source activate deeplearning
export PYTHONUNBUFFERED=1 

python main.py --config='/HOME/scz1292/run/nlp/hw1/config_300.yaml'
