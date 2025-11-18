#!/bin/bash
#
#SBATCH --partition=disc
#SBATCH --gres=gpu:1
#SBATCH --output=PNet_1Mil_%J_.txt
#SBATCH --error=PNet_1Mil_%J_.txt

#SBATCH --mem=40G
#SBATCH --time=48:00:00

#SBATCH --mail-user=hongyu.xiao-1@ou.edu      
#SBATCH --mail-type=BEGIN,END,FAIL             

python TL_PNet_1Mil_ModelTrain_FromSTEAD_ver_1.py
